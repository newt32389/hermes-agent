"""Integration coverage for profile-local MCP discovery in slash workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import yaml

_PARENT_RESPONSE_HEADROOM_S = 3.0
_DEFAULT_PARENT_TIMEOUT_S = 45.0
_HOSTED_SETUP_TIMEOUT_S = 30.0

_mcp_server_mod = pytest.importorskip("mcp.server")

if not hasattr(_mcp_server_mod, "MCPServer"):
    # `mcp.server.MCPServer` replaced `mcp.server.fastmcp.FastMCP` in mcp 2.0.
    # Skip rather than fail on a FastMCP-era SDK: the probe below is written
    # against the 2.x API, and the pinned version provides it.
    pytest.skip(
        "profile-local MCP discovery probe requires mcp >= 2.0 (MCPServer)",
        allow_module_level=True,
    )


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    (
        "discovery_delay_s",
        "slash_timeout_s",
        "pre_gate_delay_s",
        "parent_deadline",
        "warm_worker",
        "expect_mcp_tool",
    ),
    [
        pytest.param(
            3.0, None, 0.0, False, False, True, id="cold-default-shows-delayed-mcp"
        ),
        pytest.param(
            3.0,
            5.0,
            0.0,
            True,
            True,
            None,
            id="minimum-parent-deadline-returns-before-timeout",
        ),
        pytest.param(
            3.0,
            7.0,
            5.5,
            False,
            True,
            True,
            id="long-lived-worker-gets-fresh-request-budget",
        ),
    ],
)
def test_delayed_profile_local_mcp_tool_is_visible_in_slash_worker(
    tmp_path,
    discovery_delay_s,
    slash_timeout_s,
    pre_gate_delay_s,
    parent_deadline,
    warm_worker,
    expect_mcp_tool,
):
    """`/tools` reports a completed MCP or returns safely before its parent timeout."""
    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    marker = "profile-local-61922"
    gate = tmp_path / "start-mcp"
    server_started = tmp_path / "mcp-server-started"
    server = tmp_path / "mcp_probe.py"
    server.write_text(
        textwrap.dedent(
            f"""
            from mcp.server import MCPServer
            from pathlib import Path
            import time

            Path({str(server_started)!r}).write_text("started", encoding="utf-8")
            gate = Path({str(gate)!r})
            while not gate.exists():
                time.sleep(0.01)
            time.sleep({discovery_delay_s})

            mcp = MCPServer("profileprobe")

            @mcp.tool()
            def hermes_61922_profile_probe() -> str:
                return {marker!r}

            if __name__ == "__main__":
                mcp.run(transport="stdio")
            """
        ),
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({
            "mcp_servers": {
                "profileprobe": {
                    "enabled": True,
                    "command": sys.executable,
                    "args": [str(server)],
                }
            }
        }),
        encoding="utf-8",
    )

    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            env.pop(key)
    env["HERMES_HOME"] = str(profile_home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["HERMES_SLASH_WATCHDOG_GRACE_S"] = "0"
    env["HERMES_SLASH_WATCHDOG_POLL_S"] = "0.05"
    if slash_timeout_s is not None:
        env["HERMES_TUI_SLASH_TIMEOUT_S"] = str(slash_timeout_s)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "tui_gateway.slash_worker",
            "--session-key",
            "agent:main:tui:dm:mcp-profile-test",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    output: queue.Queue[str] = queue.Queue()
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        stdout = proc.stdout

        def _drain_stdout() -> None:
            for line in stdout:
                output.put(line)

        threading.Thread(target=_drain_stdout, daemon=True).start()

        def _request(request_id, command, *, deadline=None, timeout=10.0):
            request = {"id": request_id, "command": command}
            if deadline is not None:
                request["deadline_monotonic"] = deadline
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
            expires_at = time.monotonic() + timeout
            while True:
                remaining = expires_at - time.monotonic()
                if remaining <= 0:
                    pytest.fail(f"slash worker did not respond to {command}")
                try:
                    response = json.loads(output.get(timeout=remaining))
                except queue.Empty:
                    pytest.fail(f"slash worker did not respond to {command}")
                if response.get("id") == request_id:
                    return response

        if warm_worker:
            # `/help` establishes that the persistent worker is fully ready
            # while the MCP child remains deterministically blocked behind the
            # test gate.
            ready = _request(1, "/help")
            assert ready["ok"] is True, ready
            # The marker is test setup, not the `/tools` behavior under test.
            # Hosted 16-way CI can delay the child process substantially.
            startup_deadline = time.monotonic() + _HOSTED_SETUP_TIMEOUT_S
            while not server_started.exists() and time.monotonic() < startup_deadline:
                time.sleep(0.01)
            assert server_started.exists(), "MCP child did not reach its startup gate"
            if pre_gate_delay_s:
                time.sleep(pre_gate_delay_s)
            gate.write_text("go", encoding="utf-8")
        else:
            # Production's on-demand path starts the worker and immediately
            # sends `/tools`; do not hide its cold-start cost behind `/help`.
            gate.write_text("go", encoding="utf-8")
        started = time.monotonic()
        # Match `_SlashWorker.run`: parent timeout minus its 3s response/IPC
        # reserve, not the older 1s test-only allowance.
        deadline = (
            started + slash_timeout_s - _PARENT_RESPONSE_HEADROOM_S
            if parent_deadline
            else None
        )
        response = _request(
            2 if warm_worker else 1,
            "/tools",
            deadline=deadline,
            timeout=(slash_timeout_s or _DEFAULT_PARENT_TIMEOUT_S) + 2,
        )
        elapsed = time.monotonic() - started
        assert response["ok"] is True, response
        tool_name = "mcp__profileprobe__hermes_61922_profile_probe"
        if expect_mcp_tool is not None:
            assert (tool_name in response["output"]) is expect_mcp_tool
        if slash_timeout_s is not None:
            assert elapsed < slash_timeout_s - 0.25
    finally:
        # The worker owns the MCP stdio child; terminating it closes that
        # child's transport too. Keep the bounded kill fallback for a stuck
        # worker so this timing regression cannot leak a process.
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
