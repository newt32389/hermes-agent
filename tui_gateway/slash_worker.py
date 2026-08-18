"""Persistent slash-command worker — one HermesCLI per TUI session.

Protocol: reads JSON lines from stdin {id, command}, writes {id, ok, output|error} to stdout.
"""

# Stop a ``utils/`` (or ``proxy/``, ``ui/``) package in the launch directory
# from shadowing Hermes's own top-level modules.  This worker is spawned as
# ``-m tui_gateway.slash_worker`` and inherits the user's CWD, so the ``import
# cli`` below would otherwise resolve ``utils`` to a colliding local package
# and crash the child in a retry loop (issue #51286).  ``hermes_bootstrap``
# lives at the repo root, so importing it is safe before the guard runs (its
# name won't collide with a user package), and it owns the canonical
# path-hardening logic shared with the other entry points — #51693 added the
# guard to ``entry.py``/``acp_adapter/entry.py`` but missed this child.
import hermes_bootstrap

hermes_bootstrap.harden_import_path()

import argparse
import contextlib
import io
import json
import logging
import math
import os
import sys
import threading
import time

import cli as cli_mod
from cli import HermesCLI
from tui_gateway._stdin_recovery import handle_spurious_eof
from rich.console import Console

# Env-overridable so the integration test can drive sub-second timing.
def _env_float(name: str, default: float) -> float:
    """Parse a float env knob, falling back to ``default`` on absent/malformed
    values. A bare ``float(os.environ.get(...))`` would raise ValueError at
    import time on a typo (e.g. ``HERMES_SLASH_WATCHDOG_POLL_S=2s``) and kill
    the worker before it can serve a single command."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


_WATCHDOG_POLL_S = max(0.05, _env_float("HERMES_SLASH_WATCHDOG_POLL_S", 2.0))
_ORPHAN_GRACE_S = max(0.0, _env_float("HERMES_SLASH_WATCHDOG_GRACE_S", 5.0))
# Keep this aligned with the gateway's off-critical-path late-refresh bound.
# Unlike normal slash-worker startup, this only runs on a user's explicit
# `/tools` request, whose purpose is to report the registry as completely as
# possible.
_TOOLS_MCP_DISCOVERY_WAIT_S = 30.0
# The parent starts this deadline when it writes the request, not when this
# worker starts.  Reserve time for command rendering and the JSON response so
# `/tools` never turns a slow MCP into the parent's 5030 timeout.
_TOOLS_RESPONSE_SAFETY_MARGIN_S = 3.0
_in_flight = threading.Event()  # set while a command is executing
logger = logging.getLogger(__name__)


def _is_orphaned(original_ppid, getppid=os.getppid) -> bool:
    """Return whether this worker no longer has its original POSIX parent."""
    return getppid() != original_ppid


def _tools_mcp_discovery_wait_s(request_deadline: float) -> float:
    """Return the `/tools` completion wait within the parent's request budget."""
    return max(0.0, min(_TOOLS_MCP_DISCOVERY_WAIT_S, request_deadline - time.monotonic()))


def _slash_request_deadline(received_at: float) -> float:
    """Set the per-request deadline to the parent's timeout minus response headroom."""
    parent_timeout = max(5.0, _env_float("HERMES_TUI_SLASH_TIMEOUT_S", 45.0))
    return received_at + parent_timeout - _TOOLS_RESPONSE_SAFETY_MARGIN_S


def _prepare_slash_worker_runtime() -> None:
    """Start bounded MCP discovery before HermesCLI snapshots tools.

    Each slash_worker child is its own process — the parent ``hermes serve``
    discovery thread does not populate this registry (issue #61891).
    """
    import logging

    from hermes_cli.mcp_startup import (
        start_background_mcp_discovery,
        wait_for_mcp_discovery,
    )

    logger = logging.getLogger(__name__)
    start_background_mcp_discovery(
        logger=logger,
        thread_name="slash-worker-mcp-discovery",
    )
    wait_for_mcp_discovery()


def _start_parent_death_watchdog(original_ppid) -> None:
    def _loop():
        while not _is_orphaned(original_ppid):
            time.sleep(_WATCHDOG_POLL_S)
        deadline = time.monotonic() + _ORPHAN_GRACE_S
        while _in_flight.is_set() and time.monotonic() < deadline:
            time.sleep(0.05)  # let an in-flight command finish/flush
        os._exit(0)

    threading.Thread(target=_loop, daemon=True).start()


def _run(
    cli: HermesCLI, command: str, *, request_deadline: float | None = None
) -> str:
    cmd = (command or "").strip()
    if not cmd:
        return ""
    if not cmd.startswith("/"):
        cmd = f"/{cmd}"
    if request_deadline is None:
        request_deadline = _slash_request_deadline(time.monotonic())

    # HermesCLI captures the initial registry after the normal, short
    # interactive discovery wait.  A cold profile-local MCP can finish after
    # that snapshot, so an immediate `/tools` otherwise reports an incomplete
    # registry for the lifetime of this persistent worker.  Do not wait for
    # ordinary commands: that would lengthen interactive/model startup and
    # risks changing a conversation's tool snapshot.  `/tools` is a read-only
    # registry inspection, so it can safely wait for the same bounded
    # completion window used by the gateway's late refresh.
    # `/tools enable|disable|...` are state-changing commands. Only the exact
    # read-only listing gets the late discovery wait and temporary override.
    is_tools_command = cmd.lower() == "/tools"
    tools_wait_s = _tools_mcp_discovery_wait_s(request_deadline) if is_tools_command else None

    buf = io.StringIO()

    # Rich Console captures its file handle at construction time, so
    # contextlib.redirect_stdout won't affect it. Swap the console's
    # underlying file to our buffer so self.console.print() is captured.
    cli.console = Console(file=buf, force_terminal=True, width=120)

    old = getattr(cli_mod, "_cprint", None)
    listing_toolsets = None
    listing_toolsets_overridden = False
    if old is not None:
        cli_mod._cprint = lambda text: print(text)

    try:
        if tools_wait_s is None:
            wait_scope = contextlib.nullcontext()
        else:
            from hermes_cli.mcp_startup import bounded_mcp_discovery_wait

            wait_scope = bounded_mcp_discovery_wait(tools_wait_s)
        with wait_scope:
            if tools_wait_s is not None:
                from hermes_cli.mcp_startup import join_mcp_discovery

                # show_tools() reaches cli.get_tool_definitions(), which normally
                # does another startup wait.  Keep both joins within this one
                # request deadline rather than spending the parent timeout twice.
                if join_mcp_discovery(timeout=tools_wait_s):
                    # HermesCLI resolved this once during startup, before a
                    # cold MCP necessarily registered its per-server toolset.
                    # `/tools` is a read-only inspection command, so refresh
                    # that listing-only selection after discovery lands.
                    from hermes_cli.config import load_config
                    from hermes_cli.tools_config import _get_platform_tools

                    listing_toolsets = cli.enabled_toolsets
                    cli.enabled_toolsets = sorted(
                        _get_platform_tools(load_config(), "cli")
                    )
                    listing_toolsets_overridden = True
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                cli.process_command(cmd)
    finally:
        if listing_toolsets_overridden:
            cli.enabled_toolsets = listing_toolsets
        if old is not None:
            cli_mod._cprint = old

    # Desktop chat bubbles render plain text, not ANSI. A worker-routed command
    # that emits Rich color (e.g. /journey building its own Console, which picks
    # up truecolor from the gateway's inherited COLORTERM) would otherwise leak
    # raw escapes; strip them at the single choke point. (The TUI opens /journey
    # as an overlay, so it never travels this path.)
    from tools.ansi_strip import strip_ansi

    return strip_ansi(buf.getvalue().rstrip())


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--session-key", required=True)
    p.add_argument("--model", default="")
    args = p.parse_args()

    os.environ["HERMES_SESSION_KEY"] = args.session_key
    os.environ["HERMES_INTERACTIVE"] = "1"

    # Start before the (hundreds-of-ms) HermesCLI build — that window is itself
    # an orphan risk if the gateway dies mid-spawn.
    orig_ppid = os.getppid()
    _start_parent_death_watchdog(orig_ppid)
    _prepare_slash_worker_runtime()

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cli = HermesCLI(model=args.model or None, compact=True, resume=args.session_key, verbose=False)

    # Spurious stdin-EOF recovery (same O_NONBLOCK shared file-description
    # issue as the gateway entry point — any child inheriting fd 0 can flip
    # the flag and launder EAGAIN into an apparent EOF).
    _sw_recovery_times: list[float] = []

    def _sw_log(reason: str) -> None:
        print(f"[slash-worker] {reason}", file=sys.stderr, flush=True)

    while True:
        raw = sys.stdin.readline()
        if not raw:
            if not handle_spurious_eof(_sw_recovery_times, _sw_log):
                break
            continue

        line = raw.strip()
        if not line:
            continue

        _in_flight.set()
        rid = None
        try:
            received_at = time.monotonic()
            req = json.loads(line)
            rid = req.get("id")
            request_deadline = req.get("deadline_monotonic")
            if not isinstance(request_deadline, (int, float)) or not math.isfinite(
                request_deadline
            ):
                request_deadline = _slash_request_deadline(received_at)
            out = _run(
                cli,
                req.get("command", ""),
                request_deadline=request_deadline,
            )
            sys.stdout.write(json.dumps({"id": rid, "ok": True, "output": out}) + "\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(json.dumps({"id": rid, "ok": False, "error": str(e)}) + "\n")
            sys.stdout.flush()
        finally:
            _in_flight.clear()
            # Workers persist for the TUI session, so release allocator pages at
            # the same command boundary as other long-lived gateway processes.
            # trim_memory's shared cooldown coalesces this with nearby activity.
            try:
                from hermes_cli.mem_trim import trim_memory

                trim_memory(reason="slash worker command completion")
            except Exception as exc:
                # debug, not warning — a persistent failure would repeat on
                # every slash command forever.
                logger.debug(
                    "slash worker memory trim failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )


if __name__ == "__main__":
    main()
