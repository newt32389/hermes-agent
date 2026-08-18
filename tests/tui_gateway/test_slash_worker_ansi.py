"""The slash worker feeds desktop chat bubbles, which render plain text — so
any ANSI a worker-routed command emits (e.g. /journey's own Rich Console) must
be stripped from the worker's return value."""

from __future__ import annotations

from contextlib import nullcontext


class _FakeCLI:
    console = None

    def process_command(self, cmd: str) -> None:
        import sys

        sys.stdout.write("\x1b[38;2;1;2;3mcolored\x1b[0m plain")


def test_run_strips_ansi_from_output():
    from tui_gateway import slash_worker

    out = slash_worker._run(_FakeCLI(), "/anything")

    assert "\x1b[" not in out
    assert out == "colored plain"


def test_tools_listing_restores_persistent_toolsets(monkeypatch):
    """`/tools` may refresh its listing, but must not mutate later commands."""
    from hermes_cli import mcp_startup
    from hermes_cli import config as config_mod
    from hermes_cli import tools_config
    from tui_gateway import slash_worker

    seen = []

    class _ToolsCLI:
        console = None
        enabled_toolsets = None

        def process_command(self, _cmd: str) -> None:
            seen.append(self.enabled_toolsets)

    monkeypatch.setattr(mcp_startup, "join_mcp_discovery", lambda **_kw: True)
    monkeypatch.setattr(
        mcp_startup, "bounded_mcp_discovery_wait", lambda _timeout: nullcontext()
    )
    monkeypatch.setattr(config_mod, "load_config", lambda: {})
    monkeypatch.setattr(
        tools_config, "_get_platform_tools", lambda _config, _platform: {"mcp-demo"}
    )

    cli = _ToolsCLI()
    slash_worker._run(cli, "/tools")

    assert seen == [["mcp-demo"]]
    assert cli.enabled_toolsets is None
