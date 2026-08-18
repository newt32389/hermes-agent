import os
import plistlib
from unittest.mock import patch


def test_service_path_skips_nonexistent_node_modules(tmp_path):
    """Service PATH should not include node_modules/.bin if it doesn't exist."""
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    node_modules_bin = str(tmp_path / "node_modules" / ".bin")
    assert node_modules_bin not in dirs


def test_service_path_includes_node_modules_when_present(tmp_path):
    """Service PATH should include node_modules/.bin when it exists."""
    nm_bin = tmp_path / "node_modules" / ".bin"
    nm_bin.mkdir(parents=True)
    from hermes_cli.gateway import _build_service_path_dirs
    with patch("hermes_cli.gateway.get_hermes_home", return_value=tmp_path / ".hermes"):
        dirs = _build_service_path_dirs(project_root=tmp_path)
    assert str(nm_bin) in dirs


def test_launchd_service_path_omits_unsafe_missing_and_duplicate_entries(tmp_path):
    from hermes_cli.gateway import _trusted_launchd_service_path_entries

    safe = tmp_path / "safe-bin"
    unsafe = tmp_path / "unsafe-bin"
    missing = tmp_path / "missing-bin"
    safe.mkdir()
    unsafe.mkdir()
    unsafe.chmod(0o775)

    entries = _trusted_launchd_service_path_entries([
        str(safe),
        str(unsafe),
        str(missing),
        str(safe),
        "relative-bin",
    ])

    assert entries == [str(safe)]


def test_launchd_service_path_rejects_user_owned_lexical_symlink(tmp_path):
    from hermes_cli.gateway import _trusted_launchd_service_path_entries

    target = tmp_path / "safe-target"
    link = tmp_path / "user-link"
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)

    assert _trusted_launchd_service_path_entries([str(link)]) == []


def test_launchd_service_path_accepts_canonical_system_alias_when_trusted():
    from hermes_cli.gateway import _trusted_launchd_service_path_entries

    entries = _trusted_launchd_service_path_entries(["/bin", "/usr/bin"])

    assert "/bin" in entries
    assert "/usr/bin" in entries


def test_generated_profile_launchd_plist_uses_trusted_escaped_path(
    tmp_path, monkeypatch
):
    import hermes_constants
    import hermes_cli.gateway as gateway_cli

    profile_home = tmp_path / ".hermes" / "profiles" / "fitness_tracker"
    profile_home.mkdir(parents=True)
    venv = tmp_path / "runtime"
    venv_bin = venv / "bin"
    safe_tool = tmp_path / "tool&bin"
    unsafe_tool = tmp_path / "unsafe-bin"
    venv_bin.mkdir(parents=True)
    safe_tool.mkdir()
    unsafe_tool.mkdir()
    unsafe_tool.chmod(0o775)

    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(unsafe_tool), str(tmp_path / "missing"), str(safe_tool)]),
    )
    monkeypatch.setattr(gateway_cli, "get_hermes_home", lambda: profile_home)
    monkeypatch.setattr(
        hermes_constants, "get_default_hermes_root", lambda: tmp_path / ".hermes"
    )
    monkeypatch.setattr(gateway_cli, "_detect_venv_dir", lambda: venv)
    monkeypatch.setattr(
        gateway_cli, "_build_service_path_dirs", lambda: [str(venv_bin)]
    )
    monkeypatch.setattr(
        gateway_cli, "_append_node_dir_for_service", lambda entries: None
    )
    monkeypatch.setattr(gateway_cli, "get_python_path", lambda: "/usr/bin/python3")

    plist_text = gateway_cli.generate_launchd_plist()
    plist = plistlib.loads(plist_text.encode("utf-8"))
    environment = plist["EnvironmentVariables"]

    assert "tool&amp;bin" in plist_text
    assert environment["PATH"].split(os.pathsep) == [str(venv_bin), str(safe_tool)]
    assert environment["VIRTUAL_ENV"] == str(venv)
    profile_index = plist["ProgramArguments"].index("--profile")
    assert plist["ProgramArguments"][profile_index : profile_index + 2] == [
        "--profile",
        "fitness_tracker",
    ]


def test_generated_default_launchd_path_contains_only_trusted_entries():
    import hermes_cli.gateway as gateway_cli

    plist = plistlib.loads(gateway_cli.generate_launchd_plist().encode("utf-8"))
    entries = plist["EnvironmentVariables"]["PATH"].split(os.pathsep)

    assert entries
    assert entries == gateway_cli._trusted_launchd_service_path_entries(entries)

    venv = gateway_cli._detect_venv_dir()
    venv_bin = str(venv / "bin") if venv else None
    if venv_bin and gateway_cli._is_trusted_launchd_path_entry(venv_bin):
        assert entries[0] == venv_bin


def test_unsafe_installed_launchd_path_is_stale_even_when_paths_are_normalized(
    tmp_path, monkeypatch
):
    import hermes_cli.gateway as gateway_cli

    safe = tmp_path / "safe-bin"
    unsafe = tmp_path / "unsafe-bin"
    safe.mkdir()
    unsafe.mkdir()
    unsafe.chmod(0o775)
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    generated = plistlib.dumps({
        "Label": "ai.hermes.gateway",
        "EnvironmentVariables": {"PATH": str(safe)},
    }).decode("utf-8")
    plist_path.write_text(generated, encoding="utf-8")

    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: generated)

    assert gateway_cli.launchd_plist_is_current() is True

    unsafe_plist = plistlib.loads(generated.encode("utf-8"))
    unsafe_plist["EnvironmentVariables"]["PATH"] = str(unsafe)
    plist_path.write_bytes(plistlib.dumps(unsafe_plist))

    assert gateway_cli.launchd_plist_is_current() is False

    unsafe_plist["EnvironmentVariables"]["PATH"] = str(tmp_path / "missing-bin")
    plist_path.write_bytes(plistlib.dumps(unsafe_plist))

    assert gateway_cli.launchd_plist_is_current() is False


def test_safe_installed_launchd_path_variance_after_venv_is_current(
    tmp_path, monkeypatch
):
    import hermes_cli.gateway as gateway_cli

    venv_bin = tmp_path / "venv-bin"
    extra_bin = tmp_path / "extra-bin"
    venv_bin.mkdir()
    extra_bin.mkdir()
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    generated = plistlib.dumps({
        "Label": "ai.hermes.gateway",
        "EnvironmentVariables": {"PATH": str(venv_bin)},
    }).decode("utf-8")
    installed = plistlib.loads(generated.encode("utf-8"))
    installed["EnvironmentVariables"]["PATH"] = os.pathsep.join([
        str(venv_bin),
        str(extra_bin),
    ])
    plist_path.write_bytes(plistlib.dumps(installed))

    monkeypatch.setattr(gateway_cli, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gateway_cli, "generate_launchd_plist", lambda: generated)

    assert gateway_cli.launchd_plist_is_current() is True

    installed["EnvironmentVariables"]["PATH"] = os.pathsep.join([
        str(extra_bin),
        str(venv_bin),
    ])
    plist_path.write_bytes(plistlib.dumps(installed))

    assert gateway_cli.launchd_plist_is_current() is False
