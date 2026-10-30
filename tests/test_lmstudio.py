from types import SimpleNamespace

from tatg import lmstudio


def test_local_timeout_floor_applies_only_to_localhost(monkeypatch):
    monkeypatch.delenv("TATG_LMSTUDIO_ADAPT_TIMEOUTS", raising=False)
    monkeypatch.setenv("TATG_LMSTUDIO_LLM_TIMEOUT_SECONDS", "777")

    assert (
        lmstudio.local_lmstudio_timeout_seconds(
            "http://127.0.0.1:11435/v1",
            120,
            operation="llm",
        )
        == 777
    )
    assert (
        lmstudio.local_lmstudio_timeout_seconds(
            "https://provider.example/v1",
            120,
            operation="llm",
        )
        == 120
    )


def test_local_timeout_floor_preserves_disabled_e2e(monkeypatch):
    monkeypatch.delenv("TATG_LMSTUDIO_ADAPT_TIMEOUTS", raising=False)

    assert (
        lmstudio.local_lmstudio_timeout_seconds(
            "http://localhost:11435/v1",
            0,
            operation="e2e",
        )
        == 0
    )


def test_local_timeout_floor_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TATG_LMSTUDIO_ADAPT_TIMEOUTS", "0")

    assert (
        lmstudio.local_lmstudio_timeout_seconds(
            "http://localhost:11435/v1",
            120,
            operation="generation",
        )
        == 120
    )


def test_lms_path_uses_configured_executable(monkeypatch, tmp_path):
    lms = tmp_path / "lms"
    lms.write_text("", encoding="utf-8")
    monkeypatch.setenv("TATG_LMS_BIN", str(lms))
    monkeypatch.setattr(lmstudio.shutil, "which", lambda _name: None)

    assert lmstudio._lms_path() == str(lms)


def test_server_status_port_parses_lmstudio_output():
    assert lmstudio._server_status_port(
        "The server is running on port 11435."
    ) == 11435
    assert lmstudio._server_status_port("The server is stopped.") is None


def test_recovery_does_not_restart_lmstudio_for_another_local_port(
    monkeypatch,
    tmp_path,
):
    lms = tmp_path / "lms"
    lms.write_text("", encoding="utf-8")
    lms.chmod(0o755)
    commands = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        if command[1:3] == ["server", "status"]:
            return SimpleNamespace(
                returncode=0,
                stdout="The server is running on port 11435.",
            )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(lmstudio, "_lms_path", lambda: str(lms))
    monkeypatch.setattr(
        lmstudio,
        "_listed_model_ids",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ConnectionRefusedError("down")
        ),
    )
    monkeypatch.setattr(lmstudio.subprocess, "run", _run)

    result = lmstudio.recover_local_lmstudio(
        api_base="http://localhost:19999/v1",
        model="qwen/model",
        timeout_seconds=30,
    )

    assert result == {
        "attempted": True,
        "recovered": False,
        "reason": "endpoint_not_managed_by_running_lmstudio",
        "requested_port": 19999,
        "lmstudio_server_port": 11435,
    }
    assert not any(command[1:3] == ["server", "stop"] for command in commands)
    assert not any(command[1:3] == ["server", "start"] for command in commands)


def test_recovery_restarts_reported_running_server_when_endpoint_is_down(
    monkeypatch,
    tmp_path,
):
    lms = tmp_path / "lms"
    lms.write_text("", encoding="utf-8")
    lms.chmod(0o755)
    probes = iter([ConnectionRefusedError("down"), {"qwen/model"}])
    commands = []

    def _listed(*_args, **_kwargs):
        value = next(probes)
        if isinstance(value, BaseException):
            raise value
        return value

    def _run(command, **_kwargs):
        commands.append(list(command))
        if command[1:3] == ["server", "status"]:
            return SimpleNamespace(returncode=0, stdout="The server is running.")
        if command[1] == "ps":
            return SimpleNamespace(returncode=0, stdout="qwen/model IDLE")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(lmstudio, "_lms_path", lambda: str(lms))
    monkeypatch.setattr(lmstudio, "_listed_model_ids", _listed)
    monkeypatch.setattr(lmstudio.subprocess, "run", _run)

    result = lmstudio.recover_local_lmstudio(
        api_base="http://localhost:1234/v1",
        model="qwen/model",
        timeout_seconds=30,
    )

    assert result["recovered"] is True
    assert any(command[1:3] == ["server", "stop"] for command in commands)
    assert any(command[1:3] == ["server", "start"] for command in commands)


def test_forced_recovery_unloads_and_reloads_listed_model(monkeypatch, tmp_path):
    lms = tmp_path / "lms"
    lms.write_text("", encoding="utf-8")
    lms.chmod(0o755)
    commands = []

    def _run(command, **_kwargs):
        commands.append(list(command))
        if command[1:3] == ["server", "status"]:
            return SimpleNamespace(returncode=0, stdout="The server is running.")
        if command[1] == "ps":
            return SimpleNamespace(returncode=0, stdout="qwen/model IDLE")
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(lmstudio, "_lms_path", lambda: str(lms))
    monkeypatch.setattr(
        lmstudio,
        "_listed_model_ids",
        lambda *_args, **_kwargs: {"qwen/model"},
    )
    monkeypatch.setattr(lmstudio.subprocess, "run", _run)

    result = lmstudio.recover_local_lmstudio(
        api_base="http://localhost:1234/v1",
        model="qwen/model",
        timeout_seconds=30,
        force_model_reload=True,
    )

    assert result["recovered"] is True
    assert any(command[1:3] == ["unload", "qwen/model"] for command in commands)
    assert any(command[1:3] == ["load", "qwen/model"] for command in commands)
