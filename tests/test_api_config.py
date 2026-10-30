import pytest

from tatg.api_config import (
    DEFAULT_API_KEY_ENV,
    APIConfig,
    preflight_api_config,
    resolve_api_config,
    validate_api_config,
)


def test_resolve_api_config_uses_shell_environment(monkeypatch):
    monkeypatch.delenv("OPENAI_KEY_API", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    monkeypatch.delenv("API_BASE", raising=False)
    monkeypatch.setattr(
        "tatg.api_config.load_shell_api_env",
        lambda: {
            "OPENAI_KEY_API": "secret",
            "MODEL": "model-id",
            "API_BASE": "https://example.test/v1",
        },
    )

    config = resolve_api_config()

    assert config.api_key_env == DEFAULT_API_KEY_ENV
    assert config.api_key == "secret"
    assert config.model == "model-id"
    assert config.api_base == "https://example.test/v1"


def test_resolve_api_config_accepts_tatg_namespaced_environment(monkeypatch):
    monkeypatch.setattr("tatg.api_config.load_shell_api_env", lambda: {})
    config = resolve_api_config(
        env={
            "OPENAI_KEY_API": "secret",
            "TATG_MODEL": "model-id",
            "TATG_API_BASE": "https://example.test/v1",
        }
    )

    assert config.api_key == "secret"
    assert config.model == "model-id"
    assert config.api_base == "https://example.test/v1"


def test_resolve_api_config_process_environment_overrides_shell_fallback(monkeypatch):
    monkeypatch.setattr(
        "tatg.api_config.load_shell_api_env",
        lambda: {
            "OPENAI_KEY_API": "shell-secret",
            "MODEL": "remote-model",
            "API_BASE": "https://remote.example.test/v1",
        },
    )

    config = resolve_api_config(
        env={
            "OPENAI_KEY_API": "local-key",
            "MODEL": "qwen/qwen3.6-35b-a3b",
            "API_BASE": "http://127.0.0.1:11435/v1",
        }
    )

    assert config.api_key == "local-key"
    assert config.model == "qwen/qwen3.6-35b-a3b"
    assert config.api_base == "http://127.0.0.1:11435/v1"


def test_validate_api_config_rejects_chat_completions_endpoint():
    config = APIConfig(
        api_key_env="OPENAI_KEY_API",
        api_key="secret",
        model="model-id",
        api_base="https://example.test/v1/chat/completions",
    )

    with pytest.raises(ValueError, match="not the /chat/completions endpoint"):
        validate_api_config(config)


def test_preflight_api_config_uses_wider_default_token_budget(monkeypatch):
    seen = {}

    def _fake_chat_completion_with_usage(**kwargs):
        seen.update(kwargs)
        return {"content": "OK", "usage": {}}

    monkeypatch.setattr("tatg.api_config.chat_completion_with_usage", _fake_chat_completion_with_usage)
    config = APIConfig(
        api_key_env="OPENAI_KEY_API",
        api_key="secret",
        model="model-id",
        api_base="https://example.test/v1",
    )

    preflight_api_config(config, timeout_seconds=5)

    assert seen["max_tokens"] == 256


def test_preflight_api_config_allows_token_budget_override(monkeypatch):
    seen = {}

    def _fake_chat_completion_with_usage(**kwargs):
        seen.update(kwargs)
        return {"content": "OK", "usage": {}}

    monkeypatch.setenv("TATG_API_PREFLIGHT_MAX_TOKENS", "64")
    monkeypatch.setattr("tatg.api_config.chat_completion_with_usage", _fake_chat_completion_with_usage)
    config = APIConfig(
        api_key_env="OPENAI_KEY_API",
        api_key="secret",
        model="model-id",
        api_base="https://example.test/v1",
    )

    preflight_api_config(config, timeout_seconds=5)

    assert seen["max_tokens"] == 64
