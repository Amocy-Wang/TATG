import json
import io
import threading
import time
from urllib import error as urlerror

import pytest

from tatg.openai_compat import OpenAICompatError, chat_completion_with_usage


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self._body).encode("utf-8")


class _FakeStreamResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"content":"{\\"new_tests\\":["}}]}\n'
        time.sleep(0.2)
        yield b'data: {"choices":[{"delta":{"content":"late"}}]}\n'


class _FakeTextStreamResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield b'data: {"choices":[{"text":"hello"}]}\n'
        yield b'data: {"choices":[{"delta":{"text":" world"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n'
        yield b"data: [DONE]\n"


class _FakeReasoningStreamResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        yield b'data: {"choices":[{"delta":{"reasoning_content":"thinking"}}]}\n'
        yield b'data: {"choices":[{"delta":{"content":"final"}}],"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n'
        yield b"data: [DONE]\n"


class _CloseReleasedResponse:
    def __init__(self):
        self.closed = threading.Event()

    def read(self):
        self.closed.wait(5)
        raise OSError("response closed")

    def close(self):
        self.closed.set()


def test_chat_completion_with_usage_returns_content_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")

    def _urlopen(req, timeout):
        assert timeout > 0
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        )

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    result = chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )

    assert result["content"] == "ok"
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}


def test_chat_completion_with_usage_uses_env_default_max_tokens(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    monkeypatch.setenv("TATG_OPENAI_DEFAULT_MAX_TOKENS", "2048")
    seen = []

    def _urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        seen.append(payload.get("max_tokens"))
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )
    chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
        max_tokens=32,
    )

    assert seen == [2048, 32]


def test_chat_completion_with_usage_can_send_top_level_enable_thinking(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    monkeypatch.setenv("TATG_OPENAI_ENABLE_THINKING", "false")
    seen = {}

    def _urlopen(req, timeout):
        seen.update(json.loads(req.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )

    assert seen["enable_thinking"] is False


def test_local_lmstudio_disables_reasoning_without_changing_remote_payloads(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    payloads = []

    def _urlopen(req, timeout):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(
            {"choices": [{"message": {"content": "{\"ok\":true}"}}]}
        )

    monkeypatch.setattr(
        "tatg.openai_compat.urlrequest.urlopen",
        _urlopen,
    )

    for api_base in (
        "http://127.0.0.1:11435/v1",
        "https://api.example.test/v1",
    ):
        chat_completion_with_usage(
            api_base=api_base,
            model="model",
            system_prompt="system",
            user_prompt="user",
            timeout_seconds=5,
        )

    assert payloads[0]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in payloads[1]


def test_local_lmstudio_reasoning_effort_can_be_overridden_or_omitted(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    payloads = []

    def _urlopen(req, timeout):
        payloads.append(json.loads(req.data.decode("utf-8")))
        return _FakeResponse(
            {"choices": [{"message": {"content": "{\"ok\":true}"}}]}
        )

    monkeypatch.setattr(
        "tatg.openai_compat.urlrequest.urlopen",
        _urlopen,
    )

    monkeypatch.setenv("TATG_OPENAI_REASONING_EFFORT", "low")
    chat_completion_with_usage(
        api_base="http://127.0.0.1:11435/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )
    monkeypatch.setenv("TATG_OPENAI_REASONING_EFFORT", "omit")
    chat_completion_with_usage(
        api_base="http://127.0.0.1:11435/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )

    assert payloads[0]["reasoning_effort"] == "low"
    assert "reasoning_effort" not in payloads[1]


def test_chat_completion_with_usage_forwards_optional_response_format(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    seen = {}

    def _urlopen(req, timeout):
        seen.update(json.loads(req.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "payload",
            "schema": {"type": "object"},
        },
    }

    chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
        response_format=response_format,
    )

    assert seen["response_format"] == response_format


def test_chat_completion_falls_back_when_backend_rejects_response_format(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    payloads = []

    def _urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        payloads.append(payload)
        if len(payloads) == 1:
            raise urlerror.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs={},
                fp=io.BytesIO(b'{"error":"chat_template_kwargs unsupported"}'),
            )
        if len(payloads) == 2:
            raise urlerror.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs={},
                fp=io.BytesIO(
                    b'{"error":"response_format json_schema unsupported"}'
                ),
            )
        return _FakeResponse(
            {"choices": [{"message": {"content": "{\"ok\":true}"}}]}
        )

    monkeypatch.setattr(
        "tatg.openai_compat.urlrequest.urlopen",
        _urlopen,
    )

    result = chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "payload",
                "schema": {"type": "object"},
            },
        },
    )

    assert result["content"] == '{"ok":true}'
    assert "chat_template_kwargs" in payloads[0]
    assert "chat_template_kwargs" not in payloads[1]
    assert "response_format" in payloads[1]
    assert "response_format" not in payloads[2]


def test_local_lmstudio_falls_back_when_reasoning_effort_is_rejected(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    payloads = []

    def _urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        payloads.append(payload)
        if len(payloads) == 1:
            raise urlerror.HTTPError(
                req.full_url,
                400,
                "Bad Request",
                hdrs={},
                fp=io.BytesIO(
                    b'{"error":"reasoning_effort unsupported"}'
                ),
            )
        return _FakeResponse(
            {"choices": [{"message": {"content": "{\"ok\":true}"}}]}
        )

    monkeypatch.setattr(
        "tatg.openai_compat.urlrequest.urlopen",
        _urlopen,
    )

    result = chat_completion_with_usage(
        api_base="http://127.0.0.1:11435/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )

    assert result["content"] == '{"ok":true}'
    assert payloads[0]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in payloads[1]


def test_chat_completion_with_usage_retries_transient_http_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    monkeypatch.setenv("TATG_SOCKET_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("TATG_SOCKET_RETRY_BACKOFF", "0")
    calls = {"count": 0}

    def _urlopen(req, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise urlerror.HTTPError(req.full_url, 502, "Bad Gateway", hdrs={}, fp=None)
        return _FakeResponse(
            {
                "choices": [{"message": {"content": "ok-after-retry"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    result = chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )

    assert calls["count"] == 2
    assert result["content"] == "ok-after-retry"


def test_local_lmstudio_reloads_after_repeated_server_errors(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    monkeypatch.setenv("TATG_SOCKET_RETRY_ATTEMPTS", "3")
    monkeypatch.setenv("TATG_SOCKET_RETRY_BACKOFF", "0")
    recoveries = []
    calls = {"count": 0}

    def _urlopen(req, timeout):
        calls["count"] += 1
        if calls["count"] < 3:
            raise urlerror.HTTPError(
                req.full_url,
                503,
                "Service Unavailable",
                hdrs={},
                fp=io.BytesIO(b'{"error":"inference worker unavailable"}'),
            )
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok-after-reload"}}]}
        )

    def _recover(**kwargs):
        recoveries.append(dict(kwargs))
        return {"attempted": True, "recovered": True}

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)
    monkeypatch.setattr("tatg.openai_compat.recover_local_lmstudio", _recover)

    result = chat_completion_with_usage(
        api_base="http://localhost:1234/v1",
        model="qwen/model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )

    assert result["content"] == "ok-after-reload"
    assert calls["count"] == 3
    assert len(recoveries) == 1
    assert recoveries[0]["force_model_reload"] is True


def test_local_lmstudio_socket_failure_reloads_before_retry(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    monkeypatch.setenv("TATG_SOCKET_RETRY_ATTEMPTS", "2")
    monkeypatch.setenv("TATG_SOCKET_RETRY_BACKOFF", "0")
    recoveries = []
    calls = {"count": 0}

    def _urlopen(req, timeout):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConnectionResetError("local inference worker reset")
        return _FakeResponse(
            {"choices": [{"message": {"content": "ok-after-socket-reload"}}]}
        )

    def _recover(**kwargs):
        recoveries.append(dict(kwargs))
        return {"attempted": True, "recovered": True}

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)
    monkeypatch.setattr("tatg.openai_compat.recover_local_lmstudio", _recover)

    result = chat_completion_with_usage(
        api_base="http://localhost:11435/v1",
        model="qwen/model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )

    assert result["content"] == "ok-after-socket-reload"
    assert calls["count"] == 2
    assert len(recoveries) == 1
    assert recoveries[0]["force_model_reload"] is True


def test_chat_completion_with_usage_enforces_hard_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")

    def _urlopen(req, timeout):
        time.sleep(0.2)
        return _FakeResponse({"choices": [{"message": {"content": "late"}}]})

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    started = time.monotonic()
    with pytest.raises(OpenAICompatError, match="exceeded hard timeout"):
        chat_completion_with_usage(
            api_base="http://localhost:8000/v1",
            model="model",
            system_prompt="system",
            user_prompt="user",
            timeout_seconds=0.01,
        )

    assert time.monotonic() - started < 0.15


def test_hard_timeout_closes_inflight_http_response(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    response = _CloseReleasedResponse()

    monkeypatch.setattr(
        "tatg.openai_compat.urlrequest.urlopen",
        lambda req, timeout: response,
    )

    with pytest.raises(OpenAICompatError, match="exceeded hard timeout"):
        chat_completion_with_usage(
            api_base="http://localhost:8000/v1",
            model="model",
            system_prompt="system",
            user_prompt="user",
            timeout_seconds=0.01,
        )

    assert response.closed.wait(0.2)


def test_chat_completion_with_usage_returns_partial_stream_on_timeout(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")

    def _urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return _FakeStreamResponse()

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    result = chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=0.03,
        allow_partial_on_timeout=True,
    )

    assert result["partial"] is True
    assert result["partial_reason"] == "timeout"
    assert result["content"] == '{"new_tests":['


def test_chat_completion_with_usage_stream_accepts_text_delta_shapes(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")

    def _urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        assert payload["stream_options"]["include_usage"] is True
        return _FakeTextStreamResponse()

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    result = chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=1,
        allow_partial_on_timeout=True,
    )

    assert result["content"] == "hello world"
    assert result["usage"] == {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}


def test_chat_completion_with_usage_stream_ignores_reasoning_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    monkeypatch.setenv("TATG_OPENAI_FORCE_STREAMING", "1")

    def _urlopen(req, timeout):
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["stream"] is True
        return _FakeReasoningStreamResponse()

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    result = chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=1,
    )

    assert result["content"] == "final"


def test_chat_completion_with_usage_stream_can_opt_into_reasoning_content(monkeypatch):
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    monkeypatch.setenv("TATG_OPENAI_FORCE_STREAMING", "1")
    monkeypatch.setenv("TATG_STREAM_REASONING_CONTENT_AS_CONTENT", "1")

    def _urlopen(req, timeout):
        return _FakeReasoningStreamResponse()

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)

    result = chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=1,
    )

    assert result["content"] == "thinkingfinal"


def _seed_payload(monkeypatch, seed_value):
    """Return the payload built for one call, with TATG_LLM_SEED set."""
    monkeypatch.setenv("OPENAI_KEY_API", "test-key")
    if seed_value is None:
        monkeypatch.delenv("TATG_LLM_SEED", raising=False)
    else:
        monkeypatch.setenv("TATG_LLM_SEED", seed_value)
    seen = {}

    def _urlopen(req, timeout):
        seen.update(json.loads(req.data.decode("utf-8")))
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("tatg.openai_compat.urlrequest.urlopen", _urlopen)
    chat_completion_with_usage(
        api_base="http://localhost:8000/v1",
        model="model",
        system_prompt="system",
        user_prompt="user",
        timeout_seconds=5,
    )
    return seen


def test_llm_seed_is_threaded_when_set(monkeypatch):
    """A seed makes a null A/B reproduce, which the acceptance gates require.

    Measured on the S1 baseline arm: re-running one configuration unchanged
    moved per-target branch coverage by a mean 7.1pp (sd 9.3pp), so 7 of 17
    targets "regressed" past the 3pp gate with no code change at all.
    """

    assert _seed_payload(monkeypatch, "1234")["seed"] == 1234


def test_llm_seed_absent_by_default(monkeypatch):
    assert "seed" not in _seed_payload(monkeypatch, None)


def test_llm_seed_ignores_non_integer_rather_than_failing_the_call(monkeypatch):
    assert "seed" not in _seed_payload(monkeypatch, "not-an-int")
