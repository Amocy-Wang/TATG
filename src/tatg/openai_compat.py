from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Callable
from typing import TypeVar
from urllib import error as urlerror
from urllib import request as urlrequest

from tatg.lmstudio import is_local_lmstudio_base, recover_local_lmstudio

# Transient socket errors that warrant a retry (TCP RST, ECONNRESET, etc.)
_TRANSIENT_SOCKET_ERRNOS = frozenset({
    104,  # ECONNRESET
    110,  # ETIMEDOUT
    111,  # ECONNREFUSED (transient on restarts)
    54,   # ECONNRESET on macOS/BSD
    60,   # ETIMEDOUT on macOS/BSD
    61,   # ECONNREFUSED on macOS/BSD
})
_TRANSIENT_SOCKET_TYPES = (ConnectionResetError, ConnectionAbortedError, BrokenPipeError)


_T = TypeVar("_T")


class OpenAICompatError(RuntimeError):
    pass


def _is_transient_socket_error(exc: BaseException) -> bool:
    """Return True for connection-reset / socket-close errors worth retrying."""
    if isinstance(exc, _TRANSIENT_SOCKET_TYPES):
        return True
    if isinstance(exc, OSError) and exc.errno in _TRANSIENT_SOCKET_ERRNOS:
        return True
    if isinstance(exc, urlerror.URLError):
        return _is_transient_socket_error(exc.reason) if isinstance(exc.reason, Exception) else (
            "connection reset" in str(exc.reason).lower()
            or "socket" in str(exc.reason).lower()
            or "broken pipe" in str(exc.reason).lower()
        )
    return False


def _run_with_hard_timeout(
    action: Callable[[], _T],
    timeout_seconds: float,
    *,
    operation: str,
    cancel: Callable[[], None] | None = None,
) -> _T:
    timeout_seconds = max(float(timeout_seconds), 0.001)
    result_queue: queue.Queue[tuple[bool, _T | BaseException]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result_queue.put((True, action()))
        except BaseException as exc:
            result_queue.put((False, exc))

    thread = threading.Thread(target=_worker, name=f"tatg-openai-{operation}", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        if cancel is not None:
            try:
                cancel()
            except Exception:
                pass
            thread.join(min(1.0, timeout_seconds))
        raise TimeoutError(
            f"OpenAI-compatible {operation} exceeded hard timeout of {timeout_seconds:g}s"
        )

    ok, value = result_queue.get_nowait()
    if ok:
        return value  # type: ignore[return-value]
    raise value


def chat_completion_with_usage(
    *,
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key_env: str = "OPENAI_KEY_API",
    temperature: float = 0.0,
    timeout_seconds: int = 120,
    max_tokens: int | None = None,
    allow_partial_on_timeout: bool = False,
    response_format: dict | None = None,
) -> dict:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise OpenAICompatError(f"Missing API key environment variable: {api_key_env}")

    endpoint = api_base.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        # Disable explicit reasoning traces for compatible vLLM/Qwen deployments.
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    # Optional sampling seed.
    #
    # WARNING: measured 2026-08-08, the local LM Studio setup does NOT honour
    # this -- two runs at seed 1234 produced different generated code
    # (52494aaf vs c2013bba).  Either the server ignores `seed` for this model,
    # or the nondeterminism is outside the sampler (batched inference, or
    # concurrency).  It is kept because vLLM and other backends do honour it,
    # but it does not make runs on this stack reproducible and must not be
    # reported as a fix for the 9.3pp per-target noise.
    # See docs/TATG_MEASUREMENT_NOISE.md.
    seed_raw = str(os.environ.get("TATG_LLM_SEED") or "").strip()
    if seed_raw:
        try:
            payload["seed"] = int(seed_raw)
        except ValueError:
            pass
    configured_reasoning_effort = str(
        os.environ.get("TATG_OPENAI_REASONING_EFFORT") or ""
    ).strip()
    if configured_reasoning_effort.lower() not in {
        "",
        "auto",
        "default",
        "omit",
    }:
        payload["reasoning_effort"] = configured_reasoning_effort
    elif (
        not configured_reasoning_effort
        and is_local_lmstudio_base(api_base)
    ):
        # LM Studio's OpenAI-compatible endpoint does not necessarily honor
        # ``chat_template_kwargs.enable_thinking``.  Its reasoning models can
        # then consume the complete output budget in ``reasoning_content`` and
        # return an empty ``content`` field.  The top-level OpenAI-compatible
        # control is effective for LM Studio and remains local-only so remote
        # provider payloads retain their existing behavior.
        payload["reasoning_effort"] = "none"
    configured_enable_thinking = str(os.environ.get("TATG_OPENAI_ENABLE_THINKING") or "").strip().lower()
    if configured_enable_thinking in {"0", "1", "true", "false", "yes", "no", "on", "off"}:
        payload["enable_thinking"] = configured_enable_thinking in {"1", "true", "yes", "on"}
    if max_tokens is None:
        configured_default = str(os.environ.get("TATG_OPENAI_DEFAULT_MAX_TOKENS") or "").strip()
        if configured_default:
            try:
                parsed_default = int(configured_default)
            except ValueError:
                parsed_default = 0
            if parsed_default > 0:
                max_tokens = parsed_default
    if max_tokens is not None and max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if isinstance(response_format, dict) and response_format:
        payload["response_format"] = response_format
    if os.environ.get("TATG_OPENAI_FORCE_STREAMING", "").lower() in {"1", "true", "yes", "on"}:
        allow_partial_on_timeout = True

    hard_deadline = time.monotonic() + max(float(timeout_seconds), 0.001)

    def _remaining_timeout() -> float:
        remaining = hard_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"OpenAI-compatible request exceeded hard timeout of {timeout_seconds:g}s"
            )
        return remaining

    def _request_once(request_payload: dict) -> dict:
        response_holder: dict[str, object] = {}
        response_lock = threading.Lock()

        def _close_response() -> None:
            with response_lock:
                response = response_holder.get("response")
            close = getattr(response, "close", None)
            if callable(close):
                close()

        def _do_request() -> dict:
            req = urlrequest.Request(
                endpoint,
                method="POST",
                data=json.dumps(request_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Connection": "close",
                },
            )
            response = urlrequest.urlopen(
                req,
                timeout=_remaining_timeout(),
            )
            with response_lock:
                response_holder["response"] = response
            try:
                return json.loads(response.read().decode("utf-8"))
            finally:
                _close_response()
                with response_lock:
                    response_holder.pop("response", None)

        return _run_with_hard_timeout(
            _do_request,
            _remaining_timeout(),
            operation="request",
            cancel=_close_response,
        )

    def _request_stream_once(request_payload: dict) -> dict:
        partial_chunks: list[str] = []
        usage: dict = {}
        result_queue: queue.Queue[tuple[bool, dict | BaseException]] = queue.Queue(maxsize=1)
        response_holder: dict[str, object] = {}
        response_lock = threading.Lock()

        def _close_response() -> None:
            with response_lock:
                response = response_holder.get("response")
            close = getattr(response, "close", None)
            if callable(close):
                close()

        def _extract_stream_delta(line: bytes) -> tuple[str, dict]:
            text = line.decode("utf-8", errors="ignore").strip()
            if not text:
                return "", {}
            if text.startswith("data:"):
                text = text[5:].strip()
            if not text or text == "[DONE]":
                return "", {}
            try:
                event = json.loads(text)
            except json.JSONDecodeError:
                return "", {}
            event_usage = event.get("usage") if isinstance(event, dict) else None
            content = ""
            choices = event.get("choices") if isinstance(event, dict) else None
            if isinstance(choices, list) and choices:
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") if isinstance(choice, dict) else None
                if isinstance(delta, dict):
                    reasoning_as_content = os.environ.get(
                        "TATG_STREAM_REASONING_CONTENT_AS_CONTENT",
                        "",
                    ).lower() in {"1", "true", "yes", "on"}
                    content = str(
                        delta.get("content")
                        or delta.get("text")
                        or (delta.get("reasoning_content") if reasoning_as_content else "")
                        or ""
                    )
                elif isinstance(choice.get("message"), dict):
                    reasoning_as_content = os.environ.get(
                        "TATG_STREAM_REASONING_CONTENT_AS_CONTENT",
                        "",
                    ).lower() in {"1", "true", "yes", "on"}
                    content = str(
                        choice["message"].get("content")
                        or (choice["message"].get("reasoning_content") if reasoning_as_content else "")
                        or ""
                    )
                elif "text" in choice:
                    content = str(choice.get("text") or "")
            return content, event_usage if isinstance(event_usage, dict) else {}

        def _worker() -> None:
            try:
                req = urlrequest.Request(
                    endpoint,
                    method="POST",
                    data=json.dumps(request_payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                )
                response = urlrequest.urlopen(
                    req,
                    timeout=_remaining_timeout(),
                )
                with response_lock:
                    response_holder["response"] = response
                try:
                    for raw_line in response:
                        content, event_usage = _extract_stream_delta(raw_line)
                        if content:
                            partial_chunks.append(content)
                        if event_usage:
                            usage.update(event_usage)
                finally:
                    _close_response()
                    with response_lock:
                        response_holder.pop("response", None)
                result_queue.put((True, {"content": "".join(partial_chunks), "usage": usage, "raw": {}}))
            except BaseException as exc:
                if partial_chunks:
                    result_queue.put((
                        True,
                        {
                            "content": "".join(partial_chunks),
                            "usage": usage,
                            "raw": {},
                            "partial": True,
                            "partial_reason": type(exc).__name__,
                        },
                    ))
                else:
                    result_queue.put((False, exc))

        thread = threading.Thread(target=_worker, name="tatg-openai-stream", daemon=True)
        thread.start()
        thread.join(_remaining_timeout())
        if thread.is_alive():
            partial = "".join(partial_chunks)
            _close_response()
            thread.join(0.2)
            if partial.strip():
                return {
                    "content": partial,
                    "usage": usage,
                    "raw": {},
                    "partial": True,
                    "partial_reason": "timeout",
                }
            raise TimeoutError(
                f"OpenAI-compatible streaming request exceeded hard timeout of {timeout_seconds:g}s"
            )
        ok, value = result_queue.get_nowait()
        if ok:
            result = value  # type: ignore[assignment]
            normalized_usage = result.get("usage", {}) if isinstance(result, dict) else {}
            return {
                "content": str(result.get("content") or ""),
                "usage": {
                    "prompt_tokens": int(normalized_usage.get("prompt_tokens", 0) or 0),
                    "completion_tokens": int(normalized_usage.get("completion_tokens", 0) or 0),
                    "total_tokens": int(normalized_usage.get("total_tokens", 0) or 0),
                },
                "raw": result.get("raw", {}) if isinstance(result, dict) else {},
                **(
                    {
                        "partial": True,
                        "partial_reason": str(result.get("partial_reason") or "stream_interrupted"),
                    }
                    if isinstance(result, dict) and result.get("partial")
                    else {}
                ),
            }
        raise value

    _SOCKET_RETRY_ATTEMPTS = int(os.environ.get("TATG_SOCKET_RETRY_ATTEMPTS", "3"))
    _SOCKET_RETRY_BACKOFF = float(os.environ.get("TATG_SOCKET_RETRY_BACKOFF", "2.0"))

    def _request_with_socket_retry(
        request_payload: dict,
        request_func: Callable[[dict], dict] = _request_once,
    ) -> dict:
        nonlocal hard_deadline
        last_exc: Exception | None = None
        local_recovery_attempted = False

        def _recover_local(*, force_model_reload: bool) -> None:
            nonlocal hard_deadline
            recovery = recover_local_lmstudio(
                api_base=api_base,
                model=model,
                timeout_seconds=int(
                    os.environ.get(
                        "TATG_LMSTUDIO_RECOVERY_TIMEOUT_SECONDS",
                        "240",
                    )
                    or "240"
                ),
                force_model_reload=force_model_reload,
            )
            if not bool(recovery.get("recovered")):
                return
            try:
                post_recovery_timeout = float(
                    os.environ.get(
                        "TATG_LMSTUDIO_POST_RECOVERY_TIMEOUT_SECONDS",
                        str(max(float(timeout_seconds), 120.0)),
                    )
                    or max(float(timeout_seconds), 120.0)
                )
            except (TypeError, ValueError):
                post_recovery_timeout = max(float(timeout_seconds), 120.0)
            hard_deadline = max(
                hard_deadline,
                time.monotonic() + max(1.0, post_recovery_timeout),
            )

        for attempt in range(max(1, _SOCKET_RETRY_ATTEMPTS)):
            try:
                return request_func(request_payload)
            except urlerror.HTTPError as exc:
                error_text = ""
                try:
                    error_text = exc.read().decode("utf-8", errors="ignore")
                except Exception:
                    error_text = ""
                # Preserve the body for the compatibility layer outside this
                # retry helper. HTTPError.read() is destructive.
                setattr(exc, "_tatg_error_text", error_text)
                model_unavailable = (
                    exc.code in {400, 404, 503}
                    and "model" in error_text.lower()
                    and any(
                        marker in error_text.lower()
                        for marker in ("not found", "not loaded", "unavailable", "unknown")
                    )
                )
                if (
                    model_unavailable
                    and not local_recovery_attempted
                    and is_local_lmstudio_base(api_base)
                ):
                    local_recovery_attempted = True
                    _recover_local(force_model_reload=True)
                    last_exc = exc
                    continue
                local_server_failure = (
                    exc.code in {500, 502, 503, 504}
                    and attempt > 0
                    and is_local_lmstudio_base(api_base)
                    and os.environ.get(
                        "TATG_LMSTUDIO_RELOAD_ON_5XX",
                        "1",
                    ).lower() not in {"0", "false", "no", "off"}
                )
                if local_server_failure and not local_recovery_attempted:
                    local_recovery_attempted = True
                    _recover_local(force_model_reload=True)
                    last_exc = exc
                    continue
                if exc.code in {429, 502, 503, 504} and attempt < _SOCKET_RETRY_ATTEMPTS - 1:
                    last_exc = exc
                    backoff = _SOCKET_RETRY_BACKOFF * (2 ** attempt)
                    time.sleep(min(backoff, 30.0))
                    continue
                raise
            except Exception as exc:
                if (
                    _is_transient_socket_error(exc)
                    and not local_recovery_attempted
                    and is_local_lmstudio_base(api_base)
                ):
                    local_recovery_attempted = True
                    force_reload = os.environ.get(
                        "TATG_LMSTUDIO_RELOAD_ON_SOCKET_ERROR",
                        "1",
                    ).lower() not in {"0", "false", "no", "off"}
                    _recover_local(
                        force_model_reload=force_reload,
                    )
                if _is_transient_socket_error(exc) and attempt < _SOCKET_RETRY_ATTEMPTS - 1:
                    last_exc = exc
                    backoff = _SOCKET_RETRY_BACKOFF * (2 ** attempt)
                    time.sleep(min(backoff, 30.0))
                    continue
                raise
        assert last_exc is not None
        raise last_exc

    if allow_partial_on_timeout:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_payload.setdefault("stream_options", {"include_usage": True})
        try:
            stream_body = _request_with_socket_retry(stream_payload, _request_stream_once)
            return {
                "content": stream_body.get("content", ""),
                "usage": stream_body.get("usage", {}),
                "raw": stream_body.get("raw", {}),
                **({"partial": True, "partial_reason": "timeout"} if stream_body.get("partial") else {}),
            }
        except urlerror.HTTPError:
            # Some compatible backends do not implement SSE streaming; fall back
            # to the stable non-streaming path below.
            pass
        except Exception as exc:
            if "stream" not in str(exc).lower():
                raise OpenAICompatError(f"OpenAI-compatible request failed: {exc}") from exc

    try:
        body = _request_with_socket_retry(payload)
    except urlerror.HTTPError as exc:
        # OpenAI-compatible backends differ in support for chat-template and
        # structured-output extensions. Retry only by removing the feature
        # named by a 400/422 response; ordinary provider errors retain their
        # original classification.
        should_retry_without_template_kwargs = False
        err_text = str(getattr(exc, "_tatg_error_text", "") or "")
        if not err_text:
            try:
                err_text = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                err_text = ""

        if "chat_template_kwargs" in err_text or exc.code in (400, 422):
            should_retry_without_template_kwargs = True

        if should_retry_without_template_kwargs:
            payload_without_template_kwargs = dict(payload)
            payload_without_template_kwargs.pop("chat_template_kwargs", None)
            if "reasoning_effort" in err_text.lower():
                payload_without_template_kwargs.pop(
                    "reasoning_effort",
                    None,
                )
            try:
                body = _request_with_socket_retry(payload_without_template_kwargs)
            except urlerror.HTTPError as retry_exc:
                retry_err_text = str(
                    getattr(retry_exc, "_tatg_error_text", "") or ""
                )
                if not retry_err_text:
                    try:
                        retry_err_text = retry_exc.read().decode(
                            "utf-8",
                            errors="ignore",
                        )
                    except Exception:
                        retry_err_text = ""
                response_format_rejected = bool(
                    "response_format" in retry_err_text.lower()
                    or "json_schema" in retry_err_text.lower()
                    or "structured output" in retry_err_text.lower()
                )
                reasoning_effort_rejected = (
                    "reasoning_effort" in retry_err_text.lower()
                    or "reasoning effort" in retry_err_text.lower()
                )
                if (
                    retry_exc.code in (400, 422)
                    and (
                        (
                            response_format_rejected
                            and "response_format"
                            in payload_without_template_kwargs
                        )
                        or (
                            reasoning_effort_rejected
                            and "reasoning_effort"
                            in payload_without_template_kwargs
                        )
                    )
                ):
                    payload_without_extensions = dict(
                        payload_without_template_kwargs
                    )
                    if response_format_rejected:
                        payload_without_extensions.pop(
                            "response_format",
                            None,
                        )
                    if reasoning_effort_rejected:
                        payload_without_extensions.pop(
                            "reasoning_effort",
                            None,
                        )
                    try:
                        body = _request_with_socket_retry(
                            payload_without_extensions
                        )
                    except urlerror.HTTPError as final_exc:
                        final_err_text = str(
                            getattr(
                                final_exc,
                                "_tatg_error_text",
                                "",
                            )
                            or ""
                        )
                        if not final_err_text:
                            try:
                                final_err_text = final_exc.read().decode(
                                    "utf-8",
                                    errors="ignore",
                                )
                            except Exception:
                                final_err_text = ""
                        detail = f"{final_exc}"
                        if final_err_text:
                            detail = (
                                f"{detail}: {final_err_text[:2000]}"
                            )
                        raise OpenAICompatError(
                            f"OpenAI-compatible request failed: {detail}"
                        ) from final_exc
                    except Exception as final_exc:
                        raise OpenAICompatError(
                            "OpenAI-compatible request failed: "
                            f"{final_exc}"
                        ) from final_exc
                else:
                    detail = f"{retry_exc}"
                    if retry_err_text:
                        detail = f"{detail}: {retry_err_text[:2000]}"
                    raise OpenAICompatError(
                        f"OpenAI-compatible request failed: {detail}"
                    ) from retry_exc
            except Exception as retry_exc:
                raise OpenAICompatError(f"OpenAI-compatible request failed: {retry_exc}") from retry_exc
        else:
            detail = f"{exc}"
            if err_text:
                detail = f"{detail}: {err_text[:2000]}"
            raise OpenAICompatError(f"OpenAI-compatible request failed: {detail}") from exc
    except Exception as exc:
        raise OpenAICompatError(f"OpenAI-compatible request failed: {exc}") from exc

    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    return {
        "content": body["choices"][0]["message"]["content"],
        "usage": {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
        "raw": body,
    }


def chat_completion(
    *,
    api_base: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key_env: str = "OPENAI_KEY_API",
    temperature: float = 0.0,
    timeout_seconds: int = 120,
    max_tokens: int | None = None,
) -> str:
    result = chat_completion_with_usage(
        api_base=api_base,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        api_key_env=api_key_env,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
    )
    return result["content"]
