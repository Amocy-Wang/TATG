from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from urllib.parse import urlparse

from tatg.openai_compat import OpenAICompatError, chat_completion_with_usage


DEFAULT_API_KEY_ENV = "OPENAI_KEY_API"
REQUIRED_API_ENV_VARS = (
    DEFAULT_API_KEY_ENV,
    "MODEL",
    "API_BASE",
    "TATG_MODEL",
    "TATG_API_BASE",
)


@dataclass(frozen=True)
class APIConfig:
    api_key_env: str
    api_key: str
    model: str
    api_base: str


def _shell_env_command(shell_name: str) -> list[str]:
    shell = os.environ.get("SHELL") or shell_name
    if "zsh" in shell:
        return [shell, "-ic", "env"]
    if "bash" in shell:
        return [shell, "-ic", "env"]
    return ["/bin/zsh", "-ic", "env"]


def load_shell_api_env() -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        completed = subprocess.run(
            _shell_env_command("/bin/zsh"),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
            timeout=8,
        )
    except Exception:
        return values
    for line in completed.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in REQUIRED_API_ENV_VARS:
            values[key] = value
    return values


def resolve_api_config(
    *,
    api_base: str | None = None,
    model: str | None = None,
    api_key_env: str | None = None,
    env: dict[str, str] | None = None,
) -> APIConfig:
    merged = dict(os.environ if env is None else env)
    for key, value in load_shell_api_env().items():
        # Interactive-shell values are a fallback for desktop/non-login
        # launches.  Explicit process or caller-provided values must win so a
        # local-model run cannot be silently redirected to a stale remote API.
        if value.strip() and not str(merged.get(key) or "").strip():
            merged[key] = value

    resolved_api_key_env = (api_key_env or DEFAULT_API_KEY_ENV).strip() or DEFAULT_API_KEY_ENV
    resolved_api_base = (
        api_base
        or merged.get("TATG_API_BASE")
        or merged.get("API_BASE")
        or ""
    ).strip()
    resolved_model = (
        model
        or merged.get("TATG_MODEL")
        or merged.get("MODEL")
        or ""
    ).strip()
    resolved_api_key = (merged.get(resolved_api_key_env) or "").strip()
    return APIConfig(
        api_key_env=resolved_api_key_env,
        api_key=resolved_api_key,
        model=resolved_model,
        api_base=resolved_api_base,
    )


def validate_api_config(config: APIConfig) -> None:
    missing: list[str] = []
    if not config.api_key:
        missing.append(config.api_key_env)
    if not config.model:
        missing.append("MODEL")
    if not config.api_base:
        missing.append("API_BASE")
    if missing:
        raise ValueError(
            "Missing required API environment variable(s): " + ", ".join(missing)
        )

    parsed = urlparse(config.api_base)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("API_BASE must be an absolute http(s) base URL, for example https://host/v1")
    if parsed.path.rstrip("/").endswith("/chat/completions"):
        raise ValueError("API_BASE must be the API base URL, not the /chat/completions endpoint")


def apply_api_config_to_env(config: APIConfig, env: dict[str, str] | None = None) -> dict[str, str]:
    target = os.environ if env is None else env
    target[config.api_key_env] = config.api_key
    target["MODEL"] = config.model
    target["API_BASE"] = config.api_base
    return target


def preflight_api_config(config: APIConfig, *, timeout_seconds: int = 10) -> None:
    validate_api_config(config)
    apply_api_config_to_env(config)
    preflight_max_tokens = 256
    configured_max_tokens = str(os.environ.get("TATG_API_PREFLIGHT_MAX_TOKENS") or "").strip()
    if configured_max_tokens:
        try:
            parsed = int(configured_max_tokens)
        except ValueError:
            parsed = 0
        if parsed > 0:
            preflight_max_tokens = parsed
    try:
        chat_completion_with_usage(
            api_base=config.api_base,
            model=config.model,
            api_key_env=config.api_key_env,
            system_prompt="You are a connectivity probe. Reply with OK.",
            user_prompt="Reply OK.",
            temperature=0.0,
            timeout_seconds=max(1, int(timeout_seconds)),
            max_tokens=preflight_max_tokens,
        )
    except OpenAICompatError as exc:
        raise ValueError(f"API preflight failed: {exc}") from exc
