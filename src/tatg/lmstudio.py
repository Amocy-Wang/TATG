from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib import request as urlrequest
from urllib.parse import urlparse


_RECOVERY_LOCK = threading.Lock()
_SERVER_PORT_RE = re.compile(r"\bport\s+(\d{1,5})\b", re.IGNORECASE)

_LOCAL_TIMEOUT_DEFAULTS = {
    "llm": 900,
    "branch_analysis": 1200,
    "generation": 1800,
    "e2e": 7200,
}


def is_local_lmstudio_base(api_base: str) -> bool:
    if os.environ.get("TATG_LMSTUDIO_AUTO_RECOVER", "1").lower() in {
        "0", "false", "no", "off",
    }:
        return False
    parsed = urlparse(str(api_base or ""))
    return parsed.scheme in {"http", "https"} and parsed.hostname in {
        "127.0.0.1", "localhost", "::1",
    }


def local_lmstudio_timeout_seconds(
    api_base: str,
    requested_seconds: int | float,
    *,
    operation: str,
) -> int:
    """Apply a localhost-only timeout floor for slower local inference.

    Explicit hard timeouts are useful for remote providers, but the same
    defaults can systematically misclassify a healthy, resource-constrained
    LM Studio model as a failed strategy. The floor is local-only, can be
    disabled globally, and has a per-operation environment override.
    ``e2e=0`` remains an explicit request to disable the method deadline.
    """

    try:
        requested = max(0, int(float(requested_seconds)))
    except (TypeError, ValueError):
        requested = 0
    if not is_local_lmstudio_base(api_base):
        return requested
    if os.environ.get(
        "TATG_LMSTUDIO_ADAPT_TIMEOUTS",
        "1",
    ).strip().lower() in {"0", "false", "no", "off"}:
        return requested

    normalized = str(operation or "").strip().lower()
    if normalized not in _LOCAL_TIMEOUT_DEFAULTS:
        return requested
    if normalized == "e2e" and requested == 0:
        return 0
    env_name = (
        "TATG_LMSTUDIO_"
        + normalized.upper()
        + "_TIMEOUT_SECONDS"
    )
    try:
        floor = max(
            1,
            int(
                os.environ.get(
                    env_name,
                    str(_LOCAL_TIMEOUT_DEFAULTS[normalized]),
                )
                or _LOCAL_TIMEOUT_DEFAULTS[normalized]
            ),
        )
    except (TypeError, ValueError):
        floor = _LOCAL_TIMEOUT_DEFAULTS[normalized]
    return max(requested, floor)


def _lms_path() -> str | None:
    configured = str(os.environ.get("TATG_LMS_BIN") or "").strip()
    candidates = [
        configured,
        shutil.which("lms") or "",
        str(Path.home() / ".lmstudio" / "bin" / "lms"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def _models_endpoint(api_base: str) -> str:
    base = str(api_base or "").rstrip("/")
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def _server_status_port(status_output: str) -> int | None:
    """Return the port reported by ``lms server status`` when available."""

    match = _SERVER_PORT_RE.search(str(status_output or ""))
    if not match:
        return None
    try:
        port = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return port if 0 < port <= 65535 else None


def _listed_model_ids(api_base: str, *, timeout_seconds: float = 5.0) -> set[str]:
    req = urlrequest.Request(_models_endpoint(api_base), method="GET")
    with urlrequest.urlopen(req, timeout=max(0.5, timeout_seconds)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        str(item.get("id") or "")
        for item in list(payload.get("data") or [])
        if isinstance(item, dict) and str(item.get("id") or "")
    }


def recover_local_lmstudio(
    *,
    api_base: str,
    model: str,
    timeout_seconds: int = 240,
    force_model_reload: bool = False,
) -> dict[str, object]:
    """Best-effort recovery for a local LM Studio OpenAI-compatible endpoint.

    Recovery is deliberately limited to localhost. Slow inference timeouts do
    not call this function; callers invoke it only for connection/server/model
    availability failures.
    """

    if not is_local_lmstudio_base(api_base):
        return {"attempted": False, "reason": "not_local_lmstudio"}
    lms = _lms_path()
    if not lms:
        return {"attempted": False, "reason": "lms_cli_not_found"}

    parsed = urlparse(api_base)
    port = int(parsed.port or (443 if parsed.scheme == "https" else 80))
    deadline = time.monotonic() + max(10, int(timeout_seconds or 0))
    actions: list[str] = []
    with _RECOVERY_LOCK:
        endpoint_reachable = False
        try:
            ids = _listed_model_ids(api_base)
            endpoint_reachable = True
            if model in ids and not force_model_reload:
                return {
                    "attempted": True,
                    "recovered": True,
                    "actions": ["endpoint_already_healthy"],
                }
        except Exception:
            pass

        status = subprocess.run(
            [lms, "server", "status"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=min(20, max(5, int(deadline - time.monotonic()))),
        )
        server_running = (
            status.returncode == 0 and "running" in status.stdout.lower()
        )
        reported_port = _server_status_port(status.stdout)
        if (
            server_running
            and reported_port is not None
            and reported_port != port
        ):
            return {
                "attempted": True,
                "recovered": False,
                "reason": "endpoint_not_managed_by_running_lmstudio",
                "requested_port": port,
                "lmstudio_server_port": reported_port,
            }
        if server_running and not endpoint_reachable:
            stop = subprocess.run(
                [lms, "server", "stop"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=min(30, max(5, int(deadline - time.monotonic()))),
            )
            actions.append(f"server_stop_rc={stop.returncode}")
            server_running = False
        if not server_running:
            start = subprocess.run(
                [lms, "server", "start", "--port", str(port), "--bind", parsed.hostname or "127.0.0.1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=min(45, max(10, int(deadline - time.monotonic()))),
            )
            actions.append(f"server_start_rc={start.returncode}")

        loaded = subprocess.run(
            [lms, "ps"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=min(20, max(5, int(deadline - time.monotonic()))),
        )
        model_loaded = model in loaded.stdout
        if force_model_reload and model_loaded:
            unload = subprocess.run(
                [lms, "unload", model],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=min(60, max(10, int(deadline - time.monotonic()))),
            )
            actions.append(f"model_unload_rc={unload.returncode}")
            model_loaded = False
        if not model_loaded:
            load_command = [lms, "load", model, "--identifier", model, "--yes"]
            context_length = str(os.environ.get("TATG_LMSTUDIO_CONTEXT_LENGTH") or "").strip()
            parallel = str(os.environ.get("TATG_LMSTUDIO_PARALLEL") or "").strip()
            if context_length.isdigit() and int(context_length) > 0:
                load_command.extend(["--context-length", context_length])
            if parallel.isdigit() and int(parallel) > 0:
                load_command.extend(["--parallel", parallel])
            load = subprocess.run(
                load_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=max(30, int(deadline - time.monotonic())),
            )
            actions.append(f"model_load_rc={load.returncode}")

        last_error = ""
        while time.monotonic() < deadline:
            try:
                ids = _listed_model_ids(api_base, timeout_seconds=5)
                if model in ids:
                    return {
                        "attempted": True,
                        "recovered": True,
                        "actions": actions,
                        "model": model,
                        "port": port,
                    }
                last_error = f"model_not_listed:{model}"
            except Exception as exc:
                last_error = f"{type(exc).__name__}:{exc}"
            time.sleep(2)
        return {
            "attempted": True,
            "recovered": False,
            "actions": actions,
            "reason": last_error or "recovery_timeout",
        }
