"""TATG backend — registers the ``tatg`` backend name in the engine registry.

The TATG (Type-Aware Test Generation) backend extends CAT by injecting
pre-extracted type shapes, mock scaffolds, and branch hints into the LLM
prompt before generation.  It is a drop-in replacement for the ``cat`` backend
for multi-module Spring projects with complex domain types.
"""

from __future__ import annotations

import os
import queue
import threading
from pathlib import Path

from tatg.backends.base import CommandSpec
from tatg.models import GenerationRequest, PromptBundle, Target
from tatg.runner import WorkUnit
from tatg.engines.tatg_adapter import TATGEngineAdapter


class TATGBackend:
    name = "tatg"

    @staticmethod
    def _generation_timeout_seconds(request: GenerationRequest) -> float:
        override = os.getenv("TATG_GENERATION_TIMEOUT_SECONDS")
        if override:
            try:
                return max(1.0, float(override))
            except ValueError:
                pass
        llm_budget = max(1, int(getattr(request, "llm_timeout_seconds", 120) or 120))
        return float(max(90, llm_budget * 2))

    @classmethod
    def _run_generate_with_timeout(cls, action, timeout_seconds: float) -> dict:
        result_queue: queue.Queue[tuple[bool, dict | BaseException]] = queue.Queue(maxsize=1)

        def _worker() -> None:
            try:
                result_queue.put((True, action()))
            except BaseException as exc:  # noqa: BLE001
                result_queue.put((False, exc))

        thread = threading.Thread(target=_worker, name="tatg-tatg-generate", daemon=True)
        thread.start()
        thread.join(max(float(timeout_seconds), 0.001))
        if thread.is_alive():
            return {
                "status": "error",
                "reason": f"TATG generation exceeded hard timeout of {timeout_seconds:g}s",
                "timeout": True,
            }
        ok, value = result_queue.get_nowait()
        if ok and isinstance(value, dict):
            return value
        if isinstance(value, BaseException):
            raise value
        return {"status": "error", "reason": "TATG generation returned a non-dict result"}

    def build_commands(
        self,
        request: GenerationRequest,
        targets: list[Target],
        prompts: dict[str, PromptBundle],
    ) -> list:
        """Build one :class:`~tatg.runner.WorkUnit` per target.

        The units are executed in parallel by the runner's thread pool; targets
        sharing a test file are automatically serialized by
        ``TestFileRegistry``.
        """
        adapter = TATGEngineAdapter()
        results: list[WorkUnit] = []

        for target in targets:
            if not target.source_file:
                continue

            # Derive the expected test file path up-front so the runner can
            # de-duplicate targets that share a test file.
            test_file = adapter._derive_test_file(Path(target.source_file))

            def make_task(t: Target = target, tf: Path = test_file) -> callable:
                def task() -> dict:
                    a = TATGEngineAdapter()
                    api_key_env = request.api_key_env or "OPENAI_KEY_API"
                    if request.api_key_env:
                        api_key = os.environ.get(request.api_key_env)
                        if api_key:
                            os.environ[api_key_env] = api_key
                    generate_action = lambda: a.generate_test_for_target(
                        request=request,
                        target=t,
                        project_root=request.project_root,
                        test_command=request.test_command,
                        api_key_env=api_key_env,
                    )
                    if bool(getattr(request, "defer_test_writes", False)):
                        result = self._run_generate_with_timeout(
                            generate_action,
                            self._generation_timeout_seconds(request),
                        )
                    else:
                        result = generate_action()
                    status = result.get("status", "error")
                    return {
                        "status": status,
                        "timeout": bool(result.get("timeout")),
                        "reason": result.get("reason"),
                        "error": result.get("error"),
                        "results": {str(t): result},
                        "model": request.model,
                        "target_label": t.label,
                        "source_file": str(t.source_file) if t.source_file else "",
                        "test_file": str(tf),
                        "class_name": t.class_name,
                        "method_name": t.method_name,
                        # TATG-specific metadata
                        "tatg_type_shapes_count": result.get("tatg_type_shapes_count"),
                        "tatg_autowired_deps_count": result.get("tatg_autowired_deps_count"),
                        "tatg_branch_hints_count": result.get("tatg_branch_hints_count"),
                    }
                return task

            results.append(WorkUnit(
                task=make_task(),
                output_file=test_file,
                label=target.label,
                lock_condition=lambda req=request: not bool(getattr(req, "defer_test_writes", False)),
            ))

        return results
