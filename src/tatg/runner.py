from __future__ import annotations

import shlex
import subprocess
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Union

from tatg.backends.base import CommandSpec


# ---------------------------------------------------------------------------
# TestFileRegistry — centralised per-file mutual exclusion
# ---------------------------------------------------------------------------

class TestFileRegistry:
    """Class-level registry of per-file threading locks.

    Centralises the lock logic that previously lived inside each adapter.
    Any component that writes to a test file should acquire the lock via
    ``TestFileRegistry.get_lock(path)`` before reading/writing, ensuring that
    concurrent threads targeting the same output file are serialised while
    threads writing to *different* files proceed in parallel.
    """

    _locks: ClassVar[dict[str, threading.Lock]] = {}
    _mutex: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def get_lock(cls, path: Path) -> threading.Lock:
        key = str(path.resolve())
        with cls._mutex:
            if key not in cls._locks:
                cls._locks[key] = threading.Lock()
            return cls._locks[key]


# ---------------------------------------------------------------------------
# WorkUnit — the atomic parallel unit produced by backends
# ---------------------------------------------------------------------------

@dataclass
class WorkUnit:
    """A single parallel work unit returned by a backend.

    Backends produce one ``WorkUnit`` per logical generation target.
    ``run_commands`` dispatches these units.  By default it acquires per-file
    locks from ``TestFileRegistry`` so units sharing the same *output_file* are
    serialised while units with different output files run concurrently.

    Attributes:
        task:        Zero-argument callable that performs generation and
                     returns a result dict (same contract as existing engine
                     callables).
        output_file: Path of the test file that *task* will write.  Used as
                     the lock key in ``TestFileRegistry``.
        label:       Human-readable identifier for logging/reporting.
        lock_condition: Optional zero-argument predicate.  When provided, the
                        per-file lock is acquired only if it returns True.
    """

    task: Callable[[], dict]
    output_file: Path
    label: str
    lock_condition: Callable[[], bool] | None = None

    def __call__(self) -> dict:
        """Delegate to *task* so that WorkUnit passes ``callable()`` checks."""
        return self.task()


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CommandResult:
    command: Union[CommandSpec, str]
    return_code: int
    stdout: str
    stderr: str
    payload: dict | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_commands(
    commands: list[Union[WorkUnit, CommandSpec, Callable]],
    dry_run: bool,
    workers: int = 1,
    on_start: Callable[[str, str], None] | None = None,
    on_done: Callable[[str, bool, float, str], None] | None = None,
    on_result: Callable[[int, CommandResult, str], None] | None = None,
    worker_timeout_seconds: int | float | None = None,
) -> list[CommandResult]:
    """Execute a list of work units / commands / callables, optionally in parallel.

    Dispatch rules (checked in order):
    1. ``WorkUnit``    — acquire ``TestFileRegistry`` lock for *output_file*,
                         run *task*, release lock.  File-level serialisation is
                         automatic; distinct files proceed in parallel.
    2. ``Callable``    — invoke directly (legacy built-in engine callables).
    3. ``CommandSpec`` — launch as subprocess.

    Args:
        commands:  Mixed list of WorkUnit, CommandSpec, or plain callables.
        dry_run:   When True skip actual execution and return placeholder results.
        workers:   Thread-pool size for parallel dispatch (default 1 = sequential).
                   Values <= 0 are normalised to 1.
        on_start:  Optional callback ``(label, worker) -> None`` invoked just before
                   each WorkUnit task begins executing.
        on_done:   Optional callback ``(label, ok, elapsed_seconds, worker) -> None``
                   invoked after each WorkUnit task completes.
        on_result: Optional callback ``(index, result, worker) -> None`` invoked as
               soon as each command result is available.  In parallel mode
               this streams completions in finish order; in sequential mode
               it is called after each command.

    Returns:
        ``CommandResult`` list in the same order as *commands*.
    """
    effective_workers = max(1, workers)
    if effective_workers == 1:
        return _run_sequential(
            commands,
            dry_run,
            on_start=on_start,
            on_done=on_done,
            on_result=on_result,
            worker_timeout_seconds=worker_timeout_seconds,
        )
    return _run_parallel(
        commands,
        dry_run,
        effective_workers,
        on_start=on_start,
        on_done=on_done,
        on_result=on_result,
        worker_timeout_seconds=worker_timeout_seconds,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _execute_one(cmd: Union[WorkUnit, CommandSpec, Callable], dry_run: bool) -> CommandResult:
    """Execute a single work item and return a CommandResult."""

    # --- WorkUnit (preferred path for built-in backends) ---
    if isinstance(cmd, WorkUnit):
        if dry_run:
            return CommandResult(
                command=f"[dry-run] {cmd.label}",
                return_code=0,
                stdout=f"[dry-run] WorkUnit: {cmd.label}",
                stderr="",
                payload=None,
            )
        should_lock = True
        if cmd.lock_condition is not None:
            try:
                should_lock = bool(cmd.lock_condition())
            except Exception:
                should_lock = True
        if should_lock:
            lock = TestFileRegistry.get_lock(cmd.output_file)
            with lock:
                return _execute_work_unit_task(cmd)
        return _execute_work_unit_task(cmd)

    # --- Plain callable (legacy engine callables) ---
    if callable(cmd) and not isinstance(cmd, CommandSpec):
        if dry_run:
            return CommandResult(
                command="[built-in engine call]",
                return_code=0,
                stdout="[dry-run] Built-in engine invoked",
                stderr="",
                payload=None,
            )
        try:
            result = cmd()
            return CommandResult(
                command="[built-in engine]",
                return_code=0 if result.get("status") == "success" else 1,
                stdout=str(result),
                stderr="",
                payload=result if isinstance(result, dict) else None,
            )
        except Exception as exc:
            return CommandResult(
                command="[built-in engine]",
                return_code=1,
                stdout="",
                stderr=str(exc),
                payload=None,
            )

    # --- CommandSpec (subprocess) ---
    if isinstance(cmd, CommandSpec):
        if dry_run:
            return CommandResult(
                command=cmd,
                return_code=0,
                stdout="[dry-run] " + shlex.join(cmd.argv),
                stderr="",
                payload=None,
            )
        import os as _os
        merged_env = dict(_os.environ)
        merged_env.update(cmd.env)
        proc = subprocess.run(
            cmd.argv,
            cwd=str(cmd.cwd),
            env=merged_env,
            text=True,
            capture_output=True,
            check=False,
        )
        return CommandResult(
            command=cmd,
            return_code=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            payload=None,
        )

    raise TypeError(f"Unsupported command type: {type(cmd)}")


def _execute_work_unit_task(cmd: WorkUnit) -> CommandResult:
    try:
        result = cmd.task()
        if isinstance(result, dict):
            try:
                result.setdefault("_tatg_output_file", str(cmd.output_file))
                if cmd.output_file.exists():
                    result["_tatg_test_file_snapshot"] = cmd.output_file.read_text(
                        encoding="utf-8",
                        errors="ignore",
                    )
            except Exception:
                pass
        return CommandResult(
            command=cmd.label,
            return_code=0 if isinstance(result, dict) and result.get("status") == "success" else 1,
            stdout=str(result),
            stderr="",
            payload=result if isinstance(result, dict) else None,
        )
    except Exception as exc:
        return CommandResult(
            command=cmd.label,
            return_code=1,
            stdout="",
            stderr=str(exc),
            payload=None,
        )


def _run_sequential(
    commands: list[Union[WorkUnit, CommandSpec, Callable]],
    dry_run: bool,
    on_start: Callable[[str, str], None] | None = None,
    on_done: Callable[[str, bool, float, str], None] | None = None,
    on_result: Callable[[int, CommandResult, str], None] | None = None,
    worker_timeout_seconds: int | float | None = None,
) -> list[CommandResult]:
    results = []
    worker = "W1"
    timeout = float(worker_timeout_seconds or 0)
    for index, cmd in enumerate(commands):
        label = cmd.label if isinstance(cmd, WorkUnit) else None
        if label and on_start:
            on_start(label, worker)
        t0 = time.perf_counter()
        if timeout > 0 and not dry_run:
            result_queue: queue.Queue[CommandResult] = queue.Queue(maxsize=1)

            def _worker() -> None:
                try:
                    result_queue.put(_execute_one(cmd, dry_run))
                except BaseException as exc:  # noqa: BLE001
                    result_queue.put(
                        CommandResult(
                            command=label or "[built-in engine]",
                            return_code=1,
                            stdout="",
                            stderr=str(exc),
                            payload=None,
                        )
                    )

            thread = threading.Thread(
                target=_worker,
                name=f"tatg-worker-timeout-{label or index}",
                daemon=True,
            )
            thread.start()
            thread.join(timeout)
            if thread.is_alive():
                command_label = label or "[built-in engine]"
                result = CommandResult(
                    command=command_label,
                    return_code=124,
                    stdout="",
                    stderr=f"generation worker exceeded {timeout:.0f}s timeout",
                    payload={
                        "status": "error",
                        "error": "generation_worker_timeout",
                        "target_label": label,
                        "timeout_seconds": timeout,
                        "daemon_worker_unjoined": True,
                    },
                )
            else:
                result = result_queue.get_nowait()
        else:
            result = _execute_one(cmd, dry_run)
        elapsed = time.perf_counter() - t0
        if label and on_done:
            on_done(label, result.return_code == 0, elapsed, worker)
        if on_result:
            on_result(index, result, worker)
        results.append(result)
    return results


def _run_parallel(
    commands: list[Union[WorkUnit, CommandSpec, Callable]],
    dry_run: bool,
    workers: int,
    on_start: Callable[[str, str], None] | None = None,
    on_done: Callable[[str, bool, float, str], None] | None = None,
    on_result: Callable[[int, CommandResult, str], None] | None = None,
    worker_timeout_seconds: int | float | None = None,
) -> list[CommandResult]:
    """Run commands via a thread pool; preserve input order in the output."""
    results: list[CommandResult | None] = [None] * len(commands)
    worker_ids: dict[str, str] = {}
    worker_lock = threading.Lock()

    def _worker_name() -> str:
        thread_name = threading.current_thread().name
        with worker_lock:
            if thread_name not in worker_ids:
                worker_ids[thread_name] = f"W{len(worker_ids) + 1}"
            return worker_ids[thread_name]

    def _run_one(index: int, cmd: Union[WorkUnit, CommandSpec, Callable]) -> tuple[int, CommandResult, str]:
        label = cmd.label if isinstance(cmd, WorkUnit) else None
        worker = _worker_name()
        if label and on_start:
            on_start(label, worker)
        t0 = time.perf_counter()
        timeout = float(worker_timeout_seconds or 0)
        if timeout > 0 and not dry_run:
            result_queue: queue.Queue[CommandResult] = queue.Queue(maxsize=1)

            def _timed_worker() -> None:
                try:
                    result_queue.put(_execute_one(cmd, dry_run))
                except BaseException as exc:  # noqa: BLE001
                    result_queue.put(
                        CommandResult(
                            command=label or "[built-in engine]",
                            return_code=1,
                            stdout="",
                            stderr=str(exc),
                            payload=None,
                        )
                    )

            timed_thread = threading.Thread(
                target=_timed_worker,
                name=f"tatg-parallel-worker-timeout-{label or index}",
                daemon=True,
            )
            timed_thread.start()
            timed_thread.join(timeout)
            if timed_thread.is_alive():
                result = CommandResult(
                    command=label or "[built-in engine]",
                    return_code=124,
                    stdout="",
                    stderr=f"generation worker exceeded {timeout:.0f}s timeout",
                    payload={
                        "status": "error",
                        "error": "generation_worker_timeout",
                        "target_label": label,
                        "timeout_seconds": timeout,
                        "daemon_worker_unjoined": True,
                    },
                )
            else:
                result = result_queue.get_nowait()
        else:
            result = _execute_one(cmd, dry_run)
        elapsed = time.perf_counter() - t0
        if label and on_done:
            on_done(label, result.return_code == 0, elapsed, worker)
        return index, result, worker

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, i, cmd): i for i, cmd in enumerate(commands)}
        for future in as_completed(futures):
            idx, result, worker = future.result()
            results[idx] = result
            if on_result:
                on_result(idx, result, worker)

    return results  # type: ignore[return-value]
