"""Terminal progress display for TATG test generation.

Renders a live status panel to *stderr* during ``tatg generate``:

  • Per-target generation status  (pending / running / done / failed)
  • Repair-loop iteration, attempt, and current failure / repaired counts
  • Elapsed wall time

When stderr is not a TTY the panel falls back to plain log lines that can be
safely redirected to a file or CI log without ANSI pollution.

No third-party dependencies — stdlib only.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from enum import Enum
from shutil import get_terminal_size
from typing import TextIO


# ---------------------------------------------------------------------------
# Internal data model
# ---------------------------------------------------------------------------

class _Status(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"


@dataclass
class _TargetEntry:
    label:   str
    status:  _Status = _Status.PENDING
    elapsed: float | None = None


@dataclass
class _RepairState:
    iteration:      int = 0
    max_iterations: int = 0
    attempt:        int = 0
    max_attempts:   int = 0
    failed:         int = 0
    repaired:       int = 0


@dataclass
class _WorkerState:
    name: str
    status: _Status = _Status.PENDING
    label: str = "idle"
    elapsed: float | None = None
    last_ok: bool | None = None
    start_time: float | None = None  # perf_counter when this worker's current target started


# ---------------------------------------------------------------------------
# ANSI helpers & formatting
# ---------------------------------------------------------------------------

_G  = "\033[32m"   # green
_R  = "\033[31m"   # red
_Y  = "\033[33m"   # yellow
_C  = "\033[36m"   # cyan
_B  = "\033[1m"    # bold
_D  = "\033[2m"    # dim
_Z  = "\033[0m"    # reset
_UP = "\033[A"     # cursor up one line
_CL = "\033[2K\r"  # clear line + return to col 0

_BAR_WIDTH = 22    # width of the ASCII progress bar
_MIN_TERMINAL_WIDTH = 80
_DEFAULT_TERMINAL_WIDTH = 120
_LOG_PREFIX = "[tatg]"
_SEPARATOR = "  |  "
_RECENT_DONE_LIMIT = 5
_RECENT_DONE_DISPLAY = 3
_MIN_LABEL_WIDTH = 30
_MAX_LABEL_WIDTH = 100


def _bar(done: int, total: int) -> str:
    if total == 0:
        return "░" * _BAR_WIDTH
    filled = int(_BAR_WIDTH * done / total)
    return "█" * filled + "░" * (_BAR_WIDTH - filled)


def _hms(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{seconds:.1f}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def _trunc(text: str, width: int) -> str:
    return (text[: width - 1] + "…") if len(text) > width else text


def _compact_label(text: str, width: int) -> str:
    """Keep labels readable by preserving meaningful suffixes and endpoints."""
    if len(text) <= width:
        return text
    candidate = text
    if "::" in candidate:
        parts = candidate.split("::")
        head = parts[0]
        tail = "::".join(parts[-2:])
        if "/" in head:
            head_parts = [p for p in head.split("/") if p]
            head = "/" + "/".join(head_parts[-3:])
        candidate = f"{head}::{tail}"
        if len(candidate) <= width:
            return candidate
    if width <= 12:
        return _trunc(candidate, width)
    left = max(8, width // 2 - 1)
    right = max(8, width - left - 1)
    return candidate[:left] + "…" + candidate[-right:]


# ---------------------------------------------------------------------------
# ProgressDisplay
# ---------------------------------------------------------------------------

class ProgressDisplay:
    """Thread-safe, stderr progress panel.

    Args:
        total_targets:     Number of generation targets.
        max_iterations:    ``max_iterations`` from the generation request.
        repair_iterations: ``repair_iterations`` from the generation request.
        enabled:           When *False* the display is completely suppressed.
        out:               Output stream (default ``sys.stderr``).
    """

    def __init__(
        self,
        total_targets:     int,
        max_iterations:    int,
        repair_iterations: int,
        worker_count:      int = 1,
        unit_label:        str = "target",
        java_version:      str = "",
        junit_version:     str = "",
        junit_family:      str = "",
        enabled:           bool       = True,
        out:               TextIO | None = None,
    ) -> None:
        self._total   = total_targets
        self._unit    = unit_label
        self._java_version = (java_version or "").strip()
        self._junit_version = (junit_version or "").strip()
        self._junit_family = (junit_family or "").strip()
        self._enabled = enabled
        self._out     = out or sys.stderr
        self._is_tty  = enabled and getattr(self._out, "isatty", lambda: False)()
        self._lock    = threading.Lock()
        self._worker_count = max(1, worker_count)
        self._terminal_width = max(
            _MIN_TERMINAL_WIDTH,
            get_terminal_size(fallback=(_DEFAULT_TERMINAL_WIDTH, 24)).columns
        )

        self._targets: list[_TargetEntry] = []
        self._workers: dict[str, _WorkerState] = {
            f"W{i + 1}": _WorkerState(name=f"W{i + 1}") for i in range(self._worker_count)
        }
        # Each entry: (label, ok, elapsed, worker, stats_dict | None)
        # stats_dict keys: total, ok, fail
        self._recent_done: list[tuple[str, bool, float, str, dict | None]] = []
        self._last_test_stats_by_label: dict[str, tuple[int, int, int]] = {}
        self._repair = _RepairState(
            max_iterations=max_iterations,
            max_attempts=repair_iterations
        )
        self._phase = "generation"
        self._validation_workers = 0
        self._validation_jobs_submitted = 0
        self._validation_jobs_completed = 0
        self._validation_jobs_inflight = 0
        self._validation_total = 0
        self._validation_ok = 0
        self._validation_fail = 0
        self._done_count = 0
        self._fail_count = 0
        self._cov_line: float | None = None
        self._cov_branch: float | None = None
        self._start = time.perf_counter()
        self._lines_drawn = 0
        self._finished = False
        self._seq_index = 0  # monotone counter used in non-TTY log lines

    def _version_summary(self) -> str:
        java_part = f"java={self._java_version or 'unknown'}"
        junit_value = self._junit_version or self._junit_family or "unknown"
        if self._junit_version and self._junit_family:
            junit_part = f"junit={self._junit_version} ({self._junit_family})"
        else:
            junit_part = f"junit={junit_value}"
        return f"{java_part}  {junit_part}"

    def _repair_attempt_summary(self, attempt: int) -> str:
        max_attempts = int(self._repair.max_attempts or 0)
        if max_attempts <= 0:
            return f"attempt={attempt}"
        if attempt <= max_attempts:
            return f"attempt={attempt}/{max_attempts}"
        # Streaming repair updates track candidate progress, not logical attempt rounds.
        return f"attempt={attempt} (candidates)"

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_targets(self, labels: list[str]) -> None:
        """Call once with the full list of target labels before generation starts."""
        with self._lock:
            self._targets = [_TargetEntry(label=l) for l in labels]
            if self._is_tty:
                self._redraw_locked()
            else:
                self._log(
                    f"starting generation of {len(labels)} {self._unit}(s)"
                    f"  |  {self._version_summary()}"
                )

    # ------------------------------------------------------------------
    # Generation callbacks
    # ------------------------------------------------------------------

    def set_validation_workers(self, workers: int) -> None:
        with self._lock:
            self._validation_workers = max(0, int(workers or 0))
            if self._is_tty:
                self._redraw_locked()

    def on_validation_submitted(self, count: int = 1) -> None:
        with self._lock:
            n = max(0, int(count or 0))
            self._validation_jobs_submitted += n
            self._validation_jobs_inflight += n
            if self._is_tty:
                self._redraw_locked()

    def on_validation_completed(self, count: int = 1) -> None:
        with self._lock:
            n = max(0, int(count or 0))
            self._validation_jobs_completed += n
            self._validation_jobs_inflight = max(0, self._validation_jobs_inflight - n)
            if self._is_tty:
                self._redraw_locked()

    def on_target_start(self, label: str, worker: str) -> None:
        with self._lock:
            self._seq_index += 1
            for t in self._targets:
                if t.label == label:
                    t.status = _Status.RUNNING
                    break
            if worker not in self._workers:
                self._workers[worker] = _WorkerState(name=worker)
            w = self._workers[worker]
            w.status = _Status.RUNNING
            w.label = label
            w.elapsed = None
            w.last_ok = None
            w.start_time = time.perf_counter()
            if self._is_tty:
                self._redraw_locked()
            else:
                idx = self._seq_index
                self._log(
                    f"[gen {idx:>3}/{self._total}] {worker} ⟳  "
                    f"{_compact_label(label, 72)}"
                )

    def on_target_done(self, label: str, ok: bool, elapsed: float, worker: str) -> None:
        with self._lock:
            for t in self._targets:
                if t.label == label:
                    t.status  = _Status.DONE if ok else _Status.FAILED
                    t.elapsed = elapsed
                    break
            if worker not in self._workers:
                self._workers[worker] = _WorkerState(name=worker)
            w = self._workers[worker]
            w.status = _Status.DONE if ok else _Status.FAILED
            w.label = label
            w.elapsed = elapsed
            w.last_ok = ok
            w.start_time = None
            if ok:
                self._done_count += 1
            else:
                self._fail_count += 1
            self._recent_done.append((label, ok, elapsed, worker, None))
            if len(self._recent_done) > _RECENT_DONE_LIMIT:
                self._recent_done = self._recent_done[-_RECENT_DONE_LIMIT:]
            if self._is_tty:
                self._redraw_locked()
            else:
                done  = self._done_count + self._fail_count
                icon  = "✓" if ok else "✗"
                self._log(
                    f"[gen {done:>3}/{self._total}] {worker} {icon}  "
                    f"{_compact_label(label, 72)}  ({elapsed:.1f}s)"
                )

    def on_target_test_stats(
        self,
        label: str,
        total_tests: int,
        successful_tests: int,
        failing_tests: int,
    ) -> None:
        """Attach per-target test validation counts to the most recent *label* entry.

        Args:
            label:            Target label (must match an existing recent-done entry).
            total_tests:      Number of test methods in the generated candidate.
            successful_tests: Generated test methods that passed validation.
            failing_tests:    Generated test methods that failed validation.
        """
        with self._lock:
            stats = {
                "total": max(0, int(total_tests or 0)),
                "ok": max(0, int(successful_tests or 0)),
                "fail": max(0, int(failing_tests or 0)),
            }
            key = (stats["total"], stats["ok"], stats["fail"])
            prev = self._last_test_stats_by_label.get(label)
            self._last_test_stats_by_label[label] = key
            if prev is None:
                self._validation_total += stats["total"]
                self._validation_ok += stats["ok"]
                self._validation_fail += stats["fail"]
            else:
                self._validation_total += stats["total"] - prev[0]
                self._validation_ok += stats["ok"] - prev[1]
                self._validation_fail += stats["fail"] - prev[2]
            for i in range(len(self._recent_done) - 1, -1, -1):
                if self._recent_done[i][0] == label:
                    e = self._recent_done[i]
                    self._recent_done[i] = (e[0], e[1], e[2], e[3], stats)
                    break
            if self._is_tty:
                self._redraw_locked()
            else:
                if key == (0, 0, 0) or key == prev:
                    return
                parts: list[str] = [f"generated={stats['total']}"]
                parts.append(f"tests_ok={stats['ok']}")
                parts.append(f"tests_fail={stats['fail']}")
                self._log(
                    f"[validate][stats] {_compact_label(label, 72)}  " + "  ".join(parts)
                )

    # ------------------------------------------------------------------
    # Repair-loop callbacks
    # ------------------------------------------------------------------

    def on_repair_start(
        self,
        iteration:   int,
        attempt:     int,
        failed:      int,
    ) -> None:
        with self._lock:
            self._phase            = "repair"
            self._repair.iteration = iteration
            self._repair.attempt   = attempt
            self._repair.failed    = failed
            if self._is_tty:
                self._redraw_locked()
            else:
                self._log(
                    f"[repair iter={iteration}/{self._repair.max_iterations} "
                    f"{self._repair_attempt_summary(attempt)}]  "
                    f"failed={failed}"
                )

    def on_repair_end(
        self,
        iteration:     int,
        attempt:       int,
        failed_after:  int,
        repaired:      int,
    ) -> None:
        with self._lock:
            self._repair.iteration = iteration
            self._repair.attempt   = attempt
            self._repair.failed    = failed_after
            self._repair.repaired  = repaired
            if self._is_tty:
                self._redraw_locked()
            else:
                suffix = (
                    f"repaired={repaired}  still failing={failed_after}"
                    if repaired > 0
                    else f"failed={failed_after}"
                )
                self._log(
                    f"[repair iter={iteration}/{self._repair.max_iterations} "
                    f"{self._repair_attempt_summary(attempt)}]  {suffix}"
                )

    # ------------------------------------------------------------------
    # Coverage update
    # ------------------------------------------------------------------

    def on_coverage(self, line: float, branch: float | None = None, *, label: str | None = None) -> None:
        """Call after each test run with the latest coverage metrics (0.0–1.0)."""
        with self._lock:
            self._cov_line   = line
            self._cov_branch = branch
            if self._is_tty:
                self._redraw_locked()
            else:
                cov_parts = [f"line={line:.1%}"]
                if branch is not None:
                    cov_parts.append(f"branch={branch:.1%}")
                if label:
                    cov_parts.append(f"scope={label}")
                self._log("coverage  " + "  ".join(cov_parts))

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finish(
        self,
        final_failed:           int = 0,
        generation_success_rate: float | None = None,
        test_passing_rate:       float | None = None,
    ) -> None:
        """Render a final summary line and stop all future redraws."""
        with self._lock:
            if self._finished:
                return
            self._finished     = True
            self._repair.failed = final_failed
            elapsed = time.perf_counter() - self._start

            gen_ok = self._done_count
            gen_total = self._total
            pending = max(gen_total - (self._done_count + self._fail_count), 0)

            if self._is_tty:
                # Erase live panel …
                self._erase_locked()
                # … then print a compact one-line summary.
                parts = [
                    f"{_B}TATG done{_Z}",
                    _hms(elapsed),
                    f"phase=generation",
                    f"completed {gen_ok}/{gen_total}",
                    f"failed {self._fail_count}",
                    f"pending {pending}",
                ]
                if self._repair.iteration > 0:
                    parts[2] = f"phase=repair(iter {self._repair.iteration})"
                    parts.append(f"repair ok={self._repair.repaired}")
                    parts.append(f"repair remaining={self._repair.failed}")
                if generation_success_rate is not None:
                    parts.append(f"gen rate {generation_success_rate:.0%}")
                if test_passing_rate is not None:
                    parts.append(f"passing {test_passing_rate:.0%}")
                if self._cov_line is not None:
                    cov_p = f"line cov {self._cov_line:.1%}"
                    if self._cov_branch is not None:
                        cov_p += f" / branch {self._cov_branch:.1%}"
                    parts.append(f"{_G}{cov_p}{_Z}")
                if final_failed == 0:
                    parts.append(f"{_G}all passing{_Z}")
                else:
                    parts.append(f"{_R}{final_failed} blocking{_Z}")
                self._out.write(_SEPARATOR.join(parts) + "\n")
                self._out.flush()
            else:
                parts = [
                    f"done {_hms(elapsed)}",
                    f"phase=generation",
                    f"completed={gen_ok}/{gen_total}",
                    f"failed={self._fail_count}",
                    f"pending={pending}",
                ]
                if self._repair.iteration > 0:
                    parts[1] = f"phase=repair(iter {self._repair.iteration})"
                    parts.append(f"repair_ok={self._repair.repaired}")
                    parts.append(f"repair_remaining={self._repair.failed}")
                if generation_success_rate is not None:
                    parts.append(f"gen_rate={generation_success_rate:.0%}")
                if test_passing_rate is not None:
                    parts.append(f"passing={test_passing_rate:.0%}")
                if self._cov_line is not None:
                    parts.append(f"line_cov={self._cov_line:.1%}")
                if self._cov_branch is not None:
                    parts.append(f"branch_cov={self._cov_branch:.1%}")
                parts.append("all_passing" if final_failed == 0 else f"blocking={final_failed}")
                self._log(_SEPARATOR.join(parts))

    # ------------------------------------------------------------------
    # Internal rendering — must be called with self._lock held
    # ------------------------------------------------------------------

    def write(self, msg: str) -> None:
        """Write an informational line while preserving the live TTY panel."""
        with self._lock:
            if self._is_tty:
                self._erase_locked()
                self._out.write(f"{_LOG_PREFIX} {msg}\n")
                self._out.flush()
                self._redraw_locked()
            else:
                self._log(msg)

    def _log(self, msg: str) -> None:
        """Write a single plain-text log line (non-TTY path)."""
        self._out.write(f"{_LOG_PREFIX} {msg}\n")
        self._out.flush()

    def _erase_locked(self) -> None:
        if not self._is_tty or self._lines_drawn == 0:
            return
        for _ in range(self._lines_drawn):
            self._out.write(_UP + _CL)
        self._lines_drawn = 0

    def _writeln(self, text: str) -> None:
        self._out.write(text + "\n")
        self._lines_drawn += 1

    def _redraw_locked(self) -> None:
        if not self._enabled or self._finished:
            return

        elapsed = time.perf_counter() - self._start
        self._erase_locked()

        # ── Phase header ──────────────────────────────────────────────
        if self._phase == "generation":
            if self._validation_jobs_submitted > 0 or self._validation_jobs_inflight > 0 or self._validation_total > 0:
                phase_str = f"{_B}[Generation + Validation]{_Z}"
            else:
                phase_str = f"{_B}[Generation]{_Z}"
        else:
            r = self._repair
            iter_str = f" iter {r.iteration}/{r.max_iterations}" if r.max_iterations > 0 else ""
            phase_str = f"{_Y}[Repair{iter_str}]{_Z}"

        # ── Progress bar ──────────────────────────────────────────────
        completed = self._done_count + self._fail_count
        running = sum(1 for t in self._targets if t.status == _Status.RUNNING)
        pending = max(self._total - completed - running, 0)
        bar   = _bar(completed, self._total)
        pct   = int(100 * completed / self._total) if self._total else 0

        unit_plural = self._unit + ("s" if self._total != 1 else "")
        self._writeln(
            f"{phase_str} [{bar}] {completed} completed / {self._total} target {unit_plural} ({pct}%)"
            f"  {_D}{_hms(elapsed)}{_Z}"
        )
        cov_str = ""
        if self._cov_line is not None:
            cov_str = f"  |  {_G}cov line={self._cov_line:.1%}"
            if self._cov_branch is not None:
                cov_str += f"  branch={self._cov_branch:.1%}"
            cov_str += _Z
        repair_str = ""
        if self._phase == "repair" and self._repair.iteration > 0:
            r = self._repair
            repair_str = (
                f"  |  repair iter={r.iteration}/{r.max_iterations}"
                f" done={r.attempt} ok={r.repaired} remaining={r.failed}"
            )
        validation_str = ""
        if (
            self._validation_total > 0
            or self._validation_workers > 0
            or self._validation_jobs_submitted > 0
            or self._validation_jobs_inflight > 0
        ):
            awaiting_first_result = (
                self._validation_jobs_submitted > 0
                and self._validation_jobs_completed == 0
                and self._validation_jobs_inflight > 0
                and self._validation_total == 0
            )
            # Compact: "validate  12 tests (✓10 ✗2)  |  5/7 jobs · 2 in-flight"
            test_part = (
                f"{self._validation_total} tests"
                f" ({_G}✓{self._validation_ok}{_Z}"
                f" {_R}✗{self._validation_fail}{_Z})"
                if self._validation_total > 0
                else ("awaiting…" if awaiting_first_result else "0 tests")
            )
            job_part = (
                f"{self._validation_jobs_completed}/{self._validation_jobs_submitted} jobs"
                f" · {self._validation_jobs_inflight} in-flight"
            )
            workers_part = f" · {self._validation_workers}w" if self._validation_workers > 0 else ""
            validation_str = f"  |  validate  {test_part}  |  {job_part}{workers_part}"
        self._writeln(
            f"{_D}running={running}  pending={pending}  "
            f"ok={self._done_count}  fail={self._fail_count}  |  workers={self._worker_count}{_Z}"
            f"{repair_str}"
            f"{validation_str}"
            f"{cov_str}"
        )
        self._writeln(f"{_D}{self._version_summary()}{_Z}")

        # ── Worker rows ───────────────────────────────────────────────
        label_width = max(_MIN_LABEL_WIDTH, min(_MAX_LABEL_WIDTH, self._terminal_width - 34))
        now = time.perf_counter()
        for idx in range(1, self._worker_count + 1):
            worker = self._workers.get(f"W{idx}")
            if worker is None:
                worker = _WorkerState(name=f"W{idx}")
                self._workers[worker.name] = worker
            if worker.status == _Status.RUNNING:
                icon = f"{_C}⟳{_Z}"
                action = f"generating {_compact_label(worker.label, label_width)}"
                live_elapsed = now - worker.start_time if worker.start_time is not None else None
                elapsed_part = f"  {_C}{_hms(live_elapsed)}{_Z}" if live_elapsed is not None else ""
            elif worker.status == _Status.DONE:
                icon = f"{_G}✓{_Z}"
                action = f"{_D}← {_compact_label(worker.label, label_width)}{_Z}"
                elapsed_part = f"  {_D}{worker.elapsed:.1f}s{_Z}" if worker.elapsed is not None else ""
            elif worker.status == _Status.FAILED:
                icon = f"{_R}✗{_Z}"
                action = f"{_D}← {_compact_label(worker.label, label_width)}{_Z}"
                elapsed_part = f"  {_D}{worker.elapsed:.1f}s{_Z}" if worker.elapsed is not None else ""
            else:
                icon = f"{_D}○{_Z}"
                action = f"{_D}idle{_Z}"
                elapsed_part = ""
            self._writeln(f"  {icon} {worker.name:<3} {action}{elapsed_part}")

        # ── Recent completed tasks ─────────────────────────────────────
        if self._recent_done:
            self._writeln(f"{_B}[Recent]{_Z}")
            for entry in self._recent_done[-_RECENT_DONE_DISPLAY:]:
                label, ok, done_elapsed, worker = entry[0], entry[1], entry[2], entry[3]
                stats = entry[4] if len(entry) > 4 else None
                icon = f"{_G}✓{_Z}" if ok else f"{_R}✗{_Z}"
                stats_str = ""
                if stats is not None:
                    total = int(stats.get("total", 0) or 0)
                    ok_tests = int(stats.get("ok", 0) or 0)
                    fail_tests = int(stats.get("fail", 0) or 0)
                    fail_col = _R if fail_tests > 0 else _D
                    stats_str = (
                        f"  {_D}[{total} tests · "
                        f"{_G}✓{ok_tests}{_D} · "
                        f"{fail_col}✗{fail_tests}{_D}]{_Z}"
                    )
                self._writeln(
                    f"  {icon} {worker:<3} {_compact_label(label, label_width)}"
                    f"  {_D}{done_elapsed:.1f}s{_Z}{stats_str}"
                )

        # ── Repair section ─────────────────────────────────────────────
        if self._phase == "repair" and self._repair.iteration > 0:
            r = self._repair
            status_icon = (
                f"{_G}✓{_Z}" if r.failed == 0 else f"{_Y}⟳{_Z}"
            )
            self._writeln(
                f"{_B}[Repair]{_Z}"
                f"  iter {r.iteration}/{r.max_iterations}"
                f"  ·  attempt {r.attempt}/{r.max_attempts}"
                f"  ·  {status_icon} failed: {r.failed}"
                + (f"  repaired: {r.repaired}" if r.repaired else "")
                + f"  {_D}{_hms(elapsed)}{_Z}"
            )

        self._out.flush()
