#!/usr/bin/env python3
"""Run the four-class TATG batch from the current source checkout.

This script deliberately invokes ``python -m tatg.cli`` with
``PYTHONPATH=<this repo>/src`` so the run uses the working tree, not an older
installed ``tatg`` console script from a conda environment.
"""

from __future__ import annotations

import argparse
import codecs
import csv
from datetime import datetime
import json
import os
import selectors
import signal
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tatg.api_config import DEFAULT_API_KEY_ENV, apply_api_config_to_env, preflight_api_config, resolve_api_config
from tatg.evaluation.evaluator import Evaluator


@dataclass(frozen=True)
class BatchTarget:
    name: str
    project: str
    target_file: str


TARGETS: tuple[BatchTarget, ...] = (
    BatchTarget(
        name="Rotation",
        project="Math-2f",
        target_file="src/main/java/org/apache/commons/math3/geometry/euclidean/threed/Rotation.java",
    ),
    BatchTarget(
        name="GaussNewtonOptimizer",
        project="Math-2f",
        target_file="src/main/java/org/apache/commons/math3/fitting/leastsquares/GaussNewtonOptimizer.java",
    ),
    BatchTarget(
        name="FromXmlParser",
        project="JacksonXml-5f",
        target_file="src/main/java/com/fasterxml/jackson/dataformat/xml/deser/FromXmlParser.java",
    ),
    BatchTarget(
        name="ToXmlGenerator",
        project="JacksonXml-5f",
        target_file="src/main/java/com/fasterxml/jackson/dataformat/xml/ser/ToXmlGenerator.java",
    ),
    BatchTarget(
        name="Metaphone",
        project="Codec-18f",
        target_file="src/main/java/org/apache/commons/codec/language/Metaphone.java",
    ),
    BatchTarget(
        name="NumberUtils",
        project="Lang-4f",
        target_file="src/main/java/org/apache/commons/lang3/math/NumberUtils.java",
    ),
    BatchTarget(
        name="FilteringParserDelegate",
        project="JacksonCore-26f",
        target_file="src/main/java/com/fasterxml/jackson/core/filter/FilteringParserDelegate.java",
    ),
    BatchTarget(
        name="Flat3Map",
        project="Collections-28f",
        target_file="src/main/java/org/apache/commons/collections4/map/Flat3Map.java",
    ),
    BatchTarget(
        name="DateTimeFormat",
        project="Time-13f",
        target_file="src/main/java/org/joda/time/format/DateTimeFormat.java",
    ),
    BatchTarget(
        name="Entities",
        project="Jsoup-93f",
        target_file="src/main/java/org/jsoup/nodes/Entities.java",
    ),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _java_files(project_root: Path) -> dict[Path, Path]:
    """Return production/test Java files across root and nested modules."""

    files: dict[Path, Path] = {}
    excluded_parts = {
        ".git",
        ".gradle",
        ".idea",
        "build",
        "node_modules",
        "target",
    }
    if not project_root.exists():
        return files
    for path in project_root.rglob("*.java"):
        if not path.is_file():
            continue
        try:
            resolved = path.resolve()
            rel_path = resolved.relative_to(project_root.resolve())
        except (OSError, ValueError):
            continue
        parts = rel_path.parts
        if any(part in excluded_parts for part in parts):
            continue
        if not any(
            parts[index] == "src" and parts[index + 1] in {"main", "test"}
            for index in range(len(parts) - 1)
        ):
            continue
        files[rel_path] = resolved
    return files


def _backup_java_files(project_root: Path, backup_root: Path) -> dict[Path, Path]:
    files = _java_files(project_root)
    for rel_path, source in files.items():
        destination = backup_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return files


def _copy_changed_java_files(
    project_root: Path,
    before: dict[Path, Path],
    backup_root: Path,
    generated_out: Path,
) -> list[Path]:
    after = _java_files(project_root)
    generated_out.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for rel_path, path in sorted(after.items()):
        original = backup_root / rel_path
        if rel_path in before and original.exists() and path.read_bytes() == original.read_bytes():
            continue
        destination = generated_out / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def _restore_java_files(project_root: Path, before: dict[Path, Path], backup_root: Path) -> None:
    after = _java_files(project_root)
    for rel_path, path in sorted(after.items()):
        if rel_path not in before:
            path.unlink(missing_ok=True)
    for rel_path in sorted(before):
        source = backup_root / rel_path
        destination = project_root / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _java_restore_mismatches(
    project_root: Path,
    before: dict[Path, Path],
    backup_root: Path,
) -> list[dict[str, str]]:
    """Return exact Java-tree differences from the pre-run backup."""
    after = _java_files(project_root)
    mismatches: list[dict[str, str]] = []
    for rel_path in sorted(set(after) - set(before)):
        mismatches.append({"path": str(rel_path), "reason": "unexpected_java_file"})
    for rel_path in sorted(before):
        expected = backup_root / rel_path
        actual = project_root / rel_path
        if not expected.is_file():
            mismatches.append({"path": str(rel_path), "reason": "backup_missing"})
        elif not actual.is_file():
            mismatches.append({"path": str(rel_path), "reason": "original_missing"})
        elif actual.read_bytes() != expected.read_bytes():
            mismatches.append({"path": str(rel_path), "reason": "content_mismatch"})
    return mismatches


def _test_command(mode: str, class_name: str) -> str:
    if mode == "true":
        # The CLI treats exact no-op commands such as `true` as unusable and
        # falls back to inferred Maven validation. Use a lightweight command
        # that still returns 0 but is not filtered as a no-op.
        return "printf tatg-fast-validation-skip"
    return f"mvn test -pl . -Dtest={class_name}Test -Dsurefire.failIfNoSpecifiedTests=false"


def _comma_values(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _project_dirs(subjects_root: Path) -> list[str]:
    if not subjects_root.exists():
        return []
    return sorted(
        path.name
        for path in subjects_root.iterdir()
        if path.is_dir() and (path / "src" / "main" / "java").exists()
    )


def _discover_project_targets(subjects_root: Path, project: str) -> list[BatchTarget]:
    project_root = subjects_root / project
    source_root = project_root / "src" / "main" / "java"
    if not source_root.exists():
        raise SystemExit(f"Project source root not found: {source_root}")
    targets: list[BatchTarget] = []
    for source_file in sorted(source_root.rglob("*.java")):
        if source_file.name in {"package-info.java", "module-info.java"}:
            continue
        targets.append(
            BatchTarget(
                name=source_file.stem,
                project=project,
                target_file=source_file.relative_to(project_root).as_posix(),
            )
        )
    return targets


def _find_class_target(subjects_root: Path, project: str, class_name: str) -> BatchTarget:
    matches = [target for target in _discover_project_targets(subjects_root, project) if target.name == class_name]
    if not matches:
        raise SystemExit(f"Class {class_name} not found in project {project}")
    if len(matches) > 1:
        paths = ", ".join(target.target_file for target in matches[:10])
        raise SystemExit(f"Class {class_name} is ambiguous in {project}: {paths}")
    return matches[0]


def _read_class_list(
    path: Path,
    subjects_root: Path,
    projects: list[str] | None = None,
    class_specs: list[str] | None = None,
) -> list[BatchTarget]:
    if not path.exists():
        raise SystemExit(f"Class list not found: {path}")
    wanted_projects = set(projects or [])
    wanted_classes = {_parse_class_spec(spec) for spec in (class_specs or [])}
    targets: list[BatchTarget] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or not {"project", "class"}.issubset(set(reader.fieldnames)):
            raise SystemExit("Class list CSV must contain project and class columns")
        for row in reader:
            project = str(row.get("project") or "").strip()
            class_name = str(row.get("class") or "").strip()
            if wanted_projects and project not in wanted_projects:
                continue
            if wanted_classes and (None, class_name) not in wanted_classes and (project, class_name) not in wanted_classes:
                continue
            if project and class_name:
                try:
                    targets.append(_find_class_target(subjects_root, project, class_name))
                except SystemExit as exc:
                    print(f"[class-list] skip {project}:{class_name}: {exc}", file=sys.stderr)
    return targets


def _parse_class_spec(spec: str) -> tuple[str | None, str]:
    if ":" in spec:
        project, class_name = spec.split(":", 1)
        return project.strip() or None, class_name.strip()
    if "/" in spec:
        project, class_name = spec.split("/", 1)
        return project.strip() or None, class_name.strip()
    return None, spec.strip()


def _select_targets(args: argparse.Namespace) -> list[BatchTarget]:
    subjects_root = args.subjects_root.resolve()
    projects = _comma_values(args.projects)
    class_specs = _comma_values(args.classes)

    if args.class_list:
        targets = _read_class_list(Path(args.class_list).resolve(), subjects_root, projects, class_specs)
    elif projects and not class_specs:
        targets = []
        for project in projects:
            targets.extend(_discover_project_targets(subjects_root, project))
    elif class_specs:
        search_projects = projects or _project_dirs(subjects_root)
        targets = []
        for spec in class_specs:
            project, class_name = _parse_class_spec(spec)
            if not class_name:
                continue
            if project:
                targets.append(_find_class_target(subjects_root, project, class_name))
                continue
            matches = []
            for candidate_project in search_projects:
                matches.extend(
                    target
                    for target in _discover_project_targets(subjects_root, candidate_project)
                    if target.name == class_name
                )
            if not matches:
                raise SystemExit(f"Class {class_name} not found in selected projects")
            targets.extend(matches)
    else:
        targets = list(TARGETS)

    deduped: list[BatchTarget] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        key = (target.project, target.name, target.target_file)
        if key not in seen:
            seen.add(key)
            deduped.append(target)
    if not deduped:
        raise SystemExit("No targets selected")
    return deduped


def _build_command(
    args: argparse.Namespace,
    target: BatchTarget,
    project_root: Path,
    method_name: str | None = None,
) -> list[str]:
    scope = "method" if method_name else "class"
    return [
        args.python,
        "-m",
        "tatg.cli",
        "generate",
        "--project-root",
        str(project_root),
        "--target-file",
        target.target_file,
        "--scope",
        scope,
        "--class-name",
        target.name,
        *( ["--method-name", method_name] if method_name else [] ),
        "--backend",
        "tatg",
        "--api-base",
        args.api_base,
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
        "--max-iterations",
        str(args.max_iterations),
        "--repair-iterations",
        str(args.repair_iterations),
        "--streaming-repair-mode",
        args.streaming_repair_mode,
        "--llm-timeout-seconds",
        str(args.llm_timeout_seconds),
        "--tatg-preanalysis-timeout-seconds",
        str(getattr(args, "tatg_preanalysis_timeout_seconds", 180)),
        "--log-prompts",
        "--log-unrepaired-tests",
        "--test-command",
        _test_command(args.test_command_mode, target.name),
        "--json-out",
        str(args.out_dir / f"{target.name}.{method_name}.json" if method_name else args.out_dir / f"{target.name}.json"),
        "--e2e-timeout-seconds",
        str(args.e2e_timeout_seconds),
        "--generation-parallelism",
        str(args.generation_parallelism),
        "--repair-parallelism",
        str(args.repair_parallelism),
        "--validation-parallelism",
        str(args.validation_parallelism),
    ] + (["--with-mutation"] if getattr(args, "with_mutation", False) else [])


def _json_outcome(json_out: Path) -> str:
    if not json_out.exists():
        return "json=missing"
    try:
        data = json.loads(json_out.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:  # noqa: BLE001
        return f"json=unreadable:{exc.__class__.__name__}"
    final = data.get("final_validation") or {}
    runs = data.get("test_runs") or []
    latest = runs[-1] if runs else {}
    return (
        f"json=ok final_rc={final.get('return_code')} "
        f"failed={final.get('failed_test_count')} "
        f"generated={data.get('generated_test_count')} "
        f"line={latest.get('line_coverage')} "
        f"branch={latest.get('branch_coverage')} "
        f"e2e_timed_out={data.get('e2e_timed_out')}"
    )


def _post_run_evaluate_and_merge(
    *,
    project_root: Path,
    json_out: Path,
    generated_test_count: int,
    run_mutation_if_missing: bool,
) -> tuple[bool, str]:
    if not json_out.exists():
        return (False, "json_missing")
    try:
        payload = json.loads(json_out.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:  # noqa: BLE001
        return (False, f"json_unreadable:{exc.__class__.__name__}")
    if not isinstance(payload, dict):
        return (False, "json_invalid")

    started = time.perf_counter()
    evaluator = Evaluator()
    try:
        report = evaluator.evaluate(
            project_root=project_root,
            coverage_type="jacoco",
            test_count=max(0, int(generated_test_count or 0)),
            run_coverage_if_missing=True,
            run_mutation_if_missing=bool(run_mutation_if_missing),
        ).to_dict()
    except Exception as exc:  # noqa: BLE001
        payload["post_run_evaluation"] = {
            "status": "error",
            "error": f"{exc.__class__.__name__}: {exc}",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        }
        json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return (False, f"post_run_error:{exc.__class__.__name__}")

    coverage = report.get("coverage") if isinstance(report.get("coverage"), dict) else {}
    mutation = report.get("mutation") if isinstance(report.get("mutation"), dict) else {}

    summary = payload.setdefault("report_summary", {})
    if not isinstance(summary, dict):
        summary = {}
        payload["report_summary"] = summary

    coverage_summary = summary.setdefault("coverage", {})
    if not isinstance(coverage_summary, dict):
        coverage_summary = {}
        summary["coverage"] = coverage_summary

    methods_rows = int(coverage_summary.get("methods_coverage_rows") or 0)
    if methods_rows == 0 or coverage_summary.get("line_coverage") is None:
        if coverage.get("line_coverage") is not None:
            coverage_summary["line_coverage"] = coverage.get("line_coverage")
    if methods_rows == 0 or coverage_summary.get("branch_coverage") is None:
        if coverage.get("branch_coverage") is not None:
            coverage_summary["branch_coverage"] = coverage.get("branch_coverage")
    coverage_summary["source"] = "post_run"
    coverage_summary["post_run_line_covered"] = coverage.get("line_covered")
    coverage_summary["post_run_line_missed"] = coverage.get("line_missed")
    coverage_summary["post_run_branch_covered"] = coverage.get("branch_covered")
    coverage_summary["post_run_branch_missed"] = coverage.get("branch_missed")

    mutation_summary = summary.setdefault("mutation", {})
    if not isinstance(mutation_summary, dict):
        mutation_summary = {}
        summary["mutation"] = mutation_summary
    mutation_summary["post_run_total_mutations"] = mutation.get("total_mutations")
    mutation_summary["post_run_killed_mutations"] = mutation.get("killed_mutations")
    mutation_summary["post_run_survived_mutations"] = mutation.get("survived_mutations")
    mutation_summary["post_run_mutation_score_percent"] = mutation.get("mutation_score")
    if mutation_summary.get("evaluation_status") is None and mutation.get("total_mutations"):
        mutation_summary["evaluation_status"] = "post_run"

    timing_summary = summary.setdefault("timing", {})
    if isinstance(timing_summary, dict):
        timing_summary["post_run_evaluation_seconds"] = round(time.perf_counter() - started, 3)

    payload["post_run_evaluation"] = {
        "status": "ok",
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "report": report,
    }

    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return (True, "post_run_ok")


def _run_command_with_timestamped_log(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
    progress_log_file: Path,
    show_progress: bool = False,
    timeout_seconds: int | None = None,
) -> int:
    def _startup_artifact_monitor_config() -> tuple[list[Path], int]:
        raw_paths = env.get("TATG_STARTUP_ARTIFACT_PATHS", "")
        paths = [Path(part) for part in raw_paths.split(os.pathsep) if part.strip()]
        try:
            timeout = int(env.get("TATG_STARTUP_ARTIFACT_TIMEOUT_SECONDS", "0") or "0")
        except ValueError:
            timeout = 0
        return paths, max(0, timeout)

    def _artifact_signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
        signature: list[tuple[str, int, int]] = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                continue
            signature.append((str(path), int(stat.st_mtime_ns), int(stat.st_size)))
        return tuple(signature)

    def _stale_partial_monitor_config() -> tuple[list[Path], int, set[str]]:
        raw_paths = env.get("TATG_STALE_PARTIAL_PATHS", "")
        paths = [Path(part) for part in raw_paths.split(os.pathsep) if part.strip()]
        try:
            timeout = int(env.get("TATG_STALE_PARTIAL_STAGE_TIMEOUT_SECONDS", "0") or "0")
        except ValueError:
            timeout = 0
        raw_stages = env.get(
            "TATG_STALE_PARTIAL_STAGES",
            "post_coverage_generation,deferred_streaming_repair_done,post_mutation_extra,"
            "coverage_feedback_loop_done,post_generation_count_done,post_repair_pipeline_done,"
            "repair_summary_done,progress_finish_done,final_report_assembly_start",
        )
        stages = {
            part.strip()
            for part in raw_stages.split(",")
            if part.strip()
        }
        return paths, max(0, timeout), stages

    def _stale_partial_stage(path: Path, stages: set[str]) -> tuple[str, float] | None:
        try:
            stat = path.stat()
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        stage = str(payload.get("stage") or "").strip()
        if not stage:
            return None
        matched = stage in stages or any(stage.startswith(prefix) for prefix in stages)
        if not matched:
            return None
        return stage, float(stat.st_mtime)

    def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            process.wait(timeout=5)
            return
        except Exception:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    with log_file.open("w", encoding="utf-8") as out:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out.write(f"[{stamp}] command: {shlex.join(command)}\n")
        if timeout_seconds and timeout_seconds > 0:
            out.write(f"[{stamp}] method_watchdog_timeout_seconds: {int(timeout_seconds)}\n")
        # Live progress rendering writes directly to the terminal. If the
        # process is not in the foreground process group, shells with job
        # control may suspend it (SIGTTOU: "suspended (tty output)").
        # Keep progress enabled only for safe foreground TTY contexts unless
        # explicitly overridden.
        if show_progress:
            force_tty_progress = env.get("TATG_FORCE_TTY_PROGRESS", "0") == "1"
            try:
                tty_ready = sys.stdout.isatty()
            except Exception:  # noqa: BLE001
                tty_ready = False
            if tty_ready:
                try:
                    fg_ready = os.tcgetpgrp(sys.stdout.fileno()) == os.getpgrp()
                except Exception:  # noqa: BLE001
                    # macOS raises OSError [Errno 25] for some terminal emulators;
                    # treat as foreground if TTY is confirmed above.
                    fg_ready = True
            else:
                fg_ready = False
            if not tty_ready:
                out.write(
                    f"[{stamp}] show_progress requested but no interactive TTY; "
                    "falling back to log-only mode.\n"
                )
                sys.stderr.write(
                    "[tatg] progress bar disabled: stdout is not a TTY.\n"
                )
                show_progress = False
            elif not fg_ready and not force_tty_progress:
                out.write(
                    f"[{stamp}] show_progress requested outside foreground TTY; "
                    "falling back to log-only mode to avoid SIGTTOU suspension. "
                    "Set TATG_FORCE_TTY_PROGRESS=1 to force it.\n"
                )
                sys.stderr.write(
                    "[tatg] progress bar disabled: not in foreground process group. "
                    "Set TATG_FORCE_TTY_PROGRESS=1 to force.\n"
                )
                show_progress = False
        if show_progress:
            # Use `script` to preserve child TTY behavior (live progress panel)
            # while writing a full terminal transcript to a sidecar file.
            progress_log_file.unlink(missing_ok=True)
            script_bin = shutil.which("script")
            if not script_bin:
                out.write(
                    f"[{stamp}] show_progress requested but 'script' command is unavailable; "
                    "falling back to log-only mode.\n"
                )
                show_progress = False
            else:
                wrapped = [
                    script_bin,
                    "-q",
                    str(progress_log_file),
                    *command,
                ]
                out.write(f"[{stamp}] progress_log: {progress_log_file}\n")
                process = subprocess.Popen(
                    wrapped,
                    cwd=cwd,
                    env=env,
                    stdout=None,
                    stderr=None,
                    text=True,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    process.wait(timeout=timeout_seconds if timeout_seconds and timeout_seconds > 0 else None)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    timeout_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    out.write(
                        f"[{timeout_stamp}] method_watchdog_timeout: exceeded "
                        f"{int(timeout_seconds or 0)}s; terminating process tree\n"
                    )
                    _terminate_process_tree(process)
                    process.returncode = 124
                # Keep the main log useful by embedding a small tail from the
                # transcript sidecar when live TTY mode is enabled.
                if progress_log_file.exists():
                    try:
                        lines = progress_log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
                        tail = lines[-200:]
                        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        out.write(
                            f"[{stamp}] progress_log_tail_lines={len(tail)} "
                            f"(from {progress_log_file})\n"
                        )
                        for line in tail:
                            out.write(f"[{stamp}] [progress] {line}\n")
                    except Exception as exc:  # noqa: BLE001
                        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        out.write(f"[{stamp}] failed_to_read_progress_log: {exc}\n")
                if timed_out:
                    process.returncode = 124
        else:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            startup_paths, startup_timeout = _startup_artifact_monitor_config()
            stale_partial_paths, stale_partial_timeout, stale_partial_stages = _stale_partial_monitor_config()
            use_incremental_monitor = (
                bool(startup_paths)
                or bool(stale_partial_paths and stale_partial_timeout > 0)
                or bool(timeout_seconds and timeout_seconds > 0)
            )
            if use_incremental_monitor:
                started = time.monotonic()
                last_artifact_signature = _artifact_signature(startup_paths)
                # A caller may pre-write a batch-level checkpoint before the
                # child process starts.  That is useful for crash recovery, but
                # it must not satisfy the startup monitor.  Treat startup as
                # observed only after the child updates or creates one of the
                # watched artifacts.
                artifact_seen = False
                timed_out = False
                startup_timed_out = False
                stale_partial_timed_out = False
                selector = selectors.DefaultSelector()
                stdout_fd: int | None = None
                stdout_decoder = codecs.getincrementaldecoder("utf-8")(
                    errors="replace",
                )
                pending_stdout = ""

                def _write_stdout_chunk(
                    chunk: bytes,
                    *,
                    final: bool = False,
                ) -> None:
                    nonlocal pending_stdout
                    pending_stdout += stdout_decoder.decode(
                        chunk,
                        final=final,
                    )
                    while "\n" in pending_stdout:
                        line, pending_stdout = pending_stdout.split(
                            "\n",
                            1,
                        )
                        line_stamp = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        out.write(f"[{line_stamp}] {line}\n")
                    if final and pending_stdout:
                        line_stamp = datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        )
                        out.write(f"[{line_stamp}] {pending_stdout}\n")
                        pending_stdout = ""
                    out.flush()

                if process.stdout is not None:
                    stdout_fd = process.stdout.fileno()
                    os.set_blocking(stdout_fd, False)
                    selector.register(stdout_fd, selectors.EVENT_READ)
                try:
                    while process.poll() is None:
                        for key, _mask in selector.select(timeout=1.0):
                            try:
                                chunk = os.read(int(key.fd), 65536)
                            except BlockingIOError:
                                continue
                            if chunk:
                                _write_stdout_chunk(chunk)
                        now = time.monotonic()
                        if startup_paths and not artifact_seen:
                            current_signature = _artifact_signature(startup_paths)
                            if current_signature and current_signature != last_artifact_signature:
                                artifact_seen = True
                                last_artifact_signature = current_signature
                                seen_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                out.write(
                                    f"[{seen_stamp}] startup_artifact_seen: "
                                    f"{', '.join(path for path, _mtime, _size in current_signature[:4])}\n"
                                )
                                out.flush()
                            elif startup_timeout > 0 and now - started >= startup_timeout:
                                startup_timed_out = True
                                timeout_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                out.write(
                                    f"[{timeout_stamp}] startup_artifact_timeout: no JSON partial, "
                                    f"prompt sidecar, or final artifact within {startup_timeout}s; "
                                    "terminating process tree\n"
                                )
                                _terminate_process_tree(process)
                                process.returncode = 124
                                break
                        if timeout_seconds and timeout_seconds > 0 and now - started >= timeout_seconds:
                            timed_out = True
                            timeout_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            out.write(
                                f"[{timeout_stamp}] method_watchdog_timeout: exceeded "
                                f"{int(timeout_seconds)}s; terminating process tree\n"
                            )
                            _terminate_process_tree(process)
                            process.returncode = 124
                            break
                        if stale_partial_paths and stale_partial_timeout > 0:
                            stale_matches: list[tuple[Path, str, float]] = []
                            for partial_path in stale_partial_paths:
                                stale = _stale_partial_stage(partial_path, stale_partial_stages)
                                if stale is None:
                                    continue
                                stage, mtime = stale
                                if time.time() - mtime >= stale_partial_timeout:
                                    stale_matches.append((partial_path, stage, time.time() - mtime))
                            if stale_matches:
                                stale_partial_timed_out = True
                                timeout_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                rendered = ", ".join(
                                    f"{path}:{stage}:stale={age:.1f}s"
                                    for path, stage, age in stale_matches[:4]
                                )
                                out.write(
                                    f"[{timeout_stamp}] generation_worker_stalled: partial artifact stage "
                                    f"stale for >= {stale_partial_timeout}s ({rendered}); "
                                    "terminating process tree\n"
                                )
                                _terminate_process_tree(process)
                                process.returncode = 124
                                break
                    if stdout_fd is not None:
                        # Drain buffered bytes without re-entering text
                        # ``readline()``. A child may end with an unterminated
                        # progress fragment, which must never suspend the
                        # watchdog loop or final collection.
                        while True:
                            try:
                                chunk = os.read(stdout_fd, 65536)
                            except BlockingIOError:
                                break
                            if not chunk:
                                break
                            _write_stdout_chunk(chunk)
                        _write_stdout_chunk(b"", final=True)
                finally:
                    try:
                        selector.close()
                    except Exception:
                        pass
                if timed_out or startup_timed_out or stale_partial_timed_out:
                    process.returncode = 124
                elif process.returncode is None:
                    process.wait()
            else:
                assert process.stdout is not None
                for line in process.stdout:
                    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    out.write(f"[{stamp}] {line}")
                    out.flush()
                process.wait()
        end_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out.write(f"[{end_stamp}] return_code: {int(process.returncode or 0)}\n")
        return int(process.returncode or 0)


def _run_one(args: argparse.Namespace, target: BatchTarget, env: dict[str, str], method_name: str | None = None) -> int:
    subjects_root = args.subjects_root.resolve()
    project_root = subjects_root / target.project
    output_stem = f"{target.name}.{method_name}" if method_name else target.name
    json_out = args.out_dir / f"{output_stem}.json"
    log_out = args.out_dir / f"{output_stem}.log"
    progress_log_out = args.out_dir / f"{output_stem}.progress.log"
    generated_out = args.out_dir / "generated-tests" / output_stem

    command = _build_command(args, target, project_root, method_name=method_name)
    label = output_stem
    print(f"[{label}] command: {' '.join(command)}")
    if args.dry_run:
        return 0

    if json_out.exists() and not args.force:
        print(f"[{label}] skip: {json_out} already exists (use --force to rerun)")
        return 0
    if args.force:
        json_out.unlink(missing_ok=True)
        (args.out_dir / f"{output_stem}.json.partial").unlink(missing_ok=True)
        log_out.unlink(missing_ok=True)
        progress_log_out.unlink(missing_ok=True)

    backup_root = args.out_dir / ".tree-backups" / output_stem
    if backup_root.exists():
        shutil.rmtree(backup_root)
    before = _backup_java_files(project_root, backup_root)
    start = time.monotonic()
    print(f"[{label}] starting; log={log_out} progress_log={progress_log_out}")
    process_returncode = 130
    copied: list[Path] = []
    post_eval_result = "post_run_skipped"
    try:
        process_returncode = _run_command_with_timestamped_log(
            command,
            project_root,
            env,
            log_out,
            progress_log_out,
            show_progress=bool(getattr(args, "show_progress", False)),
        )
        if bool(getattr(args, "post_run_evaluate", False)) and json_out.exists():
            json_data = _safe_read_json(json_out)
            ok, msg = _post_run_evaluate_and_merge(
                project_root=project_root,
                json_out=json_out,
                generated_test_count=int(json_data.get("generated_test_count") or 0),
                run_mutation_if_missing=bool(getattr(args, "with_mutation", False)),
            )
            post_eval_result = msg
        copied = _copy_changed_java_files(project_root, before, backup_root, generated_out)
    finally:
        _restore_java_files(project_root, before, backup_root)
    elapsed = time.monotonic() - start
    print(
        f"[{label}] rc={process_returncode} elapsed={elapsed:.1f}s "
        f"changed_files_copied={len(copied)} {post_eval_result} {_json_outcome(json_out)}"
    )
    return process_returncode


def _safe_read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


def _pipeline_rc(return_codes: object) -> int | None:
    if not isinstance(return_codes, list) or not return_codes:
        return None
    values = [int(code) for code in return_codes if isinstance(code, (int, float))]
    return max(values) if values else None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="Run Rotation, GaussNewtonOptimizer, FromXmlParser, and ToXmlGenerator from source.",
    )
    p.add_argument(
        "--subjects-root",
        type=Path,
        default=Path(
            os.environ.get(
                "TATG_SUBJECTS_ROOT", "/path/to/defects4j-subjects-notests"
            )
        ),
        help="Root directory containing Math-2f and JacksonXml-5f.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "TATG_FOUR_CLASS_OUT_DIR",
                str(Path(tempfile.gettempdir()) / "tatg-four-class-source"),
            )
        ),
        help="Directory for JSON logs and copied generated test files.",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run -m tatg.cli.",
    )
    p.add_argument("--api-base", default="", help="Defaults to API_BASE from shell environment.")
    p.add_argument("--model", default="", help="Defaults to MODEL from shell environment.")
    p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    p.add_argument("--skip-api-preflight", action="store_true")
    p.add_argument("--max-iterations", type=int, default=1)
    p.add_argument("--repair-iterations", type=int, default=2)
    p.add_argument("--streaming-repair-mode", choices=["deferred", "eager"], default="deferred")
    p.add_argument("--llm-timeout-seconds", type=int, default=60)
    p.add_argument("--tatg-preanalysis-timeout-seconds", type=int, default=180)
    p.add_argument("--generation-parallelism", type=int, default=1)
    p.add_argument("--repair-parallelism", type=int, default=1)
    p.add_argument("--validation-parallelism", type=int, default=4)
    p.add_argument("--with-mutation", action="store_true", help="Run PITest mutation evaluation after validation.")
    p.add_argument(
        "--e2e-timeout-seconds",
        type=int,
        default=172800,
        help="Per-class end-to-end timeout passed to tatg (default: 48h).",
    )
    p.add_argument(
        "--test-command-mode",
        choices=("maven", "true"),
        default="maven",
        help="Use Maven validation/coverage or the fast true command.",
    )
    p.add_argument(
        "--classes",
        default="",
        help="Comma-separated class specs to run. Use Class, Project:Class, or Project/Class.",
    )
    p.add_argument(
        "--projects",
        default="",
        help="Comma-separated Defects4J project ids. With no --classes, runs all classes in those projects.",
    )
    p.add_argument(
        "--class-list",
        default="",
        help="CSV with project,class columns, e.g. PLRDGEN/panta/evaluation/class_list.csv.",
    )
    p.add_argument(
        "--methods",
        default="",
        help="Comma-separated method names for quick method-scope runs. Default: class-scope runs.",
    )
    p.add_argument("--force", action="store_true", help="Overwrite existing class JSON/log files.")
    p.add_argument(
        "--show-progress",
        action="store_true",
        help=(
            "Keep the live TTY progress panel visible in terminal. "
            "Child stdout is logged to file to avoid redraw corruption."
        ),
    )
    p.add_argument(
        "--post-run-evaluate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run a post-run coverage/mutation evaluation pass and merge results into the same JSON report.",
    )
    p.add_argument("--dry-run", action="store_true", help="Print commands without running generation.")
    args = p.parse_args(argv)
    args.repo_root = repo
    args.out_dir = args.out_dir.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    try:
        api_config = resolve_api_config(
            api_base=args.api_base,
            model=args.model,
            api_key_env=args.api_key_env,
            env=env,
        )
        if (not args.dry_run) and (not args.skip_api_preflight):
            preflight_api_config(api_config, timeout_seconds=min(15, max(5, int(args.llm_timeout_seconds))))
        apply_api_config_to_env(api_config, env)
        args.api_base = api_config.api_base
        args.model = api_config.model
        args.api_key_env = api_config.api_key_env
    except Exception as exc:
        print(f"API configuration error: {exc}", file=sys.stderr)
        return 1
    src_path = str(args.repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"

    print(f"repo_root={args.repo_root}")
    print(f"PYTHONPATH starts with {src_path}")
    print(f"out_dir={args.out_dir}")

    rc = 0
    method_names = [name.strip() for name in str(args.methods or "").split(",") if name.strip()]
    targets = _select_targets(args)
    print(f"selected_targets={len(targets)}")
    suspicious: list[tuple[str, int]] = []
    for target in targets:
        for method_name in (method_names if method_names else [None]):  # type: ignore[assignment]
            rc = max(rc, _run_one(args, target, env, method_name=method_name))
            output_stem = f"{target.name}.{method_name}" if method_name else target.name
            json_out = args.out_dir / f"{output_stem}.json"
            data = _safe_read_json(json_out)
            p_rc = _pipeline_rc(data.get("return_codes"))
            final_rc = (data.get("final_validation") or {}).get("return_code")
            if (p_rc is not None and p_rc != 0) and final_rc == 0:
                suspicious.append((output_stem, p_rc))

    if suspicious:
        print(
            f"\n[WARN] {len(suspicious)} run(s) had pipeline_rc!=0 but final_validation_rc==0"
            " — intermediate build/compile failure may have inflated coverage:"
        )
        for label, p_rc in suspicious:
            print(f"  {label}  pipeline_rc={p_rc}")

    print("batch complete")
    return rc


if __name__ == "__main__":
    sys.exit(main())
