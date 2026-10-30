#!/usr/bin/env python3
"""Run source-checkout TATG batches for hard methods or whole projects.

Default mode runs the curated HARD_METHODS list in method scope. You can also
switch to project-wide or project-class modes for broader scans.
"""

from __future__ import annotations

import csv
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tatg.api_config import DEFAULT_API_KEY_ENV, apply_api_config_to_env, preflight_api_config, resolve_api_config
from tatg.targeting import _extract_methods

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_four_class_source_batch import (
    TARGETS,
    _backup_java_files,
    _copy_changed_java_files,
    _json_outcome,
    _post_run_evaluate_and_merge,
    _repo_root,
    _restore_java_files,
    _run_command_with_timestamped_log,
    _run_one,
    _safe_read_json,
    _test_command,
)


HARD_METHODS: dict[str, tuple[str, ...]] = {
    "Rotation": ("getAngles",),
    "GaussNewtonOptimizer": ("doOptimize",),
    "FromXmlParser": ("nextTextValue",),
    "ToXmlGenerator": ("writeString",),
    "Metaphone": ("metaphone",),
    "NumberUtils": ("createNumber",),
    "FilteringParserDelegate": ("nextToken",),
    "Flat3Map": ("remove",),
    "DateTimeFormat": ("parsePatternTo",),
    "Entities": ("escape",),
}

PROJECTS: tuple[str, ...] = tuple(
    project.strip()
    for project in os.environ.get("TATG_PROJECTS", "JacksonDatabind-112f,Math-2f").split(",")
    if project.strip()
)

# Set TATG_PROJECT_CLASS_LIMIT to a positive integer for smoke testing.
# 0 means run every discovered class.
MAX_CLASSES_PER_PROJECT = int(os.environ.get("TATG_PROJECT_CLASS_LIMIT", "0") or "0")

# Project-wide mode runs one TATG invocation per project. Target discovery
# now strips Java comments/literals before class extraction, so it avoids the
# old comment false positives while keeping one final coverage/mutation pass per
# project. Set TATG_PROJECT_WIDE=0 for per-class resumable fallback.
PROJECT_WIDE = os.environ.get("TATG_PROJECT_WIDE", "1") != "0"

SUBJECTS_ROOT = Path(
    os.environ.get("TATG_SUBJECTS_ROOT", "/path/to/defects4j-subjects-notests")
)
OUT_DIR = Path(
    os.environ.get(
        "TATG_HARD_METHOD_SOURCE_OUT_DIR",
        str(Path(tempfile.gettempdir()) / "tatg-hard-method-source"),
    )
)
PYTHON = os.environ.get("TATG_PYTHON", sys.executable)
API_BASE = os.environ.get("TATG_API_BASE", os.environ.get("API_BASE", ""))
MODEL = os.environ.get("TATG_MODEL", os.environ.get("MODEL", ""))
API_KEY_ENV = os.environ.get("TATG_API_KEY_ENV", DEFAULT_API_KEY_ENV)
MAX_ITERATIONS = int(os.environ.get("TATG_MAX_ITERATIONS", "2") or "2")
REPAIR_ITERATIONS = int(os.environ.get("TATG_REPAIR_ITERATIONS", "2") or "2")
STREAMING_REPAIR_MODE = os.environ.get("TATG_STREAMING_REPAIR_MODE", "deferred")
LLM_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_LLM_TIMEOUT_SECONDS", "60") or "60"
)
TATG_PREANALYSIS_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_PREANALYSIS_TIMEOUT_SECONDS", "600") or "600"
)
TATG_GENERATION_TIMEOUT_SECONDS = int(
    os.environ.get(
        "TATG_GENERATION_TIMEOUT_SECONDS",
        str(max(480, LLM_TIMEOUT_SECONDS * 3 + 60)),
    )
    or str(max(480, LLM_TIMEOUT_SECONDS * 3 + 60))
)
BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS", "900")
    or "900"
)
# Project-wide batches include thousands of method targets; 0 disables the
# global e2e deadline while preserving per-LLM and per-test timeouts.
E2E_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_E2E_TIMEOUT_SECONDS", "0") or "0"
)
TEST_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_TEST_TIMEOUT_SECONDS", "900") or "900"
)
GENERATION_PARALLELISM = 2
REPAIR_PARALLELISM = 2
VALIDATION_PARALLELISM = 4
WITH_MUTATION = os.environ.get("TATG_WITH_MUTATION", "1") != "0"
TEST_COMMAND_MODE = "maven"
FORCE = False
DRY_RUN = os.environ.get("TATG_DRY_RUN", "0") == "1"
POST_RUN_EVALUATE = os.environ.get("TATG_POST_RUN_EVALUATE", "1") != "0"
SHOW_PROGRESS = os.environ.get("TATG_SHOW_PROGRESS", "1") != "0"


def _comma_values(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _select_hard_methods(selector: str) -> dict[str, tuple[str, ...]]:
    """Select a subset of HARD_METHODS.

    Selector tokens can be:
    - method name, e.g. ``writeString``
    - class name, e.g. ``ToXmlGenerator``
    - class.method, e.g. ``ToXmlGenerator.writeString``
    """

    tokens = _comma_values(selector)
    if not tokens:
        return HARD_METHODS

    selected: dict[str, list[str]] = {}

    def _add(class_name: str, method_name: str) -> None:
        selected.setdefault(class_name, [])
        if method_name not in selected[class_name]:
            selected[class_name].append(method_name)

    class_to_methods = {name: set(methods) for name, methods in HARD_METHODS.items()}
    method_to_classes: dict[str, list[str]] = {}
    for class_name, methods in HARD_METHODS.items():
        for method_name in methods:
            method_to_classes.setdefault(method_name, []).append(class_name)

    for token in tokens:
        if "." in token:
            class_name, method_name = token.split(".", 1)
            class_name = class_name.strip()
            method_name = method_name.strip()
            if class_name not in class_to_methods:
                raise SystemExit(f"Unknown hard-method class selector: {class_name}")
            if method_name not in class_to_methods[class_name]:
                raise SystemExit(f"Unknown hard-method selector: {class_name}.{method_name}")
            _add(class_name, method_name)
            continue

        if token in class_to_methods:
            for method_name in HARD_METHODS[token]:
                _add(token, method_name)
            continue

        classes = method_to_classes.get(token)
        if not classes:
            raise SystemExit(
                "Unknown hard-method selector: "
                f"{token}. Use method, class, or class.method from HARD_METHODS."
            )
        for class_name in classes:
            _add(class_name, token)

    ordered: dict[str, tuple[str, ...]] = {}
    for class_name, methods in HARD_METHODS.items():
        chosen = selected.get(class_name)
        if chosen:
            ordered[class_name] = tuple(method for method in methods if method in chosen)
    return ordered


@dataclass(frozen=True)
class ProjectClassTarget:
    project: str
    class_name: str
    target_file: str
    output_id: str
    method_count: int


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hard-method or project-wide TATG source batches.")
    parser.add_argument(
        "--mode",
        choices=("hard-methods", "project-wide", "project-class"),
        default="hard-methods",
        help=(
            "Execution mode. 'hard-methods' runs the HARD_METHODS list (default); "
            "'project-wide' runs one singlemodule job per project; "
            "'project-class' runs class-scope jobs across discovered classes."
        ),
    )
    parser.add_argument(
        "--methods",
        default="",
        help=(
            "Hard-method selector list (comma-separated). Supports method, class, or class.method, "
            "e.g. 'writeString,remove' or 'ToXmlGenerator.writeString,Flat3Map.remove'. "
            "Only applies when --mode hard-methods."
        ),
    )
    parser.add_argument(
        "--list-hard-methods",
        action="store_true",
        help="Print available HARD_METHODS selectors and exit.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help=(
            "Display live TTY progress panel while running. "
            "Use --no-show-progress to force log-only mode."
        ),
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=FORCE,
        help="Delete existing JSON/log outputs for each selected target before running.",
    )
    parser.add_argument(
        "--tatg-preanalysis-timeout-seconds",
        type=int,
        default=TATG_PREANALYSIS_TIMEOUT_SECONDS,
        help=(
            "Per-target timeout (seconds) for bounded TATG preanalysis before formal generation. "
            "Applies to hard-method, class, and project-wide modes."
        ),
    )
    parser.add_argument(
        "--tatg-generation-timeout-seconds",
        type=int,
        default=TATG_GENERATION_TIMEOUT_SECONDS,
        help=(
            "Per-target hard timeout (seconds) for TATG generation worker. "
            "This is exported as TATG_GENERATION_TIMEOUT_SECONDS."
        ),
    )
    return parser.parse_args()


def _target_method_coverage(json_path: Path, method_name: str) -> str:
    if not json_path.exists():
        return "target_method_cov=json_missing"
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:  # noqa: BLE001
        return f"target_method_cov=json_unreadable:{exc.__class__.__name__}"
    runs = data.get("test_runs") or []
    methods = []
    for run in runs:
        methods.extend(run.get("methods_coverage") or [])
    matches = [m for m in methods if m.get("method_name") == method_name]
    if not matches:
        return "target_method_cov=missing"
    best = max(
        matches,
        key=lambda item: float(item.get("line_coverage") or 0.0) + float(item.get("branch_coverage") or 0.0),
    )
    return (
        f"target_method_cov=line={best.get('line_coverage')} "
        f"branch={best.get('branch_coverage')} "
        f"line_missed={best.get('line_missed')} "
        f"branch_missed={best.get('branch_missed')} "
        f"descriptor={best.get('descriptor')}"
    )


def _pct(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:  # noqa: BLE001
        return ""


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pipeline_rc(return_codes: object) -> int | None:
    if not isinstance(return_codes, list) or not return_codes:
        return None
    values = [int(code) for code in return_codes if isinstance(code, (int, float))]
    return max(values) if values else None


def _has_nonzero_intermediate_rc(return_codes: object) -> bool:
    if not isinstance(return_codes, list):
        return False
    return any(int(code) != 0 for code in return_codes if isinstance(code, (int, float)))


def _best_method_cov_pair(data: dict, class_name: str, method_name: str) -> tuple[float | None, float | None]:
    runs = data.get("test_runs") or []
    methods: list[dict] = []
    for run in runs:
        methods.extend(run.get("methods_coverage") or [])
    matches = [
        row for row in methods
        if row.get("method_name") == method_name and row.get("class_name") == class_name
    ]
    if not matches:
        return (None, None)
    best = max(
        matches,
        key=lambda item: float(item.get("line_coverage") or 0.0) + float(item.get("branch_coverage") or 0.0),
    )
    line = best.get("line_coverage")
    branch = best.get("branch_coverage")
    return (
        float(line) if isinstance(line, (int, float)) else None,
        float(branch) if isinstance(branch, (int, float)) else None,
    )


def _safe_output_id(project_root: Path, source_file: Path) -> str:
    rel = source_file.relative_to(project_root / "src" / "main" / "java")
    return "__".join(rel.with_suffix("").parts)


def _has_primary_type(source: str, class_name: str) -> bool:
    return bool(
        re.search(
            rf"\b(?:class|interface|enum|record|@interface)\s+{re.escape(class_name)}\b",
            source,
        )
    )


def _discover_project_class_targets(subjects_root: Path, project: str) -> list[ProjectClassTarget]:
    project_root = subjects_root / project
    source_root = project_root / "src" / "main" / "java"
    targets: list[ProjectClassTarget] = []
    if not source_root.exists():
        return targets

    for source_file in sorted(source_root.rglob("*.java")):
        if source_file.name in {"package-info.java", "module-info.java"}:
            continue
        class_name = source_file.stem
        source = source_file.read_text(encoding="utf-8", errors="ignore")
        if not _has_primary_type(source, class_name):
            continue
        methods = _extract_methods(source_file, source, class_name)
        if not methods:
            continue
        targets.append(
            ProjectClassTarget(
                project=project,
                class_name=class_name,
                target_file=source_file.relative_to(project_root).as_posix(),
                output_id=_safe_output_id(project_root, source_file),
                method_count=len(methods),
            )
        )
    if MAX_CLASSES_PER_PROJECT > 0:
        return targets[:MAX_CLASSES_PER_PROJECT]
    return targets


def _read_json(json_path: Path) -> dict:
    if not json_path.exists():
        return {}
    try:
        return json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return {}


def _class_summary_row(project: str, target: ProjectClassTarget, json_path: Path) -> dict[str, object]:
    data = _read_json(json_path)
    runs = data.get("test_runs") or []
    latest = runs[-1] if runs else {}
    final = data.get("final_validation") or {}
    method_rows = [
        row for row in (latest.get("methods_coverage") or [])
        if row.get("class_name") == target.class_name
    ]
    method_line_values = [float(row.get("line_coverage")) for row in method_rows if isinstance(row.get("line_coverage"), (int, float))]
    method_branch_values = [float(row.get("branch_coverage")) for row in method_rows if isinstance(row.get("branch_coverage"), (int, float))]
    mutation_scores = [float(row.get("mutation_score")) for row in method_rows if isinstance(row.get("mutation_score"), (int, float))]
    mutation_total = sum(int(row.get("mutation_total") or 0) for row in method_rows)
    mutation_killed = sum(int(row.get("mutation_killed") or 0) for row in method_rows)
    mutation_survived = sum(int(row.get("mutation_survived") or 0) for row in method_rows)
    mutation_score = (mutation_killed / mutation_total * 100.0) if mutation_total else _avg(mutation_scores)
    mutation_eval = data.get("mutation_evaluation") or {}
    return_codes = data.get("return_codes")
    pipeline_rc = _pipeline_rc(return_codes)
    final_rc = final.get("return_code")
    generation_timeout_count = int((data.get("validation_policy") or {}).get("generation_timeout_count") or 0)
    return {
        "project": project,
        "class_name": target.class_name,
        "target_file": target.target_file,
        "method_targets": target.method_count,
        "json_path": str(json_path),
        # Legacy columns kept for backward compatibility.
        "rc": pipeline_rc,
        "final_rc": final_rc,
        # Explicit, non-ambiguous status columns.
        "pipeline_rc": pipeline_rc,
        "final_validation_rc": final_rc,
        "has_nonzero_intermediate_rc": _has_nonzero_intermediate_rc(return_codes),
        "generation_timeout_count": generation_timeout_count,
        "failed_tests": final.get("failed_test_count"),
        "generated_tests": data.get("generated_test_count"),
        "e2e_timed_out": data.get("e2e_timed_out"),
        "overall_line": latest.get("line_coverage"),
        "overall_branch": latest.get("branch_coverage"),
        "class_method_line_avg": _avg(method_line_values),
        "class_method_branch_avg": _avg(method_branch_values),
        "mutation_status": mutation_eval.get("status"),
        "mutation_total": mutation_total or None,
        "mutation_killed": mutation_killed or None,
        "mutation_survived": mutation_survived or None,
        "mutation_score": mutation_score,
    }


def _write_reports(out_dir: Path, targets_by_project: dict[str, list[ProjectClassTarget]]) -> None:
    rows: list[dict[str, object]] = []
    for project, targets in targets_by_project.items():
        project_out = out_dir / project
        for target in targets:
            rows.append(_class_summary_row(project, target, project_out / f"{target.output_id}.json"))

    csv_path = out_dir / "coverage_mutation_report.csv"
    md_path = out_dir / "coverage_mutation_report.md"
    json_path = out_dir / "coverage_mutation_report.json"
    fieldnames = [
        "project", "class_name", "target_file", "method_targets", "json_path",
        "rc", "final_rc", "pipeline_rc", "final_validation_rc", "has_nonzero_intermediate_rc",
        "generation_timeout_count", "failed_tests", "generated_tests", "e2e_timed_out",
        "overall_line", "overall_branch", "class_method_line_avg", "class_method_branch_avg",
        "mutation_status", "mutation_total", "mutation_killed", "mutation_survived", "mutation_score",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    lines = ["# TATG Project Coverage/Mutation Report", ""]
    for project in targets_by_project:
        project_rows = [row for row in rows if row["project"] == project]
        completed = [row for row in project_rows if Path(str(row["json_path"])).exists()]
        passed = [row for row in completed if row.get("final_rc") == 0]
        line_avg = _avg([float(row["overall_line"]) for row in completed if isinstance(row.get("overall_line"), (int, float))])
        branch_avg = _avg([float(row["overall_branch"]) for row in completed if isinstance(row.get("overall_branch"), (int, float))])
        mutation_scores = [float(row["mutation_score"]) for row in completed if isinstance(row.get("mutation_score"), (int, float))]
        lines.extend([
            f"## {project}",
            "",
            f"- classes discovered: {len(project_rows)}",
            f"- classes completed: {len(completed)}",
            f"- classes final green: {len(passed)}",
            f"- classes with pipeline warnings (pipeline_rc != 0): {sum(1 for row in completed if row.get('pipeline_rc') not in (None, 0))}",
            f"- generated tests: {sum(int(row.get('generated_tests') or 0) for row in completed)}",
            f"- average overall line coverage: {_pct(line_avg)}",
            f"- average overall branch coverage: {_pct(branch_avg)}",
            f"- average class mutation score: {_pct(_avg(mutation_scores) / 100.0 if mutation_scores else None)}",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report_csv={csv_path}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")


def _project_json_summary(project: str, json_path: Path) -> dict[str, object]:
    data = _read_json(json_path)
    runs = data.get("test_runs") or []
    latest = runs[-1] if runs else {}
    method_rows = latest.get("methods_coverage") or []
    mutation_scores = [float(row.get("mutation_score")) for row in method_rows if isinstance(row.get("mutation_score"), (int, float))]
    mutation_total = sum(int(row.get("mutation_total") or 0) for row in method_rows)
    mutation_killed = sum(int(row.get("mutation_killed") or 0) for row in method_rows)
    mutation_survived = sum(int(row.get("mutation_survived") or 0) for row in method_rows)
    mutation_score = (mutation_killed / mutation_total * 100.0) if mutation_total else _avg(mutation_scores)
    final = data.get("final_validation") or {}
    mutation_eval = data.get("mutation_evaluation") or {}
    return_codes = data.get("return_codes")
    pipeline_rc = _pipeline_rc(return_codes)
    generation_timeout_count = int((data.get("validation_policy") or {}).get("generation_timeout_count") or 0)
    return {
        "project": project,
        "json_path": str(json_path),
        "return_codes": return_codes,
        "pipeline_rc": pipeline_rc,
        "has_nonzero_intermediate_rc": _has_nonzero_intermediate_rc(return_codes),
        "final_rc": final.get("return_code"),
        "final_validation_rc": final.get("return_code"),
        "generation_timeout_count": generation_timeout_count,
        "failed_tests": final.get("failed_test_count"),
        "generated_tests": data.get("generated_test_count"),
        "e2e_timed_out": data.get("e2e_timed_out"),
        "overall_line": latest.get("line_coverage"),
        "overall_branch": latest.get("branch_coverage"),
        "covered_methods": len(method_rows),
        "mutation_status": mutation_eval.get("status"),
        "mutation_total": mutation_total or None,
        "mutation_killed": mutation_killed or None,
        "mutation_survived": mutation_survived or None,
        "mutation_score": mutation_score,
    }


def _write_project_wide_reports(out_dir: Path, projects: tuple[str, ...]) -> None:
    rows = [_project_json_summary(project, out_dir / project / f"{project}.json") for project in projects]
    csv_path = out_dir / "project_coverage_mutation_report.csv"
    md_path = out_dir / "project_coverage_mutation_report.md"
    json_path = out_dir / "project_coverage_mutation_report.json"
    fieldnames = [
        "project", "json_path", "return_codes", "pipeline_rc", "has_nonzero_intermediate_rc",
        "final_rc", "final_validation_rc", "generation_timeout_count", "failed_tests", "generated_tests",
        "e2e_timed_out", "overall_line", "overall_branch", "covered_methods", "mutation_status",
        "mutation_total", "mutation_killed", "mutation_survived", "mutation_score",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    lines = ["# TATG Project-Wide Coverage/Mutation Report", ""]
    for row in rows:
        lines.extend([
            f"## {row['project']}",
            "",
            f"- generated tests: {row.get('generated_tests')}",
            f"- pipeline rc (intermediate/process): {row.get('pipeline_rc')}",
            f"- final validation rc: {row.get('final_validation_rc')}",
            f"- generation timeout count: {row.get('generation_timeout_count')}",
            f"- e2e timed out: {row.get('e2e_timed_out')}",
            f"- overall line coverage: {_pct(row.get('overall_line'))}",
            f"- overall branch coverage: {_pct(row.get('overall_branch'))}",
            f"- covered method rows: {row.get('covered_methods')}",
            f"- mutation status: {row.get('mutation_status')}",
            f"- mutation score: {_pct((float(row['mutation_score']) / 100.0) if isinstance(row.get('mutation_score'), (int, float)) else None)}",
            f"- mutation killed/survived/total: {row.get('mutation_killed')}/{row.get('mutation_survived')}/{row.get('mutation_total')}",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"project_report_csv={csv_path}")
    print(f"project_report_json={json_path}")
    print(f"project_report_md={md_path}")


def _run_project_wide(args: Namespace, project: str, env: dict[str, str]) -> int:
    project_root = args.subjects_root.resolve() / project
    project_out = args.out_dir / project
    project_out.mkdir(parents=True, exist_ok=True)
    json_out = project_out / f"{project}.json"
    log_out = project_out / f"{project}.log"
    progress_log_out = project_out / f"{project}.progress.log"
    if json_out.exists() and not args.force:
        print(f"[{project}] skip existing {json_out}")
        return 0
    (project_out / f"{project}.json.partial").unlink(missing_ok=True)
    if args.force:
        json_out.unlink(missing_ok=True)
        log_out.unlink(missing_ok=True)
        progress_log_out.unlink(missing_ok=True)

    command = [
        args.python,
        "-m",
        "tatg.cli",
        "generate",
        "--project-root",
        str(project_root),
        "--scope",
        "singlemodule",
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
        str(args.tatg_preanalysis_timeout_seconds),
        "--test-timeout-seconds",
        str(args.test_timeout_seconds),
        "--log-prompts",
        "--log-unrepaired-tests",
        "--test-command",
        "mvn test -pl . -Dsurefire.failIfNoSpecifiedTests=false",
        "--json-out",
        str(json_out),
        "--e2e-timeout-seconds",
        str(args.e2e_timeout_seconds),
        "--generation-parallelism",
        str(args.generation_parallelism),
        "--repair-parallelism",
        str(args.repair_parallelism),
        "--validation-parallelism",
        str(args.validation_parallelism),
    ]
    if args.with_mutation:
        command.append("--with-mutation")

    print(f"[{project}] command: {' '.join(command)}")
    if args.dry_run:
        return 0
    start = time.monotonic()
    print(f"[{project}] starting; log={log_out} progress_log={progress_log_out}")
    process_returncode = _run_command_with_timestamped_log(
        command,
        project_root,
        env,
        log_out,
        progress_log_out,
        show_progress=bool(getattr(args, "show_progress", False)),
    )
    post_eval_result = "post_run_skipped"
    if bool(getattr(args, "post_run_evaluate", False)) and json_out.exists():
        ok, msg = _post_run_evaluate_and_merge(
            project_root=project_root,
            json_out=json_out,
            generated_test_count=int((_safe_read_json(json_out).get("generated_test_count") or 0)),
            run_mutation_if_missing=bool(getattr(args, "with_mutation", False)),
        )
        post_eval_result = msg
    elapsed = time.monotonic() - start
    print(f"[{project}] rc={process_returncode} elapsed={elapsed:.1f}s {post_eval_result} {_json_outcome(json_out)}")
    return process_returncode


def _run_project_class(args: Namespace, target: ProjectClassTarget, env: dict[str, str]) -> int:
    project_root = args.subjects_root.resolve() / target.project
    project_out = args.out_dir / target.project
    project_out.mkdir(parents=True, exist_ok=True)
    json_out = project_out / f"{target.output_id}.json"
    log_out = project_out / f"{target.output_id}.log"
    progress_log_out = project_out / f"{target.output_id}.progress.log"
    generated_out = project_out / "generated-tests" / target.output_id
    if json_out.exists() and not args.force:
        print(f"[{target.project}:{target.class_name}] skip existing {json_out}")
        return 0
    if args.force:
        json_out.unlink(missing_ok=True)
        (project_out / f"{target.output_id}.json.partial").unlink(missing_ok=True)
        log_out.unlink(missing_ok=True)
        progress_log_out.unlink(missing_ok=True)

    command = [
        args.python,
        "-m",
        "tatg.cli",
        "generate",
        "--project-root",
        str(project_root),
        "--target-file",
        target.target_file,
        "--scope",
        "class",
        "--class-name",
        target.class_name,
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
        str(args.tatg_preanalysis_timeout_seconds),
        "--test-timeout-seconds",
        str(args.test_timeout_seconds),
        "--log-prompts",
        "--log-unrepaired-tests",
        "--test-command",
        _test_command(args.test_command_mode, target.class_name),
        "--json-out",
        str(json_out),
        "--e2e-timeout-seconds",
        str(args.e2e_timeout_seconds),
        "--generation-parallelism",
        str(args.generation_parallelism),
        "--repair-parallelism",
        str(args.repair_parallelism),
        "--validation-parallelism",
        str(args.validation_parallelism),
    ]
    if args.with_mutation:
        command.append("--with-mutation")

    label = f"{target.project}:{target.output_id}"
    print(f"[{label}] command: {' '.join(command)}")
    if args.dry_run:
        return 0

    backup_root = project_out / ".tree-backups" / target.output_id
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
            ok, msg = _post_run_evaluate_and_merge(
                project_root=project_root,
                json_out=json_out,
                generated_test_count=int((_safe_read_json(json_out).get("generated_test_count") or 0)),
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


def main() -> int:
    cli_args = _parse_cli_args()
    if bool(getattr(cli_args, "list_hard_methods", False)):
        print("Available HARD_METHODS selectors:")
        for class_name, methods in HARD_METHODS.items():
            for method_name in methods:
                print(f"- {class_name}.{method_name}")
        return 0

    repo_root = _repo_root()
    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    try:
        api_config = resolve_api_config(
            api_base=API_BASE,
            model=MODEL,
            api_key_env=API_KEY_ENV,
            env=env,
        )
        if not DRY_RUN:
            preflight_api_config(api_config, timeout_seconds=min(15, max(5, int(LLM_TIMEOUT_SECONDS))))
        apply_api_config_to_env(api_config, env)
    except Exception as exc:
        print(f"API configuration error: {exc}", file=sys.stderr)
        return 1
    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"

    args = Namespace(
        subjects_root=SUBJECTS_ROOT,
        out_dir=out_dir,
        python=PYTHON,
        api_base=api_config.api_base,
        model=api_config.model,
        api_key_env=api_config.api_key_env,
        max_iterations=MAX_ITERATIONS,
        repair_iterations=REPAIR_ITERATIONS,
        streaming_repair_mode=STREAMING_REPAIR_MODE,
        llm_timeout_seconds=LLM_TIMEOUT_SECONDS,
        tatg_preanalysis_timeout_seconds=max(1, int(getattr(cli_args, "tatg_preanalysis_timeout_seconds", TATG_PREANALYSIS_TIMEOUT_SECONDS) or TATG_PREANALYSIS_TIMEOUT_SECONDS)),
        tatg_generation_timeout_seconds=max(1, int(getattr(cli_args, "tatg_generation_timeout_seconds", TATG_GENERATION_TIMEOUT_SECONDS) or TATG_GENERATION_TIMEOUT_SECONDS)),
        e2e_timeout_seconds=E2E_TIMEOUT_SECONDS,
        generation_parallelism=GENERATION_PARALLELISM,
        repair_parallelism=REPAIR_PARALLELISM,
        validation_parallelism=VALIDATION_PARALLELISM,
        test_timeout_seconds=TEST_TIMEOUT_SECONDS,
        with_mutation=WITH_MUTATION,
        post_run_evaluate=POST_RUN_EVALUATE,
        show_progress=bool(cli_args.show_progress),
        test_command_mode=TEST_COMMAND_MODE,
        force=bool(cli_args.force),
        dry_run=DRY_RUN,
        repo_root=repo_root,
    )

    # Make generation hard-timeout explicit for this runner invocation.
    env["TATG_GENERATION_TIMEOUT_SECONDS"] = str(args.tatg_generation_timeout_seconds)
    # Branch analysis performs several optional evidence subcalls. Keep their
    # combined budget bounded in source batches so a stalled local-model socket
    # cannot hold the target after coverage validation; callers may override it
    # explicitly for slower deployments.
    env["TATG_BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS"] = str(
        max(30, BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS)
    )
    # Local LM Studio normally applies generous timeout floors for interactive
    # use. A source batch already has explicit bounded budgets; leaving those
    # floors untouched makes a timed-out branch request keep reading its socket
    # long after the batch budget has expired.
    env["TATG_LMSTUDIO_LLM_TIMEOUT_SECONDS"] = str(
        max(1, LLM_TIMEOUT_SECONDS)
    )
    env["TATG_LMSTUDIO_BRANCH_ANALYSIS_TIMEOUT_SECONDS"] = str(
        max(30, BRANCH_ANALYSIS_STAGE_TIMEOUT_SECONDS)
    )
    env["TATG_LMSTUDIO_GENERATION_TIMEOUT_SECONDS"] = str(
        max(1, args.tatg_generation_timeout_seconds)
    )
    # When the user requests progress output, bypass the foreground-process-group
    # check (which raises OSError on macOS for some terminal emulators).
    if args.show_progress:
        env["TATG_FORCE_TTY_PROGRESS"] = "1"

    print(f"repo_root={repo_root}")
    print(f"PYTHONPATH starts with {src_path}")
    print(f"out_dir={out_dir}")
    print(
        f"test_command_mode={TEST_COMMAND_MODE} max_iterations={MAX_ITERATIONS} "
        f"repair_iterations={REPAIR_ITERATIONS} streaming_repair_mode={STREAMING_REPAIR_MODE} "
        f"llm_timeout_seconds={LLM_TIMEOUT_SECONDS} "
        f"tatg_preanalysis_timeout_seconds={args.tatg_preanalysis_timeout_seconds} "
        f"tatg_generation_timeout_seconds={args.tatg_generation_timeout_seconds} "
        f"parallelism={GENERATION_PARALLELISM}/{REPAIR_PARALLELISM}/{VALIDATION_PARALLELISM} "
        f"with_mutation={WITH_MUTATION} post_run_evaluate={POST_RUN_EVALUATE} "
        f"show_progress={args.show_progress} force={args.force} mode={cli_args.mode}"
    )

    if cli_args.mode in {"project-wide", "project-class"} and PROJECTS:
        targets_by_project = {
            project: _discover_project_class_targets(SUBJECTS_ROOT, project)
            for project in PROJECTS
        }
        for project, targets in targets_by_project.items():
            method_total = sum(target.method_count for target in targets)
            print(f"[{project}] discovered_classes={len(targets)} discovered_method_targets={method_total}")

        if cli_args.mode == "project-wide":
            rc = 0
            for project in PROJECTS:
                rc = max(rc, _run_project_wide(args, project, env))
                _write_project_wide_reports(out_dir, PROJECTS)
            _write_project_wide_reports(out_dir, PROJECTS)
            print("project-wide batch complete")
            return rc

        rc = 0
        for project in PROJECTS:
            for target in targets_by_project[project]:
                rc = max(rc, _run_project_class(args, target, env))
                _write_reports(out_dir, targets_by_project)
        _write_reports(out_dir, targets_by_project)
        print("project batch complete")
        return rc

    selected_hard_methods = _select_hard_methods(str(getattr(cli_args, "methods", "") or ""))
    targets_by_name = {target.name: target for target in TARGETS}
    hard_rows: list[dict[str, object]] = []
    rc = 0
    for class_name, methods in selected_hard_methods.items():
        target = targets_by_name[class_name]
        for method_name in methods:
            process_rc = _run_one(args, target, env, method_name=method_name)
            rc = max(rc, process_rc)
            json_path = out_dir / f"{class_name}.{method_name}.json"
            print(f"[{class_name}.{method_name}] {_target_method_coverage(json_path, method_name)}")
            data = _read_json(json_path)
            final = data.get("final_validation") or {}
            return_codes = data.get("return_codes")
            latest = (data.get("test_runs") or [{}])[-1]
            line_cov, branch_cov = _best_method_cov_pair(data, class_name, method_name)
            hard_rows.append({
                "project": target.project,
                "class_name": class_name,
                "method_name": method_name,
                "json_path": str(json_path),
                "process_rc": process_rc,
                "pipeline_rc": _pipeline_rc(return_codes),
                "return_codes": return_codes,
                "has_nonzero_intermediate_rc": _has_nonzero_intermediate_rc(return_codes),
                "final_validation_rc": final.get("return_code"),
                "final_failed_tests": final.get("failed_test_count"),
                "generation_timeout_count": int((data.get("validation_policy") or {}).get("generation_timeout_count") or 0),
                "preanalysis_elapsed_seconds": (data.get("tatg_preanalysis") or {}).get("elapsed_seconds"),
                "pre_e2e_seconds": data.get("pre_e2e_seconds"),
                "e2e_seconds": data.get("e2e_seconds"),
                "target_line_coverage": line_cov,
                "target_branch_coverage": branch_cov,
                "overall_line_coverage": latest.get("line_coverage"),
                "overall_branch_coverage": latest.get("branch_coverage"),
            })

    if hard_rows:
        csv_path = out_dir / "hard_method_report.csv"
        json_path = out_dir / "hard_method_report.json"
        md_path = out_dir / "hard_method_report.md"
        fieldnames = [
            "project", "class_name", "method_name", "json_path", "process_rc", "pipeline_rc", "return_codes",
            "has_nonzero_intermediate_rc", "final_validation_rc", "final_failed_tests", "generation_timeout_count",
            "preanalysis_elapsed_seconds", "pre_e2e_seconds", "e2e_seconds",
            "target_line_coverage", "target_branch_coverage", "overall_line_coverage", "overall_branch_coverage",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(hard_rows)
        json_path.write_text(json.dumps(hard_rows, indent=2, sort_keys=True), encoding="utf-8")

        lines = [
            "# TATG Hard-Method Report",
            "",
            "- `process_rc`: wrapper process return code for this method invocation.",
            "- `pipeline_rc`: max value in `return_codes` (intermediate/pipeline status).",
            "- `final_validation_rc`: final validation return code from JSON (`0` means final tests passed).",
            "",
            "| Method | process_rc | pipeline_rc | final_validation_rc | gen_timeout_count | preanalysis(s) | pre_e2e(s) | target line | target branch |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in hard_rows:
            lines.append(
                f"| {row['class_name']}.{row['method_name']} | {row['process_rc']} | {row['pipeline_rc']} | "
                f"{row['final_validation_rc']} | {row['generation_timeout_count']} | {row['preanalysis_elapsed_seconds']} | "
                f"{row['pre_e2e_seconds']} | {row['target_line_coverage']} | {row['target_branch_coverage']} |"
            )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"hard_method_report_csv={csv_path}")
        print(f"hard_method_report_json={json_path}")
        print(f"hard_method_report_md={md_path}")

    # Stability check: pipeline errors that produced a passing final validation
    # suggest silent test compilation failures — coverage numbers may be inflated.
    suspicious = [
        row for row in hard_rows
        if (row.get("pipeline_rc") or 0) != 0 and (row.get("final_validation_rc") or 0) == 0
    ]
    if suspicious:
        print(
            f"\n[WARN] {len(suspicious)} method(s) had pipeline_rc!=0 but final_validation_rc==0"
            " — intermediate build/compile failure may have inflated coverage:"
        )
        for row in suspicious:
            print(f"  {row['class_name']}.{row['method_name']}  pipeline_rc={row['pipeline_rc']}")

    print("hard-method batch complete")
    return rc


if __name__ == "__main__":
    sys.exit(main())
