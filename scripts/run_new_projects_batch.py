#!/usr/bin/env python3
"""TATG hard-method batch for new projects: jquick-curl and adk-java.

Each project uses its Maven module directory as the working root so that
``mvn test -pl .`` resolves correctly from the module's own pom.xml.

Usage
-----
# Dry-run (print commands only):
python scripts/run_new_projects_batch.py --dry-run

# Run all hard methods:
python scripts/run_new_projects_batch.py

# Run a subset (class name, method name, or class.method):
python scripts/run_new_projects_batch.py --select FunctionCallingUtils
python scripts/run_new_projects_batch.py --select visitCurlCommand
python scripts/run_new_projects_batch.py --select FunctionCallingUtils.buildFunctionDeclaration

Environment variables
---------------------
TATG_MAX_ITERATIONS   — LLM generation/repair iterations (default 2)
TATG_DRY_RUN          — set to 1 for dry-run mode
TATG_WITH_MUTATION    — set to 0 to skip mutation scoring
TATG_SHOW_PROGRESS    — set to 0 to suppress live progress bar
TATG_PREANALYSIS_TIMEOUT_SECONDS — timeout for TATG pre-analysis (default 600)
TATG_GENERATION_TIMEOUT_SECONDS  — timeout per LLM generation call (default 240)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tatg.api_config import (
    DEFAULT_API_KEY_ENV,
    apply_api_config_to_env,
    preflight_api_config,
    resolve_api_config,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_four_class_source_batch import (
    _backup_java_files,
    _copy_changed_java_files,
    _json_outcome,
    _pipeline_rc,
    _post_run_evaluate_and_merge,
    _repo_root,
    _restore_java_files,
    _run_command_with_timestamped_log,
    _safe_read_json,
)


# ──────────────────────────────────────────────────────────────────────────────
# Project layout
# ──────────────────────────────────────────────────────────────────────────────

# Root directory containing all three new projects as immediate subdirectories.
NEW_SUBJECTS_ROOT = Path(
    os.environ.get("TATG_NEW_SUBJECTS_ROOT", "/path/to/new_projects")
)

# For multi-module projects the "project" field in BatchTarget is a path
# *relative to NEW_SUBJECTS_ROOT* that points directly at the Maven module
# to run tests in (e.g. "adk-java/core", not just "adk-java").
# Single-module projects (jquick-curl) just use the project directory name.

# ──────────────────────────────────────────────────────────────────────────────
# Defaults — all overridable via CLI args or env vars
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR = Path(
    os.environ.get(
        "TATG_NEW_PROJECTS_OUT_DIR",
        str(Path(tempfile.gettempdir()) / "tatg-new-projects"),
    )
)
PYTHON = os.environ.get("TATG_PYTHON", sys.executable)
API_BASE = os.environ.get("TATG_API_BASE", os.environ.get("API_BASE", ""))
MODEL = os.environ.get("TATG_MODEL", os.environ.get("MODEL", ""))
API_KEY_ENV = os.environ.get("TATG_API_KEY_ENV", DEFAULT_API_KEY_ENV)
MAX_ITERATIONS = int(os.environ.get("TATG_MAX_ITERATIONS", "2") or "2")
REPAIR_ITERATIONS = 2
STREAMING_REPAIR_MODE = os.environ.get("TATG_STREAMING_REPAIR_MODE", "deferred")
LLM_TIMEOUT_SECONDS = 60
TATG_PREANALYSIS_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_PREANALYSIS_TIMEOUT_SECONDS", "600") or "600"
)
TATG_GENERATION_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_GENERATION_TIMEOUT_SECONDS", "240") or "240"
)
E2E_TIMEOUT_SECONDS = 0
TEST_TIMEOUT_SECONDS = 900
GENERATION_PARALLELISM = 2
REPAIR_PARALLELISM = 2
VALIDATION_PARALLELISM = 4
WITH_MUTATION = os.environ.get("TATG_WITH_MUTATION", "1") != "0"
FORCE = False
DRY_RUN = os.environ.get("TATG_DRY_RUN", "0") == "1"
POST_RUN_EVALUATE = os.environ.get("TATG_POST_RUN_EVALUATE", "1") != "0"
SHOW_PROGRESS = os.environ.get("TATG_SHOW_PROGRESS", "1") != "0"


# ──────────────────────────────────────────────────────────────────────────────
# Target definition
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NewBatchTarget:
    name: str          # Java class simple name (also used to derive test class name)
    project: str       # Path relative to NEW_SUBJECTS_ROOT (may include module subdir)
    target_file: str   # Java source file relative to the project (module) directory


# Hard methods: class name → tuple of method names.
# These are the specific methods we ask the LLM to test exhaustively.
HARD_METHODS: dict[str, tuple[str, ...]] = {
    # ── jquick-curl ──────────────────────────────────────────────────────────
    # ANTLR-based HTTP DSL parser; visitor pattern with complex dispatch
    "JQuickCurlCommonVisitor": ("visitCurlCommand", "visitOption"),
    # Annotation-driven parameter interpolation and curl command invocation
    "JCurlCommandProcessor": ("resolveCurlCommand",),

    # ── adk-java (core module) ───────────────────────────────────────────────
    # Recursive Jackson-based JSON Schema builder; circular-reference guard;
    # 8+ type dispatch branches (enum, POJO, collection, map, primitives…)
    "FunctionCallingUtils": ("buildFunctionDeclaration", "buildSchemaFromType"),
    # Backward sliding-window scan over event history; two-pass threshold logic;
    # complex loop invariants for compaction interval and overlap tracking
    "SlidingWindowEventCompactor": ("getCompactionEvents",),
    # Switch-driven schema type validator + recursive nested field iteration
    "SchemaUtils": ("validateMapOnSchema", "validateOutputSchema"),
    # Deep immutable Content→Part→Blob/FileData transformation; Optional chaining
    "GeminiUtil": ("sanitizeRequestForGeminiApi", "stripThoughts"),
    # LLM-request → OpenAI Chat Completions format converter; complex multi-level
    # content-part dispatch (text, inlineData, fileData, functionCall, response)
    "ChatCompletionsRequest": ("fromLlmRequest",),
    # Event reordering algorithm: pairs function-call events with their responses;
    # non-trivial order invariants under streaming and partial-response scenarios
    "Contents": ("rearrangeEventsForLatestFunctionResponse",),
    # Recursive DFS through the agent sub-tree; multiple path and name conditions
    "BaseAgent": ("findAgent",),
}


TARGETS: tuple[NewBatchTarget, ...] = (
    # ── jquick-curl ──────────────────────────────────────────────────────────
    # Single-module Maven project; project root == module root.
    NewBatchTarget(
        name="JQuickCurlCommonVisitor",
        project="jquick-curl",
        target_file="src/main/java/com/github/paohaijiao/visitor/JQuickCurlCommonVisitor.java",
    ),
    NewBatchTarget(
        name="JCurlCommandProcessor",
        project="jquick-curl",
        target_file="src/main/java/com/github/paohaijiao/support/JCurlCommandProcessor.java",
    ),

    # ── adk-java / core ──────────────────────────────────────────────────────
    # Multi-module Maven project; we point at the "core" submodule so that
    # backup/restore, source scanning, and `mvn test -pl .` all resolve
    # correctly from core/pom.xml without needing the full reactor.
    NewBatchTarget(
        name="FunctionCallingUtils",
        project="adk-java/core",
        target_file="src/main/java/com/google/adk/tools/FunctionCallingUtils.java",
    ),
    NewBatchTarget(
        name="SlidingWindowEventCompactor",
        project="adk-java/core",
        target_file="src/main/java/com/google/adk/summarizer/SlidingWindowEventCompactor.java",
    ),
    NewBatchTarget(
        name="SchemaUtils",
        project="adk-java/core",
        target_file="src/main/java/com/google/adk/SchemaUtils.java",
    ),
    NewBatchTarget(
        name="GeminiUtil",
        project="adk-java/core",
        target_file="src/main/java/com/google/adk/models/GeminiUtil.java",
    ),
    NewBatchTarget(
        name="ChatCompletionsRequest",
        project="adk-java/core",
        target_file="src/main/java/com/google/adk/models/chat/ChatCompletionsRequest.java",
    ),
    NewBatchTarget(
        name="Contents",
        project="adk-java/core",
        target_file="src/main/java/com/google/adk/flows/llmflows/Contents.java",
    ),
    NewBatchTarget(
        name="BaseAgent",
        project="adk-java/core",
        target_file="src/main/java/com/google/adk/agents/BaseAgent.java",
    ),
)

# Quick index: class name → target (for method-selector lookup)
_TARGETS_BY_NAME: dict[str, NewBatchTarget] = {t.name: t for t in TARGETS}


# ──────────────────────────────────────────────────────────────────────────────
# Command building
# ──────────────────────────────────────────────────────────────────────────────

def _test_command(class_name: str) -> str:
    """Return the maven test command for a generated test class.

    The command is always ``-pl .`` because we set the working directory
    (``cwd``) to the module root when running the subprocess — so Maven
    picks up the right ``pom.xml`` automatically.
    """
    return (
        f"mvn test -pl . --no-transfer-progress "
        f"-Dtest={class_name}Test "
        f"-Dsurefire.failIfNoSpecifiedTests=false"
    )


def _build_command(
    args: argparse.Namespace,
    target: NewBatchTarget,
    project_root: Path,
    method_name: str | None = None,
) -> list[str]:
    scope = "method" if method_name else "class"
    out_stem = f"{target.name}.{method_name}" if method_name else target.name
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
        *(["--method-name", method_name] if method_name else []),
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
        str(getattr(args, "tatg_preanalysis_timeout_seconds", TATG_PREANALYSIS_TIMEOUT_SECONDS)),
        "--log-prompts",
        "--log-unrepaired-tests",
        "--test-command",
        _test_command(target.name),
        "--json-out",
        str(args.out_dir / f"{out_stem}.json"),
        "--e2e-timeout-seconds",
        str(args.e2e_timeout_seconds),
        "--generation-parallelism",
        str(args.generation_parallelism),
        "--repair-parallelism",
        str(args.repair_parallelism),
        "--validation-parallelism",
        str(args.validation_parallelism),
    ] + (["--with-mutation"] if getattr(args, "with_mutation", False) else [])


# ──────────────────────────────────────────────────────────────────────────────
# Run one method
# ──────────────────────────────────────────────────────────────────────────────

def _run_one(
    args: argparse.Namespace,
    target: NewBatchTarget,
    env: dict[str, str],
    method_name: str | None = None,
) -> int:
    subjects_root = args.subjects_root.resolve()
    project_root = subjects_root / target.project
    out_stem = f"{target.name}.{method_name}" if method_name else target.name
    json_out = args.out_dir / f"{out_stem}.json"
    log_out = args.out_dir / f"{out_stem}.log"
    progress_log_out = args.out_dir / f"{out_stem}.progress.log"
    generated_out = args.out_dir / "generated-tests" / out_stem

    command = _build_command(args, target, project_root, method_name=method_name)
    label = out_stem
    print(f"[{label}] command: {' '.join(command)}")
    if args.dry_run:
        return 0

    if json_out.exists() and not args.force:
        print(f"[{label}] skip: {json_out} already exists (use --force to rerun)")
        return 0
    if args.force:
        json_out.unlink(missing_ok=True)
        (args.out_dir / f"{out_stem}.json.partial").unlink(missing_ok=True)
        log_out.unlink(missing_ok=True)
        progress_log_out.unlink(missing_ok=True)

    backup_root = args.out_dir / ".tree-backups" / out_stem
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
            _ok, msg = _post_run_evaluate_and_merge(
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


# ──────────────────────────────────────────────────────────────────────────────
# Method selector
# ──────────────────────────────────────────────────────────────────────────────

def _comma_values(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _select_hard_methods(selector: str) -> dict[str, tuple[str, ...]]:
    """Filter HARD_METHODS by a comma-separated selector.

    Each token may be:
    - a class name, e.g. ``FunctionCallingUtils``
    - a method name, e.g. ``visitCurlCommand``
    - class.method, e.g. ``FunctionCallingUtils.buildFunctionDeclaration``
    """
    tokens = _comma_values(selector)
    if not tokens:
        return HARD_METHODS

    selected: dict[str, list[str]] = {}

    def _add(cls: str, meth: str) -> None:
        selected.setdefault(cls, [])
        if meth not in selected[cls]:
            selected[cls].append(meth)

    class_to_methods = {k: set(v) for k, v in HARD_METHODS.items()}
    method_to_classes: dict[str, list[str]] = {}
    for cls, methods in HARD_METHODS.items():
        for meth in methods:
            method_to_classes.setdefault(meth, []).append(cls)

    for token in tokens:
        if "." in token:
            cls, meth = token.split(".", 1)
            cls, meth = cls.strip(), meth.strip()
            if cls not in class_to_methods:
                raise SystemExit(f"Unknown class selector: {cls}")
            if meth not in class_to_methods[cls]:
                raise SystemExit(f"Unknown method selector: {cls}.{meth}")
            _add(cls, meth)
        elif token in class_to_methods:
            for meth in HARD_METHODS[token]:
                _add(token, meth)
        elif token in method_to_classes:
            for cls in method_to_classes[token]:
                _add(cls, token)
        else:
            raise SystemExit(
                f"Unknown selector: {token}. "
                "Use class name, method name, or class.method."
            )

    return {k: tuple(v) for k, v in selected.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = _repo_root()
    p = argparse.ArgumentParser(
        description="TATG batch for jquick-curl and adk-java hard methods."
    )
    p.add_argument(
        "--subjects-root",
        type=Path,
        default=NEW_SUBJECTS_ROOT,
        help="Root directory containing the new project subdirectories.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for JSON reports, logs, and generated tests.",
    )
    p.add_argument(
        "--python",
        default=PYTHON,
        help="Python interpreter used to run tatg.cli.",
    )
    p.add_argument("--api-base", default=API_BASE)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--api-key-env", default=API_KEY_ENV)
    p.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    p.add_argument("--repair-iterations", type=int, default=REPAIR_ITERATIONS)
    p.add_argument("--streaming-repair-mode", default=STREAMING_REPAIR_MODE)
    p.add_argument("--llm-timeout-seconds", type=int, default=LLM_TIMEOUT_SECONDS)
    p.add_argument(
        "--tatg-preanalysis-timeout-seconds",
        type=int,
        default=TATG_PREANALYSIS_TIMEOUT_SECONDS,
    )
    p.add_argument(
        "--tatg-generation-timeout-seconds",
        type=int,
        default=TATG_GENERATION_TIMEOUT_SECONDS,
    )
    p.add_argument("--e2e-timeout-seconds", type=int, default=E2E_TIMEOUT_SECONDS)
    p.add_argument("--test-timeout-seconds", type=int, default=TEST_TIMEOUT_SECONDS)
    p.add_argument("--generation-parallelism", type=int, default=GENERATION_PARALLELISM)
    p.add_argument("--repair-parallelism", type=int, default=REPAIR_PARALLELISM)
    p.add_argument("--validation-parallelism", type=int, default=VALIDATION_PARALLELISM)
    p.add_argument(
        "--with-mutation",
        action=argparse.BooleanOptionalAction,
        default=WITH_MUTATION,
        help="Run mutation scoring after generation.",
    )
    p.add_argument(
        "--post-run-evaluate",
        action=argparse.BooleanOptionalAction,
        default=POST_RUN_EVALUATE,
    )
    p.add_argument(
        "--show-progress",
        action="store_true",
        default=SHOW_PROGRESS,
        help="Show live TTY progress panel.",
    )
    p.add_argument("--force", action="store_true", default=FORCE)
    p.add_argument("--dry-run", action="store_true", default=DRY_RUN)
    p.add_argument(
        "--select",
        default="",
        metavar="SELECTOR",
        help=(
            "Comma-separated list of class names, method names, or class.method "
            "tokens to run (default: all HARD_METHODS)."
        ),
    )
    p.add_argument(
        "--skip-api-preflight",
        action="store_true",
        help="Skip the LLM API connectivity check at startup.",
    )
    args = p.parse_args(argv)
    args.repo_root = repo
    args.out_dir = args.out_dir.resolve()
    return args


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def _target_method_coverage(json_path: Path, method_name: str) -> str:
    """Return a brief coverage summary for a single method."""
    data = _safe_read_json(json_path)
    if not data:
        return "no-json"
    runs = data.get("test_runs") or []
    latest = runs[-1] if runs else {}
    return (
        f"line={latest.get('line_coverage')} "
        f"branch={latest.get('branch_coverage')} "
        f"generated={data.get('generated_test_count')}"
    )


def _write_report(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    csv_path = out_dir / "new_projects_report.csv"
    json_path = out_dir / "new_projects_report.json"
    md_path = out_dir / "new_projects_report.md"

    fieldnames = [
        "project",
        "class_name",
        "method_name",
        "json_path",
        "process_rc",
        "pipeline_rc",
        "final_validation_rc",
        "final_failed_tests",
        "generation_timeout_count",
        "target_line_coverage",
        "target_branch_coverage",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# TATG New-Projects Report",
        "",
        "| Class | Method | project_rc | pipeline_rc | final_rc | target line | target branch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['class_name']} | {row['method_name']} "
            f"| {row['process_rc']} | {row['pipeline_rc']} | {row['final_validation_rc']} "
            f"| {row['target_line_coverage']} | {row['target_branch_coverage']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"report_csv={csv_path}")
    print(f"report_json={json_path}")
    print(f"report_md={md_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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
            preflight_api_config(
                api_config,
                timeout_seconds=min(15, max(5, int(args.llm_timeout_seconds))),
            )
        apply_api_config_to_env(api_config, env)
        args.api_base = api_config.api_base
        args.model = api_config.model
        args.api_key_env = api_config.api_key_env
    except Exception as exc:
        print(f"API configuration error: {exc}", file=sys.stderr)
        return 1

    if args.show_progress:
        env["TATG_FORCE_TTY_PROGRESS"] = "1"

    src_path = str(args.repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    )

    print(f"subjects_root={args.subjects_root}")
    print(f"out_dir={args.out_dir}")

    selected = _select_hard_methods(args.select)

    rows: list[dict[str, Any]] = []
    rc = 0

    for class_name, methods in selected.items():
        target = _TARGETS_BY_NAME.get(class_name)
        if target is None:
            print(
                f"[WARN] {class_name} not in TARGETS — no source file path defined; skipping.",
                file=sys.stderr,
            )
            continue

        for method_name in methods:
            process_rc = _run_one(args, target, env, method_name=method_name)
            rc = max(rc, process_rc)

            json_path = args.out_dir / f"{class_name}.{method_name}.json"
            print(f"[{class_name}.{method_name}] {_target_method_coverage(json_path, method_name)}")
            data = _safe_read_json(json_path)
            final = data.get("final_validation") or {}
            return_codes = data.get("return_codes")
            latest = (data.get("test_runs") or [{}])[-1]

            rows.append({
                "project": target.project,
                "class_name": class_name,
                "method_name": method_name,
                "json_path": str(json_path),
                "process_rc": process_rc,
                "pipeline_rc": _pipeline_rc(return_codes),
                "return_codes": return_codes,
                "final_validation_rc": final.get("return_code"),
                "final_failed_tests": final.get("failed_test_count"),
                "generation_timeout_count": int(
                    (data.get("validation_policy") or {}).get("generation_timeout_count") or 0
                ),
                "target_line_coverage": latest.get("line_coverage"),
                "target_branch_coverage": latest.get("branch_coverage"),
            })

    _write_report(rows, args.out_dir)

    # Stability check: pipeline errors paired with clean final validation
    # indicate a silent build/compile failure — coverage numbers may be inflated.
    suspicious = [
        row
        for row in rows
        if (row.get("pipeline_rc") or 0) != 0 and (row.get("final_validation_rc") or 0) == 0
    ]
    if suspicious:
        print(
            f"\n[WARN] {len(suspicious)} method(s) had pipeline_rc!=0 but final_validation_rc==0"
            " — intermediate build/compile failure may have inflated coverage:"
        )
        for row in suspicious:
            print(
                f"  {row['class_name']}.{row['method_name']}"
                f"  pipeline_rc={row['pipeline_rc']}"
            )

    print("new-projects batch complete")
    return rc


if __name__ == "__main__":
    sys.exit(main())
