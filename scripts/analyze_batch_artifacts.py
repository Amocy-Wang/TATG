#!/usr/bin/env python3
"""Analyze TATG batch output artifacts for anomalies and bottlenecks.

Usage:
  python scripts/analyze_batch_artifacts.py --out-dir ./artifacts/tatg-progress
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _safe_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return raw if isinstance(raw, dict) else None
    except Exception:
        return None


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{(100.0 * numerator / denominator):.1f}%"


def _analyze_one(out_dir: Path, json_path: Path) -> list[str]:
    rows: list[str] = []
    name = json_path.stem
    data = _safe_json(json_path)
    if data is None:
        return [f"- {name}: unreadable JSON"]

    final = data.get("final_validation") or {}
    cov = ((data.get("report_summary") or {}).get("coverage") or {})
    vp = data.get("validation_policy") or {}

    timeout = int(data.get("e2e_timeout_seconds") or 0)
    elapsed = float(data.get("e2e_seconds") or 0.0)
    timed_out = bool(data.get("e2e_timed_out"))

    submitted = int(vp.get("streaming_validation_submitted") or 0)
    completed = int(vp.get("streaming_validation_completed") or 0)
    passed = int(vp.get("streaming_validation_passed") or 0)
    failed = int(vp.get("streaming_validation_failed") or 0)
    errors = int(vp.get("streaming_validation_errors") or 0)
    repair_submitted = int(vp.get("streaming_validation_repair_submitted") or 0)

    line_cov = cov.get("line_coverage")
    branch_cov = cov.get("branch_coverage")
    methods_cov_rows = int(cov.get("methods_coverage_rows") or 0)

    class_log = out_dir / f"{name}.log"
    progress_log = out_dir / f"{name}.progress.log"
    generated_dir = out_dir / "generated-tests" / name
    generated_java_count = len(list(generated_dir.rglob("*.java"))) if generated_dir.exists() else 0

    rows.append(f"- {name}:")
    rows.append(
        "  summary: "
        f"target_count={data.get('target_count')} generated_test_count={data.get('generated_test_count')} "
        f"test_passing_rate={data.get('test_passing_rate')} gen_success_rate={data.get('generation_success_rate')} "
        f"final_rc={final.get('return_code')} final_failed={final.get('failed_test_count')}"
    )
    rows.append(
        "  validation: "
        f"submitted={submitted} completed={completed} passed={passed} failed={failed} errors={errors} "
        f"pass_rate={_percent(passed, completed)} error_rate={_percent(errors, completed)}"
    )
    rows.append(
        "  artifacts: "
        f"generated_java_files={generated_java_count} "
        f"log_size={class_log.stat().st_size if class_log.exists() else 0}B "
        f"progress_log_size={progress_log.stat().st_size if progress_log.exists() else 0}B"
    )
    rows.append(
        "  coverage: "
        f"line={line_cov} branch={branch_cov} methods_coverage_rows={methods_cov_rows}"
    )

    anomalies: list[str] = []
    if timed_out:
        anomalies.append(
            f"e2e timeout triggered (elapsed={elapsed:.1f}s, timeout={timeout}s, overrun={max(elapsed-timeout, 0.0):.1f}s)"
        )
    if line_cov is None or branch_cov is None or methods_cov_rows == 0:
        anomalies.append("coverage not materialized (line/branch coverage missing or methods_coverage_rows=0)")
    if completed > 0 and errors * 2 >= completed:
        anomalies.append("validation error ratio is high (future/task failures dominate)")
    if (failed > 0 or errors > 0) and repair_submitted == 0:
        anomalies.append("repair starvation: failures observed but no streaming repairs submitted")
    if generated_java_count == 0 and int(data.get("generated_test_count") or 0) > 0:
        anomalies.append("generated tests counted but no persisted generated test files")
    if class_log.exists() and class_log.stat().st_size == 0 and progress_log.exists() and progress_log.stat().st_size == 0:
        anomalies.append("both log and progress log are empty")

    if anomalies:
        rows.append("  anomalies:")
        rows.extend([f"    - {item}" for item in anomalies])

    suggestions: list[str] = []
    if timed_out:
        suggestions.append("switch test command from 'mvn clean test' to 'mvn test' to avoid repeated clean recompilation")
        suggestions.append("split large classes into method-scope batches or smaller class slices")
    if (failed > 0 or errors > 0) and repair_submitted == 0:
        suggestions.append("use eager streaming repair mode or reserve explicit repair budget")
    if line_cov is None or methods_cov_rows == 0:
        suggestions.append("add post-run coverage collection step (jacoco report) independent of streaming validation")
    if generated_java_count == 0 and int(data.get("generated_test_count") or 0) > 0:
        suggestions.append("persist promoted candidates/test files before restore to improve reproducibility")
    if suggestions:
        rows.append("  suggestions:")
        rows.extend([f"    - {item}" for item in suggestions])

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze TATG batch output artifacts")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if not out_dir.exists():
        raise SystemExit(f"Output directory not found: {out_dir}")

    json_files = sorted(out_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json files found in {out_dir}")

    print(f"batch_dir={out_dir}")
    print(f"completed_json_files={len(json_files)}")

    for jp in json_files:
        for line in _analyze_one(out_dir, jp):
            print(line)

    running_partials = []
    for pp in sorted(out_dir.glob("*.json.partial")):
        data = _safe_json(pp)
        if isinstance(data, dict) and str(data.get("status") or "") == "running":
            running_partials.append((pp.name, data.get("stage"), data.get("target_count")))
    if running_partials:
        print("running_partial_files:")
        for name, stage, count in running_partials:
            print(f"- {name}: stage={stage} target_count={count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
