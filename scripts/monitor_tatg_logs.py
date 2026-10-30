#!/usr/bin/env python3
"""Periodic analyzer for TATG hard-method batch logs.

The input and output roots come from ``TATG_MONITOR_ROOT`` and
``TATG_MONITOR_OUT_DIR``. When unset, they are derived from
``TATG_HARD_METHOD_OUT_DIR`` and the platform temporary directory.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DEFAULT_BATCH_ROOT = Path(
    os.environ.get(
        "TATG_HARD_METHOD_OUT_DIR",
        str(Path(tempfile.gettempdir()) / "tatg-all-hard-methods"),
    )
)
ROOT = Path(os.environ.get("TATG_MONITOR_ROOT", str(_DEFAULT_BATCH_ROOT / "tatg")))
OUT_DIR = Path(
    os.environ.get("TATG_MONITOR_OUT_DIR", str(_DEFAULT_BATCH_ROOT / "monitor"))
)
STATE_PATH = OUT_DIR / "tatg_monitor_state.json"
ROUNDS_PATH = OUT_DIR / "tatg_monitor_rounds.md"
SUMMARY_PATH = OUT_DIR / "tatg_monitor_summary.md"


@dataclass
class MethodRecord:
    key: str
    cls: str
    method: str
    method_dir: Path
    log_path: Path | None
    json_path: Path | None
    partial_path: Path | None
    completed: bool


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail(path: Path | None, lines: int = 40) -> str:
    if path is None or not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        return "\n".join(content[-lines:])
    except Exception:
        return ""


def _find_method_records() -> list[MethodRecord]:
    if not ROOT.exists():
        return []
    records: list[MethodRecord] = []
    for cls_dir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
        for method_dir in sorted(p for p in cls_dir.iterdir() if p.is_dir()):
            log_candidates = sorted(method_dir.glob("*.log"))
            json_candidates = sorted(method_dir.glob("*.json"))
            partial_candidates = sorted(method_dir.glob("*.json.partial"))
            json_path = next((p for p in json_candidates if not p.name.endswith(".partial")), None)
            partial_path = partial_candidates[0] if partial_candidates else None
            log_path = log_candidates[0] if log_candidates else None
            key = f"{cls_dir.name}.{method_dir.name}"
            records.append(
                MethodRecord(
                    key=key,
                    cls=cls_dir.name,
                    method=method_dir.name,
                    method_dir=method_dir,
                    log_path=log_path,
                    json_path=json_path,
                    partial_path=partial_path,
                    completed=json_path is not None,
                )
            )
    return records


def _coverage_from_payload(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    rs = payload.get("report_summary") or {}
    cov = rs.get("coverage") or {}
    line = cov.get("line_coverage")
    branch = cov.get("branch_coverage")
    if isinstance(line, (int, float)) and isinstance(branch, (int, float)):
        return float(line), float(branch)
    test_runs = payload.get("test_runs") or []
    for run in reversed(test_runs):
        line = run.get("line_coverage")
        branch = run.get("branch_coverage")
        if isinstance(line, (int, float)) or isinstance(branch, (int, float)):
            return (float(line) if isinstance(line, (int, float)) else None, float(branch) if isinstance(branch, (int, float)) else None)
    return None, None


def _mutation_from_payload(payload: dict[str, Any]) -> float | None:
    rs = payload.get("report_summary") or {}
    mut = rs.get("mutation") or {}
    avg_method = mut.get("average_method_mutation_score")
    if isinstance(avg_method, (int, float)):
        return float(avg_method)

    me = payload.get("mutation_evaluation") or {}
    tm = me.get("target_method_mutation") or {}
    score = tm.get("mutation_score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _fmt_pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def _extract_unsatisfied_obligations(payload: dict[str, Any]) -> tuple[int, int]:
    rs = payload.get("report_summary") or {}
    cov = rs.get("coverage") or {}
    sat = cov.get("coverage_obligation_satisfaction") or {}
    total = sat.get("obligation_count")
    unsat = sat.get("unsatisfied_count")
    if isinstance(total, int) and isinstance(unsat, int):
        return unsat, total
    return 0, 0


def _build_reasoning(payload: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    line, branch = _coverage_from_payload(payload)
    mutation = _mutation_from_payload(payload)

    if line is not None and line < 0.8:
        reasons.append("Coverage remains below the 80% target; this is a hard miss, not a small fluctuation.")
    if branch is not None and line is not None and (line - branch) > 0.15:
        reasons.append("Branch coverage is materially below line coverage, indicating path-condition exploration is weaker than nominal line execution.")

    unsat, total = _extract_unsatisfied_obligations(payload)
    if total > 0 and unsat > 0:
        reasons.append(
            f"Coverage-obligation closure is incomplete ({total - unsat}/{total} satisfied), suggesting obligation-guided feedback did not fully translate into valid promoted tests."
        )

    qm = payload.get("quality_metrics") or {}
    gen_rate = qm.get("generation_success_rate")
    if isinstance(gen_rate, (int, float)) and gen_rate < 1.0:
        reasons.append("Generation success rate is below 100%; strategy quality is partially offset by unstable generations/rejections.")

    vp = payload.get("validation_policy") or {}
    failed = int(vp.get("streaming_validation_failed", 0) or 0)
    passed = int(vp.get("streaming_validation_passed", 0) or 0)
    repaired = int(vp.get("streaming_validation_repair_passed", 0) or 0)
    if failed > 0:
        reasons.append(
            f"Many streaming candidates failed validation (passed={passed}, failed={failed}); pipeline relies heavily on repair, implying weak first-pass executable correctness."
        )
    if repaired > 0:
        reasons.append(
            f"Repair recovered {repaired} streaming candidates, which prevents collapse but also hides upstream prompt/spec instability."
        )

    if mutation is not None and mutation < 0.8:
        reasons.append("Mutation score is below target; generated tests may validate nominal behavior but are not adversarial enough against equivalent-looking mutants.")

    if payload.get("e2e_timed_out"):
        reasons.append("E2E timeout occurred, so the optimization loop likely terminated before strategy convergence.")

    if not reasons:
        reasons.append("No major systemic weakness observed in this snapshot; metrics are currently near or above target.")
    return reasons


def _build_framework_gap_hypotheses(payload: dict[str, Any], completed: bool) -> list[str]:
    notes: list[str] = []
    vp = payload.get("validation_policy") or {}
    failed = int(vp.get("streaming_validation_failed", 0) or 0)
    applied = int(vp.get("streaming_validation_applied_candidates", 0) or 0)

    notes.append(
        "Potential strategy non-effectiveness: obligation prompts may overfit to symbolic target IDs while under-specifying fixture realism and oracle strength, causing repeated compile/test rejection before promotion."
    )
    if failed > applied:
        notes.append(
            "Process vulnerability: the pipeline spends significant budget in generate->validate->repair loops; when failure volume exceeds promotions, exploration breadth can crowd out deep obligation closure."
        )
    notes.append(
        "Exception-handling robustness risk: if failures are primarily semantic (assertion wrong / singularity handling), current repair may converge slowly because it optimizes compilability first, semantic discriminativeness second."
    )
    notes.append(
        "Coverage blind spot: high line coverage can still coexist with low mutation strength when assertions are weak or invariant-insensitive; mutation-aware objective should be promoted earlier in the loop, not only post hoc."
    )
    if not completed:
        notes.append(
            "This method is still running; current findings are provisional and may shift after later repair/feedback rounds."
        )
    return notes


def _analyze_record(record: MethodRecord) -> dict[str, Any]:
    payload = _load_json(record.json_path) or _load_json(record.partial_path) or {}
    line, branch = _coverage_from_payload(payload)
    mutation = _mutation_from_payload(payload)
    reasons = _build_reasoning(payload)
    hypotheses = _build_framework_gap_hypotheses(payload, record.completed)

    rs = payload.get("report_summary") or {}
    counts = rs.get("counts") or {}
    generated = counts.get("generated_test_count", payload.get("generated_test_count"))

    return {
        "key": record.key,
        "class": record.cls,
        "method": record.method,
        "completed": record.completed,
        "line_coverage": line,
        "branch_coverage": branch,
        "mutation_score": mutation,
        "generated_tests": generated,
        "reasons": reasons,
        "hypotheses": hypotheses,
        "log_tail": _tail(record.log_path, lines=25),
        "json_path": str(record.json_path) if record.json_path else None,
        "partial_path": str(record.partial_path) if record.partial_path else None,
    }


def _load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {"seen": {}}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"seen": {}}


def _record_fingerprint(record: MethodRecord) -> str:
    path = record.json_path or record.partial_path or record.log_path
    if path is None or not path.exists():
        return "missing"
    try:
        st = path.stat()
        return f"{path.name}:{int(st.st_mtime)}:{st.st_size}:{int(record.completed)}"
    except Exception:
        return f"{path.name}:unknown:{int(record.completed)}"


def _append_round(round_items: list[dict[str, Any]], run_ts: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"\n## Round @ {run_ts}\n")
    for item in round_items:
        lines.append(f"### {item['key']} ({'completed' if item['completed'] else 'running'})")
        lines.append(f"- Coverage(line/branch): {_fmt_pct(item['line_coverage'])} / {_fmt_pct(item['branch_coverage'])}")
        lines.append(f"- Mutation score: {_fmt_pct(item['mutation_score'])}")
        lines.append(f"- Generated tests: {item.get('generated_tests')}")
        lines.append("- Primary diagnosis:")
        for r in item["reasons"]:
            lines.append(f"  - {r}")
        lines.append("- Framework/process hypotheses:")
        for h in item["hypotheses"]:
            lines.append(f"  - {h}")
        lines.append("- Log signals (tail):")
        lines.append("```")
        lines.append(item.get("log_tail", "") or "(no log tail)")
        lines.append("```")
        lines.append("")

    with ROUNDS_PATH.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_summary(all_items: list[dict[str, Any]], run_ts: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    low_cov = [x for x in all_items if isinstance(x.get("line_coverage"), float) and x["line_coverage"] < 0.8]
    low_mut = [x for x in all_items if isinstance(x.get("mutation_score"), float) and x["mutation_score"] < 0.8]

    lines: list[str] = []
    lines.append(f"# TATG Monitoring Summary\n")
    lines.append(f"Last update: {run_ts}\n")
    lines.append(f"- Methods discovered: {len(all_items)}")
    lines.append(f"- Methods below coverage target (<80% line): {len(low_cov)}")
    lines.append(f"- Methods below mutation target (<80%): {len(low_mut)}")
    lines.append("")

    lines.append("## Research-Oriented Improvement Insights")
    lines.append("- Strategy effect gap: move from obligation-ID completion to obligation+oracle quality objectives, so promoted tests maximize semantic fault discrimination instead of line hit counts only.")
    lines.append("- Loop-budget allocation: introduce adaptive budget shift (less generation, more targeted repair) when validation-failure ratio spikes; this can reduce ineffective candidate churn.")
    lines.append("- Robustness instrumentation: attach fine-grained failure taxonomies (compile/type/runtime/assertion/domain) to every rejected candidate and feed them back into next-prompt planning.")
    lines.append("- Mutation-in-the-loop optimization: compute lightweight mutation proxies earlier (per-iteration surrogate scores) to steer generation before final mutation evaluation.")
    lines.append("- Generalization risk control: detect when high coverage is achieved by narrow data manifolds and enforce diversity constraints across boundary conditions and invariant families.")
    lines.append("")

    lines.append("## Per-Method Snapshot")
    for item in all_items:
        lines.append(
            f"- {item['key']}: line={_fmt_pct(item['line_coverage'])}, branch={_fmt_pct(item['branch_coverage'])}, mutation={_fmt_pct(item['mutation_score'])}, status={'completed' if item['completed'] else 'running'}"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    run_ts = _now_iso()
    records = _find_method_records()
    state = _load_state()
    seen = state.setdefault("seen", {})

    changed_items: list[dict[str, Any]] = []
    all_items: list[dict[str, Any]] = []

    for rec in records:
        fingerprint = _record_fingerprint(rec)
        item = _analyze_record(rec)
        all_items.append(item)
        if seen.get(rec.key) != fingerprint:
            changed_items.append(item)
            seen[rec.key] = fingerprint

    _write_summary(all_items, run_ts)
    if changed_items:
        _append_round(changed_items, run_ts)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    print(f"[{run_ts}] monitored_methods={len(records)} changed={len(changed_items)}")
    print(f"summary={SUMMARY_PATH}")
    print(f"rounds={ROUNDS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
