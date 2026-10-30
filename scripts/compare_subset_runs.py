#!/usr/bin/env python3
"""Paired A/B comparison of two TATG subset runs.

Strategy acceptance for the TATG optimization work is decided per method against
a frozen baseline run, not on an aggregate mean alone: a change that lifts the
mean while quietly regressing a handful of targets is not an improvement.  This
script pairs the two runs by target selector, reports the per-method deltas, and
evaluates the acceptance rule.

Usage
-----
    python scripts/compare_subset_runs.py BASELINE_DIR CANDIDATE_DIR
    python scripts/compare_subset_runs.py BASELINE_DIR CANDIDATE_DIR --json out.json
    python scripts/compare_subset_runs.py ... --min-mean-delta 0.02 --max-regression 0.03

Either directory may contain ``merged_replay_rows.json`` (produced by
``merge_tatg_eval_archive.py``) or ``hard_method_report.json`` (produced
directly by ``run_all_hard_methods_batch.py``).  Both are lists of per-target
row dicts and expose the same coverage fields.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable

ROW_FILE_CANDIDATES = (
    "merged_replay_rows.json",
    "hard_method_report.json",
    "hard_method_report.incremental.jsonl",
)

# Acceptance rule defaults.  See docs/TATG_OPTIMIZATION_PLAN.md §4.3.
DEFAULT_MIN_MEAN_DELTA = 0.02
# Per-method regression gates have to be read against the sampling noise of the
# pipeline, not asserted.  Generation samples at temperature 0.2, and running the
# S1 baseline arm twice under IDENTICAL configuration moved per-target branch
# coverage by a mean 7.1pp, sd 9.31pp, max 20pp -- 12 of 17 targets moved more
# than 3pp, and 7 of 17 *regressed* more than 3pp.  A 3pp gate therefore rejects
# a candidate identical to the baseline; it measured noise, not quality.
#
# The defaults below are derived from that measured null distribution:
#   * worst regression   2 sd  ~= 0.19, so a single target has to move well
#                        outside the noise band before it counts, and
#   * regressing methods the null produced 7/17 (41%), so anything at or below
#                        roughly half the paired set is indistinguishable from
#                        chance; the gate only catches a broad collapse.
# Tighten both once TATG_LLM_SEED is in use and the null reproduces exactly.
# See docs/TATG_MEASUREMENT_NOISE.md.
MEASURED_NULL_SD = 0.0931
DEFAULT_MAX_REGRESSION = 0.19
DEFAULT_MAX_REGRESSING_METHODS_FRACTION = 0.5
DEFAULT_MAX_WALLCLOCK_GROWTH = 0.25
DEFAULT_MAX_P_VALUE = 0.10

# What makes a comparison trustworthy is that enough targets were measured in
# *both* arms, so the paired test has something to work with -- not that some
# fraction of the whole subset was measured.  The original rule asked for 95% of
# all targets measured; the reference 141-method run measures 20.6%, and even
# with the branch-count recovery in run_all_hard_methods_batch.py the ceiling is
# ~93.6%, so that gate could never pass and would have rejected every candidate
# before its coverage numbers were read.
#
# It is replaced by two checks that carry the intent:
#   * at least this many targets measured in both arms (paired-test power), and
#   * the candidate arm does not measure meaningfully fewer targets than the
#     baseline arm, which is how a change that breaks measurement shows up.
DEFAULT_MIN_MEASURED_PAIRS = 12
DEFAULT_MAX_MEASURED_FRACTION_DROP = 0.10


# ──────────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Return the per-target row list for a run directory or explicit file."""

    if path.is_file():
        files = [path]
    else:
        files = [path / name for name in ROW_FILE_CANDIDATES if (path / name).is_file()]
    if not files:
        raise SystemExit(
            f"no run rows found under {path}; expected one of {', '.join(ROW_FILE_CANDIDATES)}"
        )
    source = files[0]
    if source.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{source} does not contain a list of target rows")
    return [row for row in rows if isinstance(row, dict)]


def _selector(row: dict[str, Any]) -> str:
    class_name = str(row.get("class_name") or "").strip()
    method_name = str(row.get("method_name") or "").strip()
    if class_name and method_name:
        return f"{class_name}.{method_name}"
    return str(row.get("target") or row.get("selector") or "").strip()


def _key(row: dict[str, Any]) -> str:
    """Pair on selector plus descriptor so overloads stay distinct."""

    descriptor = str(row.get("target_descriptor") or "").strip()
    return f"{_selector(row)}::{descriptor}" if descriptor else _selector(row)


def _branch(row: dict[str, Any]) -> float | None:
    for field in ("target_branch_coverage", "target_branch_coverage_display"):
        value = row.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _is_measured(row: dict[str, Any]) -> bool:
    status = str(row.get("branch_measurement_status") or "").strip()
    return status == "measured"


def _mutation(row: dict[str, Any]) -> float | None:
    value = row.get("target_mutation_score")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _seconds(row: dict[str, Any]) -> float:
    value = row.get("e2e_seconds")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0


def _failed(row: dict[str, Any]) -> bool:
    status = str(row.get("coverage_status") or "").strip()
    return status in {
        "compilation-failure",
        "compile-type-mismatch",
        "runtime_validation_failure",
        "project_native_preflight_failed",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────────────

def _wilcoxon_signed_rank_p(deltas: Iterable[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value via a normal approximation.

    Implemented locally so the script has no SciPy dependency.  Returns None
    when there are too few non-zero pairs for the approximation to mean
    anything (n < 6).
    """

    values = [d for d in deltas if abs(d) > 1e-9]
    n = len(values)
    if n < 6:
        return None
    ordered = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.0] * n
    index = 0
    while index < n:
        stop = index
        while stop + 1 < n and abs(abs(values[ordered[stop + 1]]) - abs(values[ordered[index]])) < 1e-12:
            stop += 1
        average_rank = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[ordered[position]] = average_rank
        index = stop + 1
    w_plus = sum(rank for rank, value in zip(ranks, values) if value > 0)
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w == 0:
        return None
    z = (w_plus - mean_w) / std_w
    # Two-sided normal tail.
    return max(0.0, min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))


# ──────────────────────────────────────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────────────────────────────────────

def compare(baseline_rows: list[dict], candidate_rows: list[dict]) -> dict[str, Any]:
    baseline = {_key(row): row for row in baseline_rows if _key(row)}
    candidate = {_key(row): row for row in candidate_rows if _key(row)}
    shared = sorted(set(baseline) & set(candidate))

    paired: list[dict[str, Any]] = []
    for key in shared:
        before, after = baseline[key], candidate[key]
        b_branch, a_branch = _branch(before), _branch(after)
        paired.append(
            {
                "target": _selector(after) or key,
                "baseline_branch": b_branch,
                "candidate_branch": a_branch,
                "delta": (a_branch - b_branch) if (b_branch is not None and a_branch is not None) else None,
                "baseline_measured": _is_measured(before),
                "candidate_measured": _is_measured(after),
                "baseline_mutation": _mutation(before),
                "candidate_mutation": _mutation(after),
                "baseline_seconds": _seconds(before),
                "candidate_seconds": _seconds(after),
                "new_failure": _failed(after) and not _failed(before),
            }
        )

    deltas = [row["delta"] for row in paired if row["delta"] is not None]
    regressions = [row for row in paired if row["delta"] is not None and row["delta"] < -1e-9]
    worst = min((row["delta"] for row in regressions), default=0.0)
    new_failures = [row for row in paired if row["new_failure"]]

    candidate_measured = [row for row in paired if row["candidate_measured"]]
    measured_fraction = len(candidate_measured) / len(paired) if paired else 0.0
    baseline_measured = [row for row in paired if row["baseline_measured"]]
    baseline_measured_fraction = len(baseline_measured) / len(paired) if paired else 0.0
    # Only a target measured in both arms contributes a trustworthy pair.
    both_measured = [
        row for row in paired
        if row["baseline_measured"] and row["candidate_measured"]
    ]

    baseline_seconds = sum(row["baseline_seconds"] for row in paired)
    candidate_seconds = sum(row["candidate_seconds"] for row in paired)
    wallclock_growth = (
        (candidate_seconds - baseline_seconds) / baseline_seconds
        if baseline_seconds > 0
        else 0.0
    )

    mutation_pairs = [
        (row["baseline_mutation"], row["candidate_mutation"])
        for row in paired
        if row["baseline_mutation"] is not None and row["candidate_mutation"] is not None
    ]
    mutation_delta = (
        statistics.mean(after - before for before, after in mutation_pairs)
        if mutation_pairs
        else None
    )

    return {
        "paired_count": len(paired),
        "baseline_only": sorted(set(baseline) - set(candidate)),
        "candidate_only": sorted(set(candidate) - set(baseline)),
        "rows": paired,
        "mean_delta": statistics.mean(deltas) if deltas else 0.0,
        "median_delta": statistics.median(deltas) if deltas else 0.0,
        "improved": sum(1 for d in deltas if d > 1e-9),
        "unchanged": sum(1 for d in deltas if abs(d) <= 1e-9),
        "regressed": len(regressions),
        "worst_regression": worst,
        "new_failures": [row["target"] for row in new_failures],
        "measured_fraction": measured_fraction,
        "baseline_measured_fraction": baseline_measured_fraction,
        "measured_pairs": len(both_measured),
        "mean_delta_measured_pairs": statistics.mean(
            [row["delta"] for row in both_measured if row["delta"] is not None]
        ) if any(row["delta"] is not None for row in both_measured) else None,
        "wallclock_growth": wallclock_growth,
        "baseline_seconds": baseline_seconds,
        "candidate_seconds": candidate_seconds,
        "mutation_mean_delta": mutation_delta,
        "p_value": _wilcoxon_signed_rank_p(deltas),
        "baseline_mean_branch": statistics.mean(
            [row["baseline_branch"] for row in paired if row["baseline_branch"] is not None]
        ) if paired else 0.0,
        "candidate_mean_branch": statistics.mean(
            [row["candidate_branch"] for row in paired if row["candidate_branch"] is not None]
        ) if paired else 0.0,
    }


def evaluate_acceptance(result: dict[str, Any], args: argparse.Namespace) -> list[tuple[str, bool, str]]:
    checks: list[tuple[str, bool, str]] = []

    mean_delta = result["mean_delta"]
    checks.append((
        f"mean Δbranch ≥ {args.min_mean_delta:+.3f}",
        mean_delta >= args.min_mean_delta,
        f"{mean_delta:+.4f}",
    ))

    p_value = result["p_value"]
    checks.append((
        f"Wilcoxon p < {args.max_p_value}",
        p_value is not None and p_value < args.max_p_value,
        "n/a (too few non-zero pairs)" if p_value is None else f"{p_value:.4f}",
    ))

    checks.append((
        f"worst regression ≤ {args.max_regression:.3f}",
        abs(result["worst_regression"]) <= args.max_regression + 1e-9,
        f"{result['worst_regression']:+.4f}",
    ))

    checks.append((
        f"regressing methods ≤ {_max_regressing(args, result)}",
        result["regressed"] <= _max_regressing(args, result),
        str(result["regressed"]),
    ))

    checks.append((
        "no new compile/runtime failures",
        not result["new_failures"],
        ", ".join(result["new_failures"]) or "none",
    ))

    checks.append((
        f"wall-clock growth ≤ {args.max_wallclock_growth:.0%}",
        result["wallclock_growth"] <= args.max_wallclock_growth + 1e-9,
        f"{result['wallclock_growth']:+.1%}",
    ))

    mutation_delta = result["mutation_mean_delta"]
    checks.append((
        "mutation score not lower",
        mutation_delta is None or mutation_delta >= -1e-9,
        "not measured" if mutation_delta is None else f"{mutation_delta:+.4f}",
    ))

    checks.append((
        f"targets measured in both arms ≥ {args.min_measured_pairs}",
        result["measured_pairs"] >= args.min_measured_pairs,
        f"{result['measured_pairs']} of {result['paired_count']}",
    ))

    measured_drop = result["baseline_measured_fraction"] - result["measured_fraction"]
    checks.append((
        f"measurement not degraded (≤ {args.max_measured_fraction_drop:.0%} drop)",
        measured_drop <= args.max_measured_fraction_drop + 1e-9,
        f"{result['baseline_measured_fraction']:.1%} → {result['measured_fraction']:.1%}",
    ))

    return checks


def _print_report(result: dict[str, Any], checks: list[tuple[str, bool, str]]) -> None:
    print("=" * 78)
    print("PAIRED SUBSET COMPARISON")
    print("=" * 78)
    print(f"paired targets      : {result['paired_count']}")
    if result["baseline_only"]:
        print(f"baseline only       : {len(result['baseline_only'])} -> {', '.join(result['baseline_only'][:6])}")
    if result["candidate_only"]:
        print(f"candidate only      : {len(result['candidate_only'])} -> {', '.join(result['candidate_only'][:6])}")
    print(f"mean branch         : {result['baseline_mean_branch']:.4f} -> {result['candidate_mean_branch']:.4f}")
    print(f"mean Δbranch        : {result['mean_delta']:+.4f}   (median {result['median_delta']:+.4f})")
    print(f"improved / flat / regressed : {result['improved']} / {result['unchanged']} / {result['regressed']}")
    print(f"wall clock          : {result['baseline_seconds']/3600:.2f}h -> {result['candidate_seconds']/3600:.2f}h ({result['wallclock_growth']:+.1%})")
    print()

    print("-" * 78)
    print(f"{'target':44s} {'before':>8s} {'after':>8s} {'delta':>8s}")
    print("-" * 78)
    ordered = sorted(
        result["rows"],
        key=lambda row: (row["delta"] is None, row["delta"] if row["delta"] is not None else 0.0),
    )
    for row in ordered:
        before = f"{row['baseline_branch']:.3f}" if row["baseline_branch"] is not None else "  --  "
        after = f"{row['candidate_branch']:.3f}" if row["candidate_branch"] is not None else "  --  "
        delta = f"{row['delta']:+.3f}" if row["delta"] is not None else "   n/a"
        flag = "  !" if row["new_failure"] else ""
        print(f"{row['target'][:44]:44s} {before:>8s} {after:>8s} {delta:>8s}{flag}")
    print()

    print("=" * 78)
    print("ACCEPTANCE RULE")
    print("=" * 78)
    for label, passed, detail in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}]  {label:38s}  {detail}")
    print()
    verdict = all(passed for _, passed, _ in checks)
    print(f"VERDICT: {'ACCEPT' if verdict else 'REJECT'}")
    if not verdict:
        print("Per the anti-plateau rule, allow at most two revisions of this strategy")
        print("before retiring it and moving to the next one.")


def _max_regressing(args, result) -> int:
    """Regression cap, scaled to the paired set unless explicitly overridden.

    The measured null regresses roughly 41% of targets with no code change, so a
    fixed small cap rejects everything.  See MEASURED_NULL_SD above.
    """

    if args.max_regressing_methods is not None:
        return int(args.max_regressing_methods)
    paired = int(result.get("measured_pairs") or result.get("paired_count") or 0)
    return max(1, int(paired * DEFAULT_MAX_REGRESSING_METHODS_FRACTION))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("baseline", type=Path, help="Baseline run directory or rows file.")
    parser.add_argument("candidate", type=Path, help="Candidate run directory or rows file.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the full result as JSON.")
    parser.add_argument("--min-mean-delta", type=float, default=DEFAULT_MIN_MEAN_DELTA)
    parser.add_argument("--max-regression", type=float, default=DEFAULT_MAX_REGRESSION)
    parser.add_argument(
        "--max-regressing-methods",
        type=int,
        default=None,
        help="Absolute cap. Default: scaled from the paired count by "
             "DEFAULT_MAX_REGRESSING_METHODS_FRACTION, because the measured "
             "null regresses ~41%% of targets with no code change.",
    )
    parser.add_argument("--max-wallclock-growth", type=float, default=DEFAULT_MAX_WALLCLOCK_GROWTH)
    parser.add_argument("--min-measured-pairs", type=int, default=DEFAULT_MIN_MEASURED_PAIRS)
    parser.add_argument(
        "--max-measured-fraction-drop",
        type=float,
        default=DEFAULT_MAX_MEASURED_FRACTION_DROP,
    )
    parser.add_argument("--max-p-value", type=float, default=DEFAULT_MAX_P_VALUE)
    args = parser.parse_args(argv)

    result = compare(_load_rows(args.baseline), _load_rows(args.candidate))
    if not result["paired_count"]:
        print("no targets paired between the two runs; check the selectors", file=sys.stderr)
        return 2
    checks = evaluate_acceptance(result, args)
    _print_report(result, checks)

    if args.json:
        payload = dict(result)
        payload["acceptance"] = [
            {"criterion": label, "passed": passed, "value": detail}
            for label, passed, detail in checks
        ]
        payload["verdict"] = "accept" if all(passed for _, passed, _ in checks) else "reject"
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if all(passed for _, passed, _ in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
