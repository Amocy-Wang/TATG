#!/usr/bin/env python3
"""Merge archived TATG hard-method results with current triage logic.

This is intentionally an offline/replay utility: it never calls an LLM and
never mutates generated tests.  It reuses the current batch summary classifiers
so old archives are not analyzed with stale ``dominant_missing_reason`` values.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_batch_module() -> Any:
    path = Path(__file__).resolve().parent / "run_all_hard_methods_batch.py"
    spec = importlib.util.spec_from_file_location("run_all_hard_methods_batch", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import batch module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_report(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("results", "rows", "methods"):
            rows = data.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _read_manifest(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _class_name(row: dict[str, Any]) -> str:
    return str(row.get("class_name") or row.get("target_class") or row.get("class") or "").strip()


def _method_name(row: dict[str, Any]) -> str:
    return str(row.get("method_name") or row.get("target_method") or row.get("method") or "").strip()


def _simple_key(row: dict[str, Any]) -> str:
    klass = _class_name(row)
    method = _method_name(row)
    if klass and method:
        return f"{klass}.{method}"
    return str(row.get("target_label") or row.get("selector") or "").strip()


def _artifact_path_key(row: dict[str, Any]) -> str:
    """Recover selector identity from per-method artifact paths.

    Some legacy artifacts store the resolved source class in ``class_name`` but
    are nested under the requested hard-method selector directory, for example
    ``tatg/Flat3MapNew/get/Flat3MapNew.get.json`` with JSON class
    ``Flat3Map``.  The path key is used only as an additional merge key so
    offline replay can attach the current blocker to the original selector row;
    it does not change target resolution or coverage attribution.
    """
    raw = str(row.get("json_path") or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    parts = path.parts
    if len(parts) < 3:
        return ""
    method = _method_name(row) or path.parent.name
    klass = path.parent.parent.name
    if not klass or not method or klass in {"tatg", "target", "tmp"}:
        return ""
    return f"{klass}.{method}"


def _source_file_key(row: dict[str, Any]) -> str:
    project = str(row.get("project") or row.get("project_name") or "").strip()
    source = str(row.get("target_file") or row.get("source_file") or row.get("target_source_file") or "").strip()
    klass = _class_name(row)
    method = _method_name(row)
    descriptor = str(row.get("target_descriptor") or row.get("descriptor") or "").strip()
    if not (source and klass and method):
        return ""
    normalized_source = source.replace("\\", "/").strip("/")
    parts = [project, normalized_source, klass, method, descriptor]
    return "|".join(part for part in parts if part)


def _normalized_source_path(row: dict[str, Any]) -> str:
    raw = str(
        row.get("target_file")
        or row.get("source_file")
        or row.get("target_source_file")
        or ""
    ).strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/")
    for marker in ("/src/main/java/", "/src/test/java/"):
        if marker in normalized:
            return marker.lstrip("/") + normalized.split(marker, 1)[1]
    return normalized.strip("/")


def _project_identity(row: dict[str, Any]) -> str:
    explicit = str(row.get("project") or row.get("project_name") or "").strip()
    if explicit:
        return explicit
    project_root = str(row.get("project_root") or "").strip()
    if project_root:
        normalized_root = project_root.replace("\\", "/").rstrip("/")
        if normalized_root:
            return normalized_root.rsplit("/", 1)[-1]
    raw = str(
        row.get("target_file")
        or row.get("source_file")
        or row.get("target_source_file")
        or ""
    ).replace("\\", "/")
    for marker in ("/src/main/java/", "/src/test/java/"):
        if marker in raw:
            prefix = raw.split(marker, 1)[0].rstrip("/")
            return prefix.rsplit("/", 1)[-1]
    return ""


def _descriptor(row: dict[str, Any]) -> str:
    return str(row.get("target_descriptor") or row.get("descriptor") or "").strip()


def _strict_identity_compatible(base: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Match result artifacts without silently crossing source/overload targets."""
    if _class_name(base) != _class_name(candidate) or _method_name(base) != _method_name(candidate):
        return False
    base_project = _project_identity(base)
    candidate_project = _project_identity(candidate)
    if base_project and candidate_project and base_project != candidate_project:
        return False
    base_source = _normalized_source_path(base)
    candidate_source = _normalized_source_path(candidate)
    if base_source and candidate_source and base_source != candidate_source:
        return False
    base_descriptor = _descriptor(base)
    candidate_descriptor = _descriptor(candidate)
    if base_descriptor and candidate_descriptor and base_descriptor != candidate_descriptor:
        return False
    return bool(
        (base_project and candidate_project)
        or (base_source and candidate_source)
        or (base_descriptor and candidate_descriptor)
    )


def _strict_candidates_for_base(
    base: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matches = [row for row in candidates if _strict_identity_compatible(base, row)]
    base_descriptor = _descriptor(base)
    if base_descriptor:
        return [row for row in matches if not _descriptor(row) or _descriptor(row) == base_descriptor]

    # A descriptor-less legacy row may be upgraded only when all matching live
    # artifacts describe the same overload.  Otherwise the legacy identity is
    # ambiguous and replacing it would merge sibling overload coverage.
    descriptors = {_descriptor(row) for row in matches if _descriptor(row)}
    if len(descriptors) > 1:
        return []
    return matches


def _row_variants(row: dict[str, Any]) -> set[str]:
    values = {_simple_key(row)}
    source_key = _source_file_key(row)
    if source_key:
        values.add(source_key)
    path_key = _artifact_path_key(row)
    if path_key:
        values.add(path_key)
    descriptor = str(row.get("target_descriptor") or row.get("descriptor") or "").strip()
    overload = str(row.get("overload_short") or row.get("overload_id") or "").strip()
    if descriptor:
        values.add(f"{_simple_key(row)}#{descriptor}")
    if overload:
        values.add(f"{_simple_key(row)}#{overload}")
        values.add(f"{_simple_key(row)}__{overload}")
    for key in ("target_label", "selector", "method_selector", "target"):
        value = row.get(key)
        if value:
            values.add(str(value).strip())
    return {value for value in values if value}


def _num(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                stripped = value.strip()
                if stripped and stripped.lower() not in {"nan", "none", "null", "n/a"}:
                    return float(stripped)
            except Exception:
                pass
    return None


def _line(row: dict[str, Any]) -> float | None:
    return _num(row, "target_line_coverage", "target_line_coverage_display", "line_coverage")


def _branch(row: dict[str, Any]) -> float | None:
    return _num(row, "target_branch_coverage", "target_branch_coverage_display", "branch_coverage")


def _is_measured(row: dict[str, Any]) -> bool:
    status = str(row.get("coverage_measurement_status") or row.get("coverage_status") or "").strip().lower()
    return (
        status in {
            "measured",
            "measured_partial_artifact",
            "covered",
            "low_coverage",
            "covered_with_intermediate_failures",
            "candidate_measured_final_coverage_missing",
            "candidate_measured_best_snapshot_only",
            "candidate_measured_combination_rejected",
            "best_snapshot_only_measured",
            "combination_rejected_best_snapshot_measured",
        }
        or str(row.get("coverage_status") or "").strip().lower()
        in {
            "covered_with_candidate_evidence",
            "best_snapshot_only",
            "combination_rejected_best_snapshot_only",
            "best_snapshot_only_measured",
            "combination_rejected_best_snapshot_measured",
        }
    ) and _line(row) is not None


def _is_api_like(row: dict[str, Any]) -> bool:
    status_text = " ".join(
        str(row.get(key) or "").lower()
        for key in ("coverage_status", "coverage_measurement_status", "dominant_missing_reason", "failure_category", "error_hint", "status")
    )
    return any(token in status_text for token in ("bad_gateway", "provider", "connection", "llm_timeout", "api"))


def _best_measured_by_simple(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not _is_measured(row):
            continue
        for key in _row_variants(row):
            if not key:
                continue
            if key not in best or (_line(row) or -1.0) > (_line(best[key]) or -1.0):
                best[key] = row
    return best


def _best_retry_replacement_by_simple(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick the best retry result, including unmeasured non-API blockers.

    API retries are allowed to replace an old API error even when the retry did
    not produce measured coverage.  In that case the point is to replace
    ``llm_api_bad_gateway`` with the real blocker surfaced by the retry
    (compile-surface, target resolution, lifecycle, etc.).
    """
    best: dict[str, dict[str, Any]] = {}

    def score(row: dict[str, Any]) -> tuple[int, float]:
        api_like = _is_api_like(row)
        if _is_measured(row):
            return (3, _line(row) or 0.0)
        if api_like:
            return (0, -1.0)
        return (2, 0.0)

    for row in rows:
        for key in _row_variants(row):
            if not key:
                continue
            if key not in best or score(row) > score(best[key]):
                best[key] = row
    return best


def _candidate_rows_from_archive(archive_root: Path, low_manifest: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in ("focused-*/hard_method_report.json", "infra_fix_validation_20260624/*hard_method_report.json"):
        for path in sorted(archive_root.glob(pattern)):
            for row in _read_report(path):
                if _is_measured(row) and (_row_variants(row) & low_manifest):
                    rows.append(row)
    return rows


def _artifact_rows_from_root(root: Path) -> list[dict[str, Any]]:
    """Read per-method JSON/final or partial artifacts from *root*.

    This deliberately ignores aggregate reports and sidecar diagnostics.  It is
    used for replay/merge only; no generation or validation is triggered.
    """
    if not root.exists():
        return []
    candidates = sorted([*root.rglob("*.json"), *root.rglob("*.json.partial")])
    rows: list[dict[str, Any]] = []
    for path in candidates:
        name = path.name
        if name in {"hard_method_report.json", "summary.json", "merged_replay_rows.json", "merged_replay_summary.json"}:
            continue
        if name.endswith(".candidate_salvage_audit.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        if not (_class_name(data) and _method_name(data)):
            continue
        out = dict(data)
        out.setdefault("json_path", str(path))
        if path.name.endswith(".partial"):
            out["partial_artifact_used"] = True
        rows.append(out)
    return rows


def _best_current_replacement_by_simple(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Pick current artifact replacements without letting API failures win.

    Current measured artifacts can replace stale zero/unmeasured results, and
    can replace measured rows only when non-regressing via _merge_rows. Current
    non-API blockers can replace old unmeasured rows so stale generic
    ``no_target_positive_candidate`` reasons become actionable. API-like rows
    never win here; they are tracked separately for retry.
    """
    best: dict[str, dict[str, Any]] = {}

    def score(row: dict[str, Any]) -> tuple[int, float]:
        if _is_measured(row):
            return (3, _line(row) or 0.0)
        if not _is_api_like(row):
            return (2, 0.0)
        return (-1, -1.0)

    for row in rows:
        if score(row)[0] < 0:
            continue
        for key in sorted(_row_variants(row)):
            if not key:
                continue
            if key not in best or score(row) > score(best[key]):
                best[key] = row
    return best


def _merge_branch_non_regressing(
    base_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select strictly identified branch gains without sacrificing line coverage."""
    merged: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for base in base_rows:
        base_measured = _is_measured(base)
        base_line = _line(base)
        base_branch = _branch(base)
        eligible: list[dict[str, Any]] = []
        for candidate in _strict_candidates_for_base(base, candidate_rows):
            if not _is_measured(candidate) or _is_api_like(candidate):
                continue
            candidate_line = _line(candidate)
            candidate_branch = _branch(candidate)
            if candidate_line is None or candidate_branch is None:
                continue
            if base_measured:
                if base_line is not None and candidate_line + 1e-12 < base_line:
                    continue
                if (
                    base_branch is not None
                    and candidate_branch <= base_branch + 1e-12
                ):
                    continue
            eligible.append(candidate)
        if not eligible:
            merged.append(dict(base))
            continue
        replacement = max(
            eligible,
            key=lambda row: (
                _branch(row) or 0.0,
                _line(row) or 0.0,
                int(row.get("target_branch_covered") or 0),
            ),
        )
        out = dict(replacement)
        out["merged_replaces_line_coverage"] = base_line
        out["merged_replaces_branch_coverage"] = base_branch
        out["merged_selection_policy"] = (
            "branch_non_regressing"
            if base_measured
            else "measured_recovery"
        )
        merged.append(out)
        applied.append(out)
    return merged, applied


def _enrich_branch_measurement_evidence(
    rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Backfill branch applicability only from explicit matching JaCoCo counts."""
    enriched: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        explicit: list[tuple[int, int]] = []
        for candidate in _strict_candidates_for_base(row, candidate_rows):
            covered = candidate.get("target_branch_covered")
            missed = candidate.get("target_branch_missed")
            if isinstance(covered, (int, float)) and isinstance(missed, (int, float)):
                explicit.append((int(covered), int(missed)))
        if explicit and all(covered + missed == 0 for covered, missed in explicit):
            out["target_branch_covered"] = 0
            out["target_branch_missed"] = 0
            out["branch_measurement_evidence"] = "matching_descriptor_jacoco_zero_denominator"
        enriched.append(out)
    return enriched


def _current_api_retry_targets(rows: list[dict[str, Any]]) -> list[str]:
    """Return current API failures that do not already have measured siblings.

    Descriptor-expanded retries can leave an older method-name aggregate JSON
    beside newer descriptor-level measured artifacts.  The aggregate API error
    should not keep the simple method in the retry manifest once any descriptor
    sibling has produced measured coverage.
    """
    measured_keys = {_simple_key(row) for row in rows if _is_measured(row) and _simple_key(row)}
    return sorted({
        _simple_key(row)
        for row in rows
        if _is_api_like(row) and _simple_key(row) and _simple_key(row) not in measured_keys
    })


def _target_positive_candidate_coverage(cov: Any, gate: Any = None) -> bool:
    if not isinstance(cov, dict):
        return False
    if str(cov.get("status") or "").strip().lower() != "measured":
        return False
    if gate is None:
        gate = {}
    if isinstance(gate, dict) and not bool(gate.get("accepted", True)):
        return False
    if cov.get("has_target_coverage") is False:
        return False
    if cov.get("has_target_coverage") is True:
        return True
    return any(
        isinstance(cov.get(key), (int, float)) and float(cov.get(key) or 0.0) > 0.0
        for key in ("line_coverage", "branch_coverage", "line_covered", "branch_covered")
    )


def _best_target_positive_passed_candidate_coverage(validation_policy: dict[str, Any]) -> dict[str, Any] | None:
    rows: list[Any] = []
    for key in (
        "streaming_candidate_coverage_audit_details",
        "streaming_validation_passed_details",
        "coverage_positive_repair_passed_details",
    ):
        value = validation_policy.get(key)
        if isinstance(value, list):
            rows.extend(value)
    best: dict[str, Any] | None = None
    best_score = -1.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "passed").strip().lower() not in {"", "passed"}:
            continue
        cov = (
            row.get("candidate_target_coverage")
            or row.get("post_repair_candidate_target_coverage")
            or row.get("target_coverage")
        )
        gate = row.get("coverage_gate") if isinstance(row.get("coverage_gate"), dict) else None
        if not _target_positive_candidate_coverage(cov, gate):
            continue
        assert isinstance(cov, dict)
        line = float(cov.get("line_coverage") or 0.0)
        branch = float(cov.get("branch_coverage") or 0.0)
        line_covered = int(cov.get("line_covered") or 0)
        branch_covered = int(cov.get("branch_covered") or 0)
        score = line + branch + (line_covered + branch_covered) / 100000.0
        if score > best_score:
            best_score = score
            best = dict(cov)
            if row.get("test_method"):
                best["test_method"] = row.get("test_method")
    return best


def _merge_rows(
    base_rows: list[dict[str, Any]],
    replacements: dict[str, dict[str, Any]],
    *,
    non_regressing_only: bool = True,
    allow_unmeasured_non_api_replacement: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    for row in base_rows:
        replacement = None
        for key in sorted(
            _row_variants(row),
            key=lambda item: (
                0 if "|" in item else 1,
                0 if "#" in item else 1,
                0 if "__" in item else 1,
                item,
            ),
        ):
            replacement = replacements.get(key)
            if replacement:
                break
        replace = False
        if replacement:
            if (
                allow_unmeasured_non_api_replacement
                and not _is_measured(replacement)
                and not _is_measured(row)
            ):
                replace = True
            elif (
                not non_regressing_only
                or not _is_measured(row)
                or (_line(replacement) is not None and (_line(replacement) or -1.0) >= (_line(row) or -1.0))
            ):
                replace = True
        if replace:
            out = dict(replacement)
            out["merged_replaces_line_coverage"] = _line(row)
            out["merged_replaces_branch_coverage"] = _branch(row)
            merged.append(out)
            if _line(replacement) != _line(row) or _branch(replacement) != _branch(row):
                applied.append(out)
        else:
            merged.append(dict(row))
    return merged, applied


def _recompute_row(batch: Any, row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    validation_policy = out.get("validation_policy") if isinstance(out.get("validation_policy"), dict) else {}
    if validation_policy:
        try:
            streaming_summary = batch._streaming_validation_summary(out)
            for key, value in streaming_summary.items():
                if value not in (None, "", [], {}):
                    out.setdefault(key, value)
        except Exception:
            pass
        if (
            out.get("candidate_best_target_line_coverage") is None
            and out.get("candidate_best_target_branch_coverage") is None
        ):
            candidate_best = _best_target_positive_passed_candidate_coverage(validation_policy)
            if isinstance(candidate_best, dict):
                out["candidate_best_target_line_coverage"] = candidate_best.get("line_coverage")
                out["candidate_best_target_branch_coverage"] = candidate_best.get("branch_coverage")
                out["candidate_best_target_coverage_source"] = candidate_best.get("source")
                out["candidate_best_target_test_method"] = candidate_best.get("test_method")
                out["candidate_best_target_descriptor"] = candidate_best.get("descriptor")
        if (
            out.get("candidate_failed_best_target_line_coverage") is None
            and out.get("candidate_failed_best_target_branch_coverage") is None
        ):
            try:
                candidate_failed_best = batch._candidate_best_failed_target_coverage(out)
            except Exception:
                candidate_failed_best = None
            if isinstance(candidate_failed_best, dict):
                out["candidate_failed_best_target_line_coverage"] = candidate_failed_best.get("line_coverage")
                out["candidate_failed_best_target_branch_coverage"] = candidate_failed_best.get("branch_coverage")
                out["candidate_failed_best_target_failure_category"] = candidate_failed_best.get("failure_category")
    if out.get("json_coverage_measurement_status") is None and out.get("coverage_measurement_status") is not None:
        out["json_coverage_measurement_status"] = out.get("coverage_measurement_status")
    status_text = str(out.get("coverage_measurement_status") or out.get("coverage_status") or "").strip().lower()
    if status_text in {"measured", "measured_partial_artifact", "covered", "low_coverage", "covered_with_intermediate_failures"}:
        if out.get("target_line_coverage") is None:
            line_display = _num(out, "target_line_coverage_display", "line_coverage_display")
            if line_display is not None:
                out["target_line_coverage"] = line_display
        if out.get("target_branch_coverage") is None:
            branch_display = _num(out, "target_branch_coverage_display", "branch_coverage_display")
            if branch_display is not None:
                out["target_branch_coverage"] = branch_display
    out["coverage_status"] = batch._coverage_status(out)
    out["coverage_measurement_status"] = batch._coverage_measurement_status(out)
    out["dominant_missing_reason"] = batch._dominant_missing_reason(out, out)
    batch._refine_generic_measurement_status(out)
    precise_unmeasured_statuses = {
        "compile_blocker",
        "compile-type-mismatch",
        "compile_surface_blocker",
        "compilation-failure",
        "wrong_overload_or_arity",
        "unavailable_type",
        "project_native_preflight_failed",
        "residual_generated_test_compile_failure",
        "java_runtime_version_mismatch",
        "missing-required-initialization",
        "runtime_validation_failure",
        "target_descriptor_mismatch",
        "source_path_mismatch",
    }
    measurement_status = str(out.get("coverage_measurement_status") or "").strip()
    if (
        measurement_status in precise_unmeasured_statuses
        and out.get("target_line_coverage") is None
        and out.get("target_branch_coverage") is None
        and out.get("candidate_best_target_line_coverage") is None
        and out.get("candidate_best_target_branch_coverage") is None
    ):
        out["coverage_status"] = measurement_status
    line_display, branch_display = batch._coverage_display_values(out)
    out["target_line_coverage_display"] = line_display
    out["target_branch_coverage_display"] = branch_display
    out["branch_measurement_status"] = batch._branch_measurement_status(out)
    out["target_branch_coverage_applicable"] = (
        out["branch_measurement_status"] != "not_applicable_no_branches"
    )
    return out


def _bucket(row: dict[str, Any]) -> str:
    if not _is_measured(row):
        return "unmeasured"
    line = _line(row)
    if line is None:
        return "unmeasured"
    if line < 0.30:
        return "lt30"
    if line < 0.70:
        return "30to70"
    return "gte70"


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if _is_measured(row)]
    line_values = [_line(row) for row in measured if _line(row) is not None]
    branch_values = [_branch(row) for row in measured if _branch(row) is not None]
    branch_applicable = [
        row
        for row in measured
        if str(row.get("branch_measurement_status") or "") != "not_applicable_no_branches"
        and _branch(row) is not None
    ]
    applicable_branch_values = [_branch(row) or 0.0 for row in branch_applicable]
    buckets: dict[str, list[dict[str, Any]]] = {"lt30": [], "30to70": [], "gte70": [], "unmeasured": []}
    for row in rows:
        buckets[_bucket(row)].append(row)
    return {
        "total_rows": len(rows),
        "measured_count": len(measured),
        "avg_line_coverage": (sum(line_values) / len(line_values)) if line_values else None,
        "avg_branch_coverage": (sum(branch_values) / len(branch_values)) if branch_values else None,
        "branch_applicable_count": len(branch_applicable),
        "branch_not_applicable_count": len(measured) - len(branch_applicable),
        "avg_branch_coverage_applicable": (
            sum(applicable_branch_values) / len(applicable_branch_values)
            if applicable_branch_values
            else None
        ),
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "bucket_avg_line_coverage": {
            key: (
                sum(_line(row) or 0.0 for row in value if _is_measured(row)) / len([row for row in value if _is_measured(row)])
                if [row for row in value if _is_measured(row)]
                else None
            )
            for key, value in buckets.items()
        },
        "low_coverage_rows": [
            {
                "target": _simple_key(row),
                "line": _line(row),
                "branch": _branch(row),
                "bucket": _bucket(row),
                "coverage_status": row.get("coverage_status"),
                "coverage_measurement_status": row.get("coverage_measurement_status"),
                "dominant_missing_reason": row.get("dominant_missing_reason"),
                "retained_target_positive_count": row.get("retained_target_positive_count"),
                "coverage_positive_failed_count": row.get("coverage_positive_failed_count"),
            }
            for row in rows
            if _is_measured(row) and (_line(row) or 0.0) < 0.70
        ],
        "api_like_rows": [
            {
                "target": _simple_key(row),
                "coverage_status": row.get("coverage_status"),
                "coverage_measurement_status": row.get("coverage_measurement_status"),
                "dominant_missing_reason": row.get("dominant_missing_reason"),
            }
            for row in rows
            if any(
                token in str(row.get(key) or "").lower()
                for key in ("coverage_status", "coverage_measurement_status", "dominant_missing_reason", "failure_category", "error_hint")
                for token in ("bad_gateway", "provider", "connection", "llm_timeout", "api")
            )
        ],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "project",
        "class_name",
        "method_name",
        "overload_short",
        "target_descriptor",
        "target_line_coverage_display",
        "target_branch_coverage_display",
        "coverage_status",
        "coverage_measurement_status",
        "dominant_missing_reason",
        "generated_test_count",
        "streaming_validation_submitted",
        "streaming_validation_passed",
        "retained_target_positive_count",
        "coverage_positive_failed_count",
        "candidate_best_target_line_coverage",
        "candidate_best_target_branch_coverage",
        "candidate_best_target_coverage_source",
        "candidate_best_target_test_method",
        "candidate_failed_best_target_line_coverage",
        "candidate_failed_best_target_branch_coverage",
        "candidate_failed_best_target_failure_category",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge archived TATG eval results and replay current triage status")
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument(
        "--base-report",
        type=Path,
        default=None,
        help="Explicit baseline JSON report; defaults to <archive-root>/hard_method_report.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--current-artifact-root", type=Path, default=None)
    parser.add_argument(
        "--additional-artifact-root",
        type=Path,
        action="append",
        default=[],
        help="Additional read-only per-target artifact root; may be repeated.",
    )
    parser.add_argument(
        "--selection-policy",
        choices=("legacy", "branch-non-regressing"),
        default="legacy",
        help="Use strict source/descriptor matching and prefer branch gains without line regressions.",
    )
    args = parser.parse_args()

    archive_root = args.archive_root.resolve()
    output_dir = (args.output_dir or (archive_root / "merged_replay")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    batch = _load_batch_module()
    base_report = args.base_report.resolve() if args.base_report else archive_root / "hard_method_report.json"
    base_rows = _read_report(base_report)
    if not base_rows:
        raise SystemExit(f"No baseline report rows found in {base_report}")
    low_manifest = _read_manifest(archive_root / "low_coverage_28.txt")
    focused_replacements = _best_measured_by_simple(_candidate_rows_from_archive(archive_root, low_manifest))
    retry_replacements = _best_retry_replacement_by_simple(
        _read_report(archive_root / "retry_502_after_20260624-014300" / "hard_method_report.json")
    )
    current_artifact_rows = (
        _artifact_rows_from_root(args.current_artifact_root.resolve())
        if args.current_artifact_root
        else []
    )
    additional_artifact_rows: list[dict[str, Any]] = []
    for root in args.additional_artifact_root:
        additional_artifact_rows.extend(_artifact_rows_from_root(root.resolve()))
    current_replacements = _best_current_replacement_by_simple(current_artifact_rows)

    low_merged, low_applied = _merge_rows(base_rows, focused_replacements)
    merged, retry_applied = _merge_rows(
        low_merged,
        retry_replacements,
        allow_unmeasured_non_api_replacement=True,
    )
    if args.selection_policy == "branch-non-regressing":
        branch_candidates = [*current_artifact_rows, *additional_artifact_rows]
        merged, current_applied = _merge_branch_non_regressing(
            merged,
            branch_candidates,
        )
        merged = _enrich_branch_measurement_evidence(merged, branch_candidates)
    else:
        merged, current_applied = _merge_rows(
            merged,
            current_replacements,
            allow_unmeasured_non_api_replacement=True,
        )
    recomputed = [_recompute_row(batch, row) for row in merged]
    summary = _summary(recomputed)
    summary["applied_low_replacements"] = [
        {
            "target": _simple_key(row),
            "old_line": row.get("merged_replaces_line_coverage"),
            "new_line": _line(row),
            "old_branch": row.get("merged_replaces_branch_coverage"),
            "new_branch": _branch(row),
        }
        for row in low_applied
    ]
    summary["applied_retry_replacements"] = [
        {
            "target": _simple_key(row),
            "old_line": row.get("merged_replaces_line_coverage"),
            "new_line": _line(row),
            "old_branch": row.get("merged_replaces_branch_coverage"),
            "new_branch": _branch(row),
        }
        for row in retry_applied
    ]
    summary["applied_current_artifact_replacements"] = [
        {
            "target": _simple_key(row),
            "old_line": row.get("merged_replaces_line_coverage"),
            "new_line": _line(row),
            "old_branch": row.get("merged_replaces_branch_coverage"),
            "new_branch": _branch(row),
        }
        for row in current_applied
    ]
    current_api_retry_targets = _current_api_retry_targets(current_artifact_rows)
    summary["current_api_retry_targets"] = current_api_retry_targets
    summary["current_api_retry_target_count"] = len(current_api_retry_targets)

    (output_dir / "merged_replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "merged_replay_rows.json").write_text(json.dumps(recomputed, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(output_dir / "merged_replay_rows.csv", recomputed)
    (output_dir / "retry_api_current.txt").write_text(
        "\n".join(current_api_retry_targets) + ("\n" if current_api_retry_targets else ""),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "total_rows": summary["total_rows"],
        "measured_count": summary["measured_count"],
        "avg_line_coverage": summary["avg_line_coverage"],
        "avg_branch_coverage": summary["avg_branch_coverage"],
        "bucket_counts": summary["bucket_counts"],
        "api_like_count": len(summary["api_like_rows"]),
        "current_api_retry_target_count": len(current_api_retry_targets),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
