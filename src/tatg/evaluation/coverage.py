from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CoverageStats:
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    instruction_coverage: float = 0.0
    class_coverage: float = 0.0
    method_coverage: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_coverage": self.line_coverage,
            "branch_coverage": self.branch_coverage,
            "instruction_coverage": self.instruction_coverage,
            "class_coverage": self.class_coverage,
            "method_coverage": self.method_coverage,
            **self.metadata,
        }


def parse_jacoco_csv(csv_path: Path) -> CoverageStats:
    """
    Parse JaCoCo CSV coverage report.
    
    JaCoCo CSV format (per row):
    NAME, INSTRUCTION_MISSED, INSTRUCTION_COVERED, BRANCH_MISSED, BRANCH_COVERED,
    LINE_MISSED, LINE_COVERED, COMPLEXITY_MISSED, COMPLEXITY_COVERED, METHOD_MISSED, METHOD_COVERED
    """
    if not csv_path.exists():
        return CoverageStats()

    total_instruction_missed = 0
    total_instruction_covered = 0
    total_branch_missed = 0
    total_branch_covered = 0
    total_line_missed = 0
    total_line_covered = 0
    total_method_missed = 0
    total_method_covered = 0
    total_class_missed = 0
    total_class_covered = 0

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    total_instruction_missed += int(row.get("INSTRUCTION_MISSED", 0) or 0)
                    total_instruction_covered += int(row.get("INSTRUCTION_COVERED", 0) or 0)
                    total_branch_missed += int(row.get("BRANCH_MISSED", 0) or 0)
                    total_branch_covered += int(row.get("BRANCH_COVERED", 0) or 0)
                    total_line_missed += int(row.get("LINE_MISSED", 0) or 0)
                    total_line_covered += int(row.get("LINE_COVERED", 0) or 0)
                    total_method_missed += int(row.get("METHOD_MISSED", 0) or 0)
                    total_method_covered += int(row.get("METHOD_COVERED", 0) or 0)
                    total_class_missed += int(row.get("CLASS_MISSED", 0) or 0)
                    total_class_covered += int(row.get("CLASS_COVERED", 0) or 0)
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        raise ValueError(f"Error parsing JaCoCo CSV {csv_path}: {e}")

    line_total = total_line_missed + total_line_covered
    branch_total = total_branch_missed + total_branch_covered
    instruction_total = total_instruction_missed + total_instruction_covered
    class_total = total_class_missed + total_class_covered
    method_total = total_method_missed + total_method_covered

    return CoverageStats(
        line_coverage=total_line_covered / line_total if line_total > 0 else 0.0,
        branch_coverage=total_branch_covered / branch_total if branch_total > 0 else 0.0,
        instruction_coverage=total_instruction_covered / instruction_total if instruction_total > 0 else 0.0,
        class_coverage=total_class_covered / class_total if class_total > 0 else 0.0,
        method_coverage=total_method_covered / method_total if method_total > 0 else 0.0,
        metadata={
            "line_covered": total_line_covered,
            "line_missed": total_line_missed,
            "branch_covered": total_branch_covered,
            "branch_missed": total_branch_missed,
        },
    )


def parse_pytest_coverage_json(json_path: Path) -> CoverageStats:
    """
    Parse pytest-cov JSON coverage report (.coverage.json or coverage.json).
    """
    if not json_path.exists():
        return CoverageStats()

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Error parsing pytest-cov JSON {json_path}: {e}")

    summary = data.get("totals", {})
    percent_covered = summary.get("percent_covered", 0.0)

    return CoverageStats(
        line_coverage=percent_covered / 100.0 if percent_covered > 0 else 0.0,
        metadata={
            "num_statements": summary.get("num_statements", 0),
            "covered_lines": summary.get("covered_lines", 0),
            "missing_lines": summary.get("missing_lines", 0),
        },
    )


def parse_jacoco_xml(
    xml_path: Path,
    *,
    class_fqn: str | None = None,
    method_name: str | None = None,
) -> CoverageStats:
    """Parse a JaCoCo XML report, optionally scoping to a single class/method.

    * If ``class_fqn`` is ``None``, the report-level totals are used (whole project).
    * If ``class_fqn`` is set and ``method_name`` is ``None``, the matching ``<class>``'s
      counters are aggregated.
    * If both are set, only the matching ``<method>``'s counters are used.

    Class FQN may be supplied with either ``.`` or ``/`` separators
    (e.g. ``"com.foo.MyClass"`` or ``"com/foo/MyClass"``); both are normalised.
    """
    if not xml_path.exists():
        return CoverageStats()

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ValueError(f"Error parsing JaCoCo XML {xml_path}: {e}")
    root = tree.getroot()

    counter_sources: list[ET.Element] = []
    target_class_name: str | None = None
    if class_fqn:
        target_class_name = class_fqn.replace(".", "/")

    if target_class_name is None:
        # Project-wide: use top-level <counter> children of <report>.
        counter_sources = [root]
    else:
        for cls in root.iter("class"):
            if cls.get("name") != target_class_name:
                continue
            if method_name is None:
                counter_sources.append(cls)
                continue
            for meth in cls.findall("method"):
                if meth.get("name") == method_name:
                    counter_sources.append(meth)

    if not counter_sources:
        return CoverageStats(
            metadata={"matched": False, "class_fqn": class_fqn, "method_name": method_name}
        )

    totals: dict[str, tuple[int, int]] = {}  # type -> (missed, covered)
    for src in counter_sources:
        # Direct <counter> children only — descendants would double-count.
        for ctr in src.findall("counter"):
            try:
                t = ctr.get("type", "")
                m = int(ctr.get("missed", "0") or 0)
                c = int(ctr.get("covered", "0") or 0)
            except (ValueError, TypeError):
                continue
            pm, pc = totals.get(t, (0, 0))
            totals[t] = (pm + m, pc + c)

    def _ratio(t: str) -> float:
        m, c = totals.get(t, (0, 0))
        tot = m + c
        return (c / tot) if tot > 0 else 0.0

    return CoverageStats(
        line_coverage=_ratio("LINE"),
        branch_coverage=_ratio("BRANCH"),
        instruction_coverage=_ratio("INSTRUCTION"),
        class_coverage=_ratio("CLASS"),
        method_coverage=_ratio("METHOD"),
        metadata={
            "matched": True,
            "class_fqn": class_fqn,
            "method_name": method_name,
            "line_covered": totals.get("LINE", (0, 0))[1],
            "line_missed": totals.get("LINE", (0, 0))[0],
            "branch_covered": totals.get("BRANCH", (0, 0))[1],
            "branch_missed": totals.get("BRANCH", (0, 0))[0],
        },
    )


def parse_jacoco_xml_per_method(
    xml_path: Path,
    class_fqn: str,
) -> list[dict[str, Any]]:
    """Return one row per ``<method>`` under the matching ``<class>``.

    Each row carries:
    ``{method_name, descriptor, line_coverage, branch_coverage,
       instruction_coverage, line_covered, line_missed,
       branch_covered, branch_missed}``.

    Ratios are floats in ``[0, 1]``. Returns an empty list when the report
    or class cannot be found — callers should treat that as "no data".
    """
    if not class_fqn or not xml_path.exists():
        return []

    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        raise ValueError(f"Error parsing JaCoCo XML {xml_path}: {e}")
    root = tree.getroot()
    target = class_fqn.replace(".", "/")

    rows: list[dict[str, Any]] = []
    for cls in root.iter("class"):
        if cls.get("name") != target:
            continue
        for meth in cls.findall("method"):
            totals: dict[str, tuple[int, int]] = {}
            for ctr in meth.findall("counter"):
                try:
                    t = ctr.get("type", "")
                    m = int(ctr.get("missed", "0") or 0)
                    c = int(ctr.get("covered", "0") or 0)
                except (ValueError, TypeError):
                    continue
                pm, pc = totals.get(t, (0, 0))
                totals[t] = (pm + m, pc + c)

            def _ratio(t: str) -> float:
                m, c = totals.get(t, (0, 0))
                tot = m + c
                return (c / tot) if tot > 0 else 0.0

            line_m, line_c = totals.get("LINE", (0, 0))
            br_m, br_c = totals.get("BRANCH", (0, 0))
            try:
                method_start_line = int(meth.get("line") or 0)
            except (ValueError, TypeError):
                method_start_line = 0
            rows.append({
                "method_name": meth.get("name"),
                "descriptor": meth.get("desc"),
                "method_start_line": method_start_line or None,
                "line_coverage": _ratio("LINE"),
                "branch_coverage": _ratio("BRANCH"),
                "instruction_coverage": _ratio("INSTRUCTION"),
                "line_covered": line_c,
                "line_missed": line_m,
                "branch_covered": br_c,
                "branch_missed": br_m,
            })
    return rows
