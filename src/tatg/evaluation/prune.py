"""
Prune failing JUnit test methods from generated test sources before invoking
mutation analysis (PITest aborts unless the test suite is green).

The strategy is non-destructive:
  1. Read JUnit XML reports under ``target/surefire-reports/`` (Maven) or
     ``build/test-results/test/`` (Gradle) to collect failing
     ``(test_class_fqn, method_name)`` pairs.
  2. Locate the test source file under ``src/test/java/`` (or
     ``src/test/kotlin/``) and, for each failing method, insert
     ``@org.junit.Ignore`` (JUnit 4) or ``@org.junit.jupiter.api.Disabled``
     (JUnit 5) immediately above its ``@Test`` annotation.

The function is idempotent: pre-existing ``@Ignore`` / ``@Disabled``
annotations are left untouched. A summary list of pruned methods is returned
so callers can record the action in the run's JSON payload.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


@dataclass
class PrunedTest:
    test_class: str
    method: str
    file: str
    annotation: str  # "@Ignore" or "@Disabled"
    reason: str  # short snippet of failure/error message


@dataclass
class PruneResult:
    failures: list[tuple[str, str, str]] = field(default_factory=list)  # (class, method, reason)
    pruned: list[PrunedTest] = field(default_factory=list)
    skipped: list[tuple[str, str, str]] = field(default_factory=list)  # (class, method, reason)

    def to_dict(self) -> dict:
        return {
            "failure_count": len(self.failures),
            "pruned_count": len(self.pruned),
            "skipped_count": len(self.skipped),
            "pruned": [
                {
                    "test_class": p.test_class,
                    "method": p.method,
                    "file": p.file,
                    "annotation": p.annotation,
                    "reason": p.reason,
                }
                for p in self.pruned
            ],
            "skipped": [
                {"test_class": c, "method": m, "reason": r} for c, m, r in self.skipped
            ],
        }


def _iter_surefire_reports(project_root: Path) -> Iterable[Path]:
    candidates = [
        project_root / "target" / "surefire-reports",
        project_root / "target" / "failsafe-reports",
        project_root / "build" / "test-results" / "test",
    ]
    for d in candidates:
        if d.is_dir():
            for p in d.glob("TEST-*.xml"):
                yield p


def parse_surefire_failures(project_root: Path) -> list[tuple[str, str, str]]:
    """
    Return ``[(test_class_fqn, method_name, reason)]`` for all failing or
    erroring test cases discovered in Surefire/Failsafe/Gradle reports.
    """
    out: list[tuple[str, str, str]] = []
    for xml_path in _iter_surefire_reports(Path(project_root)):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            continue
        for tc in root.iter("testcase"):
            classname = (tc.get("classname") or "").strip()
            name = (tc.get("name") or "").strip()
            if not classname or not name:
                continue
            # Parameterized methods may include "[0]", "[bracket]" suffix —
            # strip to the bare identifier so we can find the source method.
            base_name = re.sub(r"\[.*\]$", "", name)
            failure = tc.find("failure")
            error = tc.find("error")
            if failure is None and error is None:
                continue
            node = failure if failure is not None else error
            msg = (node.get("message") or "").strip() if node is not None else ""
            kind = "failure" if failure is not None else "error"
            reason = f"{kind}: {msg[:200]}" if msg else kind
            out.append((classname, base_name, reason))
    # Deduplicate (class, method)
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str, str]] = []
    for cls, mtd, reason in out:
        key = (cls, mtd)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((cls, mtd, reason))
    return deduped


_TEST_ANNOTATION_RE = re.compile(r"^(\s*)@Test\b")


def _disable_method_in_source(
    source: str,
    method_name: str,
    annotation_import: str,
    annotation_name: str,
) -> tuple[str, bool]:
    """
    Insert ``@<annotation_name>("auto-pruned: failing test")`` immediately
    above the ``@Test``-annotated method declaration ``method_name``.

    Returns the modified source and a bool indicating whether an edit was made.
    Idempotent: returns ``(source, False)`` if the method is already disabled
    or cannot be located.
    """
    # Find ``<modifiers...> <return-type> method_name (`` — robust enough for
    # ordinary JUnit methods (void m() / public void m()) without parsing Java.
    decl_re = re.compile(
        r"^(?P<indent>[ \t]*)(?P<modifiers>(?:public|protected|private|static|final|\s)+)?"
        r"(?P<retype>[\w<>\[\],?\s.]+?)\s+"
        + re.escape(method_name)
        + r"\s*\([^)]*\)\s*(?:throws[^{;]+)?\{",
        re.MULTILINE,
    )
    lines = source.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    edited = False
    while i < len(lines):
        line = lines[i]
        # Look for the method declaration line.
        if decl_re.search(line):
            # Walk backwards to find @Test annotation block.
            test_idx = None
            j = i - 1
            while j >= 0 and (lines[j].strip().startswith("@") or lines[j].strip() == ""):
                if _TEST_ANNOTATION_RE.match(lines[j]):
                    test_idx = j
                    break
                j -= 1
            if test_idx is not None:
                # Check for existing disable annotation in the same block.
                block_start = test_idx
                k = test_idx - 1
                while k >= 0 and (lines[k].strip().startswith("@") or lines[k].strip() == ""):
                    block_start = k if lines[k].strip().startswith("@") else block_start
                    k -= 1
                existing_block = "".join(lines[block_start : i + 1])
                if "@Ignore" in existing_block or "@Disabled" in existing_block:
                    out.append(line)
                    i += 1
                    continue
                # Determine indent matching @Test annotation.
                m_indent = _TEST_ANNOTATION_RE.match(lines[test_idx])
                indent = m_indent.group(1) if m_indent else ""
                # Replace the lines up to but not including test_idx, then
                # inject the disable annotation, then the rest unchanged.
                # `out` already contains lines[0..i-1]; pop entries we already
                # appended starting from test_idx (relative to out length).
                # Simpler: rebuild — but `out` has appended up to i-1, with
                # test_idx < i. Since we walked the file linearly, lines
                # [test_idx..i-1] are already in `out`. Insert the annotation
                # just before the @Test line in `out`.
                inject_pos = len(out) - (i - test_idx)
                disable_line = f"{indent}@{annotation_name}(\"auto-pruned: failing test\")\n"
                out.insert(inject_pos, disable_line)
                out.append(line)
                edited = True
                i += 1
                continue
        out.append(line)
        i += 1

    new_source = "".join(out)
    if not edited:
        return source, False

    # Inject the import if missing.
    if annotation_import not in new_source:
        new_source = _inject_import(new_source, annotation_import)
    return new_source, True


def _inject_import(source: str, fqn: str) -> str:
    import_line = f"import {fqn};\n"
    if import_line in source:
        return source
    lines = source.splitlines(keepends=True)
    # Insert after the package statement or after the last existing import.
    last_import = -1
    package_idx = -1
    for idx, ln in enumerate(lines):
        stripped = ln.lstrip()
        if stripped.startswith("package "):
            package_idx = idx
        elif stripped.startswith("import "):
            last_import = idx
    insert_at = last_import + 1 if last_import >= 0 else package_idx + 1
    if insert_at <= 0:
        insert_at = 0
        lines.insert(insert_at, import_line)
    else:
        # Ensure a blank line separation when inserting right after `package`.
        lines.insert(insert_at, import_line)
    return "".join(lines)


def _find_test_source(project_root: Path, test_class_fqn: str) -> Path | None:
    rel = test_class_fqn.replace(".", "/") + ".java"
    for base in (
        project_root / "src" / "test" / "java",
        project_root / "src" / "test" / "kotlin",
        project_root / "src" / "test" / "groovy",
    ):
        candidate = base / rel
        if candidate.is_file():
            return candidate
    # Fall back: search anywhere under src/test.
    for path in (project_root / "src" / "test").rglob(Path(rel).name):
        return path
    return None


def prune_failing_tests(
    project_root: Path,
    junit_family: str | None = None,
    allowed_test_files: Iterable[Path | str] | None = None,
) -> PruneResult:
    """
    Detect failing tests via Surefire reports and disable them in source by
    inserting ``@Ignore`` / ``@Disabled`` annotations. Returns a structured
    summary suitable for embedding in a JSON run report.

    Args:
        project_root: project root containing ``target/`` or ``build/``.
        junit_family: ``"junit4"`` or ``"junit5"``; defaults to ``"junit5"``
            when not specified (modern projects). The annotation chosen
            follows the family.
    """
    project_root = Path(project_root)
    result = PruneResult()
    failures = parse_surefire_failures(project_root)
    result.failures = failures
    allowed: set[str] | None = None
    if allowed_test_files is not None:
        allowed = set()
        for item in allowed_test_files:
            try:
                allowed.add(str(Path(item).resolve()))
            except Exception:
                continue

    if not failures:
        return result

    family = (junit_family or "junit5").lower()
    if family == "junit4":
        annotation_import = "org.junit.Ignore"
        annotation_name = "Ignore"
    else:
        annotation_import = "org.junit.jupiter.api.Disabled"
        annotation_name = "Disabled"

    # Group failures by class so we open each source file once.
    by_class: dict[str, list[tuple[str, str]]] = {}
    for cls, mtd, reason in failures:
        by_class.setdefault(cls, []).append((mtd, reason))

    for cls, methods in by_class.items():
        src_path = _find_test_source(project_root, cls)
        if src_path is None:
            for mtd, reason in methods:
                result.skipped.append((cls, mtd, f"source not found ({reason})"))
            continue
        if allowed is not None and str(src_path.resolve()) not in allowed:
            for mtd, reason in methods:
                result.skipped.append((cls, mtd, f"source outside allowed generated files ({reason})"))
            continue
        try:
            text = src_path.read_text(encoding="utf-8")
        except OSError as e:
            for mtd, reason in methods:
                result.skipped.append((cls, mtd, f"read error: {e}"))
            continue

        modified = text
        for mtd, reason in methods:
            new_text, did_edit = _disable_method_in_source(
                modified, mtd, annotation_import, annotation_name
            )
            if did_edit:
                modified = new_text
                result.pruned.append(
                    PrunedTest(
                        test_class=cls,
                        method=mtd,
                        file=str(src_path),
                        annotation=f"@{annotation_name}",
                        reason=reason,
                    )
                )
            else:
                result.skipped.append((cls, mtd, f"method not located ({reason})"))

        if modified != text:
            try:
                src_path.write_text(modified, encoding="utf-8")
            except OSError as e:
                # Roll back accounting for any edits we recorded for this file.
                for p in list(result.pruned):
                    if p.file == str(src_path):
                        result.pruned.remove(p)
                        result.skipped.append((p.test_class, p.method, f"write error: {e}"))

    return result
