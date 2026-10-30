"""Tests for ``tatg.evaluation.prune.prune_failing_tests``."""
from __future__ import annotations

from pathlib import Path

from tatg.evaluation.prune import (
    parse_surefire_failures,
    prune_failing_tests,
)


def _write_surefire_xml(reports_dir: Path, class_fqn: str, failing_methods: list[str]) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    testcases = "\n".join(
        f'<testcase classname="{class_fqn}" name="{m}"><failure message="boom"/></testcase>'
        for m in failing_methods
    )
    (reports_dir / f"TEST-{class_fqn}.xml").write_text(
        f'<?xml version="1.0"?>\n<testsuite name="{class_fqn}">\n{testcases}\n</testsuite>\n',
        encoding="utf-8",
    )


def _write_test_source(project_root: Path, class_fqn: str, body: str) -> Path:
    pkg = class_fqn.rsplit(".", 1)[0]
    cls = class_fqn.rsplit(".", 1)[1]
    test_dir = project_root / "src" / "test" / "java" / pkg.replace(".", "/")
    test_dir.mkdir(parents=True, exist_ok=True)
    src = test_dir / f"{cls}.java"
    src.write_text(body, encoding="utf-8")
    return src


def test_parse_surefire_failures(tmp_path: Path) -> None:
    _write_surefire_xml(
        tmp_path / "target" / "surefire-reports",
        "com.example.FooTest",
        ["badMethod"],
    )
    failures = parse_surefire_failures(tmp_path)
    assert failures == [("com.example.FooTest", "badMethod", "failure: boom")]


def test_prune_inserts_disabled_annotation_for_junit5(tmp_path: Path) -> None:
    _write_surefire_xml(
        tmp_path / "target" / "surefire-reports",
        "com.example.FooTest",
        ["badMethod"],
    )
    src = _write_test_source(
        tmp_path,
        "com.example.FooTest",
        """package com.example;
import org.junit.jupiter.api.Test;

public class FooTest {
    @Test
    public void goodMethod() {
    }

    @Test
    public void badMethod() {
        throw new RuntimeException("boom");
    }
}
""",
    )
    result = prune_failing_tests(tmp_path, junit_family="junit5")
    assert len(result.pruned) == 1
    assert result.pruned[0].annotation == "@Disabled"

    updated = src.read_text(encoding="utf-8")
    assert "import org.junit.jupiter.api.Disabled;" in updated
    # The disable annotation is attached to badMethod, not goodMethod.
    bad_index = updated.index("badMethod")
    good_index = updated.index("goodMethod")
    disabled_index = updated.index("@Disabled")
    assert good_index < disabled_index < bad_index


def test_prune_uses_ignore_for_junit4(tmp_path: Path) -> None:
    _write_surefire_xml(
        tmp_path / "target" / "surefire-reports",
        "com.example.BarTest",
        ["broken"],
    )
    src = _write_test_source(
        tmp_path,
        "com.example.BarTest",
        """package com.example;
import org.junit.Test;

public class BarTest {
    @Test
    public void broken() {}
}
""",
    )
    result = prune_failing_tests(tmp_path, junit_family="junit4")
    assert len(result.pruned) == 1
    assert result.pruned[0].annotation == "@Ignore"
    updated = src.read_text(encoding="utf-8")
    assert "import org.junit.Ignore;" in updated
    assert "@Ignore" in updated


def test_prune_is_idempotent(tmp_path: Path) -> None:
    _write_surefire_xml(
        tmp_path / "target" / "surefire-reports",
        "com.example.BazTest",
        ["alreadyOff"],
    )
    _write_test_source(
        tmp_path,
        "com.example.BazTest",
        """package com.example;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

public class BazTest {
    @Disabled
    @Test
    public void alreadyOff() {}
}
""",
    )
    result = prune_failing_tests(tmp_path, junit_family="junit5")
    # Existing @Disabled is detected and no duplicate annotation is added.
    assert result.pruned == []
    assert len(result.skipped) == 1


def test_prune_returns_empty_when_no_failures(tmp_path: Path) -> None:
    (tmp_path / "target" / "surefire-reports").mkdir(parents=True, exist_ok=True)
    result = prune_failing_tests(tmp_path)
    assert result.failures == []
    assert result.pruned == []


def test_prune_skips_failures_outside_allowed_files(tmp_path: Path) -> None:
    _write_surefire_xml(
        tmp_path / "target" / "surefire-reports",
        "com.example.ExistingTest",
        ["broken"],
    )
    existing = _write_test_source(
        tmp_path,
        "com.example.ExistingTest",
        """package com.example;
import org.junit.jupiter.api.Test;

public class ExistingTest {
    @Test
    public void broken() {}
}
""",
    )
    generated = _write_test_source(
        tmp_path,
        "com.example.GeneratedTest",
        """package com.example;
import org.junit.jupiter.api.Test;

public class GeneratedTest {
    @Test
    public void ok() {}
}
""",
    )

    result = prune_failing_tests(
        tmp_path,
        junit_family="junit5",
        allowed_test_files=[generated],
    )

    assert result.pruned == []
    assert len(result.skipped) == 1
    assert "outside allowed generated files" in result.skipped[0][2]
    assert "@Disabled" not in existing.read_text(encoding="utf-8")
