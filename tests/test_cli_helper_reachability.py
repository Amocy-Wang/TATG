"""Unit tests for helper-identification + reachability utilities used by the
per-method isolation probe to ensure that helpers added to the final test
file are validated (i.e. referenced by at least one passing @Test method)."""

from __future__ import annotations

from tatg.cli import (
    _drop_helpers_by_name,
    _extract_failing_methods_from_compile_output,
    _helper_reachability,
    _junit_lifecycle_method_ranges,
    _non_test_method_ranges,
    _strip_orphan_junit_annotations,
)


SRC_WITH_TWO_HELPERS = """
package p;

import java.util.List;

public class FooTest {

    private int helperUsedByA() { return 1; }
    private int helperUsedByNobody() { return 2; }

    @Test
    public void testA() {
        int x = helperUsedByA();
        assertEquals(1, x);
    }

    @Test
    public void testB() {
        assertTrue(true);
    }
}
"""


SRC_HELPER_CHAIN = """
package p;

public class FooTest {

    private int leaf() { return 0; }
    private int mid() { return leaf() + 1; }
    private int orphan() { return 99; }

    @Test
    public void testChain() {
        assertEquals(1, mid());
    }
}
"""


def test_non_test_method_ranges_excludes_at_test_methods():
    ranges = _non_test_method_ranges(SRC_WITH_TWO_HELPERS)
    names = {n for (n, _s, _e) in ranges}
    assert "helperUsedByA" in names
    assert "helperUsedByNobody" in names
    assert "testA" not in names
    assert "testB" not in names


def test_non_test_method_ranges_excludes_lifecycle_methods():
    src = """
package p;

public class FooTest {
    private String value;

    @Before
    public void setUp() { value = "ready"; }

    @After
    public void tearDown() { value = null; }

    private String helper() { return value; }

    @Test
    public void testA() { assertEquals("ready", helper()); }
}
"""
    ranges = _non_test_method_ranges(src)
    names = {n for (n, _s, _e) in ranges}
    assert names == {"helper"}
    assert "setUp" not in names
    assert "tearDown" not in names


def test_lifecycle_methods_can_be_dropped_when_diagnostic_hits_them(tmp_path):
    test_file = tmp_path / "FooTest.java"
    src = """
package p;

public class FooTest {
    @Before
    public void setUp() { MissingType value = null; }

    @Test
    public void testA() { assertTrue(true); }
}
"""
    test_file.write_text(src, encoding="utf-8")
    failure_output = f"{test_file}:[6,27] cannot find symbol"

    assert _extract_failing_methods_from_compile_output(failure_output, test_file) == ["setUp"]

    ranges = _junit_lifecycle_method_ranges(src)
    trimmed = _drop_helpers_by_name(src, ranges, {"setUp"})
    assert "MissingType" not in trimmed
    assert "@Before" not in trimmed
    assert "testA" in trimmed


def test_helper_reachability_keeps_only_referenced_helpers():
    helper_ranges = _non_test_method_ranges(SRC_WITH_TWO_HELPERS)
    reachable = _helper_reachability(
        SRC_WITH_TWO_HELPERS,
        seed_method_names={"testA"},
        helper_ranges=helper_ranges,
    )
    assert reachable == {"helperUsedByA"}


def test_helper_reachability_walks_transitively():
    helper_ranges = _non_test_method_ranges(SRC_HELPER_CHAIN)
    reachable = _helper_reachability(
        SRC_HELPER_CHAIN,
        seed_method_names={"testChain"},
        helper_ranges=helper_ranges,
    )
    assert reachable == {"mid", "leaf"}
    assert "orphan" not in reachable


def test_drop_helpers_by_name_removes_only_targeted_helpers():
    helper_ranges = _non_test_method_ranges(SRC_WITH_TWO_HELPERS)
    trimmed = _drop_helpers_by_name(
        SRC_WITH_TWO_HELPERS,
        helper_ranges,
        drop_names={"helperUsedByNobody"},
    )
    assert "helperUsedByNobody" not in trimmed
    assert "helperUsedByA" in trimmed
    # @Test methods untouched
    assert "testA" in trimmed
    assert "testB" in trimmed


def test_helper_reachability_no_validated_tests_returns_empty():
    helper_ranges = _non_test_method_ranges(SRC_WITH_TWO_HELPERS)
    reachable = _helper_reachability(
        SRC_WITH_TWO_HELPERS,
        seed_method_names=set(),
        helper_ranges=helper_ranges,
    )
    assert reachable == set()


def test_strip_orphan_junit_annotations_drops_dangling_before():
    src = (
        "import org.junit.Before;\n"
        "import org.junit.Test;\n"
        "class FooTest {\n"
        "  @Before\n"
        "\n"
        "\n"
        "  @Test\n"
        "  public void testA() {}\n"
        "}\n"
    )
    cleaned = _strip_orphan_junit_annotations(src)
    assert "@Before" not in cleaned
    assert "@Test" in cleaned
