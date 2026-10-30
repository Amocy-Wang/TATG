"""Live regression tests for SootUp callgraph analysis on the mall project.

These tests are **skipped** unless both environment variables are set:
  * ``SOOTUP_CG_JAR``  — absolute path to the fat-JAR
  * ``SOOTUP_MALL_ADMIN_CP`` — colon-separated classpath for mall-admin
  * ``SOOTUP_MALL_PORTAL_CP`` — colon-separated classpath for mall-portal

Quick-start:
    export SOOTUP_CG_JAR=/path/to/sootup-callgraph-pathfinder.jar
    export SOOTUP_MALL_ADMIN_CP=$(cat /tmp/mall_admin_cp.txt)
    export SOOTUP_MALL_PORTAL_CP=$(cat /tmp/mall_portal_cp.txt)
    pytest tests/test_sootup_callgraph_live.py -v

Alternatively, pre-generate classpath files with Maven and pass them via env:
    cd /path/to/mall
    mvn -pl mall-admin -am -DskipTests dependency:build-classpath \\
        -Dmdep.outputFile=/tmp/mall_admin_cp.txt
    mvn -pl mall-portal -am -DskipTests dependency:build-classpath \\
        -Dmdep.outputFile=/tmp/mall_portal_cp.txt
"""

from __future__ import annotations

import os
import time

import pytest

from tatg.analysis.base import AnalysisRequest
from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

# ---------------------------------------------------------------------------
# Skip condition
# ---------------------------------------------------------------------------

_JAR = os.environ.get("SOOTUP_CG_JAR", "")
_ADMIN_CP = os.environ.get("SOOTUP_MALL_ADMIN_CP", "")
_PORTAL_CP = os.environ.get("SOOTUP_MALL_PORTAL_CP", "")

_SKIP_REASON = (
    "Live SootUp regression tests require SOOTUP_CG_JAR, "
    "SOOTUP_MALL_ADMIN_CP, and SOOTUP_MALL_PORTAL_CP env vars."
)
_LIVE = bool(_JAR and _ADMIN_CP and _PORTAL_CP)

pytestmark = pytest.mark.skipif(not _LIVE, reason=_SKIP_REASON)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------
# Each entry: (classpath_env_key, target_class, target_method, expected_callsites)
# expected_callsites: substrings that must appear in the raw output

_CASES: list[tuple[str, str, str, list[str]]] = [
    # mall-admin cases
    (
        "portal",
        "com.macro.mall.portal.service.impl.OmsPromotionServiceImpl",
        "calcCartPromotion",
        [
            "OmsPromotionServiceImpl",
            "calcCartPromotion",
        ],
    ),
    (
        "admin",
        "com.macro.mall.service.impl.UmsRoleServiceImpl",
        "allocResource",
        [
            "UmsRoleServiceImpl",
            "allocResource",
        ],
    ),
    (
        "admin",
        "com.macro.mall.service.impl.UmsAdminServiceImpl",
        "getResourceList",
        [
            "UmsAdminServiceImpl",
            "getResourceList",
        ],
    ),
    (
        "admin",
        "com.macro.mall.service.impl.PmsProductServiceImpl",
        "create",
        [
            "PmsProductServiceImpl",
            "create",
        ],
    ),
    (
        "admin",
        "com.macro.mall.service.impl.OmsOrderServiceImpl",
        "delivery",
        [
            "OmsOrderServiceImpl",
            "delivery",
        ],
    ),
    (
        "admin",
        "com.macro.mall.service.impl.UmsAdminServiceImpl",
        "login",
        [
            "UmsAdminServiceImpl",
            "login",
        ],
    ),
    # mall-portal cases
    (
        "portal",
        "com.macro.mall.portal.service.impl.OmsPortalOrderServiceImpl",
        "generateOrder",
        [
            "OmsPortalOrderServiceImpl",
            "generateOrder",
        ],
    ),
    (
        "portal",
        "com.macro.mall.portal.service.impl.OmsCartItemServiceImpl",
        "listPromotion",
        [
            "OmsCartItemServiceImpl",
            "listPromotion",
        ],
    ),
    (
        "portal",
        "com.macro.mall.portal.service.impl.HomeServiceImpl",
        "content",
        [
            "HomeServiceImpl",
            "content",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cp(module: str) -> str:
    return _ADMIN_CP if module == "admin" else _PORTAL_CP


def _analyzer() -> SootUpCallGraphAnalyzer:
    return SootUpCallGraphAnalyzer(
        jar_path=_JAR,
        timeout=300,
        use_cache=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module,target_class,target_method,expected_fragments",
    _CASES,
    ids=[f"{c[1].split('.')[-1]}.{c[2]}" for c in _CASES],
)
def test_sootup_finds_paths(
    module: str,
    target_class: str,
    target_method: str,
    expected_fragments: list[str],
) -> None:
    """SootUp must find at least one call-graph path for each target."""
    request = AnalysisRequest(
        classpath=_cp(module),
        target_class=target_class,
        target_method=target_method,
        max_depth=4,
        root_mode="PUBLIC",
        limit_package_roots=False,
    )
    t0 = time.perf_counter()
    result = _analyzer().analyze(request)
    elapsed = time.perf_counter() - t0

    assert result.structured is not None, (
        f"{target_class}.{target_method}: SootUp returned no structured result"
    )
    paths = result.structured.paths
    assert len(paths) >= 1, (
        f"{target_class}.{target_method}: expected ≥1 path, got 0 "
        f"(elapsed {elapsed:.1f}s)"
    )

    raw = result.raw_text or ""
    for fragment in expected_fragments:
        assert fragment in raw, (
            f"{target_class}.{target_method}: expected fragment {fragment!r} not "
            f"found in raw output (elapsed {elapsed:.1f}s)"
        )


@pytest.mark.parametrize(
    "module,target_class,target_method,_expected",
    _CASES,
    ids=[f"{c[1].split('.')[-1]}.{c[2]}" for c in _CASES],
)
def test_sootup_has_indirect_paths(
    module: str,
    target_class: str,
    target_method: str,
    _expected: list[str],
) -> None:
    """Each target must have at least one multi-hop (indirect) path.

    This validates that SootUp correctly traverses Controller → Service →
    target chains, not just direct call-sites.
    """
    request = AnalysisRequest(
        classpath=_cp(module),
        target_class=target_class,
        target_method=target_method,
        max_depth=4,
        root_mode="PUBLIC",
        limit_package_roots=False,
    )
    result = _analyzer().analyze(request)
    paths = result.structured.paths if result.structured else []
    indirect = [p for p in paths if len(p.steps) >= 2]
    assert len(indirect) >= 1, (
        f"{target_class}.{target_method}: expected ≥1 indirect path (len(steps)≥2), "
        f"got {len(paths)} total paths, 0 indirect"
    )


def test_all_nine_cases_pass() -> None:
    """Smoke test: run all 9 cases and assert no failures."""
    analyzer = _analyzer()
    failures: list[str] = []
    for module, target_class, target_method, expected in _CASES:
        request = AnalysisRequest(
            classpath=_cp(module),
            target_class=target_class,
            target_method=target_method,
            max_depth=4,
            root_mode="PUBLIC",
            limit_package_roots=False,
        )
        try:
            result = analyzer.analyze(request)
            paths = result.structured.paths if result.structured else []
            if not paths:
                failures.append(f"FAIL {target_class}.{target_method}: 0 paths")
        except Exception as exc:
            failures.append(f"ERROR {target_class}.{target_method}: {exc}")

    assert not failures, "Some SootUp cases failed:\n" + "\n".join(failures)
