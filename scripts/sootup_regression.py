#!/usr/bin/env python3
"""Standalone SootUp callgraph regression runner for the mall project.

Usage
-----
# Build classpath automatically from a local mall clone (Maven):
    python scripts/sootup_regression.py \\
        --jar /path/to/sootup-callgraph-pathfinder.jar \\
        --mall-root /path/to/mall

# Provide pre-built classpath files instead of re-running Maven:
    python scripts/sootup_regression.py \\
        --jar /path/to/sootup-callgraph-pathfinder.jar \\
        --admin-cp /path/to/admin-classpath.txt \\
        --portal-cp /path/to/portal-classpath.txt

# Use the SOOTUP_CG_JAR env var (auto-discovered) and classpath files:
    SOOTUP_CG_JAR=/path/to/jar python scripts/sootup_regression.py \\
        --admin-cp /path/to/admin-classpath.txt \\
        --portal-cp /path/to/portal-classpath.txt

Exit code 0 if all cases pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure tatg is importable when running from the repo root
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tatg.analysis.base import AnalysisRequest, AnalysisError  # noqa: E402
from tatg.analysis.classpath import build_classpath_for_project, ClasspathError  # noqa: E402
from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer  # noqa: E402


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

_CASES: list[dict] = [
    # ---- mall-admin -------------------------------------------------------
    {
        "module": "portal",
        "class": "com.macro.mall.portal.service.impl.OmsPromotionServiceImpl",
        "method": "calcCartPromotion",
        "expected_fragments": ["OmsPromotionServiceImpl", "calcCartPromotion"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    {
        "module": "admin",
        "class": "com.macro.mall.service.impl.UmsRoleServiceImpl",
        "method": "allocResource",
        "expected_fragments": ["UmsRoleServiceImpl", "allocResource"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    {
        "module": "admin",
        "class": "com.macro.mall.service.impl.UmsAdminServiceImpl",
        "method": "getResourceList",
        "expected_fragments": ["UmsAdminServiceImpl", "getResourceList"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    {
        "module": "admin",
        "class": "com.macro.mall.service.impl.PmsProductServiceImpl",
        "method": "create",
        "expected_fragments": ["PmsProductServiceImpl", "create"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    {
        "module": "admin",
        "class": "com.macro.mall.service.impl.OmsOrderServiceImpl",
        "method": "delivery",
        "expected_fragments": ["OmsOrderServiceImpl", "delivery"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    {
        "module": "admin",
        "class": "com.macro.mall.service.impl.UmsAdminServiceImpl",
        "method": "login",
        "expected_fragments": ["UmsAdminServiceImpl", "login"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    # ---- mall-portal ------------------------------------------------------
    {
        "module": "portal",
        "class": "com.macro.mall.portal.service.impl.OmsPortalOrderServiceImpl",
        "method": "generateOrder",
        "expected_fragments": ["OmsPortalOrderServiceImpl", "generateOrder"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    {
        "module": "portal",
        "class": "com.macro.mall.portal.service.impl.OmsCartItemServiceImpl",
        "method": "listPromotion",
        "expected_fragments": ["OmsCartItemServiceImpl", "listPromotion"],
        "min_paths": 1,
        "min_indirect": 1,
    },
    {
        "module": "portal",
        "class": "com.macro.mall.portal.service.impl.HomeServiceImpl",
        "method": "content",
        "expected_fragments": ["HomeServiceImpl", "content"],
        "min_paths": 1,
        "min_indirect": 1,
    },
]


# ---------------------------------------------------------------------------
# Classpath helpers
# ---------------------------------------------------------------------------


def _build_cp_from_mall(mall_root: Path, module_name: str) -> str:
    """Run Maven to build classpath for *module_name* inside *mall_root*."""
    print(f"  [classpath] Running mvn dependency:build-classpath for {module_name} …", flush=True)
    return build_classpath_for_project(
        mall_root,
        build_system="maven",
        module=module_name,
        timeout=180,
    )


def _read_cp_file(path: str) -> str:
    """Read a pre-built classpath file (plain text, one classpath line)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Classpath file not found: {p}")
    return p.read_text(encoding="utf-8").strip()


def _augment_with_class_dirs(dep_cp: str, mall_root: Path, module_name: str) -> str:
    """Prepend all compiled class dirs under *mall_root* to *dep_cp*.

    We scan the entire mall root (all sub-modules) because a target class
    may live in a sibling module (e.g. OmsPromotionServiceImpl is in
    mall-service, not mall-admin).  The module_name parameter is kept for
    future filtering but not used for path restriction.
    """
    from tatg.analysis.classpath import _collect_maven_class_dirs  # noqa: PLC0415
    class_dirs = _collect_maven_class_dirs(mall_root, include_test=False)
    if not class_dirs:
        print(
            f"  [warn] No target/classes dirs found under {mall_root}. "
            f"Run: mvn -DskipTests compile (in {mall_root})",
            flush=True,
        )
    entries = class_dirs + [e for e in dep_cp.split(os.pathsep) if e.strip()]
    return os.pathsep.join(entries)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_cases(
    cases: list[dict],
    admin_cp: str,
    portal_cp: str,
    jar_path: str | None,
    output_json: Path | None,
) -> int:
    analyzer = SootUpCallGraphAnalyzer(
        jar_path=jar_path,
        timeout=300,
        use_cache=False,
    )

    results: list[dict] = []
    ok_count = 0
    fail_count = 0

    col_w = 70
    print()
    print(f"{'TARGET':<{col_w}} {'PATHS':>6} {'INDIRECT':>9} {'STATUS':>8}")
    print("-" * (col_w + 28))

    for case in cases:
        cp = admin_cp if case["module"] == "admin" else portal_cp
        label = f"{case['class'].split('.')[-1]}.{case['method']}"
        target_full = f"{case['class']}.{case['method']}"

        request = AnalysisRequest(
            classpath=cp,
            target_class=case["class"],
            target_method=case["method"],
            max_depth=4,
            root_mode="PUBLIC",
            limit_package_roots=False,
        )

        t0 = time.perf_counter()
        error: str | None = None
        path_count = 0
        indirect_count = 0
        fragments_ok = True
        missing_fragments: list[str] = []

        try:
            result = analyzer.analyze(request)
            elapsed = time.perf_counter() - t0
            paths = result.structured.paths if result.structured else []
            path_count = len(paths)
            indirect_count = sum(1 for p in paths if len(p.steps) >= 2)

            raw = result.raw_text or ""
            for frag in case.get("expected_fragments", []):
                if frag not in raw:
                    fragments_ok = False
                    missing_fragments.append(frag)

        except (AnalysisError, Exception) as exc:
            elapsed = time.perf_counter() - t0
            error = str(exc)

        passed = (
            error is None
            and path_count >= case.get("min_paths", 1)
            and indirect_count >= case.get("min_indirect", 0)
            and fragments_ok
        )

        status = "PASS" if passed else "FAIL"
        if passed:
            ok_count += 1
        else:
            fail_count += 1

        row = {
            "target": target_full,
            "module": case["module"],
            "status": status,
            "path_count": path_count,
            "indirect_path_count": indirect_count,
            "fragments_ok": fragments_ok,
            "missing_fragments": missing_fragments,
            "elapsed_s": round(elapsed, 2),
            "error": error,
        }
        results.append(row)

        print(
            f"{label:<{col_w}} {path_count:>6} {indirect_count:>9} {status:>8}"
            + (f"  ← {error[:60]!r}" if error else "")
            + (f"  ← missing: {missing_fragments}" if missing_fragments else "")
        )

    print("-" * (col_w + 28))
    total = ok_count + fail_count
    print(f"Result: {ok_count}/{total} passed, {fail_count} failed")
    print()

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"JSON results written to: {output_json}")

    return 0 if fail_count == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SootUp callgraph regression runner for the mall project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--jar",
        metavar="PATH",
        default=os.environ.get("SOOTUP_CG_JAR", ""),
        help="Path to sootup-callgraph-pathfinder fat-JAR. "
             "Defaults to $SOOTUP_CG_JAR env var.",
    )

    cp_group = p.add_argument_group("Classpath source (pick one)")
    cp_group.add_argument(
        "--mall-root",
        metavar="DIR",
        help="Path to a local mall clone. Maven will run to build classpath.",
    )
    cp_group.add_argument(
        "--admin-cp",
        metavar="FILE",
        default=os.environ.get("SOOTUP_MALL_ADMIN_CP_FILE", ""),
        help="Pre-built classpath file for mall-admin. "
             "Defaults to $SOOTUP_MALL_ADMIN_CP_FILE env var.",
    )
    cp_group.add_argument(
        "--portal-cp",
        metavar="FILE",
        default=os.environ.get("SOOTUP_MALL_PORTAL_CP_FILE", ""),
        help="Pre-built classpath file for mall-portal. "
             "Defaults to $SOOTUP_MALL_PORTAL_CP_FILE env var.",
    )

    p.add_argument(
        "--output",
        metavar="JSON",
        default="",
        help="Optional path for JSON results output.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Resolve classpath
    admin_cp: str
    portal_cp: str

    if args.mall_root:
        mall_root = Path(args.mall_root).resolve()
        if args.admin_cp and args.portal_cp:
            # Pre-built dep classpath + class dirs from mall-root
            print(f"Using pre-built classpath files, augmenting with class dirs from: {mall_root}")
            try:
                raw_admin = _read_cp_file(args.admin_cp)
                raw_portal = _read_cp_file(args.portal_cp)
                admin_cp = _augment_with_class_dirs(raw_admin, mall_root, "mall-admin")
                portal_cp = _augment_with_class_dirs(raw_portal, mall_root, "mall-portal")
            except FileNotFoundError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
        else:
            # Full Maven build of classpath
            print(f"Building classpath from mall root: {mall_root}")
            try:
                admin_cp = _build_cp_from_mall(mall_root, "mall-admin")
                portal_cp = _build_cp_from_mall(mall_root, "mall-portal")
            except ClasspathError as exc:
                print(f"ERROR: Failed to build classpath: {exc}", file=sys.stderr)
                return 1
    elif args.admin_cp and args.portal_cp:
        try:
            admin_cp = _read_cp_file(args.admin_cp)
            portal_cp = _read_cp_file(args.portal_cp)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    else:
        print(
            "ERROR: Provide either --mall-root or both --admin-cp and --portal-cp.",
            file=sys.stderr,
        )
        return 1

    jar: str | None = args.jar or None

    output_json = Path(args.output) if args.output else None

    return run_cases(_CASES, admin_cp, portal_cp, jar, output_json)


if __name__ == "__main__":
    sys.exit(main())
