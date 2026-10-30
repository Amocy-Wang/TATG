"""Context enrichment helpers for LLM prompts.

This module provides the glue between the static-analysis layer
(:mod:`tatg.analysis`) and the test-generation engines.

A single public function — :func:`build_static_analysis_context` — is the only
entry-point the engine adapters need.  It handles all three resolution
strategies in priority order:

1. **Pre-computed file** — if ``request.static_analysis_json`` is set and the
   file exists, its content is used directly (fast, offline).
2. **Live analysis** — if ``request.cat_engine_root`` points to a classpath
   and a supported analyzer is registered, the analysis is run on-the-fly.
3. **Graceful skip** — if neither strategy produces a result, the function
   returns an empty string and the prompt is built without extra context.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from tatg.models import GenerationRequest, Target

log = logging.getLogger(__name__)


def build_static_analysis_context(
    request: GenerationRequest,
    target: Target,
    source_file: Path,
) -> str:
    """Return a static-analysis context string suitable for LLM prompt injection.

    Args:
        request:     The current generation request (carries all config).
        target:      The :class:`~tatg.models.Target` being tested.
        source_file: Resolved path of the Java source file.

    Returns:
        A plain-text context string, or an empty string if no analysis is
        available.
    """
    # --- Strategy 1: pre-computed file (static_analysis_json field) ---
    if request.static_analysis_json:
        ctx = _load_precomputed(request.static_analysis_json)
        if ctx:
            return ctx

    # --- Strategy 2: look for a pre-computed file next to the source ---
    if target.class_name and target.method_name:
        ctx = _probe_adjacent_cg_file(source_file, target.class_name, target.method_name)
        if ctx:
            return ctx

    # --- Strategy 3: live analysis via the registry ---
    if request.cat_engine_root:
        ctx = _run_live_analysis(request, target, source_file)
        if ctx:
            return ctx

    return ""


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _load_precomputed(path: Path) -> str:
    """Read a pre-computed analysis file and return its content."""
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        log.debug("Could not read pre-computed analysis file %s: %s", path, exc)
    return ""


def _probe_adjacent_cg_file(
    source_file: Path,
    class_name: str,
    method_name: str,
) -> str:
    """Search for a call-graph file produced by the Java tool in well-known locations.

    The Java tool writes files with a naming convention like::

        <ProjectId>_<ClassName>_<methodName>_cg_withdep.txt

    We look in the ``cg_info_defects4j/`` sibling directory of the source
    file's project root, and in a ``cg_info/`` directory next to the source.
    """
    short_class = class_name.split(".")[-1]
    candidates = [
        f"*{short_class}_{method_name}_cg_withdep.txt",
        f"*{short_class}_{method_name}_cg*.txt",
    ]

    search_dirs: list[Path] = []
    # Conventional sibling directory for defects4j-style projects
    for parent in source_file.parents:
        for dirname in ("cg_info_defects4j", "cg_info", "callgraph"):
            d = parent / dirname
            if d.is_dir():
                search_dirs.append(d)
            if len(search_dirs) >= 3:
                break
        if len(search_dirs) >= 3:
            break

    for search_dir in search_dirs:
        for pattern in candidates:
            matches = sorted(search_dir.glob(pattern))
            if matches:
                try:
                    return matches[0].read_text(encoding="utf-8").strip()
                except Exception as exc:
                    log.debug("Could not read adjacent cg file %s: %s", matches[0], exc)
    return ""


def _run_live_analysis(
    request: GenerationRequest,
    target: Target,
    source_file: Path,
) -> str:
    """Attempt a live call-graph analysis using the registered SootUp analyzer."""
    try:
        from tatg.analysis import AnalysisRequest, get_registry
        from tatg.analysis.base import AnalysisError

        class_name = target.class_name
        method_name = target.method_name
        if not class_name or not method_name:
            return ""

        classpath, dep_dir = _resolve_live_analysis_classpath(request)

        if not classpath:
            return ""

        registry = get_registry()
        if not registry.has("sootup-callgraph"):
            return ""

        started = time.perf_counter()
        budget_seconds = _env_positive_int("TATG_STATIC_ANALYSIS_SOOTUP_BUDGET_SECONDS", 60)
        per_run_timeout = _env_positive_int("SOOTUP_CG_TIMEOUT_SECONDS", 30)
        best_result: Any | None = None
        for depth in _sootup_depth_schedule():
            remaining = budget_seconds - (time.perf_counter() - started)
            if remaining <= 0.5:
                break
            ar = AnalysisRequest(
                classpath=classpath,
                target_class=class_name,
                target_method=method_name,
                max_depth=depth,
                dep_dir=dep_dir,
                root_mode="PUBLIC",
                limit_package_roots=True,
                max_paths=_env_positive_int("SOOTUP_CG_MAX_PATHS", 24),
                max_search_states=_env_positive_int("SOOTUP_CG_MAX_SEARCH_STATES", 50000),
                threads=max(1, int(os.environ.get("SOOTUP_CG_THREADS", "1") or "1")),
                extra={"timeout_seconds": max(1.0, min(float(per_run_timeout), remaining))},
            )
            try:
                result = registry.run("sootup-callgraph", ar)
            except AnalysisError as exc:
                log.debug("Live call-graph analysis depth %s skipped: %s", depth, exc)
                continue
            best_result = result
            structured = getattr(result, "structured", None)
            paths = list(getattr(structured, "paths", []) or [])
            if paths:
                return _render_live_analysis_context(result)

        if best_result is not None:
            return _render_live_analysis_context(best_result)
        return ""

    except Exception as exc:
        log.debug("Live call-graph analysis skipped: %s", exc)
        return ""


def _render_live_analysis_context(result: Any) -> str:
    """Render SootUp output as a bounded TATG prompt context.

    The Java tool can emit very large raw text.  TATG only needs the shortest
    source-visible entry paths plus construction options for the object types
    on those paths, so keep the raw result in analyzer metadata/cache and inject
    a compact structured view into the prompt.
    """
    structured = getattr(result, "structured", None)
    paths = list(getattr(structured, "paths", []) or [])
    initializations = list(getattr(structured, "initializations", []) or [])
    if not structured or (not paths and not initializations):
        return result.as_prompt_context() if hasattr(result, "as_prompt_context") else ""
    target_class = str(getattr(structured, "target_class", "") or "")
    target_method = str(getattr(structured, "target_method", "") or "")
    if target_method:
        paths = [path for path in paths if _path_ends_at_target(path, target_class, target_method)]

    max_paths = _env_positive_int("TATG_STATIC_ANALYSIS_PROMPT_PATHS", 6)
    max_init_types = _env_positive_int("TATG_STATIC_ANALYSIS_PROMPT_INIT_TYPES", 12)
    max_ctors = _env_positive_int("TATG_STATIC_ANALYSIS_PROMPT_CTORS_PER_TYPE", 3)
    max_factories = _env_positive_int("TATG_STATIC_ANALYSIS_PROMPT_FACTORIES_PER_TYPE", 3)
    max_impls = _env_positive_int("TATG_STATIC_ANALYSIS_PROMPT_IMPLS_PER_TYPE", 3)

    # Prefer shorter paths whose entry step is public (direct call chains).
    # Tie-break by label so output is deterministic.
    def _path_sort_key(p: Any) -> tuple[int, int, int, str]:
        steps = list(getattr(p, "steps", []) or [])
        entry_vis = str(getattr(steps[0], "visibility", "") or "") if steps else ""
        entry_class = str(getattr(steps[0], "class_fqn", "") or "") if steps else ""
        target_entry_score = 0 if target_class and entry_class == target_class else 1
        # Public entry = 0, protected = 1, other = 2 (lower is better)
        vis_score = 0 if entry_vis.lower() == "public" else (1 if "protected" in entry_vis.lower() else 2)
        return (target_entry_score, len(steps), vis_score, _path_label(p))

    selected_paths = sorted(paths, key=_path_sort_key)[:max_paths]
    relevant_classes = _classes_relevant_to_paths(selected_paths)
    selected_inits = _select_relevant_initializations(initializations, relevant_classes, max_init_types)

    analyzer = getattr(result, "analyzer_name", "static-analysis")
    target = getattr(result, "target", "")
    lines: list[str] = [
        f"[Static analysis: {analyzer} for {target}]",
        "This is a compact summary of SootUp/CAT output; raw tool text is intentionally not injected.",
        f"Paths shown: {len(selected_paths)}/{len(paths)}. Initialization types shown: {len(selected_inits)}/{len(initializations)}.",
        "Use entry paths as preferred source-visible calls for private targets; construct only the receiver and parameters needed by the chosen path.",
    ]

    if selected_paths:
        lines.append("[Reachable Entry Paths]")
        for path in selected_paths:
            lines.append("Path:")
            for step in getattr(path, "steps", []) or []:
                lines.append(f"  {_method_label(step)}")
            lines.append("---")

    if selected_inits:
        lines.append("[Construction Options For Path Types]")
        lines.append(
            "// Only non-standard types are listed; JDK standard types (java.*, javax.*) "
            "are omitted because the LLM already knows how to construct them."
        )
        for init in selected_inits:
            class_fqn = str(getattr(init, "class_fqn", "") or "")
            if not class_fqn:
                continue
            # Skip standard JDK types — the LLM already knows new ArrayList<>(), etc.
            if class_fqn.startswith(("java.", "javax.")):
                continue
            modifiers = str(getattr(init, "class_modifiers", "") or "").strip()
            source = str(getattr(init, "source_jar", "") or "").strip()
            suffix = f" ({modifiers})" if modifiers else ""
            if source and source not in ("UNKNOWN", "JDK_RUNTIME"):
                suffix += f" from {source}"
            lines.append(f"- {class_fqn}{suffix}")

            ctors = list(getattr(init, "constructors", []) or [])[:max_ctors]
            if ctors:
                ctor_labels = []
                simple = class_fqn.split(".")[-1]
                for ctor in ctors:
                    params = ", ".join(str(p) for p in (getattr(ctor, "params", []) or []))
                    ctor_labels.append(f"new {simple}({params})")
                lines.append(f"  constructors: {'; '.join(ctor_labels)}")

            factories = [str(f) for f in (getattr(init, "factory_methods", []) or [])[:max_factories]]
            if factories:
                lines.append(f"  factories: {'; '.join(factories)}")

            impls = [str(i) for i in (getattr(init, "known_implementations", []) or [])[:max_impls]]
            if impls:
                lines.append(f"  implementations: {'; '.join(impls)}")

    return "\n".join(lines).strip()


def _path_ends_at_target(path: Any, target_class: str, target_method: str) -> bool:
    steps = list(getattr(path, "steps", []) or [])
    if not steps:
        return False
    last = steps[-1]
    method_name = str(getattr(last, "method_name", "") or "")
    class_fqn = str(getattr(last, "class_fqn", "") or "")
    if method_name != target_method:
        return False
    if not target_class:
        return True
    return class_fqn == target_class or class_fqn.endswith("." + target_class) or target_class.endswith("." + class_fqn)


def _env_positive_int(env_name: str, default: int) -> int:
    raw = str(os.environ.get(env_name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _sootup_depth_schedule() -> list[int]:
    raw_schedule = str(os.environ.get("SOOTUP_CG_DEPTH_SCHEDULE", "") or "").strip()
    if raw_schedule:
        depths: list[int] = []
        for part in raw_schedule.split(","):
            try:
                depth = int(part.strip())
            except ValueError:
                continue
            if depth > 0 and depth not in depths:
                depths.append(depth)
        if depths:
            return depths

    max_depth = _env_positive_int(
        "SOOTUP_CG_MAX_DEPTH",
        _env_positive_int("TATG_STATIC_ANALYSIS_MAX_DEPTH", 4),
    )
    start_depth = min(max_depth, _env_positive_int("SOOTUP_CG_START_DEPTH", 2))
    return list(range(start_depth, max_depth + 1)) or [max_depth]


def _path_label(path: Any) -> str:
    steps = list(getattr(path, "steps", []) or [])
    return " -> ".join(_method_label(step) for step in steps)


def _method_label(step: Any) -> str:
    label = str(getattr(step, "label", "") or "")
    if label:
        return label
    class_fqn = str(getattr(step, "class_fqn", "") or "")
    method_name = str(getattr(step, "method_name", "") or "")
    params = ", ".join(str(p) for p in (getattr(step, "params", []) or []))
    return_type = str(getattr(step, "return_type", "") or "void")
    visibility = str(getattr(step, "visibility", "") or "")
    vis = f" ({visibility})" if visibility else ""
    return f"{class_fqn}#{method_name}({params}): {return_type}{vis}"


def _classes_relevant_to_paths(paths: list[Any]) -> set[str]:
    classes: set[str] = set()
    for path in paths:
        for step in getattr(path, "steps", []) or []:
            class_fqn = str(getattr(step, "class_fqn", "") or "")
            if class_fqn:
                classes.add(class_fqn)
            for type_name in list(getattr(step, "params", []) or []) + [getattr(step, "return_type", "")]:
                normalized = _normalize_reference_type(str(type_name or ""))
                if normalized:
                    classes.add(normalized)
    return classes


def _normalize_reference_type(type_name: str) -> str:
    text = type_name.strip()
    if not text:
        return ""
    text = text.replace("[]", "")
    text = text.split("<", 1)[0].strip()
    text = text.replace("...", "")
    primitive_or_common = {
        "void", "boolean", "byte", "short", "char", "int", "long", "float", "double",
        "java.lang.String", "String", "java.lang.Object", "Object",
    }
    if text in primitive_or_common:
        return ""
    return text if "." in text else ""


_INTERNAL_JDK_PREFIXES = (
    "sun.",
    "jdk.internal.",
    "jdk.vm.",
    "jdk.jshell.",
    "com.sun.",
    "sunw.",
)


def _is_internal_jdk_type(class_fqn: str) -> bool:
    """Return True for JDK-internal types that are not useful for test construction.

    Keeps ``java.*`` and ``javax.*`` since those are standard API types that
    tests legitimately construct (e.g. ``java.util.ArrayList``).  Filters the
    truly internal packages (``sun.*``, ``jdk.internal.*``, ``com.sun.*``, etc.)
    that only appear because SootUp traversed VM internals.
    """
    return any(class_fqn.startswith(pfx) for pfx in _INTERNAL_JDK_PREFIXES)


def _select_relevant_initializations(
    initializations: list[Any],
    relevant_classes: set[str],
    limit: int,
) -> list[Any]:
    if not initializations or limit <= 0:
        return []
    exact: list[Any] = []
    rest: list[Any] = []
    relevant_simple = {name.split(".")[-1] for name in relevant_classes}
    for init in initializations:
        class_fqn = str(getattr(init, "class_fqn", "") or "")
        if _is_internal_jdk_type(class_fqn):
            continue
        simple = class_fqn.split(".")[-1]
        if class_fqn in relevant_classes or simple in relevant_simple:
            exact.append(init)
        else:
            rest.append(init)
    return (exact + rest)[:limit]


def _optional_positive_int(env_name: str) -> int | None:
    raw = str(os.environ.get(env_name, "") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_live_analysis_classpath(request: GenerationRequest) -> tuple[str, str | None]:
    override = str(os.environ.get("TATG_STATIC_ANALYSIS_CLASSPATH", "") or "").strip()
    if override:
        return override, _discover_dependency_dir(Path(override.split(os.pathsep)[0]))

    try:
        from tatg.analysis.classpath import build_classpath_for_project

        timeout = max(1, int(os.environ.get("TATG_STATIC_ANALYSIS_CLASSPATH_TIMEOUT", "90") or "90"))
        module_text = ""
        module_path = getattr(request, "module_path", None)
        if module_path:
            try:
                module_text = str(Path(module_path).resolve().relative_to(Path(request.project_root).resolve()))
            except Exception:
                module_text = str(module_path)
        classpath = build_classpath_for_project(
            request.project_root,
            module=module_text or None,
            timeout=timeout,
            include_test_classes=False,
        )
        if classpath:
            return classpath, None
    except Exception as exc:  # noqa: BLE001
        log.debug("Live analysis classpath extraction skipped: %s", exc)

    fallback_root = Path(getattr(request, "cat_engine_root", None) or getattr(request, "project_root", "") or "")
    if not str(fallback_root):
        return "", None
    return str(fallback_root), _discover_dependency_dir(fallback_root)


def _discover_dependency_dir(root: Path) -> str | None:
    try:
        if not root.is_dir():
            return None
        for dep_name in ("lib", "libs", "dependency", "dependencies"):
            candidate = root / dep_name
            if candidate.is_dir():
                return str(candidate)
    except Exception:
        return None
    return None
