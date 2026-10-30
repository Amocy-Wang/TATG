"""Type-Aware context extraction for richer LLM test generation prompts.

Architecture
------------
This sub-package bridges the gap between what the LLM *knows* (Java source
code) and what it *needs* to generate high-coverage tests for complex Spring
services with many domain-type dependencies::

    source_extractor  – scan .java source files → TypeShape / AutowiredDep
    mock_synthesizer  – TypeShape + AutowiredDep → MockScaffold (Mockito code)
    models            – shared dataclass definitions

Typical usage inside an engine adapter::

    from tatg.analysis.typeinfo import build_type_aware_context

    ctx = build_type_aware_context(source_file, project_root)
    # ctx.mock_scaffold  → ready-to-embed Mockito scaffold string
    # ctx.type_shapes    → dict[str, TypeShape] for all referenced types
    # ctx.branch_hints   → list of discovered if/guard branches
"""

from tatg.analysis.typeinfo.models import (
    AnnotationInfo,
    AutowiredDep,
    BranchHint,
    ConstructorInfo,
    ExecutionCapsule,
    ExecutionRecipe,
    FieldInfo,
    MethodInfo,
    MockScaffold,
    TypeAwareContext,
    TypeShape,
)
from tatg.analysis.typeinfo.source_extractor import (
    JavaSourceExtractor,
    build_type_index,
    extract_autowired_deps,
    extract_branch_hints,
    extract_type_shape,
)
from tatg.analysis.typeinfo.mock_synthesizer import (
    MockScaffoldBuilder,
    build_mock_scaffold,
)
from tatg.analysis.typeinfo.recipe_synthesizer import (
    infer_capsule_required_call_order,
    render_execution_capsules_section,
    render_execution_recipes_section,
    synthesize_execution_recipes,
)

__all__ = [
    # models
    "AnnotationInfo",
    "AutowiredDep",
    "BranchHint",
    "ConstructorInfo",
    "ExecutionCapsule",
    "ExecutionRecipe",
    "FieldInfo",
    "MethodInfo",
    "MockScaffold",
    "TypeAwareContext",
    "TypeShape",
    # extraction
    "JavaSourceExtractor",
    "build_type_index",
    "extract_autowired_deps",
    "extract_branch_hints",
    "extract_type_shape",
    # synthesis
    "MockScaffoldBuilder",
    "build_mock_scaffold",
    "synthesize_execution_recipes",
    "infer_capsule_required_call_order",
    "render_execution_capsules_section",
    "render_execution_recipes_section",
]


def build_type_aware_context(
    source_file,
    project_root,
    *,
    method_name=None,
    class_name=None,
    max_scan_files: int = 500,
) -> "TypeAwareContext":
    """High-level convenience entry point used by engine adapters.

    Scans the project source tree for all domain-type definitions referenced by
    the given source file and synthesises a :class:`TypeAwareContext` containing
    the mock scaffold and type-shape information ready for prompt injection.

    Parameters
    ----------
    source_file:
        The ``.java`` file for the class under test.
    project_root:
        Root directory of the Maven/Gradle project.
    method_name:
        If set, branch analysis is focused on this specific method; otherwise
        the entire class is analysed.
    max_scan_files:
        Upper bound on ``.java`` files scanned when building the type index.
        Increase for very large monorepos; decrease for speed.

    Returns
    -------
    TypeAwareContext
        Ready-to-use context object.
    """
    from pathlib import Path

    source_file = Path(source_file)
    project_root = Path(project_root)

    extractor = JavaSourceExtractor(project_root, max_scan_files=max_scan_files)

    # Parse the target class
    target_shape = extract_type_shape(source_file, class_name=class_name)
    autowired = extract_autowired_deps(source_file, class_name=class_name)
    branches = extract_branch_hints(
        source_file,
        method_name=method_name,
        class_name=class_name,
    )

    # Collect all type names we need to resolve
    referenced_types: set[str] = set()
    for dep in autowired:
        referenced_types.add(dep.type_name)
    if target_shape:
        for f in target_shape.fields:
            referenced_types.add(_simple_type(f.java_type))
        # Method-scope prompts should not recursively expand helper types that
        # are mentioned only by unrelated methods on the same owner. Keep the
        # complete target shape for callable/state recipes, but bound external
        # type resolution to the selected method's signatures.
        signature_methods = (
            [
                method
                for method in target_shape.methods
                if method.name == method_name
            ]
            if method_name
            else target_shape.methods
        )
        for m in signature_methods:
            referenced_types.add(_simple_type(m.return_type))
            for ptype, _ in m.params:
                referenced_types.add(_simple_type(ptype))

    # Build type index for all referenced types
    type_index = extractor.build_index_for(referenced_types)
    if target_shape:
        type_index[target_shape.class_name] = target_shape

    # Build mock scaffold
    scaffold = build_mock_scaffold(
        target_shape=target_shape,
        autowired_deps=autowired,
        type_index=type_index,
    )

    # Walk the superclass chain and collect likely configuration methods from
    # ancestors.  This is intentionally generic: we do not rely on specific
    # prefixes (like withX/setX), and instead score methods by signature/role.
    ancestor_setup_methods: list[str] = []
    seen_setup_signatures: set[str] = set()
    if target_shape:
        current = target_shape
        for _ in range(4):  # max 4 levels up
            sc = current.superclass
            if not sc:
                break
            import re as _re
            sc_name = _re.sub(r"<.*>", "", sc).strip().rsplit(".", 1)[-1]
            ancestor = type_index.get(sc_name)
            if ancestor is None:
                # Try to scan for it now (parent class may not have been in referenced_types)
                ancestor_files = list(source_file.parent.parent.rglob(sc_name + ".java"))
                if not ancestor_files:
                    ancestor_files = list(project_root.rglob(sc_name + ".java"))
                if ancestor_files:
                    ancestor = extract_type_shape(ancestor_files[0])
                    if ancestor:
                        type_index[sc_name] = ancestor
            if not ancestor:
                break
            # Collect likely pre-target configuration methods.
            for m in ancestor.methods:
                if m.visibility != "public":
                    continue
                if not _is_likely_setup_method(
                    method_name=m.name,
                    params=m.params,
                    return_type=m.return_type,
                    target_class=(target_shape.class_name if target_shape else ""),
                    owner_class=ancestor.class_name,
                ):
                    continue
                param_str = ", ".join(f"{pt} {pn}" for pt, pn in m.params) if m.params else ""
                sig = f"{ancestor.class_name}.{m.name}({param_str}) -> {m.return_type}"
                if sig in seen_setup_signatures:
                    continue
                seen_setup_signatures.add(sig)
                ancestor_setup_methods.append(sig)
                referenced_types.add(_simple_type(m.return_type))
                for ptype, _pname in m.params:
                    referenced_types.add(_simple_type(ptype))
            current = ancestor

        extra_referenced = {
            name for name in referenced_types
            if name and name not in type_index and not _is_jdk_or_primitive_type(name)
        }
        if extra_referenced:
            type_index.update(extractor.build_index_for(extra_referenced))

    execution_recipes = synthesize_execution_recipes(
        source_file=source_file,
        target_shape=target_shape,
        type_index=type_index,
        method_name=method_name,
        class_name=class_name,
        ancestor_setup_methods=ancestor_setup_methods,
        project_root=project_root,
    )

    return TypeAwareContext(
        target_class=target_shape.class_name if target_shape else source_file.stem,
        type_shapes=type_index,
        autowired_deps=autowired,
        mock_scaffold=scaffold,
        branch_hints=branches,
        execution_recipes=execution_recipes,
        ancestor_setup_methods=ancestor_setup_methods,
    )


def _simple_type(java_type: str) -> str:
    """Strip generics and array markers to get the base type name."""
    import re
    # Handle: List<OmsOrderItem> → OmsOrderItem
    # But also capture the outer: List<...> → List
    base = re.sub(r"<.*>", "", java_type).strip()
    base = base.replace("[]", "").strip()
    # Drop qualified prefix: com.acme.Foo → Foo
    return base.rsplit(".", 1)[-1] if "." in base else base


def _is_jdk_or_primitive_type(type_name: str) -> bool:
    simple = _simple_type(type_name)
    return simple in {
        "", "void", "Void", "boolean", "Boolean", "byte", "Byte", "short", "Short",
        "int", "Integer", "long", "Long", "float", "Float", "double", "Double",
        "char", "Character", "String", "Object", "Class", "Enum", "List", "Set",
        "Map", "Collection", "Iterable", "Iterator", "Optional", "BigDecimal", "BigInteger",
    }


def _is_likely_setup_method(
    *,
    method_name: str,
    params: list[tuple[str, str]],
    return_type: str,
    target_class: str,
    owner_class: str,
) -> bool:
    """Heuristic for selecting ancestor methods that likely configure state.

    Generic scoring avoids hard-coding specific prefixes such as withX/setX.
    We prefer methods that:
    - mutate/configure behavior (name semantics),
    - consume configuration inputs (has parameters),
    - or participate in fluent chaining (returns self-like type).
    """
    import re

    name = (method_name or "").strip()
    if not name:
        return False

    lower = name.lower()
    # Exclude non-configuration utility/query methods.
    if lower in {
        "equals", "hashcode", "tostring", "clone", "compareto",
        "iterator", "stream", "spliterator", "getclass", "wait",
        "notify", "notifyall", "finalize",
    }:
        return False
    if lower.startswith(("get", "is", "has", "can", "should", "to", "as")):
        return False
    if re.match(
        r"^(compute|calc|calculate|eval|evaluate|increment|decrement|trigger|"
        r"run|execute|optimize|solve|process|parse|generate|read|write|next|"
        r"advance|handle|check)",
        lower,
    ):
        return False

    score = 0
    if params:
        score += 2

    rt_simple = _simple_type(return_type)
    if rt_simple in {owner_class, target_class, "void", "OPTIM", "SELF", "THIS", "T"}:
        score += 2
    elif re.match(r"^[A-Z][A-Z0-9_]*$", rt_simple or ""):
        # Generic type vars such as PAIR/CFG often appear in fluent bases.
        score += 1

    # Broad action verbs commonly used by configuration APIs.
    if re.match(
        r"^(set|with|add|put|enable|disable|configure|register|attach|use|apply|"
        r"setup|init|initialize|start|open|load|bind|assign|target|weight)",
        lower,
    ):
        score += 2

    # Need at least moderate confidence, and at least one of:
    # - explicit configuration verb semantics, or
    # - fluent/self-like chaining signature.
    has_config_verb = bool(
        re.match(
            r"^(set|with|add|put|enable|disable|configure|register|attach|use|apply|"
            r"setup|init|initialize|start|open|load|bind|assign|target|weight)",
            lower,
        )
    )
    has_fluent_shape = rt_simple in {owner_class, target_class, "OPTIM", "SELF", "THIS", "T"}
    return score >= 2 and (has_config_verb or has_fluent_shape)
