"""Dataclass models for the type-aware analysis pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class AnnotationInfo:
    """A Java annotation with its attributes."""
    name: str                              # e.g. "NotNull", "Min"
    attributes: dict[str, str] = field(default_factory=dict)  # e.g. {"value": "0"}

    def __str__(self) -> str:
        if not self.attributes:
            return f"@{self.name}"
        attrs = ", ".join(f"{k}={v}" for k, v in self.attributes.items())
        return f"@{self.name}({attrs})"


@dataclass
class FieldInfo:
    """A single field in a Java class."""
    name: str
    java_type: str                         # e.g. "BigDecimal", "List<OmsOrderItem>"
    annotations: list[AnnotationInfo] = field(default_factory=list)
    is_nullable: bool = True               # false if @NotNull / @NonNull present
    visibility: str = "private"            # "public" | "protected" | "private"

    @property
    def is_collection(self) -> bool:
        return any(c in self.java_type for c in ("List<", "Set<", "Map<", "Collection<"))

    @property
    def base_type(self) -> str:
        """Strip generics: List<E> → E; Map<K,V> → V (value type).

        For non-generic types, returns the type name with ``[]`` stripped.
        """
        inner = re.search(r"<(?:[^,>]+,\s*)?([^,>]+)>", self.java_type)
        if inner:
            return inner.group(1).strip()
        return self.java_type.replace("[]", "").strip()

    def sample_value(self) -> str:
        """Best-effort literal Java value for use in fixture helpers."""
        t = self.java_type.replace("[]", "").strip()
        if t in ("Long", "long"):
            return "1L"
        if t in ("Integer", "int"):
            return "1"
        if t in ("String",):
            return f'"{self.name}"'
        if t in ("BigDecimal",):
            return 'new BigDecimal("1.00")'
        if t in ("Boolean", "boolean"):
            return "true"
        if t in ("Double", "double"):
            return "1.0"
        if t in ("Date",):
            return "new Date()"
        if t.startswith("List"):
            return "new ArrayList<>()"
        if t.startswith("Map"):
            return "new HashMap<>()"
        # Domain object — use new + no-arg constructor
        return f"new {t}()"


@dataclass
class ConstructorInfo:
    """A constructor signature."""
    params: list[tuple[str, str]] = field(default_factory=list)   # [(type, name)]
    visibility: str = "public"


@dataclass
class MethodInfo:
    """A method signature (setter or factory)."""
    name: str
    return_type: str
    params: list[tuple[str, str]] = field(default_factory=list)
    annotations: list[AnnotationInfo] = field(default_factory=list)
    is_setter: bool = False
    visibility: str = "public"
    is_static: bool = False


@dataclass
class TypeShape:
    """Complete structural description of a Java class or interface."""
    class_name: str                        # simple name, e.g. "OmsOrderItem"
    fqn: str                               # FQN, e.g. "com.macro.mall.model.OmsOrderItem"
    package: str                           # e.g. "com.macro.mall.model"
    fields: list[FieldInfo] = field(default_factory=list)
    constructors: list[ConstructorInfo] = field(default_factory=list)
    methods: list[MethodInfo] = field(default_factory=list)
    interfaces: list[str] = field(default_factory=list)
    superclass: str | None = None
    is_interface: bool = False
    is_abstract: bool = False
    annotations: list[AnnotationInfo] = field(default_factory=list)
    source_file: str | None = None
    # public static final constants — each value may need its own test branch
    static_constants: list[str] = field(default_factory=list)  # "TYPE NAME"

    @property
    def setters(self) -> list[MethodInfo]:
        return [m for m in self.methods if m.is_setter]

    @property
    def has_no_arg_constructor(self) -> bool:
        return not self.constructors or any(
            not c.params and c.visibility != "private" for c in self.constructors
        )

    def fixture_helper(self, var_name: str | None = None) -> str:
        """Generate a Java fixture-helper method for this type.

        Returns a Java method string that creates a populated instance.
        Setter-style population is used when there are no non-trivial
        constructors.
        """
        name = var_name or f"build{self.class_name}"
        lines = [
            f"private {self.class_name} {name}() {{",
            f"    var obj = new {self.class_name}();",
        ]
        for f in self.fields:
            if f.visibility != "public" and not any(
                m.is_setter and m.name == f"set{f.name[0].upper()}{f.name[1:]}"
                for m in self.methods
            ):
                continue  # skip fields with no visible setter
            setter_name = f"set{f.name[0].upper()}{f.name[1:]}"
            lines.append(f"    obj.{setter_name}({f.sample_value()});")
        lines += ["    return obj;", "}"]
        return "\n".join(lines)


@dataclass
class AutowiredDep:
    """A Spring @Autowired (or @Inject) dependency field."""
    field_name: str
    type_name: str                         # simple class/interface name
    fqn: str | None = None                 # FQN if resolvable


@dataclass
class BranchHint:
    """A detected branching condition in a method body."""
    method_name: str
    condition: str                         # raw condition string
    hint: str                              # human-readable test hint
    line_number: int = 0
    setup_hints: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        loc = f" (line {self.line_number})" if self.line_number else ""
        return f"[{self.method_name}{loc}] {self.hint}"


@dataclass
class ExecutionRecipe:
    """A generic, type-derived recipe for reaching target behavior.

    Recipes are intentionally advisory. They describe state requirements,
    object construction routes, path knobs, and API sequencing without binding
    generation to one specific project or benchmark class.
    """
    method_name: str
    category: str
    title: str
    observations: list[str] = field(default_factory=list)
    setup_steps: list[str] = field(default_factory=list)
    path_goals: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    receipt: dict[str, object] = field(default_factory=dict)


@dataclass
class ExecutionCapsule:
    """A compact skeleton plan compiled from execution recipes.

    Capsules are used only for high-risk targets where free-form generation is
    likely to miss resource/state/numeric setup.  They keep TATG's existing
    recipe flow intact while giving the LLM a stricter Arrange/Act/Assert
    scaffold to fill in.
    """
    method_name: str
    strategy: str
    confidence: float
    dependency_context: list[str] = field(default_factory=list)
    skeleton_steps: list[str] = field(default_factory=list)
    holes: list[str] = field(default_factory=list)
    preserve: list[str] = field(default_factory=list)


@dataclass
class MockScaffold:
    """Generated JUnit 5 + Mockito 5 scaffold for a Spring @Service class."""
    target_class: str
    mock_declarations: list[str] = field(default_factory=list)
    # "@Mock UmsMemberService memberService;"
    inject_mocks_declaration: str = ""
    # "@InjectMocks OmsPortalOrderServiceImpl service;"
    fixture_helpers: list[str] = field(default_factory=list)
    # full Java fixture method strings
    suggested_when_stubs: list[str] = field(default_factory=list)
    # "when(memberService.getCurrentMember()).thenReturn(buildUmsMember());"
    import_hints: list[str] = field(default_factory=list)
    # additional imports that should appear in the test file


@dataclass
class TypeAwareContext:
    """Aggregated context for enriching an LLM test-generation prompt."""
    target_class: str
    type_shapes: dict[str, TypeShape] = field(default_factory=dict)
    autowired_deps: list[AutowiredDep] = field(default_factory=list)
    mock_scaffold: MockScaffold | None = None
    branch_hints: list[BranchHint] = field(default_factory=list)
    execution_recipes: list[ExecutionRecipe] = field(default_factory=list)
    # Builder/setter methods inherited from ancestor classes that are needed
    # to set up the class under test before calling doOptimize()-style methods.
    # Each entry is a string like "withTarget(double[])" or "withConvergenceChecker(ConvergenceChecker)".
    ancestor_setup_methods: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return (
            not self.type_shapes
            and not self.autowired_deps
            and self.mock_scaffold is None
            and not self.branch_hints
            and not self.execution_recipes
        )
