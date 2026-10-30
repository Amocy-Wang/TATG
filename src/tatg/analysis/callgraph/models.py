"""Data models for call-graph analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MethodSig:
    """A single method signature as emitted by the call-graph tool.

    Example raw text::

        org.apache.commons.cli.HelpFormatter#printWrapped(java.io.PrintWriter,int,java.lang.String): void (public)
    """

    class_fqn: str
    method_name: str
    params: list[str]
    return_type: str
    visibility: str  # "public" | "protected" | "private" | ""
    source_jar: str | None = None

    @property
    def label(self) -> str:
        params_str = ", ".join(self.params)
        vis = f" ({self.visibility})" if self.visibility else ""
        return f"{self.class_fqn}#{self.method_name}({params_str}): {self.return_type}{vis}"

    def to_dict(self) -> dict:
        return {
            "class_fqn": self.class_fqn,
            "method_name": self.method_name,
            "params": list(self.params),
            "return_type": self.return_type,
            "visibility": self.visibility,
            "source_jar": self.source_jar,
            "label": self.label,
        }


@dataclass
class CallPath:
    """An ordered sequence of method signatures forming one call chain."""

    steps: list[MethodSig]

    def __len__(self) -> int:
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "length": len(self.steps),
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class ConstructorInfo:
    """A single public constructor for a class."""

    params: list[str]        # raw param types (may include modifiers like "(public)")
    visibility: str = "public"

    def to_dict(self) -> dict:
        return {
            "params": list(self.params),
            "visibility": self.visibility,
        }


@dataclass
class InitInfo:
    """Initialization options for a single class."""

    class_fqn: str
    class_modifiers: str               # e.g. "public", "public final"
    constructors: list[ConstructorInfo] = field(default_factory=list)
    known_implementations: list[str] = field(default_factory=list)
    factory_methods: list[str] = field(default_factory=list)
    source_jar: str | None = None

    def to_dict(self) -> dict:
        return {
            "class_fqn": self.class_fqn,
            "class_modifiers": self.class_modifiers,
            "constructors": [c.to_dict() for c in self.constructors],
            "known_implementations": list(self.known_implementations),
            "factory_methods": list(self.factory_methods),
            "source_jar": self.source_jar,
        }


@dataclass
class CallGraphResult:
    """Full structured output of one call-graph analysis run.

    Attributes:
        target_class:   FQN of the analysed class.
        target_method:  Simple method name.
        root_mode:      Root selection strategy used (e.g. ``"PUBLIC"``).
        max_depth:      Max depth used.
        limit_package:  Whether package-root limiting was active.
        paths:          All unique call paths found.
        initializations: Init info for every class touched by any path.
    """

    target_class: str
    target_method: str
    root_mode: str
    max_depth: int
    limit_package: bool
    paths: list[CallPath] = field(default_factory=list)
    initializations: list[InitInfo] = field(default_factory=list)

    @property
    def has_paths(self) -> bool:
        return bool(self.paths)

    def to_dict(self) -> dict:
        return {
            "target": {
                "class": self.target_class,
                "method": self.target_method,
            },
            "config": {
                "root_mode": self.root_mode,
                "max_depth": self.max_depth,
                "limit_package": self.limit_package,
            },
            "stats": {
                "path_count": len(self.paths),
                "indirect_path_count": sum(1 for p in self.paths if len(p) > 1),
                "initialization_count": len(self.initializations),
            },
            "paths": [p.to_dict() for p in self.paths],
            "initializations": [i.to_dict() for i in self.initializations],
        }
