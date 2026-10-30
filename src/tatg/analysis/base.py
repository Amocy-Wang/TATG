"""Abstract base protocol and shared data models for static analysis.

Any static analysis tool — call-graph, data-flow, type-inference, etc. —
should implement the :class:`StaticAnalyzer` Protocol so it can be registered
and invoked uniformly by the TATG framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass
class AnalysisRequest:
    """Input specification for a static-analysis run.

    Attributes:
        classpath:      Colon-separated list of JARs / class-directory paths
                        to load (same syntax as Java ``-cp``).
        target_class:   Fully-qualified class name of the unit under test,
                        e.g. ``org.apache.commons.cli.HelpFormatter``.
        target_method:  Simple method name, e.g. ``printWrapped``.
        max_depth:      Maximum call-chain depth to explore (default 3).
        dep_dir:        Optional directory that contains additional dependency
                        JARs to auto-resolve external classes.
        output_file:    If given, the analyzer writes its raw text output
                        there in addition to returning a structured result.
        root_mode:      Entry-point selection strategy: ``"PUBLIC"`` (default),
                        ``"ALL"``, or ``"MAINS"``.
        limit_package_roots:
                        When *True* the analyzer restricts call-graph roots to
                        the same package as the target class.
        max_paths:      Optional upper bound on the number of unique paths to
                return. ``None`` means unlimited.
        max_search_states:
                Optional upper bound on DFS states visited by the
                analyzer. ``None`` means unlimited.
        threads:        Number of threads for parallel search. Default is 1
                (single-threaded). Set to > 1 for parallel root exploration.
        extra:          Arbitrary analyzer-specific kwargs for extensibility.
    """

    classpath: str
    target_class: str
    target_method: str
    max_depth: int = 3
    dep_dir: str | None = None
    output_file: Path | None = None
    root_mode: str = "PUBLIC"
    limit_package_roots: bool = True
    max_paths: int | None = None
    max_search_states: int | None = None
    threads: int = 1
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Structured output produced by a :class:`StaticAnalyzer`.

    Attributes:
        target:         ``<class>.<method>`` of the analysed unit.
        analyzer_name:  Name of the analyzer that produced this result.
        structured:     Analyzer-specific structured data (e.g. a
                        :class:`~tatg.analysis.callgraph.models.CallGraphResult`).
        raw_text:       Raw text output exactly as written by the analyzer.
                        Used for prompt injection and to cache pre-computed
                        files.
        metadata:       Timing, tool version, and other diagnostics.
    """

    target: str
    analyzer_name: str
    structured: Any
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_prompt_context(self) -> str:
        """Return a concise string suitable for embedding in an LLM prompt."""
        if not self.raw_text:
            return ""
        header = f"[Static analysis: {self.analyzer_name} for {self.target}]\n"
        return header + self.raw_text


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class StaticAnalyzer(Protocol):
    """Duck-typed interface that every static-analysis tool must satisfy.

    Implementors only need to provide ``name`` and ``analyze``; the
    :class:`~tatg.analysis.registry.AnalyzerRegistry` will accept any
    object that matches this structural shape.
    """

    #: Unique short identifier for the analyzer, e.g. ``"sootup-callgraph"``.
    name: str

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Run the analysis and return a structured result.

        Args:
            request: Parameters that describe what to analyse and how.

        Returns:
            An :class:`AnalysisResult` containing both structured data and
            the raw text output.

        Raises:
            AnalysisError: When the analysis fails unrecoverably.
        """
        ...  # pragma: no cover


class AnalysisError(RuntimeError):
    """Raised when a static-analysis tool encounters an unrecoverable error."""
