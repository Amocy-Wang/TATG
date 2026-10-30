"""Analyzer registry — a lightweight registry for :class:`~tatg.analysis.base.StaticAnalyzer` instances.

Usage example::

    from tatg.analysis import get_registry
    from tatg.analysis.callgraph import SootUpCallGraphAnalyzer

    registry = get_registry()
    registry.register(SootUpCallGraphAnalyzer())

    result = registry.run("sootup-callgraph", request)

Future analysis tools (data-flow, type-inference, coverage-guided, etc.) only
need to satisfy the :class:`~tatg.analysis.base.StaticAnalyzer` Protocol
and be registered here to become available across the framework.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from tatg.analysis.base import AnalysisRequest, AnalysisResult, AnalysisError

if TYPE_CHECKING:
    from tatg.analysis.base import StaticAnalyzer


class AnalyzerRegistry:
    """Thread-safe (single-process) registry for static-analysis tools."""

    def __init__(self) -> None:
        self._analyzers: dict[str, "StaticAnalyzer"] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, analyzer: "StaticAnalyzer") -> None:
        """Register *analyzer* under its ``name`` attribute.

        If an analyzer with the same name already exists it is replaced.
        """
        self._analyzers[analyzer.name] = analyzer

    def unregister(self, name: str) -> None:
        """Remove the analyzer identified by *name* (no-op if absent)."""
        self._analyzers.pop(name, None)

    def has(self, name: str) -> bool:
        """Return *True* if an analyzer with *name* is registered."""
        return name in self._analyzers

    @property
    def names(self) -> list[str]:
        """Sorted list of registered analyzer names."""
        return sorted(self._analyzers)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, name: str, request: AnalysisRequest) -> AnalysisResult:
        """Run the analyzer identified by *name*.

        Args:
            name:    Analyzer name, e.g. ``"sootup-callgraph"``.
            request: Analysis parameters.

        Returns:
            An :class:`~tatg.analysis.base.AnalysisResult`.

        Raises:
            AnalysisError: If *name* is not registered.
        """
        if name not in self._analyzers:
            available = ", ".join(self.names) or "(none)"
            raise AnalysisError(
                f"No analyzer named '{name}' is registered. "
                f"Available: {available}"
            )
        return self._analyzers[name].analyze(request)

    def run_all(self, request: AnalysisRequest) -> dict[str, AnalysisResult]:
        """Run all registered analyzers and return a ``{name: result}`` mapping.

        Errors from individual analyzers are caught and stored as
        :class:`~tatg.analysis.base.AnalysisResult` objects with an empty
        ``raw_text`` and the error message in ``metadata["error"]``.
        """
        results: dict[str, AnalysisResult] = {}
        for name, analyzer in self._analyzers.items():
            try:
                results[name] = analyzer.analyze(request)
            except Exception as exc:  # noqa: BLE001
                results[name] = AnalysisResult(
                    target=f"{request.target_class}.{request.target_method}",
                    analyzer_name=name,
                    structured=None,
                    raw_text="",
                    metadata={"error": str(exc)},
                )
        return results


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------

_default_registry: AnalyzerRegistry | None = None


def get_registry() -> AnalyzerRegistry:
    """Return the process-wide default :class:`AnalyzerRegistry`.

    The registry is lazily constructed and pre-populated with the
    :class:`~tatg.analysis.callgraph.SootUpCallGraphAnalyzer` if the
    SootUp JAR can be found automatically.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = AnalyzerRegistry()
        _try_register_defaults(_default_registry)
    return _default_registry


def _try_register_defaults(registry: AnalyzerRegistry) -> None:
    """Attempt to register built-in analyzers; silently skip if unavailable."""
    try:
        from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

        timeout = _env_positive_int("SOOTUP_CG_TIMEOUT_SECONDS", 30)
        registry.register(SootUpCallGraphAnalyzer(java_exe=_resolve_sootup_java(), timeout=timeout))
    except Exception:  # noqa: BLE001
        pass


def _resolve_sootup_java() -> str:
    explicit = str(os.environ.get("SOOTUP_JAVA", "") or "").strip()
    if explicit:
        return explicit
    java_home = str(os.environ.get("SOOTUP_JAVA_HOME", "") or "").strip()
    if java_home:
        return str(os.path.join(java_home, "bin", "java"))
    return "java"


def _env_positive_int(env_name: str, default: int) -> int:
    raw = str(os.environ.get(env_name, "") or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default
