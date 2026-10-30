"""Call-graph analysis subpackage."""

from tatg.analysis.callgraph.models import (
    CallGraphResult,
    CallPath,
    ConstructorInfo,
    InitInfo,
    MethodSig,
)
from tatg.analysis.callgraph.parser import (
    parse_callgraph_file,
    parse_callgraph_text,
)
from tatg.analysis.callgraph.sootup_analyzer import SootUpCallGraphAnalyzer

__all__ = [
    "CallGraphResult",
    "CallPath",
    "ConstructorInfo",
    "InitInfo",
    "MethodSig",
    "parse_callgraph_file",
    "parse_callgraph_text",
    "SootUpCallGraphAnalyzer",
]
