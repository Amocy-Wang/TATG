from __future__ import annotations

import json
from pathlib import Path
import re

from tatg.models import GenerationRequest, PromptBundle, PromptInsertPosition, Target
from tatg.prompting.base import PromptPass, insert_text


_JAVA_CONTROL_WORDS = {
    "if", "for", "while", "switch", "catch", "return", "throw", "new",
    "super", "this", "synchronized", "assert", "try", "do",
}


def _find_matching_delimiter(
    text: str,
    start: int,
    open_char: str,
    close_char: str,
) -> int:
    if start < 0 or start >= len(text) or text[start] != open_char:
        return -1
    depth = 0
    quote = ""
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {'"', "'"}:
            quote = char
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return -1


def _extract_declared_method_source(source_text: str, method_name: str) -> str:
    """Return a bounded method declaration/body, never a same-name call site."""
    if not source_text or not method_name:
        return ""
    declaration = re.compile(
        rf"(?m)^[ \t]*"
        rf"(?:@[A-Za-z_$][\w.$]*(?:\([^)\r\n]*\))?[ \t]+)*"
        rf"(?:(?:public|protected|private|static|final|synchronized|"
        rf"abstract|native|strictfp|default)\s+)*"
        rf"(?:<[^>{{}};]+>\s+)?"
        rf"[A-Za-z_$][\w$<>\[\].?, \t]*?[ \t]+"
        rf"{re.escape(method_name)}\s*\("
    )
    for match in declaration.finditer(source_text):
        open_paren = source_text.find("(", match.start(), match.end() + 1)
        close_paren = _find_matching_delimiter(source_text, open_paren, "(", ")")
        if close_paren < 0:
            continue
        open_brace = source_text.find("{", close_paren)
        semicolon = source_text.find(";", close_paren)
        if open_brace < 0 or (semicolon >= 0 and semicolon < open_brace):
            continue
        close_brace = _find_matching_delimiter(source_text, open_brace, "{", "}")
        if close_brace > open_brace:
            return source_text[match.start():close_brace + 1]
    return ""


def _declared_field_names(source_text: str) -> set[str]:
    fields: set[str] = set()
    for match in re.finditer(
        r"(?m)^[ \t]*(?:(?:public|protected|private)\s+)?"
        r"(?:(?:static|final|transient|volatile)\s+)*"
        r"[A-Za-z_$][\w$<>\[\].?, \t]*?[ \t]+"
        r"(?P<name>[A-Za-z_$][\w$]*)\s*(?:=|;)",
        source_text[:100000],
    ):
        fields.add(match.group("name"))
    return fields


def _declared_parameter_names(method_source: str) -> set[str]:
    open_paren = method_source.find("(")
    close_paren = _find_matching_delimiter(method_source, open_paren, "(", ")")
    if open_paren < 0 or close_paren < 0:
        return set()
    names: set[str] = set()
    for raw_parameter in method_source[open_paren + 1:close_paren].split(","):
        cleaned = re.sub(r"@\w+(?:\([^)]*\))?", " ", raw_parameter)
        cleaned = re.sub(r"\b(?:final|volatile|transient)\b", " ", cleaned)
        identifiers = re.findall(r"[A-Za-z_$][\w$]*", cleaned)
        if identifiers:
            names.add(identifiers[-1])
    return names


def _control_conditions(method_source: str) -> list[str]:
    conditions: list[str] = []
    pattern = re.compile(r"\b(?:if|for|while|switch)\s*\(")
    for match in pattern.finditer(method_source):
        open_paren = method_source.find("(", match.start(), match.end())
        close_paren = _find_matching_delimiter(
            method_source,
            open_paren,
            "(",
            ")",
        )
        if close_paren > open_paren:
            conditions.append(method_source[open_paren + 1:close_paren])
    return conditions


def _stateful_protocol_clues(target: Target) -> list[str]:
    """Derive protocol clues from Java structure, not names or API keywords."""
    target_file = Path(target.source_file)
    method_name = str(target.method_name or "").strip()
    if not target_file.exists() or not method_name:
        return []
    source_text = target_file.read_text(encoding="utf-8", errors="ignore")
    method_source = _extract_declared_method_source(source_text, method_name)
    if not method_source:
        return []

    fields = _declared_field_names(source_text)
    parameters = _declared_parameter_names(method_source)
    conditions = _control_conditions(method_source)
    branch_fields = sorted(
        field
        for field in fields
        if any(
            re.search(
                rf"(?:\bthis\s*\.\s*)?\b{re.escape(field)}\b",
                condition,
            )
            for condition in conditions
        )
    )
    written_fields = sorted(
        field
        for field in fields
        if re.search(
            rf"(?:\bthis\s*\.\s*)?\b{re.escape(field)}\b\s*"
            rf"(?:=|\+=|-=|\+\+|--)",
            method_source,
        )
    )
    declared_receivers = fields | parameters
    called_receivers = sorted({
        match.group("receiver")
        for match in re.finditer(
            r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*"
            r"(?P<method>[A-Za-z_$][\w$]*)\s*\(",
            method_source,
        )
        if match.group("receiver") in declared_receivers
        and match.group("method") not in _JAVA_CONTROL_WORDS
    })

    clues: list[str] = []
    if branch_fields:
        clues.append(
            "target control flow reads receiver field(s): "
            + ", ".join(branch_fields[:6])
        )
    if len(branch_fields) >= 2:
        clues.append(
            "one target decision relates multiple receiver fields: "
            + ", ".join(branch_fields[:6])
        )
    if written_fields and conditions:
        clues.append(
            "target both branches and updates receiver field(s): "
            + ", ".join(written_fields[:6])
        )
    if called_receivers and conditions:
        clues.append(
            "target branches while calling declared field/parameter receiver(s): "
            + ", ".join(called_receivers[:6])
        )
    return clues


class StatefulProtocolTargetPass(PromptPass):
    """Add guidance only when the target source proves stateful control flow."""

    def __init__(self, position: PromptInsertPosition = PromptInsertPosition.BEFORE_USER):
        self.position = position

    def apply(
        self,
        prompt: PromptBundle,
        request: GenerationRequest,
        target: Target,
    ) -> PromptBundle:
        clues = _stateful_protocol_clues(target)
        if not clues:
            return prompt
        text = "\n".join(
            [
                "[Source-Proven Stateful/Protocol Guidance]",
                "The target source contains receiver-state or declared-collaborator control flow.",
                "Use only source-visible constructors, factories, and calls to establish the required pre-state; do not infer behavior from class or method names and do not invent setup APIs.",
                "Vary one proven state/input relation per test, invoke the exact target or a verified source-visible entry, and assert an observable return, public state, output, collaborator effect, or specific exception.",
                "Do not assert private fields directly and do not use a no-throw or tautological assertion as the oracle.",
                "[Source evidence]",
                *[f"- {clue}" for clue in clues[:5]],
            ]
        )
        return insert_text(prompt, self.position, text)


class AdditionalPromptPass(PromptPass):
    def __init__(self, text: str, position: PromptInsertPosition):
        self.text = text.strip()
        self.position = position

    def apply(
        self,
        prompt: PromptBundle,
        request: GenerationRequest,
        target: Target,
    ) -> PromptBundle:
        if not self.text:
            return prompt
        header = "[User Additional Prompt Content]"
        return insert_text(prompt, self.position, f"{header}\n{self.text}")


class StaticAnalysisJsonPass(PromptPass):
    def __init__(self, json_path: Path, position: PromptInsertPosition = PromptInsertPosition.BEFORE_USER):
        self.json_path = json_path
        self.position = position

    def apply(
        self,
        prompt: PromptBundle,
        request: GenerationRequest,
        target: Target,
    ) -> PromptBundle:
        if not self.json_path.exists():
            return prompt
        try:
            payload = json.loads(self.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            import warnings
            warnings.warn(f"StaticAnalysisJsonPass: could not parse {self.json_path}: {exc}")
            return prompt

        def _render_value(value) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value.strip()
            return json.dumps(value, ensure_ascii=True, indent=2)

        text = ""
        if isinstance(payload, dict):
            # PLRDGEN-style keys and a few common aliases.
            method_under_test = payload.get("method_under_test") or payload.get("methods_under_test")
            call_graph_paths = payload.get("call_graph_paths") or payload.get("cg") or payload.get("call_graph")
            related_source_files = (
                payload.get("related_source_files")
                or payload.get("cgsrc")
                or payload.get("related_sources")
            )

            sections = [
                "[Static Analysis Context]",
                "The following analysis result can guide test generation and path coverage:",
            ]
            if method_under_test is not None:
                rendered = _render_value(method_under_test)
                if rendered:
                    sections.append("[Methods Under Test]")
                    sections.append(rendered)
            if call_graph_paths is not None:
                rendered = _render_value(call_graph_paths)
                if rendered:
                    sections.append("[Call Graph Paths]")
                    sections.append(rendered)
            if related_source_files is not None:
                rendered = _render_value(related_source_files)
                if rendered:
                    sections.append("[Related Source Files]")
                    sections.append(rendered)

            # Fallback to raw JSON when structured sections are unavailable.
            if len(sections) <= 2:
                compact = json.dumps(payload, ensure_ascii=True, indent=2)
                sections.append(compact)
            text = "\n".join(sections)
        else:
            compact = json.dumps(payload, ensure_ascii=True, indent=2)
            text = (
                "[Static Analysis Context]\n"
                "The following analysis result can guide test generation and path coverage:\n"
                f"{compact}"
            )
        return insert_text(prompt, self.position, text)


class CallableSurfacePass(PromptPass):
    """Inject the Signature-Truth Anchor (DACR component A).

    The pass extracts the **exact** callable surface of the class under test
    plus all top-level types declared in sibling ``.java`` files of the same
    package, and prepends it to the user prompt as a deterministic anchor.

    This is the single most effective lever against the three failure modes
    observed in the Binance baseline run:

    * hallucinated symbols (e.g. ``CryptoException``),
    * wrong-arity method calls on the API class (e.g. 11 args vs. required 10),
    * ``new`` on enum types declared in the same package.
    """

    def __init__(self, position: PromptInsertPosition = PromptInsertPosition.BEFORE_USER):
        self.position = position

    def apply(
        self,
        prompt: PromptBundle,
        request: GenerationRequest,
        target: Target,
    ) -> PromptBundle:
        try:
            from tatg.analysis.api_surface import extract_surface, render_surface_anchor
        except Exception:
            return prompt
        target_file = Path(target.source_file)
        target_class = target.class_name or target_file.stem
        if not target_file.exists():
            return prompt
        surface = extract_surface(target_file, target_class)
        text = render_surface_anchor(surface)
        if not text:
            return prompt
        return insert_text(prompt, self.position, text)


class CATPass(StaticAnalysisJsonPass):
    """Alias for CFA-guided prompt enrichment aligned with CAT terminology."""
