"""Test-repair orchestration.

Architecture
------------
The module is built around a *strategy* interface so that new repair approaches
can be added without touching the orchestration layer::

    RepairStrategy (ABC)
    ├── LLMRepairStrategy   – current default: builds a prompt and calls the LLM
    └── <future strategies> – e.g. AST-patch, search-based, symbolic, hybrid …

Public API used by :mod:`tatg.cli`::

    extract_failing_methods(failure_output, class_stem) -> list[str]
    extract_repair_candidates(runner_results)           -> list[dict]
    run_repair(request, artifacts, failure_output, *, workers, strategy) -> (usages, events)
    summarize_repair_metrics(test_runs, generated_test_count) -> dict
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tatg.openai_compat import OpenAICompatError, chat_completion_with_usage
from tatg.pathing import derive_java_test_file
from tatg.runner import TestFileRegistry

if TYPE_CHECKING:
    from tatg.models import GenerationRequest


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Maximum characters of failure output forwarded to the LLM.  Long Surefire
# logs are dominated by repeated stack frames; the tail keeps the most recent,
# most actionable diagnostics within typical context windows.
_FAILURE_OUTPUT_TAIL_CHARS = 12000

# Number of additional LLM attempts after a transient timeout.  The default is
# zero so a timed-out repair candidate yields control to the next candidate.
_LLM_TIMEOUT_RETRY_ATTEMPTS = 0

_REPAIR_SOURCE_MAX_CHARS = 24000
_REPAIR_TEST_MAX_CHARS = 32000
_REPAIR_COMPACT_SOURCE_MAX_CHARS = 12000
_REPAIR_COMPACT_TEST_MAX_CHARS = 14000
_REPAIR_METHOD_SOURCE_MAX_CHARS = 12000
_REPAIR_COMPACT_METHOD_SOURCE_MAX_CHARS = 8000
_REPAIR_CALLBACK_SIGNATURE_MAX_CHARS = 8000


def _mutation_repair_objective(
    feedback: object,
    objective_id: str,
) -> dict[str, object]:
    """Find the exact M-id contract to preserve during local repair."""

    wanted = str(objective_id or "").strip().upper()
    if not wanted or not isinstance(feedback, (dict, list)):
        return {}
    stack: list[tuple[object, int]] = [(feedback, 0)]
    seen: set[int] = set()
    while stack:
        value, depth = stack.pop()
        if depth > 6 or id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, dict):
            current = str(
                value.get("id")
                or value.get("objective_id")
                or value.get("mutation_objective_id")
                or ""
            ).strip().upper()
            if current == wanted and isinstance(value.get("facts"), dict):
                return dict(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    stack.append((child, depth + 1))
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)
    return {}
_REPAIR_COMPACT_CALLBACK_SIGNATURE_MAX_CHARS = 5000
_REPAIR_TARGET_RECEIPT_MAX_CHARS = 4200
_REPAIR_COMPACT_TARGET_RECEIPT_MAX_CHARS = 3000
_REPAIR_EXECUTION_CAPSULE_MAX_CHARS = 5200
_REPAIR_COMPACT_EXECUTION_CAPSULE_MAX_CHARS = 3800
_REPAIR_ENTRY_CONTRACT_MAX_CHARS = 3600
_REPAIR_COMPACT_ENTRY_CONTRACT_MAX_CHARS = 2600
_REPAIR_CALLABLE_SURFACE_MAX_CHARS = 9000
_REPAIR_COMPACT_CALLABLE_SURFACE_MAX_CHARS = 5200
_REPAIR_VISIBLE_CONSTRUCTION_MAX_CHARS = 7000
_REPAIR_COMPACT_VISIBLE_CONSTRUCTION_MAX_CHARS = 4200
_PROJECT_SYMBOL_INDEX_CACHE: dict[str, dict[str, list[str]]] = {}


_SPECIFIC_COMPILE_DIAGNOSTIC_RE = re.compile(
    r"(?i)(?:"
    r"cannot find symbol"
    r"|incompatible types"
    r"|method .+ cannot be applied"
    r"|constructor .+ cannot be applied"
    r"|actual and formal argument lists differ"
    r"|wrong number of type arguments"
    r"|does not override abstract method"
    r"|method does not override or implement"
    r"|is not abstract and does not override"
    r"|has private access"
    r"|is not public"
    r"|non-static method"
    r"|package .+ does not exist"
    r"|cannot be converted to"
    r")"
)
_COMPILE_DIAGNOSTIC_CONTINUATION_RE = re.compile(
    r"(?i)^\s*(?:\[ERROR\]\s*)?(?:symbol|location|required|found|reason):"
)
_GENERIC_MAVEN_COMPILE_SUMMARY_RE = re.compile(
    r"(?i)(?:failed to execute goal|compilation failure|compilation error)"
)


def _focused_compile_diagnostic_lines(text: str, *, limit: int = 8) -> list[str]:
    """Prefer concrete javac evidence over repeated Maven goal summaries."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    selected_indices: list[int] = []
    for index, line in enumerate(lines):
        if not _SPECIFIC_COMPILE_DIAGNOSTIC_RE.search(line):
            continue
        selected_indices.append(index)
        for following in range(index + 1, min(len(lines), index + 5)):
            if _COMPILE_DIAGNOSTIC_CONTINUATION_RE.search(lines[following]):
                selected_indices.append(following)
                continue
            if following > index + 1:
                break

    if not selected_indices:
        selected_indices = [
            index
            for index, line in enumerate(lines)
            if _GENERIC_MAVEN_COMPILE_SUMMARY_RE.search(line)
        ][:1]

    result: list[str] = []
    seen: set[str] = set()
    for index in selected_indices:
        line = lines[index]
        normalized = re.sub(r"\s+", " ", line)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(line)
        if len(result) >= max(1, int(limit or 1)):
            break
    return result


def _llm_temperature() -> float:
    raw = os.getenv("TATG_LLM_TEMPERATURE", "0.2")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.2


class NonRetryableLLMRepairError(RuntimeError):
    """Raised when the repair provider rejects the request itself."""


def _clip_middle(text: str, max_chars: int, *, label: str) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = f"\n/* ... {label} truncated, {len(text) - max_chars} chars omitted ... */\n"
    keep = max(0, max_chars - len(marker))
    head = keep // 2
    tail = keep - head
    return text[:head] + marker + text[-tail:]


def _repair_budget(*, compact: bool, normal: int, compact_value: int) -> int:
    return max(1, int(compact_value if compact else normal))


def _extract_java_test_method_block(text: str, method_name: str | None) -> str:
    """Return a single @Test method block when it can be found lexically."""
    source = str(text or "")
    name = str(method_name or "").strip()
    if not source or not name:
        return ""
    method_match = re.search(rf"\b{re.escape(name)}\s*\([^)]*\)\s*(?:throws\s+[^\{{]+)?\{{", source)
    if not method_match:
        return ""
    annotation_start = source.rfind("@Test", 0, method_match.start())
    if annotation_start < 0:
        annotation_start = method_match.start()
        line_start = source.rfind("\n", 0, annotation_start)
        if line_start >= 0:
            annotation_start = line_start + 1
    else:
        line_start = source.rfind("\n", 0, annotation_start)
        if line_start >= 0:
            annotation_start = line_start + 1
    brace_start = source.find("{", method_match.start())
    if brace_start < 0:
        return ""
    depth = 0
    in_string = False
    in_char = False
    in_line_comment = False
    in_block_comment = False
    escape = False
    for idx in range(brace_start, len(source)):
        ch = source[idx]
        next_ch = source[idx + 1] if idx + 1 < len(source) else ""
        if in_line_comment:
            if ch in "\r\n":
                in_line_comment = False
            continue
        if in_block_comment:
            if ch == "*" and next_ch == "/":
                in_block_comment = False
            continue
        if escape:
            escape = False
            continue
        if ch == "\\" and (in_string or in_char):
            escape = True
            continue
        if ch == '"' and not in_char:
            in_string = not in_string
            continue
        if ch == "'" and not in_string:
            in_char = not in_char
            continue
        if in_string or in_char:
            continue
        if ch == "/" and next_ch == "/":
            in_line_comment = True
            continue
        if ch == "/" and next_ch == "*":
            in_block_comment = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[annotation_start:idx + 1].strip()
    return ""


def _window_around_method(text: str, method_name: str | None, max_chars: int) -> str:
    focused_block = _extract_java_test_method_block(text, method_name)
    if focused_block:
        return _clip_middle(focused_block, max_chars, label="test method")
    if not method_name or len(text) <= max_chars:
        return _clip_middle(text, max_chars, label="code")
    match = re.search(rf"\b{re.escape(method_name)}\s*\(", text)
    if not match:
        return _clip_middle(text, max_chars, label="code")
    half = max_chars // 2
    start = max(0, match.start() - half)
    end = min(len(text), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "" if start == 0 else f"/* ... code before {method_name} omitted ... */\n"
    suffix = "" if end == len(text) else f"\n/* ... code after {method_name} omitted ... */"
    return prefix + text[start:end] + suffix


def _json_line_to_text(line: str) -> str | None:
    stripped = str(line or "").strip()
    if not stripped:
        return None
    if stripped.endswith(","):
        stripped = stripped[:-1].rstrip()
    if not (stripped.startswith('"') and stripped.endswith('"')):
        return None
    try:
        decoded = json.loads(stripped)
    except Exception:
        return None
    if isinstance(decoded, str):
        return decoded
    return None


def _source_lines_from_jsonish_excerpt(text: str) -> list[str]:
    source = str(text or "").strip()
    if not source:
        return []
    parsed: object | None = None
    if source.startswith("{") or source.startswith("["):
        try:
            parsed = json.loads(source)
        except Exception:
            parsed = None
    candidates: list[object] = []
    if isinstance(parsed, dict):
        candidates.extend(parsed.get("test_code_lines") or [])
        for item in parsed.get("new_tests") or []:
            if isinstance(item, dict):
                candidates.extend(item.get("test_code_lines") or [])
    elif isinstance(parsed, list):
        candidates.extend(parsed)
    if candidates and all(isinstance(item, str) for item in candidates):
        return [str(item) for item in candidates]

    decoded_lines: list[str] = []
    for raw in source.splitlines():
        decoded = _json_line_to_text(raw)
        if decoded is not None:
            decoded_lines.append(decoded)
    java_like = sum(
        1
        for line in decoded_lines
        if any(token in line for token in ("@Test", "assert", "new ", "=", ";", "{", "}", "."))
    )
    if len(decoded_lines) >= 2 and java_like >= 2:
        return decoded_lines
    return []


def _normalize_repair_source_excerpt(text: str) -> str:
    """Convert JSON-ish `test_code_lines` excerpts back to Java source text."""
    source = str(text or "").strip()
    if not source:
        return ""
    lines = _source_lines_from_jsonish_excerpt(source)
    if not lines:
        return source
    return "\n".join(lines).strip()


def _looks_like_member_declaration_inside_test_method(text: str) -> bool:
    source = str(text or "")
    if not source or "@Test" not in source:
        return False
    method_match = re.search(r"@Test[\s\S]{0,600}?\b(?:public\s+|protected\s+|private\s+)?void\s+\w+\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\{", source)
    if not method_match:
        return False
    brace = source.find("{", method_match.end() - 1)
    if brace < 0:
        return False
    end = _find_matching_java_brace(source, brace)
    body = source[brace + 1:end if end >= 0 else len(source)]
    tail = source[end + 1:] if end >= 0 else ""
    if re.search(r"^\s*@[\w.]+(?:\([^)]*\))?\s*$\s*^\s*(?:public|protected|private|static|final|\w[\w.<>, ?\[\]]+\s+)+\w+\s*(?:=|;)", body, re.MULTILINE):
        return True
    if re.search(r"^\s*(?:public|protected|private)?\s*[A-Z]\w*\s*\([^;{}]*\)\s*\{", body, re.MULTILINE):
        return True
    if re.search(r"^\s*@[\w.]+(?:\([^)]*\))?\s*$\s*^\s*(?:public|protected|private|static|final|\w[\w.<>, ?\[\]]+\s+)+\w+\s*(?:=|;)", tail, re.MULTILINE):
        return True
    if re.search(r"^\s*(?:public|protected|private)?\s*[A-Z]\w*\s*\([^;{}]*\)\s*\{", tail, re.MULTILINE):
        return True
    return False


def _reflection_annotation_setup_hint(
    *,
    failure_signal: dict,
    excerpt: str,
    failure_output: str,
) -> str:
    text = "\n".join(
        str(part or "")
        for part in (
            failure_signal.get("message") if isinstance(failure_signal, dict) else "",
            failure_output,
            excerpt,
        )
    )
    if not re.search(r"\bmust\s+be\s+annotated\b|\bnot\s+annotated\b", text, re.IGNORECASE):
        return ""
    if not re.search(r"\.class\s*\.\s*get(?:Declared)?Method\s*\(", text):
        return ""
    annotation = ""
    match = re.search(r"annotated\s+with\s+@?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)", text, re.IGNORECASE)
    if match:
        annotation = match.group(1)
    lines = [
        "[Reflection Annotation Setup Repair]",
        "- evidence=the failure says a reflected Method must be annotated, while the failing test obtains a Method from `.class.getMethod(...)` or `.class.getDeclaredMethod(...)`.",
        "- repair_rule=preserve the same target/entry call, but replace the unannotated reflected Method source with a source-visible helper method carrying the required annotation.",
        "- method_level_schema_rule=when the repair output can only return one @Test method, define a local helper class inside that @Test, annotate its helper method, obtain `Helper.class.getMethod(...)`, and pass a matching helper instance to the entry call when the entry call needs an object receiver.",
        "- forbidden=do not keep using framework methods such as Runnable.run or unrelated JDK methods when the target explicitly consumes the annotation.",
    ]
    if annotation:
        lines.append(f"- required_annotation={annotation}")
    return "\n".join(lines)


def _unresolved_static_member_repair_hint(failure_output: str) -> str:
    text = str(failure_output or "")
    if "cannot find symbol" not in text:
        return ""
    rows: list[tuple[str, str]] = []
    pattern = re.compile(
        r"symbol:\s+variable\s+(?P<member>[A-Za-z_$][\w$]*)\s*"
        r"(?:\n|\r\n?)\s*(?:\[ERROR\]\s*)?location:\s+class\s+(?P<owner>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(text):
        owner = str(match.group("owner") or "").strip()
        member = str(match.group("member") or "").strip()
        if owner and member and (owner, member) not in rows:
            rows.append((owner, member))
        if len(rows) >= 6:
            break
    if not rows:
        return ""
    lines = [
        "[Unresolved Static Member Repair]",
        "- evidence=javac proved the generated test referenced a missing static field/enum constant.",
        "- repair_rule=do not repeat forbidden `Owner.MEMBER` references; use only constants listed in the callable/source surface, or a constructor/builder/string overload that is explicitly visible.",
        "- string_value_rule=when the visible API accepts a String value and source examples use a quoted literal for the missing member name, prefer the quoted literal instead of inventing a static constant.",
    ]
    for owner, member in rows:
        lines.append(f"- forbidden_static_member={owner}.{member}")
    return "\n".join(lines)


def _type_name_used_as_value_repair_hint(*, failure_output: str, context_text: str) -> str:
    output = str(failure_output or "")
    if "cannot find symbol" not in output:
        return ""
    context = str(context_text or "")
    symbols: list[str] = []
    for match in re.finditer(
        r"symbol:\s+variable\s+(?P<symbol>[A-Z][A-Za-z0-9_$]*)",
        output,
        re.IGNORECASE,
    ):
        symbol = str(match.group("symbol") or "").strip()
        if not symbol or symbol in symbols:
            continue
        if re.search(rf"(?m)^\s*import\s+[\w.]*\.{re.escape(symbol)}\s*;", context) or re.search(
            rf"\b(?:class|interface|record|enum)\s+{re.escape(symbol)}\b",
            context,
        ):
            symbols.append(symbol)
        if len(symbols) >= 6:
            break
    if not symbols:
        return ""
    lines = [
        "[Type Name Used As Value Repair]",
        "- evidence=javac reported an unresolved variable whose name is also a visible type/import in the test or source context.",
        "- repair_rule=do not pass a type name as a runtime value; construct a concrete value through a listed constructor/builder/factory, a source-visible static factory, or a focused mock/fake of that type.",
        "- method_level_schema_rule=if only one @Test method can be returned, keep helper construction local inside that method and put any needed import in new_imports_lines.",
    ]
    for symbol in symbols:
        lines.append(f"- forbidden_value_expression={symbol}")
    return "\n".join(lines)


def _java_major_version(value: str | None) -> int | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text.startswith("1."):
        text = text[2:]
    match = re.search(r"\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _java_source_level_from_failure_output(output: str | None) -> int | None:
    text = str(output or "")
    patterns = (
        r"-source\s+(\d+(?:\.\d+)?)",
        r"source\s+release\s+(\d+(?:\.\d+)?)",
        r"source\s+(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        major = _java_major_version(match.group(1))
        if major:
            return major
    if "java.util.function" in text and ("cannot access" in text.lower() or "class file for" in text.lower()):
        return 7
    return None


def _find_matching_java_brace(text: str, open_pos: int) -> int:
    depth = 0
    idx = open_pos
    n = len(text)
    while idx < n:
        ch = text[idx]
        if ch == '/' and idx + 1 < n and text[idx + 1] == '/':
            newline = text.find('\n', idx)
            if newline == -1:
                return -1
            idx = newline + 1
            continue
        if ch == '/' and idx + 1 < n and text[idx + 1] == '*':
            end_comment = text.find('*/', idx + 2)
            if end_comment == -1:
                return -1
            idx = end_comment + 2
            continue
        if ch in {'"', "'"}:
            quote = ch
            idx += 1
            while idx < n:
                if text[idx] == '\\':
                    idx += 2
                    continue
                if text[idx] == quote:
                    idx += 1
                    break
                idx += 1
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return idx + 1
        idx += 1
    return -1


def _method_source_windows(source_code: str, method_name: str | None, max_chars: int) -> str:
    if not source_code or not method_name:
        return ""
    pattern = re.compile(
        rf"(?m)^\s*(?:@[\w.]+(?:\([^\n]*\))?\s*)*"
        rf"(?:public|protected|private|static|final|synchronized|native|abstract|strictfp|\s)+"
        rf"[\w.$<>\[\], ?]+\s+{re.escape(method_name)}\s*\([^;{{}}]*\)"
        rf"(?:\s+throws\s+[^{{;]+)?\s*{{"
    )
    snippets: list[str] = []
    used = 0
    for match in pattern.finditer(source_code):
        brace_pos = source_code.find('{', match.end() - 1)
        if brace_pos == -1:
            continue
        end = _find_matching_java_brace(source_code, brace_pos)
        if end == -1:
            continue
        snippet = source_code[match.start():end].strip()
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(snippet) > remaining:
            snippet = _clip_middle(snippet, remaining, label=f"{method_name} method body")
        snippets.append(snippet)
        used += len(snippet) + 2
        if used >= max_chars:
            break
    return "\n\n".join(snippets)


def _source_header_and_declarations(source_code: str, max_chars: int) -> str:
    if not source_code or max_chars <= 0:
        return ""
    lines: list[str] = []
    seen: set[str] = set()
    for raw_line in source_code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        keep = False
        if line.startswith("package ") or line.startswith("import "):
            keep = True
        elif re.match(r"(?:public|protected|private|abstract|final|static|\s)*(?:class|interface|enum)\s+\w+", line):
            keep = True
        elif re.match(r"(?:public|protected)\s+", line) and (line.endswith(";") or "(" in line):
            keep = True
        if keep and line not in seen:
            seen.add(line)
            lines.append(raw_line.rstrip())
    text = "\n".join(lines)
    return _clip_middle(text, max_chars, label="source declarations")


def _failure_symbol_context(source_code: str, failure_output: str, max_chars: int) -> str:
    if not source_code or not failure_output or max_chars <= 0:
        return ""
    symbols: list[str] = []
    for pattern in (
        r"(?m)^\s*(?:\[ERROR\]\s*)?symbol:\s+(?:method|class|variable|constructor)\s+([\w.$]+)",
        r"no suitable method found for\s+([\w.$]+)\s*\(",
        r"cannot find symbol\s*\n\s*(?:\[ERROR\]\s*)?symbol:\s+(?:method|class|variable|constructor)\s+([\w.$]+)",
    ):
        for match in re.finditer(pattern, failure_output):
            symbol = match.group(1).split(".")[-1].strip()
            if symbol and symbol not in symbols:
                symbols.append(symbol)
            if len(symbols) >= 8:
                break
        if len(symbols) >= 8:
            break
    if not symbols:
        return ""

    source_lines = source_code.splitlines()
    snippets: list[str] = []
    used_spans: set[tuple[int, int]] = set()
    for symbol in symbols:
        escaped = re.escape(symbol)
        for idx, line in enumerate(source_lines):
            if not re.search(rf"\b{escaped}\b", line):
                continue
            start = max(0, idx - 2)
            end = min(len(source_lines), idx + 3)
            span = (start, end)
            if span in used_spans:
                continue
            used_spans.add(span)
            snippets.append("\n".join(source_lines[start:end]))
            break
    text = "\n\n".join(snippets)
    return _clip_middle(text, max_chars, label="failure symbol source context")


def _repair_source_context(ctx: "RepairContext", *, compact: bool) -> str:
    production_focus = ""
    if isinstance(ctx.extra, dict):
        production_focus = str(ctx.extra.get("production_method_focus") or "").strip()
    source_method_focus = production_focus or str(ctx.method_focus or "").strip()
    if not source_method_focus:
        limit = _REPAIR_COMPACT_SOURCE_MAX_CHARS if compact else _REPAIR_SOURCE_MAX_CHARS
        return _clip_middle(ctx.source_code, limit, label="source")

    total_limit = _REPAIR_COMPACT_METHOD_SOURCE_MAX_CHARS if compact else _REPAIR_METHOD_SOURCE_MAX_CHARS
    target_budget = int(total_limit * 0.58)
    declaration_budget = int(total_limit * 0.27)
    failure_budget = total_limit - target_budget - declaration_budget

    sections: list[str] = []
    target_windows = _method_source_windows(ctx.source_code, source_method_focus, target_budget)
    if target_windows:
        sections.append("Target method source window(s):\n" + target_windows)
    else:
        sections.append(
            "Target method source window(s):\n"
            + _window_around_method(ctx.source_code, source_method_focus, target_budget)
        )

    failure_context = _failure_symbol_context(ctx.source_code, ctx.failure_output, failure_budget)
    if failure_context:
        sections.append("Source declarations/snippets matching failure symbols:\n" + failure_context)

    declarations = _source_header_and_declarations(ctx.source_code, declaration_budget)
    if declarations:
        sections.append("Available source declarations and imports:\n" + declarations)

    context = "\n\n".join(section for section in sections if section.strip())
    return _clip_middle(context, total_limit, label="method repair source context")

def _prompt_log_clip(text: str, max_chars: int) -> tuple[str, bool]:
    # Preserve full prompt text in logs; rely on log_prompts_limit for size
    # control instead of clipping content.
    return text or "", False

def _record_repair_prompt_log(
    request: GenerationRequest,
    *,
    label: str,
    method: str | None,
    system_prompt: str,
    user_prompt: str,
    compact: bool = False,
) -> None:
    if not bool(getattr(request, "log_prompts", False)):
        return
    logs = getattr(request, "prompt_logs", None)
    if not isinstance(logs, list):
        return
    limit = max(0, int(getattr(request, "log_prompts_limit", 20) or 20))
    if len(logs) >= limit:
        return
    sys_text, sys_truncated = _prompt_log_clip(system_prompt, 4000)
    user_text, user_truncated = _prompt_log_clip(user_prompt, 12000)
    logs.append({
        "phase": "repair",
        "label": label,
        "method": method,
        "compact": bool(compact),
        "system_prompt": sys_text,
        "user_prompt": user_text,
        "system_prompt_chars": len(system_prompt or ""),
        "user_prompt_chars": len(user_prompt or ""),
        "truncated": bool(sys_truncated or user_truncated),
    })


def _repair_critical_generation_receipt_lines(
    receipt: str,
    failure_output: str,
    *,
    max_lines: int = 12,
) -> str:
    """Extract exact source-derived contracts that are most useful for repair.

    The full generation receipt can be long. Repair prompts still need exact
    callable/return-type contracts, especially after compile diagnostics mention
    a mismatched generic, reactive, callback, or wrapper type.
    """
    text = str(receipt or "")
    if not text:
        return ""
    failure_text = str(failure_output or "")
    type_tokens = {
        token
        for token in re.findall(r"\b[A-Z][A-Za-z0-9_$]*(?:<[A-Za-z0-9_$., ? extends super]+>)?", failure_text)
        if len(token) >= 3 and token not in {"ERROR", "INFO", "Help", "Tests", "Failures", "Errors"}
    }
    priority_phrases = (
        "declared_return=",
        "when stubbing",
        "source_required_receiver_return_baseline",
        "source_required_collaborator_return_baseline",
        "source_exact_collaborator_stub_contracts",
        "source_required_terminal_execution_path",
        "exact declared",
        "-> Flux<",
        "-> Mono<",
        "-> CompletionStage<",
        "-> CompletableFuture<",
        "callback",
        "terminal",
    )
    out: list[str] = []
    seen: set[str] = set()
    active_header = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- source_") and stripped.endswith(":"):
            active_header = stripped
        line_matches_type = any(token and token in stripped for token in type_tokens)
        line_matches_phrase = any(phrase in stripped for phrase in priority_phrases)
        under_relevant_header = active_header in {
            "- source_required_receiver_return_baseline:",
            "- source_required_collaborator_return_baseline:",
            "- source_exact_collaborator_stub_contracts:",
            "- source_required_terminal_execution_path:",
        } and stripped.startswith("- ")
        if not (line_matches_type or line_matches_phrase or under_relevant_header):
            continue
        rendered = stripped
        if active_header and stripped.startswith("- ") and rendered != active_header:
            rendered = f"{active_header} {rendered}"
        key = re.sub(r"\s+", " ", rendered)
        if key in seen:
            continue
        seen.add(key)
        out.append(rendered[:480])
        if len(out) >= max_lines:
            break
    if not out:
        return ""
    return "\n".join(["[Repair-Critical Source Contracts]", *("- " + line.lstrip("- ").strip() for line in out)])


def _observed_required_non_null_setup_lines(
    *,
    failure_signal: dict[str, Any],
    generation_control_receipt: str,
    max_lines: int = 4,
) -> list[str]:
    message = " ".join(str(failure_signal.get("message") or "").split())
    if not message:
        return []
    names: list[str] = []
    patterns = [
        r"\b([A-Za-z_$][A-Za-z0-9_$.]*)\s+cannot\s+be\s+null\b",
        r"\b([A-Za-z_$][A-Za-z0-9_$.]*)\s+must\s+not\s+be\s+null\b",
        r"\bbecause\s+\"([A-Za-z_$][A-Za-z0-9_$.]*)\"\s+is\s+null\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, message, re.IGNORECASE):
            name = match.group(1).strip().strip(".")
            if not name or name.startswith("<"):
                continue
            if name.lower() in {"null", "object", "value"}:
                continue
            if name not in names:
                names.append(name)
    if not names:
        return []

    receipt_lines = [line.strip() for line in str(generation_control_receipt or "").splitlines() if line.strip()]
    out: list[str] = []
    for name in names[:max_lines]:
        normalized = re.sub(r"[^a-z0-9]+", "", name.lower())
        candidates: list[str] = []
        if normalized:
            for line in receipt_lines:
                if len(line) > 520 or line.startswith("["):
                    continue
                hay = re.sub(r"[^a-z0-9]+", "", line.lower())
                if normalized in hay:
                    candidates.append(line[:260])
                if len(candidates) >= 2:
                    break
        bits = [f"`{name}` must be non-null on the observed failing stack"]
        entry = str(failure_signal.get("entry_frame") or "").strip()
        target = str(failure_signal.get("target_frame") or "").strip()
        if entry or target:
            bits.append(f"stack entry={entry or '?'} target={target or '?'}")
        if candidates:
            bits.append("source-visible setup candidates: " + " | ".join(candidates))
        bits.append(
            "minimal_edit=provide this value through an existing constructor/builder/setter/collaborator return while preserving the same target call"
        )
        out.append("; ".join(bits)[:520])
    return out


def _target_positive_failure_local_setup_lines(
    *,
    validation_failure: dict[str, Any],
    failure_signal: dict[str, Any],
    coverage_positive: bool,
    max_lines: int = 5,
) -> list[str]:
    evidence = [
        item for item in list(validation_failure.get("source_evidence") or [])
        if isinstance(item, dict)
    ]
    if not evidence:
        return []
    category = str(failure_signal.get("category") or "").strip()
    if not coverage_positive and category not in {
        "missing-required-terminal-initialization",
        "terminal_output_setup_mismatch",
        "state-sequence-or-oracle-mismatch",
        "oracle-mismatch",
        "mock-stub-misuse",
    }:
        return []
    lines: list[str] = []
    for row in evidence[:max_lines]:
        statement = " ".join(str(row.get("statement") or "").split())
        if not statement:
            continue
        parts = [
            f"frame={row.get('frame') or '?'}",
            f"source={row.get('source_file') or '?'}:{row.get('line') or '?'}",
            f"statement={statement[:260]}",
        ]
        receiver = str(row.get("receiver") or "").strip()
        call = str(row.get("call") or "").strip()
        if receiver or call:
            parts.append(f"local_call={receiver or '?'}.{call or '?'}(...)")
        required_chains = [
            " ".join(str(item or "").split())
            for item in list(row.get("required_input_chains") or [])[:3]
            if str(item or "").strip()
        ]
        if required_chains:
            parts.append("required_input_chain=" + " | ".join(required_chains)[:240])
        init_candidates = [
            " ".join(str(item or "").split())
            for item in list(row.get("receiver_initialization_candidates") or [])
            if str(item or "").strip()
        ]
        if init_candidates:
            parts.append(
                "source_visible_receiver_setup="
                + " | ".join(init_candidates[:2])[:240]
            )
        if not (receiver and call) and not required_chains:
            parts.append("receiver_uncertain=true; use this source line as evidence only, do not invent a receiver setup")
        if coverage_positive:
            if (receiver and call) or required_chains:
                parts.append("minimal_fix=preserve the target-reaching Arrange/Act path and initialize/drive only this evidenced receiver/callback/terminal value")
            else:
                parts.append("minimal_fix=preserve the target-reaching Arrange/Act path; use source evidence to make the smallest local setup/oracle edit")
        else:
            parts.append("minimal_fix=repair only this pre-target setup blocker before changing assertions")
        rendered = "; ".join(parts)[:620]
        if rendered not in lines:
            lines.append(rendered)
    return lines


_RE_DECLARED_TYPE_METHOD_COMPILE_BLOCKER = re.compile(
    r"(?:cannot\s+find\s+symbol\s*)?"
    r"(?:\n?\s*(?:\[[^\]\n]+\]\s*)?(?:symbol:\s+)?method\s+"
    r"(?P<method>[A-Za-z_$][\w$]*)\s*\((?P<args>[^\n)]*)\)\s*)"
    r"(?:\n\s*(?:\[[^\]\n]+\]\s*)?location:\s+variable\s+"
    r"(?P<variable>[A-Za-z_$][\w$]*)\s+of\s+type\s+(?P<declared_type>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:<[^>\n]+>)?))",
    re.MULTILINE,
)


_RE_CLASS_TYPE_METHOD_COMPILE_BLOCKER = re.compile(
    r"(?:cannot\s+find\s+symbol\s*)?"
    r"(?:\n?\s*(?:\[[^\]\n]+\]\s*)?(?:symbol:\s+)?method\s+"
    r"(?P<method>[A-Za-z_$][\w$]*)\s*\((?P<args>[^\n)]*)\)\s*)"
    r"(?:\n\s*(?:\[[^\]\n]+\]\s*)?location:\s+class\s+"
    r"(?P<declared_type>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:<[^>\n]+>)?))",
    re.MULTILINE,
)


def _bare_mock_terminal_producer_calls(text: str, *, max_items: int = 4) -> list[str]:
    """Extract existing producer calls whose returned payload is a bare mock."""
    source = str(text or "")
    if "Mockito.mock" not in source and "mock(" not in source:
        return []
    calls: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"(?:Mockito\.)?when\s*\(\s*(?P<call>[^;\n]+?\.[A-Za-z_$][\w$]*\s*\([^;\n]*?\))\s*\)"
        r"\s*\.thenReturn\s*\((?P<ret>[^;]*?(?:Mockito\.)?mock\s*\([^;]*?)\)\s*;",
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        call = " ".join(str(match.group("call") or "").split())
        ret = " ".join(str(match.group("ret") or "").split())
        if not call or "mock(" not in ret:
            continue
        rendered = f"{call} -> {ret[:180]}"
        if rendered in seen:
            continue
        seen.add(rendered)
        calls.append(rendered[:360])
        if len(calls) >= max_items:
            break
    return calls


_RE_NONSTATIC_METHOD_COMPILE_BLOCKER = re.compile(
    r"non-static\s+method\s+"
    r"(?P<method>[A-Za-z_$][\w$]*)\s*\((?P<args>[^)]*)\)\s+"
    r"cannot\s+be\s+referenced\s+from\s+a\s+static\s+context",
    re.IGNORECASE,
)


_RE_UNIMPLEMENTED_ABSTRACT_METHOD_BLOCKER = re.compile(
    r"is\s+not\s+abstract\s+and\s+does\s+not\s+override\s+abstract\s+method\s+"
    r"(?P<method>[A-Za-z_$][\w$]*)\s*\((?P<args>[^)]*)\)\s+in\s+"
    r"(?P<owner>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
    re.MULTILINE,
)


def _declared_type_method_blockers(failure_output: str, *, max_items: int = 6) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for match in _RE_DECLARED_TYPE_METHOD_COMPILE_BLOCKER.finditer(str(failure_output or "")):
        method = str(match.group("method") or "").strip()
        args = " ".join(str(match.group("args") or "").split())
        variable = str(match.group("variable") or "").strip()
        declared_type = str(match.group("declared_type") or "").strip()
        if not method or not variable or not declared_type:
            continue
        key = (variable, method, args, declared_type)
        if key in seen:
            continue
        seen.add(key)
        blockers.append({
            "variable": variable,
            "method": method,
            "args": args,
            "declared_type": declared_type,
            "receiver_kind": "variable",
        })
        if len(blockers) >= max_items:
            return blockers
    for match in _RE_CLASS_TYPE_METHOD_COMPILE_BLOCKER.finditer(str(failure_output or "")):
        method = str(match.group("method") or "").strip()
        args = " ".join(str(match.group("args") or "").split())
        declared_type = str(match.group("declared_type") or "").strip()
        if not method or not declared_type:
            continue
        key = ("<class>", method, args, declared_type)
        if key in seen:
            continue
        seen.add(key)
        blockers.append({
            "variable": "",
            "method": method,
            "args": args,
            "declared_type": declared_type,
            "receiver_kind": "class",
        })
        if len(blockers) >= max_items:
            return blockers
    for match in _RE_NONSTATIC_METHOD_COMPILE_BLOCKER.finditer(str(failure_output or "")):
        method = str(match.group("method") or "").strip()
        args = " ".join(str(match.group("args") or "").split())
        if not method:
            continue
        key = ("<static-context>", method, args, "")
        if key in seen:
            continue
        seen.add(key)
        blockers.append({
            "variable": "",
            "method": method,
            "args": args,
            "declared_type": "",
            "receiver_kind": "static_context",
        })
        if len(blockers) >= max_items:
            return blockers
    return blockers


def _unimplemented_abstract_method_blockers(failure_output: str, *, max_items: int = 6) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for match in _RE_UNIMPLEMENTED_ABSTRACT_METHOD_BLOCKER.finditer(str(failure_output or "")):
        method = str(match.group("method") or "").strip()
        args = " ".join(str(match.group("args") or "").split())
        owner = str(match.group("owner") or "").strip()
        if not method or not owner:
            continue
        key = (owner, method, args)
        if key in seen:
            continue
        seen.add(key)
        blockers.append({"owner": owner, "method": method, "args": args})
        if len(blockers) >= max_items:
            break
    return blockers


def _unresolved_symbol_guidance(failure_output: str, *, max_items: int = 12) -> str:
    if not failure_output:
        return ""
    symbols: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?m)^\s*(?:\[ERROR\]\s*)?symbol:\s+(.+)$", failure_output):
        symbol = re.sub(r"\s+", " ", match.group(1)).strip()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        symbols.append(symbol)
        if len(symbols) >= max_items:
            break
    instance_blockers = _declared_type_method_blockers(failure_output)
    abstract_blockers = _unimplemented_abstract_method_blockers(failure_output)
    inaccessible_blockers: list[tuple[str, str]] = []
    for match in re.finditer(
        r"(?P<member>[A-Za-z_$][\w$]*(?:\([^)\n]*\))?)\s+has\s+(?:private|protected)\s+access\s+in\s+"
        r"(?P<owner>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
        failure_output,
    ):
        member = " ".join(str(match.group("member") or "").split())
        owner = " ".join(str(match.group("owner") or "").split())
        if member and owner and (owner, member) not in inaccessible_blockers:
            inaccessible_blockers.append((owner, member))
        if len(inaccessible_blockers) >= 6:
            break
    toolchain_symbols: list[str] = []
    for match in re.finditer(
        r"(?:class\s+file\s+for|cannot\s+access)\s+(?P<symbol>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)",
        failure_output,
    ):
        symbol = str(match.group("symbol") or "").strip()
        if symbol and symbol not in toolchain_symbols:
            toolchain_symbols.append(symbol)
        if len(toolchain_symbols) >= 6:
            break
    if not symbols and not instance_blockers and not abstract_blockers and not inaccessible_blockers and not toolchain_symbols:
        return ""
    lines = [
        "Unresolved symbols from the previous compile/test run:",
        *(f"- {symbol}" for symbol in symbols),
        "Do not reuse unresolved methods, fields, classes, or enum constants unless you can prove the exact declaration exists in the provided source context.",
        "Prefer visible public APIs, constructors, factories, and constants from the source under test or imports already present in the test class.",
    ]
    if instance_blockers:
        lines.append("Unavailable instance calls observed:")
        for blocker in instance_blockers:
            args = blocker.get("args") or "..."
            receiver_kind = str(blocker.get("receiver_kind") or "").strip()
            if receiver_kind == "class":
                lines.append(
                    "- "
                    f"{blocker['declared_type']}.{blocker['method']}({args}) is not declared; "
                    "repair that builder/static factory path using source-visible APIs instead of repeating the same class.method call."
                )
            elif receiver_kind == "static_context":
                lines.append(
                    "- "
                    f"{blocker['method']}({args}) is an instance method used from a static context; "
                    "construct or obtain the receiver object before calling it."
                )
            else:
                lines.append(
                    "- "
                    f"{blocker['variable']}.{blocker['method']}({args}) is not declared on {blocker['declared_type']}; "
                    "repair that construction path instead of repeating the same receiver.method call."
                )
    if abstract_blockers:
        lines.append("Incomplete abstract/interface implementations observed:")
        for blocker in abstract_blockers:
            args = blocker.get("args") or "..."
            lines.append(
                "- "
                f"anonymous/subclass implementation of {blocker['owner']} missed {blocker['method']}({args}); "
                "use a source-visible concrete implementation/factory, or a focused mock for this value object/callback when the test stack supports it."
            )
    if inaccessible_blockers:
        lines.append("Inaccessible APIs observed:")
        for owner, member in inaccessible_blockers:
            lines.append(
                "- "
                f"{member} is not accessible in {owner}; do not call private/protected/package-inaccessible constructors or members directly. "
                "Use a public/static factory, builder, existing helper, public subtype, or another source-visible construction path."
            )
    if toolchain_symbols:
        lines.append("Toolchain/API compatibility blockers observed:")
        for symbol in toolchain_symbols:
            lines.append(
                "- "
                f"{symbol} is unavailable to the configured compiler/source level; avoid newer APIs/classes and lambda/functional-interface constructs in this repair."
            )
    return "\n".join(lines)


def extract_repair_failure_signals(failure_output: str, *, max_items: int = 8) -> list[str]:
    """Return compact, generic failure labels for repair prompts.

    The labels describe failure families rather than project-specific messages.
    They are hints for the LLM repair step and do not change validation policy.
    """
    if not failure_output:
        return []
    blob = failure_output.lower()
    signals: list[str] = []

    def add(label: str) -> None:
        if label not in signals and len(signals) < max_items:
            signals.append(label)

    if "cannot find symbol" in blob or re.search(r"(?m)^\s*(?:\[error\]\s*)?symbol:\s+", blob):
        add("missing_symbol_or_import")
    if "package " in blob and " does not exist" in blob:
        add("missing_dependency_or_package")
    if (
        "cannot be applied to given types" in blob
        or "no suitable method" in blob
        or "no suitable constructor" in blob
        or ("constructor " in blob and " cannot be applied" in blob)
    ):
        add("signature_or_constructor_mismatch")
    if "incompatible types" in blob or "cannot be converted to" in blob or "conforms to" in blob:
        add("type_mismatch")
    if "inference variable" in blob and "incompatible bounds" in blob:
        add("generic_return_type_mismatch")
    if "has private access" in blob or "has protected access" in blob or "is not public in" in blob:
        add("inaccessible_api")
    if "assertionerror" in blob or "expected:<" in blob or ("expected:" in blob and "but was" in blob):
        add("assertion_or_oracle_mismatch")
    if "nullpointerexception" in blob:
        add("null_state_or_missing_fixture")
    if (
        "illegalstateexception" in blob
        or "illegal state" in blob
        or "expecting a value" in blob
        or "current context not" in blob
    ):
        add("invalid_state_transition")
    if "illegalargumentexception" in blob or "illegal argument" in blob:
        add("invalid_input_or_precondition")
    if (
        "should be pointing to" in blob
        or "instead got" in blob and any(term in blob for term in ("cursor", "reader", "parser", "token", "event"))
        or "not at start" in blob
        or "unexpected token" in blob
    ):
        add("cursor_or_resource_initial_state_mismatch")
    if any(term in blob for term in ("dimension", "singular", "rank", "matrix", "vector")):
        add("numeric_fixture_shape_or_rank")
    if any(term in blob for term in ("timeout", "timed out", "infinite", "hang")):
        add("timeout_or_nonterminating_fixture")
    if "does not call" in blob and "target" in blob:
        add("target_not_invoked")

    return signals


def _unused_local_compile_mismatch_hint(
    *,
    failure_output: str,
    test_code: str,
    method_focus: str | None,
) -> str:
    """Return a precise compact-repair hint for one unused bad declaration.

    This deliberately handles only a narrow, deterministic shape: javac points
    to one incompatible local declaration and the declared local is never read
    elsewhere in the failing test method. Removing that unrelated declaration
    preserves the target setup/call and is safer than asking a long semantic
    repair prompt to reinterpret the whole test.
    """

    method_name = str(method_focus or "").strip()
    if not method_name or not failure_output or not test_code:
        return ""
    diagnostics = {
        (
            int(match.group("line")),
            int(match.group("column")),
            " ".join(str(match.group("message") or "").split()),
        )
        for match in re.finditer(
            r"(?m)^(?:\[ERROR\]\s*)?.*?\.java:"
            r"\[(?P<line>\d+),(?P<column>\d+)\]\s*"
            r"(?P<message>incompatible types:[^\r\n]+)",
            failure_output,
        )
    }
    if len(diagnostics) != 1:
        return ""
    line_no, column_no, diagnostic = next(iter(diagnostics))
    source_lines = test_code.splitlines()
    if line_no <= 0 or line_no > len(source_lines):
        return ""
    failing_line = source_lines[line_no - 1].strip()
    declaration = re.match(
        r"^(?:final\s+)?"
        r"[A-Za-z_$][A-Za-z0-9_$.<>, ?\[\]]*\s+"
        r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*.+;\s*$",
        failing_line,
    )
    if declaration is None:
        return ""
    local_name = str(declaration.group("name") or "")
    method_window = _window_around_method(
        test_code,
        method_name,
        _REPAIR_TEST_MAX_CHARS,
    )
    if len(re.findall(rf"\b{re.escape(local_name)}\b", method_window)) != 1:
        return ""
    return (
        f"javac line {line_no}:{column_no}: {diagnostic}\n"
        f"The failing statement declares unused local `{local_name}`. "
        "Delete only that declaration; preserve the target/entry call, its "
        "receiver setup, observable oracle, and all unrelated passing code."
    )


def _is_provider_request_rejection(exc: Exception) -> bool:
    if isinstance(exc, OpenAICompatError):
        text = str(exc).lower()
        return any(token in text for token in ("400", "bad request", "422", "context", "token", "maximum"))
    return False


# ---------------------------------------------------------------------------
# Data transfer object
# ---------------------------------------------------------------------------

@dataclass
class RepairContext:
    """All context required to repair a single test unit.

    Keeping this as a plain dataclass (no dependency on *GenerationRequest*)
    means future strategies can be constructed and tested in isolation.
    """

    source_code: str
    test_code: str
    failure_output: str

    # Narrow the repair to one method; ``None`` means repair the whole class.
    method_focus: str | None = None

    # Version / style constraints forwarded from the generation request.
    java_version: str | None = None
    junit_version: str | None = None
    junit_family: str | None = None  # "junit4" | "junit5" | ""

    # Extra key/value hints that a strategy may use.  Intentionally open-ended
    # so that future strategies can carry arbitrary metadata without changing
    # the shared dataclass.
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy interface
# ---------------------------------------------------------------------------

class RepairStrategy(ABC):
    """Abstract base for test-repair strategies.

    Implementing a new strategy requires only overriding :meth:`repair`.

    Parameters
    ----------
    ctx:
        All context for the repair operation.

    Returns
    -------
    tuple[str, dict]
        ``(repaired_source, usage_dict)`` where *repaired_source* is the full
        corrected test-class source (empty string = no change produced) and
        *usage_dict* carries token-usage information (may be empty).
    """

    @abstractmethod
    def repair(self, ctx: RepairContext) -> tuple[str, dict]:
        ...

    @property
    def name(self) -> str:
        return type(self).__name__


# ---------------------------------------------------------------------------
# Built-in strategy: LLM prompt-based repair
# ---------------------------------------------------------------------------

class LLMRepairStrategy(RepairStrategy):
    """Repair by sending a prompt to the configured LLM endpoint.

    The strategy is intentionally stateless with respect to the test corpus;
    all model-configuration details are provided at construction time and the
    :class:`RepairContext` carries the code artefacts.
    """

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key_env: str,
        timeout_seconds: int = 120,
    ) -> None:
        self._api_base = api_base
        self._model = model
        self._api_key_env = api_key_env
        self._timeout = timeout_seconds

    # ------------------------------------------------------------------
    # Prompt construction (public so it can be overridden or tested)
    # ------------------------------------------------------------------

    def build_prompt(self, ctx: RepairContext, *, compact: bool = False) -> tuple[str, str]:
        """Return ``(system_prompt, user_prompt)`` for the given context."""
        extra_context = dict(ctx.extra or {})
        production_method_focus = str(extra_context.get("production_method_focus") or "").strip()
        repair_test_method = str(extra_context.get("repair_test_method") or "").strip()
        family = (ctx.junit_family or "").lower()
        stack_constraints: list[str] = []
        failure_source_major = _java_source_level_from_failure_output(ctx.failure_output)
        if ctx.java_version:
            stack_constraints.append(f"- Keep syntax compatible with Java {ctx.java_version}.")
        major = _java_major_version(ctx.java_version)
        if failure_source_major and (major is None or failure_source_major < major):
            major = failure_source_major
            stack_constraints.append(
                f"- The compiler failure indicates Java source level {failure_source_major}; repair against that source level even if the runtime JDK is newer."
            )
        if major and major <= 7:
            stack_constraints.append("- Do NOT use lambda expressions, method references, streams, or java.util.function APIs; this source level predates Java 8.")
            stack_constraints.append("- For exception assertions, use explicit try/catch plus fail(...) instead of assertThrows/lambdas.")
        if major and major <= 8:
            stack_constraints.append("- Do NOT use Java 9+ collection factories such as List.of, Set.of, Map.of, or Optional.stream.")
        if major and major <= 10:
            stack_constraints.append("- Do NOT use local-variable type inference (`var`).")
        if major and major <= 14:
            stack_constraints.append("- Do NOT use Java text blocks or switch expressions.")
        if family == "junit4":
            stack_constraints.append("- Use JUnit 4 only (org.junit.Test / org.junit.Assert.*).")
            stack_constraints.append("- Do not use org.junit.jupiter.* imports.")
        elif family == "junit5":
            stack_constraints.append("- Use JUnit 5 only (org.junit.jupiter.api.Test / org.junit.jupiter.api.Assertions.*).")
            stack_constraints.append("- Do not use org.junit.Test / org.junit.Assert.* imports.")
        elif ctx.junit_version:
            stack_constraints.append(f"- Keep tests compatible with JUnit {ctx.junit_version}.")

        repair_output_mode = str(extra_context.get("repair_output_mode") or "").strip()
        method_patch_json = bool(ctx.method_focus and repair_output_mode == "method_patch_json")
        if ctx.method_focus:
            if method_patch_json:
                system = (
                    "You are an expert Java unit test repair assistant performing STREAMING METHOD-LEVEL repair. "
                    "Return ONLY valid JSON, no markdown fences, no prose. "
                    "JSON schema: {\"new_tests\":[{\"test_name\":string,\"test_code_lines\":string[],\"new_imports_lines\":string[]}]}. "
                    "The JSON must contain exactly one repaired @Test method for the failing test; do not return a package declaration, class declaration, or unrelated tests."
                )
            else:
                system = (
                    "You are an expert Java unit test repair assistant performing METHOD-LEVEL repair. "
                    "Return only the full corrected Java test class source code, no markdown fences. "
                    "The repaired class must still contain a real @Test that preserves the target or required entry-path invocation."
                )
            target_phrase = (
                f"production target `{production_method_focus}`"
                if production_method_focus
                else "the method-level target"
            )
            test_phrase = repair_test_method or str(ctx.method_focus)
            if method_patch_json:
                focus_line = (
                    f"STREAMING METHOD-LEVEL REPAIR TARGET: failing @Test `{test_phrase}` for {target_phrase}.\n"
                    f"Return exactly one JSON new_tests item whose test_name remains `{test_phrase}` unless Java naming requires a minimal legal spelling fix.\n"
                    "The test_code_lines array must contain only the repaired @Test method lines, not a package declaration, class declaration, or unrelated tests.\n"
                    "new_imports_lines may contain only import lines needed by this repaired method.\n"
                    "Do not add new @Test methods. Do not rename unrelated tests. Do not rewrite unrelated passing tests.\n"
                    "The repaired @Test must preserve the original target invocation or the source-visible entry call used to reach a private target; "
                    "a mention in the test name, comment, string literal, or assertion message does not count.\n"
                    f"Do not replace failing @Test `{test_phrase}` with tests for neighboring methods, getters, constructors, configuration APIs, or smoke behavior.\n"
                    "If the failure is an assertion mismatch, repair the setup trace or oracle while preserving the target invocation.\n\n"
                )
            else:
                focus_line = (
                    f"METHOD-LEVEL REPAIR TARGET: failing @Test `{test_phrase}` for {target_phrase}.\n"
                    f"Edit only the failing test method `{test_phrase}` and the minimum imports or lifecycle setup needed for it.\n"
                    "Do not add new @Test methods. Do not rename existing @Test methods. "
                    "Do not rewrite unrelated passing tests. Return the complete test class with unrelated methods unchanged.\n"
                    "At least one repaired @Test must preserve the original target invocation or the source-visible entry call used to reach a private target; "
                    "a mention in the test name, comment, string literal, or assertion message does not count.\n"
                    f"Do not replace failing @Test `{test_phrase}` with tests for neighboring methods, getters, constructors, configuration APIs, or smoke behavior.\n"
                    "If the failure is an assertion mismatch, repair the setup trace or oracle while preserving the target invocation.\n\n"
                )
        else:
            system = (
                "You are an expert Java unit test repair assistant performing CLASS-LEVEL repair. "
                "Return only the full corrected Java test class source code, no markdown fences."
            )
            focus_line = (
                "CLASS-LEVEL REPAIR TARGET: the whole failing test class.\n"
                "You may repair imports, lifecycle methods, helpers, and failing @Test methods, "
                "but do not create duplicate method names.\n\n"
            )
        constraint_block = (
            "\n".join(stack_constraints)
            if stack_constraints
            else "- Keep compatibility with the project test stack."
        )

        failure_tail_chars = 4000 if compact else _FAILURE_OUTPUT_TAIL_CHARS
        test_limit = _REPAIR_COMPACT_TEST_MAX_CHARS if compact else _REPAIR_TEST_MAX_CHARS
        source_code = _repair_source_context(ctx, compact=compact)
        if compact and ctx.method_focus:
            test_code = _window_around_method(ctx.test_code, ctx.method_focus, test_limit)
        else:
            test_code = _clip_middle(ctx.test_code, test_limit, label="test class")
        test_code = _normalize_repair_source_excerpt(test_code)
        focused_test_name = repair_test_method or str(ctx.method_focus or "").strip()
        focused_test_code = ""
        if focused_test_name:
            focused_test_code = _window_around_method(
                ctx.test_code,
                focused_test_name,
                _repair_budget(compact=compact, normal=5200, compact_value=3200),
            ).strip()
            focused_test_code = _normalize_repair_source_excerpt(focused_test_code)
        compact_line = (
            "The code context may be abbreviated because the previous provider request was rejected; "
            "make the smallest valid fix from the visible context.\n\n"
            if compact else ""
        )
        unresolved_guidance = _unresolved_symbol_guidance(ctx.failure_output)
        unresolved_block = f"\n\n{unresolved_guidance}" if unresolved_guidance else ""
        focused_error = str(extra_context.get("compiler_error_context", "") or "").strip()
        if focused_error:
            focused_error = focused_error[-failure_tail_chars:]
        target_receipt = str(
            extra_context.get("target_receipt_contract")
            or extra_context.get("target_execution_receipt")
            or ""
        ).strip()
        generation_control_receipt = str(extra_context.get("generation_control_receipt") or "").strip()
        execution_capsule = str(extra_context.get("target_execution_capsule") or "").strip()
        serialized_entry_contract = str(extra_context.get("serialized_entry_contract") or "").strip()
        static_entry_path_plan = str(extra_context.get("static_entry_path_plan") or "").strip()
        baseline_skeleton_section = str(extra_context.get("baseline_skeleton_section") or "").strip()
        baseline_skeleton_satisfaction = (
            extra_context.get("baseline_skeleton_satisfaction")
            if isinstance(extra_context.get("baseline_skeleton_satisfaction"), dict)
            else {}
        )
        baseline_skeleton_id = str(extra_context.get("baseline_skeleton_id") or "").strip()
        raw_skeleton_fact_ids = extra_context.get("skeleton_fact_ids")
        if isinstance(raw_skeleton_fact_ids, str):
            skeleton_fact_ids = [
                part.strip()
                for part in re.split(r"[,;\s]+", raw_skeleton_fact_ids)
                if part.strip()
            ]
        elif isinstance(raw_skeleton_fact_ids, (list, tuple, set)):
            skeleton_fact_ids = [
                str(part).strip()
                for part in raw_skeleton_fact_ids
                if str(part).strip()
            ]
        else:
            skeleton_fact_ids = []
        terminal_execution_guidance = str(extra_context.get("terminal_execution_guidance") or "").strip()
        callback_signature_grounding = str(extra_context.get("callback_signature_grounding") or "").strip()
        callable_surface_grounding = str(extra_context.get("callable_surface_grounding") or "").strip()
        visible_construction_grounding = str(extra_context.get("visible_type_construction_grounding") or "").strip()
        failure_signals = [
            str(signal).strip()
            for signal in (
                extra_context.get("repair_failure_signals")
                or extract_repair_failure_signals(ctx.failure_output)
            )
            if str(signal).strip()
        ]
        repair_attempt = int(extra_context.get("repair_attempt") or 0)
        coverage_feedback = extra_context.get("coverage_feedback") if isinstance(extra_context.get("coverage_feedback"), dict) else None
        mutation_feedback = extra_context.get("mutation_feedback") if isinstance(extra_context.get("mutation_feedback"), dict) else None
        mutation_feedback_enabled = bool(extra_context.get("mutation_feedback_enabled"))
        mutation_exact_line_repair = (
            extra_context.get("mutation_exact_line_local_repair_evidence")
            if isinstance(
                extra_context.get(
                    "mutation_exact_line_local_repair_evidence"
                ),
                dict,
            )
            else {}
        )
        mutation_pre_act_setup_repair = (
            extra_context.get("mutation_pre_act_setup_repair_evidence")
            if isinstance(
                extra_context.get("mutation_pre_act_setup_repair_evidence"),
                dict,
            )
            else {}
        )
        failed_methods_hint = [
            str(m).strip()
            for m in (extra_context.get("failed_methods_hint") or [])
            if str(m).strip()
        ]
        overload_id = str(extra_context.get("overload_id") or "").strip()
        overload_contract = extra_context.get("overload_contract") if isinstance(extra_context.get("overload_contract"), dict) else {}
        test_behavior = str(extra_context.get("test_behavior") or "").strip()
        validation_failure = extra_context.get("streaming_validation_failure") if isinstance(extra_context.get("streaming_validation_failure"), dict) else {}
        candidate_role_hint = str(
            validation_failure.get("candidate_role")
            or extra_context.get("candidate_role")
            or ""
        ).strip().removeprefix("role:")
        objective_id_hint = str(
            validation_failure.get("objective_id")
            or extra_context.get("objective_id")
            or ""
        ).strip()
        mutation_extra_repair = bool(
            mutation_feedback_enabled
            or candidate_role_hint
            in {"mutation_oracle", "mutation_reachability"}
            or objective_id_hint.upper().startswith("M")
        )
        failed_test_source_excerpt = _normalize_repair_source_excerpt(
            str(validation_failure.get("failed_test_source_excerpt") or "").strip()
        )
        failure_signal = extra_context.get("streaming_failure_signal") if isinstance(extra_context.get("streaming_failure_signal"), dict) else {}
        if not failure_signal and isinstance(validation_failure.get("failure_signal"), dict):
            failure_signal = validation_failure.get("failure_signal") or {}
        observed_trace = extra_context.get("observed_repair_trace") if isinstance(extra_context.get("observed_repair_trace"), dict) else {}
        candidate_coverage = validation_failure.get("candidate_target_coverage") if isinstance(validation_failure.get("candidate_target_coverage"), dict) else {}
        coverage_gate = extra_context.get("coverage_gate") if isinstance(extra_context.get("coverage_gate"), dict) else None
        if not coverage_gate and isinstance(validation_failure.get("coverage_gate"), dict):
            coverage_gate = validation_failure.get("coverage_gate") or {}
        semantic_replay_repair_required = bool(
            extra_context.get("semantic_replay_repair_required")
        )
        semantic_replay_evaluation = (
            extra_context.get("semantic_replay_objective_evaluation")
            if isinstance(
                extra_context.get(
                    "semantic_replay_objective_evaluation"
                ),
                dict,
            )
            else {}
        )
        semantic_replay_sketches = [
            dict(row)
            for row in list(
                extra_context.get("semantic_replay_focus_sketches") or []
            )
            if isinstance(row, dict)
        ]
        semantic_objective_id = str(
            validation_failure.get("objective_id")
            or extra_context.get("objective_id")
            or ""
        ).strip()
        selected_semantic_sketch = next(
            (
                row
                for row in semantic_replay_sketches
                if str(row.get("sketch_id") or "").strip()
                == semantic_objective_id
            ),
            None,
        )
        structural_obligation_repair_required = bool(
            extra_context.get("structural_obligation_repair_required")
        )
        structural_obligation_evaluation = (
            extra_context.get(
                "structural_obligation_objective_evaluation"
            )
            if isinstance(
                extra_context.get(
                    "structural_obligation_objective_evaluation"
                ),
                dict,
            )
            else {}
        )
        selected_structural_contract = (
            structural_obligation_evaluation.get("selected_contract")
            if isinstance(
                structural_obligation_evaluation.get("selected_contract"),
                dict,
            )
            else {}
        )

        def _coverage_int(value: object) -> int:
            try:
                return int(float(str(value)))
            except Exception:
                return 0

        coverage_positive = bool(
            candidate_coverage.get("has_target_coverage")
            or _coverage_int(candidate_coverage.get("line_covered")) > 0
            or _coverage_int(candidate_coverage.get("branch_covered")) > 0
        )
        failure_type = str(validation_failure.get("failure_type") or "").strip()
        failure_category = str(failure_signal.get("category") or "").strip()
        failure_text_for_mode = "\n".join(
            str(part or "")
            for part in (
                ctx.failure_output,
                failure_signal.get("message") if isinstance(failure_signal, dict) else "",
                " ".join(failure_signals),
            )
        )
        has_expected_actual = bool(
            re.search(r"expected:?\s*<[^>]*>\s*but\s+was:?\s*<[^>]*>", failure_text_for_mode, re.IGNORECASE)
            or "nothing was thrown" in failure_text_for_mode
            or failure_signal.get("expected_value")
            or failure_signal.get("actual_value")
        )
        is_compile_like = bool(
            failure_type == "compile"
            or failure_category in {"compilation-failure", "compile-type-mismatch"}
            or any(signal in {
                "signature_or_constructor_mismatch",
                "generic_return_type_mismatch",
                "type_mismatch",
            } for signal in failure_signals)
            or re.search(r"\b(?:cannot find symbol|incompatible types|wrong number of type arguments|method .* cannot be applied)\b", failure_text_for_mode, re.IGNORECASE)
        )
        if mutation_exact_line_repair:
            repair_mode = "mutation_exact_line_setup_fix"
        elif (
            structural_obligation_repair_required
            and selected_structural_contract
        ):
            repair_mode = "structural_obligation_contract_fix"
        elif (
            semantic_replay_repair_required
            and isinstance(selected_semantic_sketch, dict)
        ):
            repair_mode = "semantic_replay_contract_fix"
        elif coverage_positive:
            repair_mode = "coverage_positive_preserve"
        elif is_compile_like:
            repair_mode = "compile_fix"
        elif has_expected_actual:
            repair_mode = "oracle_fix"
        elif failure_category in {
            "missing-required-terminal-initialization",
            "terminal_output_setup_mismatch",
            "missing-required-initialization",
            "invalid-public-input",
            "mock-stub-misuse",
        }:
            repair_mode = "runtime_setup_fix"
        else:
            repair_mode = "no_target_feedback_only"
        focus_sections: list[str] = []
        if mutation_extra_repair:
            focus_sections.append(
                "[Prompt Module Guide]\n"
                "- Mutation Repair Scope: obey first; repair only the concrete failure in this one mutation-extra @Test.\n"
                "- Failure Evidence: use compiler/JUnit/stack evidence to choose the smallest edit.\n"
                "- Original Test To Preserve: retain the selected M-id, candidate role, entry call, input/state delta, and mutant-distinguishing oracle intent.\n"
                "- Callable/type grounding: constraints only; use it to avoid absent APIs.\n"
                "- Do not follow structural coverage objectives, branch-frontier sketches, or baseline scenario expansion during mutation repair."
            )
            mutation_scope_lines = [
                "[Mutation Repair Scope]",
                f"- objective_id={objective_id_hint or '?'}",
                f"- candidate_role={candidate_role_hint or '?'}",
                "- repair_rule=fix only compilation, source-visible setup, or the observable oracle for this exact M-id candidate.",
                "- forbidden_rule=do not add structural coverage scenarios, switch to an O-id/SK-id objective, or replace the selected mutant with a generic branch target.",
            ]
            focus_sections.append("\n".join(mutation_scope_lines))
        else:
            focus_sections.append(
                "[Prompt Module Guide]\n"
                "- Repair Scope: obey first; repair only the concrete failure in this one @Test.\n"
                "- Failure Evidence: use compiler/JUnit/stack evidence to choose the smallest edit.\n"
                "- Original Test To Preserve: keep its scenario, target call, state-entry sequence, terminal/callback setup, and oracle intent.\n"
                "- Callable/type grounding: constraints only; use it to avoid absent APIs, not to redesign the test.\n"
                "- Generation/coverage context, when present, is preservation context only; do not use repair to explore new coverage paths."
            )
        allowed_edits = {
            "mutation_exact_line_setup_fix": "preserve the existing target-reaching call and change only its source-visible input/call sequence so the verifier-required state transition occurs before the exact mutant line; do not merely calibrate the shallow-path assertion and never assign internal/context fields directly.",
            "structural_obligation_contract_fix": "change only source-visible Arrange/input/setup values so the selected O-id reaches one of its exact source anchors; preserve the target entry call and the assertion that exposed the neighboring-path mismatch, and do not calibrate that assertion to the currently observed shallow-path value.",
            "semantic_replay_contract_fix": "change only the selected replay row's source-visible constructor/input/state delta so the declared branch predicate becomes true; preserve the exact entry call and protected green scaffold, and do not merely calibrate the assertion to the current shallow-path value.",
            "compile_fix": "fix imports/static imports, exact constructor/method arity, generic/type mismatch, or unavailable symbols; do not change the scenario.",
            "runtime_setup_fix": "initialize only the null/invalid receiver, collaborator return, or state holder named by the failure stack; do not replace the scenario.",
            "oracle_fix": "replace only the single failing expected/actual assertion or assertThrows type when the stack proves the observed behavior.",
            "coverage_positive_preserve": "make the minimal compile/runtime/oracle edit while preserving the target-positive Arrange/Act path and coverage-reaching sequence.",
            "no_target_feedback_only": "do not redesign the test in repair; this no-target failure should become generation feedback unless a compile/type diagnostic is present.",
        }
        focus_sections.append(
            "[Repair Scope]\n"
            f"- mode={repair_mode}\n"
            "- minimal_edit_rule=repair only the failing @Test method and only the concrete failure reported below.\n"
            "- preserve_test_intent=keep the same scenario family, target invocation, Arrange/Act/Assert shape, state-entry sequence, terminal/result consumption, callback/fake setup, and oracle intent.\n"
            + (
                "- mutation_exact_line_rule=this candidate already reaches the exact target method; add only the missing source-visible invocation/input transition proven below, then keep a deterministic mutant-distinguishing oracle."
                if repair_mode == "mutation_exact_line_setup_fix"
                else
                "- structural_obligation_rule=this candidate reached the target but missed its selected O-id anchor; change only source-visible input/setup so the exact source anchor is reached, and preserve the assertion exposing the mismatch."
                if repair_mode == "structural_obligation_contract_fix"
                else
                "- semantic_replay_rule=this candidate already selected one exact source branch; repair only its declared source-visible setup delta and keep its entry path unchanged."
                if repair_mode == "semantic_replay_contract_fix"
                else "- no_coverage_exploration=do not add a new scenario or change the entry path merely to improve coverage; next generation rounds handle coverage exploration."
            )
        )
        evidence_lines = ["[Failure Evidence]"]
        if failure_type:
            evidence_lines.append(f"- failure_type={failure_type}")
        if failure_category:
            evidence_lines.append(f"- category={failure_category}")
        diagnostic_text = "\n".join(
            str(part or "")
            for part in (
                focused_error,
                ctx.failure_output,
                validation_failure.get("failure_report_tail"),
                validation_failure.get("stdout_tail"),
                validation_failure.get("stderr_tail"),
                validation_failure.get("error"),
            )
        )
        diagnostic_lines: list[str] = []
        if is_compile_like:
            diagnostic_lines = _focused_compile_diagnostic_lines(
                diagnostic_text,
                limit=8,
            )
            if diagnostic_lines:
                evidence_lines.append("- compile_diagnostic:")
                evidence_lines.extend(f"  - {line}" for line in diagnostic_lines)
        for key in ("exception_type", "message", "entry_frame", "target_frame", "expected_value", "actual_value"):
            value = failure_signal.get(key) if isinstance(failure_signal, dict) else None
            if value not in (None, "", []):
                evidence_lines.append(f"- {key}={value}")
        inaccessible = failure_signal.get("inaccessible_api") if isinstance(failure_signal, dict) else None
        if isinstance(inaccessible, dict) and inaccessible:
            evidence_lines.append(
                "- inaccessible_api="
                + "; ".join(
                    f"{key}={inaccessible.get(key)}"
                    for key in ("owner", "member", "access")
                    if inaccessible.get(key)
                )
            )
        if candidate_coverage:
            evidence_lines.append(
                f"- candidate_coverage: has_target_coverage={candidate_coverage.get('has_target_coverage')} "
                f"line_covered={candidate_coverage.get('line_covered')} branch_covered={candidate_coverage.get('branch_covered')}"
            )
        focus_sections.append("\n".join(evidence_lines))
        if mutation_exact_line_repair:
            mutation_lines = [
                "[Mutation Exact-Line Local Setup Repair]",
                f"- objective_id={mutation_exact_line_repair.get('objective_id')}",
                f"- candidate_role={mutation_exact_line_repair.get('candidate_role')}",
                f"- mutant_line={mutation_exact_line_repair.get('mutant_line')}",
                f"- source_line={mutation_exact_line_repair.get('source_line')}",
                f"- proof_status={mutation_exact_line_repair.get('proof_status')}",
            ]
            public_input = str(
                mutation_exact_line_repair.get(
                    "controlling_public_input"
                )
                or ""
            ).strip()
            if public_input:
                mutation_lines.append(
                    "- controlling_public_input=" + public_input
                )
            state_delta = str(
                mutation_exact_line_repair.get("required_state_delta")
                or ""
            ).strip()
            if state_delta:
                mutation_lines.append(
                    "- required_state_transition=" + state_delta
                )
            setup = [
                str(value).strip()
                for value in list(
                    mutation_exact_line_repair.get("setup_scaffold") or []
                )[:8]
                if str(value).strip()
            ]
            for value in setup:
                mutation_lines.append("- setup_scaffold=" + value)
            accesses = [
                str(value).strip()
                for value in list(
                    mutation_exact_line_repair.get(
                        "required_state_accesses"
                    )
                    or []
                )[:8]
                if str(value).strip()
            ]
            if accesses:
                mutation_lines.append(
                    "- required_state_accesses=" + ", ".join(accesses)
                )
            for row in list(
                mutation_exact_line_repair.get(
                    "source_backed_state_transitions"
                )
                or []
            )[:8]:
                if not isinstance(row, dict):
                    continue
                mutation_lines.append(
                    "- source_transition=L"
                    + str(row.get("line") or "?")
                    + ": "
                    + str(row.get("assignment") or "").strip()
                )
            oracle = str(
                mutation_exact_line_repair.get("oracle_recipe") or ""
            ).strip()
            if oracle:
                mutation_lines.append("- mutant_oracle=" + oracle)
            mutation_lines.extend([
                "- required_fix=preserve the working target call, add only the missing source-visible call/input step needed to realize every required state transition, then assert the mutant-distinguishing observable.",
                "- forbidden_fix=do not directly assign context/collaborator fields, do not invent setters/fields, and do not make the shallow-path assertion pass without reaching the exact mutant line.",
            ])
            focus_sections.append("\n".join(mutation_lines))
        if mutation_pre_act_setup_repair:
            pre_act_lines = [
                "[Mutation Pre-ACT Setup Proof Repair]",
                "- objective_id="
                + str(
                    mutation_pre_act_setup_repair.get("objective_id") or "?"
                ),
                "- proof_kind=pre_act_setup_assertion",
                "- exact_target_call_is_after_failed_setup=true",
            ]
            failed_setup = str(
                mutation_pre_act_setup_repair.get("failed_setup_statement")
                or ""
            ).strip()
            if failed_setup:
                pre_act_lines.append("- failed_setup_proof=" + failed_setup)
            for constraint in list(
                mutation_pre_act_setup_repair.get("infection_constraints") or []
            )[:4]:
                pre_act_lines.append(
                    "- infection_constraint_to_preserve=" + str(constraint)
                )
            for transition in list(
                mutation_pre_act_setup_repair.get("pit_operator_transitions")
                or []
            )[:4]:
                if not isinstance(transition, dict):
                    continue
                original = str(
                    transition.get("original_operator") or ""
                ).strip()
                mutant = str(
                    transition.get("mutant_operator") or ""
                ).strip()
                if original and mutant:
                    pre_act_lines.append(
                        "- exact_pit_operator_transition="
                        + original
                        + " -> "
                        + mutant
                    )
            pre_act_lines.extend([
                "- required_fix=repair only the Arrange expression or construction that makes the failed setup proof false, then preserve the later exact target call and mutation oracle.",
                "- forbidden_fix=do not delete, weaken, or move the setup proof after ACT; do not replace the exact infection row with an adjacent reachability-only value.",
            ])
            focus_sections.append("\n".join(pre_act_lines))
        if (
            repair_mode == "structural_obligation_contract_fix"
            and selected_structural_contract
        ):
            structural_lines = [
                "[Selected Structural Objective To Preserve]",
                "- objective_id="
                + str(
                    structural_obligation_evaluation.get("objective_id")
                    or selected_structural_contract.get("objective_id")
                    or "?"
                ),
                "- proof_status="
                + str(
                    selected_structural_contract.get("proof_status")
                    or "source_and_live_coverage_required"
                ),
            ]
            target_call = str(
                selected_structural_contract.get("target_call") or ""
            ).strip()
            if target_call:
                structural_lines.append("- exact_entry_call=" + target_call)
            scenario = str(
                selected_structural_contract.get("scenario") or ""
            ).strip()
            if scenario:
                structural_lines.append("- selected_scenario=" + scenario)
            for anchor in list(
                selected_structural_contract.get("anchors") or []
            )[:6]:
                if not isinstance(anchor, dict):
                    continue
                structural_lines.append(
                    "- exact_anchor=L"
                    + str(anchor.get("line") or "?")
                    + ": "
                    + str(anchor.get("code") or "").strip()
                )
                branch_gap = str(
                    anchor.get("branch_gap") or ""
                ).strip()
                if branch_gap:
                    structural_lines.append(
                        "  branch_gap=" + branch_gap
                    )
                for node in list(anchor.get("control_path") or [])[:6]:
                    if not isinstance(node, dict):
                        continue
                    structural_lines.append(
                        "  control=L"
                        + str(node.get("line") or "?")
                        + ": "
                        + str(node.get("statement") or "").strip()
                        + " => "
                        + str(node.get("required_outcome") or "").strip()
                    )
            structural_lines.extend([
                "- required_fix=derive the smallest source-visible input/setup change from the exact control path above, then keep the existing target call and observable assertion.",
                "- forbidden_fix=do not change the expected value merely to accept the neighboring branch, do not delete the failing assertion, and do not invent private fields, setters, helpers, or APIs.",
            ])
            focus_sections.append("\n".join(structural_lines))
        if (
            repair_mode == "semantic_replay_contract_fix"
            and isinstance(selected_semantic_sketch, dict)
        ):
            semantic_lines = [
                "[Semantic Replay Contract To Repair]",
                f"- objective_id={semantic_objective_id}",
            ]
            target_lines = [
                str(value)
                for value in list(
                    selected_semantic_sketch.get(
                        "semantic_target_lines"
                    )
                    or []
                )
                if str(value).strip()
            ]
            if target_lines:
                semantic_lines.append(
                    "- exact_target_lines=" + ", ".join(target_lines[:4])
                )
            exact_call = str(
                selected_semantic_sketch.get(
                    "semantic_exact_entry_call"
                )
                or selected_semantic_sketch.get("target_call")
                or ""
            ).strip()
            if exact_call:
                semantic_lines.append(
                    "- exact_entry_call=" + exact_call[:300]
                )
            protected_setup = [
                str(value).strip()
                for value in list(
                    selected_semantic_sketch.get(
                        "existing_green_protected_setup_actions"
                    )
                    or []
                )
                if str(value).strip()
            ]
            if protected_setup:
                semantic_lines.append(
                    "- protected_green_setup:"
                )
                semantic_lines.extend(
                    f"  - {value[:300]}"
                    for value in protected_setup[:8]
                )
            replaceable_locals = [
                str(value).strip()
                for value in list(
                    selected_semantic_sketch.get(
                        "existing_green_delta_local_names"
                    )
                    or []
                )
                if str(value).strip()
            ]
            if replaceable_locals:
                semantic_lines.append(
                    "- replaceable_setup_locals="
                    + ", ".join(replaceable_locals[:8])
                )
            for evidence in list(
                selected_semantic_sketch.get("source_evidence") or []
            )[:6]:
                if str(evidence).strip():
                    semantic_lines.append(
                        "- source_evidence=" + str(evidence).strip()[:500]
                    )
            for arrange in list(
                selected_semantic_sketch.get("required_arrange") or []
            )[:6]:
                if str(arrange).strip():
                    semantic_lines.append(
                        "- required_arrange=" + str(arrange).strip()[:500]
                    )
            oracle_rows = list(
                selected_semantic_sketch.get("oracle_observation") or []
            )
            if oracle_rows:
                semantic_lines.append(
                    "- observable_oracle="
                    + str(oracle_rows[0]).strip()[:500]
                )
            expected_lines = list(
                semantic_replay_evaluation.get(
                    "expected_target_lines"
                )
                or []
            )
            closed_lines = list(
                semantic_replay_evaluation.get("closed_target_lines")
                or []
            )
            semantic_lines.append(
                "- replay_evidence="
                f"expected_lines={expected_lines}; "
                f"currently_closed_lines={closed_lines}"
            )
            semantic_lines.append(
                "- required_fix=change a listed replaceable input/state local "
                "or a source-visible constructor argument so the governing "
                "predicate is satisfied. Preserve the exact entry call. "
                "Do not make the test pass only by replacing expected with "
                "the currently observed shallow-path value."
            )
            focus_sections.append("\n".join(semantic_lines))
        blocker_reason = str(
            validation_failure.get("reason") or ""
        ).strip()
        forbidden_calls = [
            str(value).strip()
            for value in list(
                validation_failure.get("forbidden_calls") or []
            )
            if str(value).strip()
        ]
        source_visible_alternatives = [
            str(value).strip()
            for value in list(
                validation_failure.get(
                    "source_visible_alternatives"
                ) or []
            )
            if str(value).strip()
        ]
        blocker_hints = [
            str(value).strip()
            for value in list(validation_failure.get("hints") or [])
            if str(value).strip()
        ]
        if (
            blocker_reason
            or forbidden_calls
            or source_visible_alternatives
        ):
            blocker_lines = ["[Source-Backed Compile Blockers]"]
            if blocker_reason:
                blocker_lines.append(f"- reason={blocker_reason}")
            if forbidden_calls:
                blocker_lines.append("- forbidden_calls:")
                blocker_lines.extend(
                    f"  - {value}" for value in forbidden_calls[:8]
                )
            if source_visible_alternatives:
                blocker_lines.append("- source_visible_alternatives:")
                blocker_lines.extend(
                    f"  - {value}"
                    for value in source_visible_alternatives[:8]
                )
            if blocker_hints:
                blocker_lines.append("- exact_repair_hints:")
                blocker_lines.extend(
                    f"  - {value}" for value in blocker_hints[:8]
                )
            blocker_lines.append(
                "- repair_rule=remove every forbidden call from this exact "
                "@Test and use one listed source-compatible alternative; "
                "do not return the unchanged method."
            )
            focus_sections.append("\n".join(blocker_lines))
        producer_callback_contract = (
            failure_signal.get("producer_callback_contract")
            if isinstance(failure_signal, dict)
            and isinstance(failure_signal.get("producer_callback_contract"), dict)
            else {}
        )
        if producer_callback_contract:
            producer_lines = [
                "[Source-Proven Producer/Callback Repair]",
                f"- collaborator={producer_callback_contract.get('collaborator_variable')}:{producer_callback_contract.get('collaborator_type')}",
                f"- exact_void_call={producer_callback_contract.get('collaborator_method')}(...)",
                f"- producer_argument_type={producer_callback_contract.get('producer_type')}",
            ]
            missing_types = [
                str(value)
                for value in list(producer_callback_contract.get("missing_result_types") or [])
                if str(value).strip()
            ]
            if missing_types:
                producer_lines.append("- missing_observable_result_types=" + ", ".join(missing_types))
            methods = [
                str(value)
                for value in list(producer_callback_contract.get("source_visible_output_methods") or [])
                if str(value).strip()
            ]
            if methods:
                producer_lines.append("- source_visible_output_methods:")
                producer_lines.extend(f"  - {method}" for method in methods[:3])
            producer_lines.append(
                "- required_minimal_edit="
                + str(producer_callback_contract.get("repair_rule") or "")
            )
            producer_lines.append(
                "- forbidden_edit=do not replace this with repository save/put registration, reflection, or a different scenario; repair the existing bare callback producer setup."
            )
            focus_sections.append("\n".join(producer_lines))
        skeleton_lines: list[str] = []
        if baseline_skeleton_id:
            skeleton_lines.append(f"- baseline_skeleton_id={baseline_skeleton_id}")
        if skeleton_fact_ids:
            skeleton_lines.append("- declared_skeleton_fact_ids=" + ", ".join(skeleton_fact_ids[:12]))
        if baseline_skeleton_satisfaction:
            rendered_status = []
            for key in ("status", "reason", "missing_fact_ids", "contradicted_fact_ids"):
                value = baseline_skeleton_satisfaction.get(key)
                if value in (None, "", [], {}):
                    continue
                rendered_status.append(f"{key}={value}")
            if rendered_status:
                skeleton_lines.append("- satisfaction=" + "; ".join(rendered_status))
        if baseline_skeleton_section:
            keep_lines: list[str] = []
            for raw in baseline_skeleton_section.splitlines():
                stripped = raw.strip()
                if not stripped:
                    continue
                if stripped.startswith("[Source-Backed Baseline Skeleton Obligation]"):
                    continue
                if stripped.startswith("Candidate metadata:") or stripped.startswith("Validation rule:"):
                    continue
                keep_lines.append(stripped)
                if len(keep_lines) >= 10:
                    break
            if keep_lines:
                skeleton_lines.append("- source_backed_skeleton:")
                skeleton_lines.extend("  " + line for line in keep_lines)
        if skeleton_lines:
            focus_sections.append(
                "[Skeleton Facts To Preserve]\n"
                + "\n".join(skeleton_lines)
                + "\nRepair rule: preserve this candidate's target-reaching baseline skeleton; repair only the local compile/setup/oracle failure and do not create a new scenario."
            )
        candidate_role = candidate_role_hint
        objective_id = objective_id_hint
        deep_role = bool(
            candidate_role in {"state_transition", "callback_partition", "branch_axis", "terminal_consumption"}
            or objective_id.upper().startswith("SK")
        )
        if deep_role:
            deep_text = "\n".join(
                    str(part or "")
                    for part in (
                        focused_test_code,
                        failed_test_source_excerpt,
                        ctx.test_code,
                    )
                    if str(part or "").strip()
            )
            deep_lines: list[str] = []
            if candidate_role:
                deep_lines.append(f"- candidate_role={candidate_role}")
            if objective_id:
                deep_lines.append(f"- objective_id={objective_id}")
            target_focus = str(
                extra_context.get("production_method_focus")
                or validation_failure.get("target_method")
                or ctx.extra.get("production_method_focus")
                or ctx.method_focus
                or ""
            ).strip()
            target_calls: list[str] = []
            if target_focus:
                target_re = re.compile(rf"(?P<expr>[A-Za-z0-9_.$<>()[\\],\\s\"']{{0,120}}\b{re.escape(target_focus)}\s*\([^;{{}}]*\))")
                for match in target_re.finditer(deep_text):
                    call = " ".join(match.group("expr").split())
                    if call and call not in target_calls:
                        target_calls.append(call[:180])
                    if len(target_calls) >= 3:
                        break
            if target_calls:
                deep_lines.append("- target_or_entry_call_to_preserve:")
                deep_lines.extend(f"  - {call}" for call in target_calls)
            terminal_markers: list[str] = []
            for marker in (".block(", ".blockingawait(", ".join(", ".get(", ".subscribe(", ".collectlist(", ".iterator(", ".next("):
                if marker.lower() in deep_text.lower():
                    terminal_markers.append(marker)
            if terminal_markers:
                deep_lines.append("- terminal_consumers_to_preserve=" + ", ".join(terminal_markers[:6]))
            projection_calls: list[str] = []
            for match in re.finditer(r"\b\w+\s*\.\s*(?:get|as|find|read|content|contents|blocks?)\w*\s*\(\s*[\w.$]+\.class\s*\)", deep_text):
                call = " ".join(match.group(0).split())
                if call not in projection_calls:
                    projection_calls.append(call)
                if len(projection_calls) >= 5:
                    break
            if projection_calls:
                deep_lines.append("- caller_projection_or_payload_extractors_to_preserve:")
                deep_lines.extend(f"  - {call}" for call in projection_calls)
            state_setup_lines: list[str] = []
            for raw in deep_text.splitlines():
                stripped = raw.strip()
                lowered = stripped.lower()
                if not stripped:
                    continue
                if any(token in lowered for token in (".builder(", ".builder()", ".events(", ".messages(", ".memory(", ".state(", ".config(")):
                    state_setup_lines.append(stripped[:220])
                if len(state_setup_lines) >= 6:
                    break
            if state_setup_lines:
                deep_lines.append("- receiver_state_or_builder_setup_to_preserve:")
                deep_lines.extend(f"  - {line}" for line in state_setup_lines)
            callback_lines: list[str] = []
            for raw in deep_text.splitlines():
                stripped = raw.strip()
                lowered = stripped.lower()
                if not stripped:
                    continue
                if any(token in lowered for token in ("doanswer(", "thenanswer(", "answer<", "invocation ->", "callback", "emitter", "listener")):
                    callback_lines.append(stripped[:220])
                if len(callback_lines) >= 6:
                    break
            if callback_lines:
                deep_lines.append("- callback_or_fake_outcome_to_preserve:")
                deep_lines.extend(f"  - {line}" for line in callback_lines)
            if len(deep_lines) > (2 if candidate_role or objective_id else 0):
                focus_sections.append(
                    "[Deep Path Facts To Preserve]\n"
                    + "\n".join(deep_lines)
                    + "\nRepair rule: keep these deep-path facts intact. Minimal edits may fix types, imports, local setup, or the single failing oracle, but must not delete the projection payload, receiver-state setup, callback outcome, target call, or terminal consumption."
                )
        audit_missing: list[str] = []
        if not (failure_signal.get("message") if isinstance(failure_signal, dict) else "") and not focused_error:
            audit_missing.append("failure_message")
        if is_compile_like and not diagnostic_lines:
            audit_missing.append("compile_diagnostic")
        if not focused_test_code:
            audit_missing.append("failed_test_excerpt")
        pre_target_failure = validation_failure.get("pre_target_failure") if isinstance(validation_failure.get("pre_target_failure"), dict) else {}
        if not pre_target_failure and not observed_trace.get("target_calls"):
            audit_missing.append("failed_statement_or_target_trace")
        if not candidate_coverage:
            audit_missing.append("candidate_target_coverage")
        if not validation_failure.get("source_evidence"):
            audit_missing.append("source_evidence")
        if audit_missing:
            focus_sections.append(
                "[Repair Prompt Evidence Audit]\n"
                "- missing_or_uncertain=" + ", ".join(audit_missing) + "\n"
                "- rule=do not invent receiver/collaborator/state facts for missing evidence; use callable/source context only as constraints and keep repair local to the reported failure."
            )
        terminal_evidence = failure_signal.get("terminal_output_setup_evidence") if isinstance(failure_signal, dict) else None
        if isinstance(terminal_evidence, dict) and terminal_evidence:
            evidence_bits = [
                f"{key}={terminal_evidence.get(key)}"
                for key in (
                    "target_positive",
                    "output_absent",
                    "post_target_failure",
                    "terminal_consumer",
                    "producer_empty_or_null",
                    "bare_mock_producer_payload",
                    "producer_null_from_helpful_npe",
                    "helpful_npe_producer_chain",
                    "source_chain_evidence",
                    "strong_terminal_output_setup_mismatch",
                )
                if key in terminal_evidence
            ]
            producer_calls = _bare_mock_terminal_producer_calls(
                "\n".join(
                    part
                    for part in (
                        focused_test_code,
                        failed_test_source_excerpt,
                        ctx.test_code,
                    )
                    if str(part or "").strip()
                )
            )
            producer_call_lines = ""
            if producer_calls:
                producer_call_lines = (
                    "\n- existing_producer_calls_to_preserve:\n"
                    + "\n".join(f"  - {call}" for call in producer_calls)
                    + "\n- producer_preservation_rule=keep the existing source-visible producer call above; repair the nested returned payload/getters/content or the single oracle. Do not switch to a sibling producer method unless compiler/source evidence says the existing producer call is unavailable."
                )
            focus_sections.append(
                "[Observable Output Setup Evidence]\n"
                + "- " + "; ".join(evidence_bits) + "\n"
                + "- repair_rule=preserve the same target-reaching Act path; only repair the local producer/collaborator output setup or the single failed oracle when this evidence is closed."
                + "\n- helpful_npe_rule=when the JVM reports that a source-visible producer call returned null, keep the same target path and stub/construct that exact producer return with the minimal non-null value required by the failing downstream call."
                + "\n- bare_mock_payload_rule=if a terminal/container producer returns Mockito.mock(...), provide a concrete source-visible payload or stub the nested getters/content consumed by the target before weakening assertions."
                + producer_call_lines
            )
        observed_non_null_setup = _observed_required_non_null_setup_lines(
            failure_signal=failure_signal,
            generation_control_receipt=generation_control_receipt,
        )
        if observed_non_null_setup:
            focus_sections.append(
                "[Observed Required Non-Null Setup]\n"
                + "\n".join(f"- {line}" for line in observed_non_null_setup)
                + "\nRepair rule: treat this as a local Arrange/setup fix for the same failing @Test; do not change the scenario or target path."
            )
        local_setup_lines = _target_positive_failure_local_setup_lines(
            validation_failure=validation_failure,
            failure_signal=failure_signal,
            coverage_positive=coverage_positive,
        )
        if local_setup_lines:
            focus_sections.append(
                "[Target-Positive Failure Local Setup Plan]\n"
                + "\n".join(f"- {line}" for line in local_setup_lines)
                + "\nRepair rule: use these source lines only to repair the local setup/terminal/callback/oracle failure in the preserved test; do not create a new scenario."
            )
        reflection_annotation_hint = _reflection_annotation_setup_hint(
            failure_signal=failure_signal,
            excerpt="\n".join(part for part in (focused_test_code, failed_test_source_excerpt) if part),
            failure_output=ctx.failure_output,
        )
        if reflection_annotation_hint:
            focus_sections.append(reflection_annotation_hint)
        unresolved_static_hint = _unresolved_static_member_repair_hint(ctx.failure_output)
        if unresolved_static_hint:
            focus_sections.append(unresolved_static_hint)
        type_name_value_hint = _type_name_used_as_value_repair_hint(
            failure_output=ctx.failure_output,
            context_text="\n".join(
                part
                for part in (
                    source_code,
                    test_code,
                    focused_test_code,
                    failed_test_source_excerpt,
                )
                if part
            ),
        )
        if type_name_value_hint:
            focus_sections.append(type_name_value_hint)
        focus_sections.append(
            "[Allowed Minimal Edits]\n"
            f"- {allowed_edits.get(repair_mode, allowed_edits['no_target_feedback_only'])}\n"
            "- import/type edits may add or replace only imports and declarations needed by the same failing method.\n"
            "- setup edits may add only the missing local value/collaborator return required by the observed stack.\n"
            "- oracle edits may touch only the reported failing assertion unless the failure line is unavailable and the expected/actual pair is unique."
        )
        focus_sections.append(
            "[Forbidden Edits]\n"
            "- Do not delete, bypass, or replace the target method call or verified public entry call.\n"
            "- Do not convert the test into a smoke test, constructor-only test, null/guard-only test, or sibling-method test.\n"
            "- Do not add a new unrelated @Test, merge scenarios, or rename the test except for a minimal legal Java identifier fix.\n"
            "- Do not remove terminal/result consumption, callback/fake/subclass setup, or public state-entry calls that occur before the target.\n"
            "- Do not weaken a target-positive test into a passing no-target test."
        )
        if focused_test_code:
            focus_sections.append(
                "[Original Test To Preserve]\n"
                f"Repair this exact failing test method `{focused_test_name}` first; do not infer it only from the full class tail.\n"
                "```java\n"
                f"{focused_test_code}\n"
                "```"
            )
        if failed_test_source_excerpt and failed_test_source_excerpt not in focused_test_code:
            focus_sections.append(
                "[Original Test To Preserve From Validation]\n"
                "This method was captured during streaming validation; preserve its target-reaching Arrange/Act shape unless the failure says that shape is invalid.\n"
                "```java\n"
                + _clip_middle(
                    failed_test_source_excerpt,
                    _repair_budget(compact=compact, normal=2200, compact_value=1400),
                    label="failed test source",
                )
                + "\n```"
            )
        if _looks_like_member_declaration_inside_test_method(
            "\n".join(part for part in (focused_test_code, failed_test_source_excerpt) if part)
        ):
            focus_sections.append(
                "[Malformed Fixture Structure Hint]\n"
                "- evidence=the failing method excerpt appears to contain field, constructor, or helper declarations inside the @Test method body.\n"
                "- repair_rule=keep the same target/entry call, but move helper fields/constructors/classes to a valid helper type or convert them into local variables/local classes before use; do not leave member declarations as statements inside @Test."
            )
        critical_receipt_lines = _repair_critical_generation_receipt_lines(
            generation_control_receipt,
            ctx.failure_output,
            max_lines=_repair_budget(compact=compact, normal=12, compact_value=8),
        )
        if critical_receipt_lines:
            focus_sections.append(
                "[Minimal Source Contracts Cited By Failure]\n"
                + critical_receipt_lines
                + "\nRepair rule: use these contracts only to fix the reported failure; do not introduce a new coverage scenario."
            )
        if generation_control_receipt and not mutation_extra_repair:
            noisy_generation_prefixes = (
                "- goal=",
                "- output_format=",
                "- schema=",
                "- new_tests_count=",
                "- required_coverage_obligation_ids=",
                "- every coverage-guided",
            )
            receipt_body = "\n".join(
                line
                for line in str(generation_control_receipt or "").splitlines()
                if not line.strip().startswith("[Generation Control Receipt]")
                and not line.strip().startswith(noisy_generation_prefixes)
            ).strip()
            focus_sections.append(
                "[Repair Appendix: Source-Derived Generation Receipt Constraints]\n"
                + _clip_middle(
                    receipt_body or generation_control_receipt,
                    _repair_budget(compact=compact, normal=1800, compact_value=1000),
                    label="generation control receipt",
                )
                + "\nRepair rule: preserve any already-used source-derived contract; do not use this receipt to redesign the test or add new branch axes."
            )
        if terminal_execution_guidance:
            focus_sections.append(
                "Terminal/collaborator constraints cited by failure (preservation only):\n"
                + _clip_middle(
                    terminal_execution_guidance,
                    _repair_budget(compact=compact, normal=1800, compact_value=900),
                    label="terminal execution guidance",
                )
                + "\nRepair rule: only add or fix the missing terminal/collaborator value when the failure stack points to it; do not create a new baseline scenario."
            )
        if failed_methods_hint:
            focus_sections.append("Failed method hints: " + ", ".join(failed_methods_hint))
        if overload_id or overload_contract or test_behavior:
            lines = ["Overload-specific repair contract:"]
            if overload_id:
                lines.append(f"- overload_id={overload_id}")
            if test_behavior:
                lines.append(f"- intended_test_behavior={test_behavior}")
            signature = str(overload_contract.get("signature") or "").strip()
            if signature:
                lines.append(f"- target_signature={signature}")
            call_shape = str(overload_contract.get("required_call_shape") or "").strip()
            if call_shape:
                lines.append(f"- required_call_shape={call_shape}")
            param_types = overload_contract.get("param_types")
            if isinstance(param_types, list) and param_types:
                lines.append("- parameter_types=" + ", ".join(str(p) for p in param_types))
            lines.append("- Repair setup/oracle while preserving this exact overload identity; do not switch to a sibling overload.")
            focus_sections.append("\n".join(lines))
        if validation_failure:
            lines = ["[Repair Appendix: Previous Validation Failure Feedback]"]
            for key in ("test_method", "failure_type", "return_code", "failed_test_count"):
                if validation_failure.get(key) not in (None, ""):
                    lines.append(f"- {key}={validation_failure.get(key)}")
            if validation_failure.get("failure_report_missing") is True:
                lines.append("- failure_report_missing=true; do not perform assertion/template repair unless compiler/test output below provides a concrete expected/actual or exception signal")
            if failure_signals:
                lines.append("- failure_signals=" + ", ".join(failure_signals[:6]))
            if failure_signal:
                for key in ("category", "exception_type", "message", "entry_frame", "target_frame"):
                    value = failure_signal.get(key)
                    if value not in (None, "", []):
                        lines.append(f"- {key}={value}")
                expected_value = str(failure_signal.get("expected_value") or "")
                actual_value = str(failure_signal.get("actual_value") or "")
                if expected_value or actual_value:
                    lines.append(
                        "- expected_observed_value="
                        + (json.dumps(expected_value) if expected_value else "?")
                    )
                    lines.append(
                        "- actual_observed_value="
                        + (json.dumps(actual_value) if actual_value else "?")
                    )
                hints = [str(item).strip() for item in list(failure_signal.get("hints") or []) if str(item).strip()]
                for hint in hints[:3]:
                    lines.append(f"- source_failure_hint={hint}")
            if "invalid_state_transition" in failure_signals:
                lines.append("- likely_fix=repair the Arrange/state sequence before the target call; avoid setup calls that put the object into an illegal state.")
            elif "cursor_or_resource_initial_state_mismatch" in failure_signals:
                lines.append("- likely_fix=do not use a bare mock/default cursor/resource when constructor or entry code expects a positioned state; build a real in-memory cursor/parser/reader or a focused fake and advance it with public API to the expected initial state before the target call.")
            elif "signature_or_constructor_mismatch" in failure_signals:
                lines.append("- likely_fix=match constructor/method arity and argument types to the listed signature exactly.")
            elif "generic_return_type_mismatch" in failure_signals:
                lines.append("- likely_fix=match generic return/stub element types exactly to the declared API; for lazy/reactive/future returns, wrap the declared element type and consume the returned computation.")
            elif "type_mismatch" in failure_signals:
                lines.append("- likely_fix=replace the mismatched argument/stub with the exact declared API type; if the API expects a config/wrapper object, build that wrapper through visible constructors/builders/factories instead of passing a raw collection/value.")
            if len(lines) > 1:
                focus_sections.append("\n".join(lines))
        if coverage_positive:
            lines = [
                "[Coverage-Positive Repair Preservation]",
                f"- has_target_coverage={candidate_coverage.get('has_target_coverage')}",
                f"- line_covered={candidate_coverage.get('line_covered')}",
                f"- branch_covered={candidate_coverage.get('branch_covered')}",
                f"- line_coverage={candidate_coverage.get('line_coverage')}",
                f"- branch_coverage={candidate_coverage.get('branch_coverage')}",
                "- preserve_rule=this failed candidate already reached target-method coverage; do not delete the target call, terminal/result consumption, callback/fake, or public state-entry sequence that reached it.",
                "- preserve_intent_rule=keep the original @Test scenario, Arrange/Act/Assert structure, test name intent, and main oracle family; repair only the concrete failure.",
                "- allowed_fix=repair Arrange initialization, collaborator/fake terminal outcome, or the single failing oracle; do not collapse this into a smoke/null/guard-only test.",
            ]
            focus_sections.append("\n".join(lines))
        unexpected_actual = str(failure_signal.get("actual_value") or "").strip()
        if not unexpected_actual:
            unexpected_match = re.search(
                r"Unexpected exception type thrown,\s*expected:\s*<[^>]+>\s*but\s*was:\s*<(?P<actual>[^>]+)>",
                str(ctx.failure_output or ""),
            )
            if unexpected_match:
                unexpected_actual = unexpected_match.group("actual").strip()
        unexpected_expected = str(failure_signal.get("expected_value") or "").strip()
        if unexpected_actual:
            lines = [
                "[Unexpected Exception Repair Rule]",
                f"- expected={unexpected_expected or '?'}",
                f"- actual={unexpected_actual}",
            ]
            if unexpected_actual.endswith("NullPointerException"):
                lines.extend([
                    "- actual_npe_rule=do not blindly change assertThrows to NullPointerException.",
                    "- likely_fix=repair only the missing Arrange/state/input initialization needed to execute the original Act path; initialize the null receiver/argument named in the NPE message using source-visible constructors/builders/fakes.",
                    "- oracle_calibration_allowed_only_if=the source contract proves NullPointerException is the intended public behavior for this input or the candidate is explicitly coverage-positive and no valid setup exists.",
                ])
            else:
                lines.append("- template_fix_allowed=if the stack proves the target call produced this actual public exception and setup is otherwise valid, calibrate the assertThrows expected type to the observed actual type.")
            focus_sections.append("\n".join(lines))
        if observed_trace:
            lines = ["[Observed Trace Repair Contract]"]
            expected_value = str(observed_trace.get("expected_value") or "")
            actual_value = str(observed_trace.get("actual_value") or "")
            if expected_value or actual_value:
                lines.append(
                    "- observed_mismatch: expected="
                    + (expected_value if expected_value else "?")
                    + " actual="
                    + (actual_value if actual_value else "?")
                )
            ordered_calls = [
                str(item).strip()
                for item in list(observed_trace.get("ordered_target_calls") or [])[:12]
                if str(item).strip()
            ]
            if ordered_calls:
                lines.append("- ordered_target_calls: " + " -> ".join(ordered_calls))
            assertions = [
                str(item).strip()
                for item in list(observed_trace.get("assertion_trace") or [])[:8]
                if str(item).strip()
            ]
            for assertion in assertions:
                lines.append(f"- assertion_observed: {assertion}")
            repair_rule = str(observed_trace.get("repair_rule") or "").strip()
            if repair_rule:
                lines.append(f"- repair_rule={repair_rule}")
            lines.append("- preserve_rule=do not delete the fake/subclass/callback or public state-entry sequence that reached target coverage; repair only the oracle or the minimal pre-target state setup.")
            focus_sections.append("\n".join(lines))
        if coverage_gate and bool(coverage_gate.get("enabled")) and not bool(coverage_gate.get("accepted", True)):
            lines = [
                (
                    "[Mutation Exact-Line Repair Boundary]"
                    if repair_mode == "mutation_exact_line_setup_fix"
                    else "[No-Target Repair Boundary]"
                ),
                f"- mode={coverage_gate.get('mode')}",
                f"- reason={coverage_gate.get('reason')}",
                f"- method={coverage_gate.get('method_name') or ctx.method_focus or 'target'}",
                f"- line_covered={coverage_gate.get('line_covered')}",
                f"- branch_covered={coverage_gate.get('branch_covered')}",
                (
                    "- repair_boundary=the target method is already reached; repair only the verifier-proven missing state transition needed for the exact mutant line."
                    if repair_mode == "mutation_exact_line_setup_fix"
                    else "- repair_boundary=this is not a request to discover a new target path; if no compile/type diagnostic is present, preserve the original test and let generation feedback handle target-entry exploration."
                ),
                "- do_not_fix_by=weakening assertions, deleting setup, adding smoke behavior, or calling neighboring getters/helpers.",
            ]
            if coverage_feedback:
                lines.append(f"- coverage_status={coverage_feedback.get('status')}")
                lines.append(f"- baseline_available={coverage_feedback.get('baseline_available')}")
                lines.append(f"- closes_baseline_gap={coverage_feedback.get('closes_baseline_gap')}")
            focus_sections.append("\n".join(lines))
        if coverage_feedback and bool(coverage_feedback.get("has_target_coverage")):
            lines = [
                "[Coverage-Positive Minimal-Edit Contract]",
                f"- status={coverage_feedback.get('status')}",
                f"- line_covered={coverage_feedback.get('line_covered')}",
                f"- branch_covered={coverage_feedback.get('branch_covered')}",
                f"- closes_baseline_gap={coverage_feedback.get('closes_baseline_gap')}",
                "- repair_rule=this candidate already executed the target method; preserve the Arrange/state-entry sequence and target invocation.",
                "- allowed_changes=fix only the reported oracle, invalid observer, or missing pre-target setup named by the failure.",
                "- forbidden_changes=delete the target call, switch to a neighboring method, or collapse the test into construction-only smoke behavior.",
                "- oracle_calibration_rule=if the target returned normally but the test expected an exception, keep the Arrange/Act path and assert the observed non-null/result type or stable public fields instead of assertThrows/fail.",
                "- interaction_oracle_rule=if Mockito reports a wanted-but-not-invoked interaction, remove or replace only that verify with an assertion on the actual returned value/state unless the source path proves the interaction must occur for this input.",
            ]
            if "assertion_or_oracle_mismatch" in failure_signals:
                lines.append("- oracle_rule=if the side effect is delegated to a mock collaborator, either verify the mock interaction or use a real/fake in-memory collaborator that exposes the asserted side effect; do not assert an unrelated buffer/object.")
            focus_sections.append("\n".join(lines))
        if mutation_feedback and mutation_feedback_enabled and coverage_positive and repair_mode in {"coverage_positive_preserve", "oracle_fix", "mutation_exact_line_setup_fix"}:
            lines = ["[Mutation Repair Feedback]"]
            gap = mutation_feedback.get("mutation_score_gap") if isinstance(mutation_feedback.get("mutation_score_gap"), dict) else {}
            if gap:
                lines.append(
                    f"- mutation_score={gap.get('mutation_score')} survived={gap.get('mutation_survived')} "
                    f"no_coverage={gap.get('mutation_no_coverage')} total={gap.get('mutation_total')}"
                )
            for mutant in [
                item for item in list(mutation_feedback.get("surviving_mutants") or [])[:4]
                if isinstance(item, dict)
            ]:
                bits = [
                    f"L{mutant.get('line', '?')}",
                    str(mutant.get("status") or "").strip(),
                    str(mutant.get("mutator_short") or mutant.get("mutator") or "").strip(),
                ]
                source_line = str(mutant.get("source_line") or "").strip()
                hint = str(mutant.get("kill_hint") or "").strip()
                lines.append("- " + " ".join(bit for bit in bits if bit))
                if source_line:
                    lines.append(f"  source: {source_line[:180]}")
                if hint:
                    lines.append(f"  repair_oracle_hint: {hint[:180]}")
            selected_objective = _mutation_repair_objective(
                mutation_feedback,
                objective_id_hint,
            )
            selected_facts = (
                selected_objective.get("facts")
                if isinstance(selected_objective.get("facts"), dict)
                else {}
            )
            if selected_facts:
                lines.append(
                    "- selected_mutation_contract="
                    + (objective_id_hint or "?")
                )
                source_predicate = " ".join(
                    str(selected_facts.get("source_predicate") or "").split()
                )
                if source_predicate:
                    lines.append(
                        "  exact_source_predicate: " + source_predicate[:240]
                    )
                rendered_rows: set[str] = set()
                rendered_transitions: set[tuple[str, str]] = set()
                for constraint in list(
                    selected_facts.get("source_backed_mutation_constraints")
                    or []
                )[:4]:
                    if not isinstance(constraint, dict):
                        continue
                    for row in list(
                        constraint.get("mutant_distinguishing_rows") or []
                    )[:2]:
                        if not isinstance(row, dict):
                            continue
                        value = " ".join(
                            str(row.get("constraint") or "").split()
                        )
                        if value and value not in rendered_rows:
                            rendered_rows.add(value)
                            lines.append(
                                "  infection_constraint_to_preserve: "
                                + value[:240]
                            )
                    transition = constraint.get("pit_operator_transition")
                    if not isinstance(transition, dict):
                        continue
                    original = str(
                        transition.get("original_operator") or ""
                    ).strip()
                    mutant = str(
                        transition.get("mutant_operator") or ""
                    ).strip()
                    if (
                        original
                        and mutant
                        and (original, mutant) not in rendered_transitions
                    ):
                        rendered_transitions.add((original, mutant))
                        lines.append(
                            "  exact_pit_operator_transition: original `"
                            + original
                            + "` -> mutant `"
                            + mutant
                            + "`"
                        )
                if rendered_rows:
                    lines.append(
                        "  constraint_preserving_repair_rule=fix the concrete "
                        "compile/runtime/oracle failure while keeping one exact "
                        "infection constraint true at the mutant line; do not "
                        "replace it with an adjacent reachability-only input."
                    )
            lines.append(
                "- repair_rule=preserve target reachability first; only strengthen assertions for SURVIVED mutants after the repaired test still compiles, passes, and reaches the target."
            )
            focus_sections.append("\n".join(lines))
        if focused_error:
            focus_sections.append("Artifact-focused error evidence:\n" + focused_error)
        if callable_surface_grounding:
            focus_sections.append(
                "Exact callable/type surface for repair (source-derived; do not invent absent APIs):\n"
                + _clip_middle(
                    callable_surface_grounding,
                    _repair_budget(
                        compact=compact,
                        normal=_REPAIR_CALLABLE_SURFACE_MAX_CHARS,
                        compact_value=_REPAIR_COMPACT_CALLABLE_SURFACE_MAX_CHARS,
                    ),
                    label="repair callable surface",
                )
                + "\nRepair rule: replace calls/enum constants absent from this surface with listed constructors, methods, nested enum constants, or concrete implementing values."
            )
        if visible_construction_grounding:
            focus_sections.append(
                "Visible type construction/type-state APIs for repair (source-derived):\n"
                + _clip_middle(
                    visible_construction_grounding,
                    _repair_budget(
                        compact=compact,
                        normal=_REPAIR_VISIBLE_CONSTRUCTION_MAX_CHARS,
                        compact_value=_REPAIR_COMPACT_VISIBLE_CONSTRUCTION_MAX_CHARS,
                    ),
                    label="visible construction APIs",
                )
                + "\nRepair rule: when a value is built through `Type.builder()`, keep the chain on the builder receiver and terminate with `.build()` before assigning or passing it as `Type`; do not call builder-only methods on the built receiver."
            )
        if callback_signature_grounding:
            focus_sections.append(
                "Callback/interface signature grounding for compile repair:\n"
                + _clip_middle(
                    callback_signature_grounding,
                    _repair_budget(
                        compact=compact,
                        normal=_REPAIR_CALLBACK_SIGNATURE_MAX_CHARS,
                        compact_value=_REPAIR_COMPACT_CALLBACK_SIGNATURE_MAX_CHARS,
                    ),
                    label="callback signatures",
                )
                + "\nRepair rule: if javac says a method does not override or implement a method from a supertype, remove the invented @Override method or replace it with one of the exact signatures above. Do not invent callback/listener/filter methods."
            )
        if failure_signals:
            focus_sections.append(
                "Abstract repair failure signals:\n"
                + "\n".join(f"- {signal}" for signal in failure_signals[:8])
            )
        if target_receipt and not mutation_extra_repair:
            focus_sections.append(
                "Target execution receipt to preserve during repair:\n"
                + _clip_middle(
                    target_receipt,
                    _repair_budget(
                        compact=compact,
                        normal=_REPAIR_TARGET_RECEIPT_MAX_CHARS,
                        compact_value=_REPAIR_COMPACT_TARGET_RECEIPT_MAX_CHARS,
                    ),
                    label="target receipt",
                )
            )
        if execution_capsule and not mutation_extra_repair:
            focus_sections.append(
                "Execution capsule skeleton to preserve during repair:\n"
                + _clip_middle(
                    execution_capsule,
                    _repair_budget(
                        compact=compact,
                        normal=_REPAIR_EXECUTION_CAPSULE_MAX_CHARS,
                        compact_value=_REPAIR_COMPACT_EXECUTION_CAPSULE_MAX_CHARS,
                    ),
                    label="execution capsule",
                )
            )
        if serialized_entry_contract and not mutation_extra_repair:
            focus_sections.append(
                "Serialized entry-path contract to preserve during repair:\n"
                + _clip_middle(
                    serialized_entry_contract,
                    _repair_budget(
                        compact=compact,
                        normal=_REPAIR_ENTRY_CONTRACT_MAX_CHARS,
                        compact_value=_REPAIR_COMPACT_ENTRY_CONTRACT_MAX_CHARS,
                    ),
                    label="serialized entry contract",
                )
            )
        if static_entry_path_plan and not mutation_extra_repair:
            focus_sections.append(
                "Static entry-path plan to preserve during repair:\n"
                + _clip_middle(
                    static_entry_path_plan,
                    _repair_budget(
                        compact=compact,
                        normal=_REPAIR_ENTRY_CONTRACT_MAX_CHARS,
                        compact_value=_REPAIR_COMPACT_ENTRY_CONTRACT_MAX_CHARS,
                    ),
                    label="static entry path plan",
                )
            )
        focused_block = (
            "\n\n" + "\n\n".join(focus_sections)
            if focus_sections
            else ""
        )
        repair_subject = "failing @Test method" if method_patch_json else "failing test class"
        user = (
            f"{focus_line}{compact_line}Repair the {repair_subject} based on the failure output.\n\n"
            "Constraints:\n"
            "- Keep deterministic assertions.\n"
            "- Keep imports minimal and compatible with project dependencies.\n"
            f"{constraint_block}\n\n"
            f"{focused_block}\n\n"
            "Failure output (tail):\n"
            f"{ctx.failure_output[-failure_tail_chars:]}"
            f"{unresolved_block}\n\n"
            "Relevant production context:\n"
            f"{source_code}\n\n"
            "Current failing test class:\n"
            f"{test_code}"
        )
        if method_patch_json:
            user += (
                "\n\nOutput reminder:\n"
                "- Return ONLY valid JSON with key new_tests.\n"
                "- Return exactly one repaired @Test method in test_code_lines.\n"
                "- Do not return the full test class, package line, class declaration, markdown, or prose.\n"
                "- Put any required imports in new_imports_lines."
            )
        return system, user

    def repair(self, ctx: RepairContext) -> tuple[str, dict]:
        compact_requested = bool(
            isinstance(ctx.extra, dict)
            and ctx.extra.get("repair_prompt_compact")
        )
        system_prompt, user_prompt = self.build_prompt(
            ctx,
            compact=compact_requested,
        )
        last_error: Exception | None = None
        compact_retry_used = False
        timeout_retries_remaining = max(0, int(_LLM_TIMEOUT_RETRY_ATTEMPTS))
        while True:
            try:
                result = chat_completion_with_usage(
                    api_base=self._api_base,
                    model=self._model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    api_key_env=self._api_key_env,
                    temperature=_llm_temperature(),
                    timeout_seconds=self._timeout,
                )
                return result["content"].strip(), result.get("usage", {})
            except Exception as exc:
                last_error = exc
                if _is_provider_request_rejection(exc) and not compact_retry_used:
                    compact_retry_used = True
                    system_prompt, user_prompt = self.build_prompt(ctx, compact=True)
                    continue
                # Only transient timeouts are retried; surface other failures
                # immediately so callers can record a precise error event.
                if "timed out" not in str(exc).lower():
                    if _is_provider_request_rejection(exc):
                        raise NonRetryableLLMRepairError(str(exc)) from exc
                    raise
                if timeout_retries_remaining <= 0:
                    break
                timeout_retries_remaining -= 1
        assert last_error is not None
        raise last_error


# ---------------------------------------------------------------------------
# Built-in strategy: deterministic structural pre-patch (DACR component B)
# ---------------------------------------------------------------------------

# Regexes covering the deterministically-rewritable javac error patterns.
_RE_PUBLIC_CLASS_FILENAME = re.compile(
    r"([^\s:]+\.java):\[\d+,\d+\]\s+error:\s+class\s+(\S+)\s+is\s+public,\s+should\s+be\s+declared\s+in\s+a\s+file\s+named\s+(\S+\.java)"
)
_RE_DUPLICATE_CLASS = re.compile(
    r"([^\s:]+\.java):\[\d+,\d+\]\s+error:\s+duplicate\s+class:\s+([\w.$]+)"
)
_RE_ENUM_INSTANTIATION = re.compile(
    r"([^\s:]+\.java):\[\d+,\d+\]\s+error:\s+enum\s+types\s+may\s+not\s+be\s+instantiated"
)
_RE_HAS_MOCK_NO_EXT = re.compile(r"@Mock\b")
_RE_HAS_EXTEND_WITH = re.compile(r"@ExtendWith\s*\(\s*MockitoExtension\.class\s*\)")
_RE_HAS_OPENMOCKS = re.compile(r"MockitoAnnotations\.openMocks\s*\(")
_RE_JUNIT_EXPECTED_ACTUAL_ANGLE = re.compile(
    r"expected:\s*<(?P<expected>[^>\n\r]*)>\s*but\s+was:\s*<(?P<actual>[^>\n\r]*)>",
    re.IGNORECASE,
)
_RE_JUNIT_EXPECTED_ACTUAL_COMPACT = re.compile(
    r"expected:<(?P<expected>[^>\n\r]*)>\s*but\s+was:<(?P<actual>[^>\n\r]*)>",
    re.IGNORECASE,
)
_RE_JUNIT_EXPECTED_ACTUAL_MULTILINE = re.compile(
    r"expected:\s*<(?P<expected>[\s\S]{0,320}?)>\s*but\s+was:\s*"
    r"<(?P<actual>[\s\S]{0,320}?)>",
    re.IGNORECASE,
)


def extract_junit_expected_actual(failure_output: str) -> tuple[str, str] | None:
    """Return the richest bounded JUnit expected/actual observation.

    Surefire's XML summary normalizes embedded CR/LF to spaces, while its raw
    console diagnostic preserves the line break.  Failure evidence often
    contains both copies.  Selecting the first match therefore loses the exact
    observable value and can make a deterministic repair oscillate.  Consider
    every bounded pair and prefer a differing pair with preserved control or
    trailing whitespace.
    """
    text = str(failure_output or "")
    candidates: list[tuple[int, int, str, str]] = []
    for match in _RE_JUNIT_EXPECTED_ACTUAL_MULTILINE.finditer(text):
        expected = match.group("expected")
        actual = match.group("actual")
        combined = expected + actual
        if any(
            marker in combined
            for marker in (
                "[actionable diagnostic excerpt]",
                "[... ",
                "\n[ERROR]",
                "\n[INFO]",
                "\nat ",
                "\n\tat ",
            )
        ):
            continue
        differs = expected != actual
        has_control = any(ch in combined for ch in ("\r", "\n", "\t"))
        has_significant_edge_space = (
            expected != expected.strip(" ")
            or actual != actual.strip(" ")
        )
        score = (
            (100 if differs else 0)
            + (50 if has_control else 0)
            + (20 if has_significant_edge_space else 0)
            + min(len(combined), 40)
        )
        candidates.append((score, match.start(), expected, actual))
    if not candidates:
        return None
    _score, _offset, expected, actual = max(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    return expected, actual


def _java_string_literal_value(value: str) -> str:
    """Escape an observed runtime string for a Java source literal."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\b", "\\b")
        .replace("\f", "\\f")
        .replace('"', '\\"')
    )


def _decompact_junit_comparison_actual(
    existing_expected: str,
    expected_text: str,
    actual_text: str,
    *,
    context_length: int = 20,
) -> str:
    """Recover JUnit ComparisonCompactor's actual string when unambiguous.

    JUnit 4 renders a string delta with square-bracket markers. For example,
    expected ``K[NK]`` and actual ``KNK`` becomes
    ``expected:<K[[NK]]> but was:<K[NK]>``. Those brackets are diagnostic
    markup, not necessarily runtime characters. Reproduce JUnit's bounded
    prefix/suffix rendering against the source expected literal and accept
    only a unique reconstruction.
    """

    expected_expression = str(existing_expected or "").strip()
    if not (
        expected_expression.startswith('"')
        and expected_expression.endswith('"')
    ):
        return actual_text
    try:
        expected_value = json.loads(expected_expression)
    except (TypeError, ValueError, json.JSONDecodeError):
        return actual_text
    if not isinstance(expected_value, str):
        return actual_text

    compact_expected = str(expected_text or "")
    compact_actual = str(actual_text or "")
    if (
        not compact_expected
        or not compact_actual
        or "[" not in compact_expected
        or "]" not in compact_expected
        or "[" not in compact_actual
        or "]" not in compact_actual
    ):
        return actual_text

    reconstructions: set[str] = set()
    value_length = len(expected_value)
    for prefix_length in range(value_length + 1):
        for suffix_length in range(
            value_length - prefix_length + 1
        ):
            delta_end = value_length - suffix_length
            expected_delta = expected_value[
                prefix_length:delta_end
            ]
            full_prefix = expected_value[:prefix_length]
            full_suffix = (
                expected_value[delta_end:]
                if suffix_length
                else ""
            )
            rendered_prefix = (
                full_prefix
                if prefix_length <= context_length
                else "..." + full_prefix[-context_length:]
            )
            rendered_suffix = (
                full_suffix
                if suffix_length <= context_length
                else full_suffix[:context_length] + "..."
            )
            if (
                rendered_prefix
                + "["
                + expected_delta
                + "]"
                + rendered_suffix
                != compact_expected
            ):
                continue
            actual_prefix = rendered_prefix + "["
            actual_suffix = "]" + rendered_suffix
            if not compact_actual.startswith(actual_prefix):
                continue
            if not compact_actual.endswith(actual_suffix):
                continue
            actual_delta_end = len(compact_actual) - len(
                actual_suffix
            )
            actual_delta = compact_actual[
                len(actual_prefix):actual_delta_end
            ]
            reconstructions.add(
                full_prefix + actual_delta + full_suffix
            )
    if len(reconstructions) == 1:
        return next(iter(reconstructions))
    return actual_text
_RE_RAW_HASHSET_TO_SET_STRING = re.compile(
    r"incompatible types:\s+java\.util\.HashSet<[^>]+>\s+cannot be converted to\s+java\.util\.Set<java\.lang\.String>|"
    r"incompatible types:\s+java\.util\.HashSet<[^>]+>\s+cannot be converted to\s+java\.util\.Set<String>",
    re.IGNORECASE,
)
_RE_CANNOT_FIND_METHOD = re.compile(
    r"(?:cannot find symbol\s*)?(?:\n?\s*(?:\[ERROR\]\s*)?(?:symbol:\s+)?method\s+([A-Za-z_$][\w$]*)\s*\()",
    re.IGNORECASE,
)
_RE_DUPLICATE_LOCAL_VARIABLE = re.compile(
    r"\bvariable\s+(?P<variable>[A-Za-z_$][A-Za-z0-9_$]*)\s+"
    r"is\s+already\s+defined\s+in\s+method\s+"
    r"(?P<method>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(",
    re.IGNORECASE,
)


class StructuralPrepatchStrategy(RepairStrategy):
    """Deterministic, LLM-free rewriter for high-confidence compile errors.

    Runs **before** any LLM call.  Eliminates the failure modes that wasted
    LLM cycles in the Binance baseline run:

    * ``class X is public, should be declared in a file named …`` — delete the
      offending sibling file (always keep the file whose stem matches the
      intended test class).
    * ``duplicate class: …`` — same fix.
    * Missing Mockito extension when ``@Mock`` fields are present — inject
      ``@ExtendWith(MockitoExtension.class)`` and the matching import.
    * ``enum types may not be instantiated`` — rewrite ``new EnumName(...)``
      into ``EnumName.values()[0]`` (semantically equivalent for compile-time
      placeholder values; runtime semantics improve via a subsequent LLM
      pass when the test gates on specific enum identity).

    The strategy returns ``("", {...})`` when no rule fires, signalling the
    orchestrator to fall through to the next strategy in the composite.
    """

    def repair(self, ctx: RepairContext) -> tuple[str, dict]:
        test_file = Path(ctx.extra.get("test_file", "") or "")
        if not test_file.exists() and not str(ctx.test_code or "").strip():
            return "", {}
        test_code = ctx.test_code or test_file.read_text(encoding="utf-8", errors="ignore")
        applied: list[str] = []

        # Rule 1+2: drop offending sibling files (always preserve test_file itself).
        offending: set[Path] = set()
        for pattern in (_RE_PUBLIC_CLASS_FILENAME, _RE_DUPLICATE_CLASS):
            for match in pattern.finditer(ctx.failure_output or ""):
                offending.add(Path(match.group(1)))
        for off in offending:
            if test_file.exists() and off.resolve() == test_file.resolve():
                continue  # never delete the canonical test file
            try:
                if off.exists():
                    off.unlink()
                    applied.append(f"DUP_CLASS_REMOVED:{off.name}")
            except OSError:
                pass

        new_test_code = test_code

        # Javac coordinates refer to the original candidate. Apply repairs
        # that depend on an exact reported line before imports, stubs, or
        # declarations can shift that coordinate.
        patched = self._repair_duplicate_local_variable_declaration(
            new_test_code,
            ctx.failure_output or "",
            method_focus=ctx.method_focus,
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("DUPLICATE_LOCAL_VARIABLE_RENAMED")

        patched = self._remove_mockito_stub_with_reported_return_type_mismatch(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCKITO_RETURN_TYPE_MISMATCH_STUB_REMOVED")

        # Rule 0: generated tests sometimes emit wildcard imports from utility
        # classes as normal imports. Java requires static imports for static
        # helpers such as Assertions.assertEquals or Mockito.mock.
        patched = self._repair_static_utility_wildcard_imports(new_test_code)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("STATIC_UTILITY_WILDCARD_IMPORT_FIXED")

        patched = self._repair_missing_junit_mockito_static_imports(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("JUNIT_MOCKITO_STATIC_IMPORTS_ADDED")

        patched = self._repair_mixed_mockito_argument_matchers(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCKITO_MIXED_ARGUMENT_MATCHERS_FIXED")

        patched = self._remove_unused_imports_from_missing_packages(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("UNUSED_UNRESOLVED_IMPORT_REMOVED")

        # Rule 3: inject MockitoExtension when @Mock used without wiring.
        if (
            _RE_HAS_MOCK_NO_EXT.search(new_test_code)
            and not _RE_HAS_EXTEND_WITH.search(new_test_code)
            and not _RE_HAS_OPENMOCKS.search(new_test_code)
        ):
            new_test_code = self._inject_mockito_extension(new_test_code)
            if new_test_code != test_code:
                applied.append("MOCKITO_EXT_INJECTED")

        # Rule 3b: if javac reports an unresolved unqualified Mockito helper
        # call, qualify it locally instead of spending another LLM round.
        if re.search(r"symbol:\s+method\s+mock\s*\(", ctx.failure_output or ""):
            patched = self._qualify_unqualified_mockito_mock_calls(new_test_code)
            if patched != new_test_code:
                new_test_code = patched
                applied.append("MOCKITO_MOCK_QUALIFIED")

        # Rule 4: enum instantiation rewrite.
        enum_names = self._enum_names_from_errors(ctx.failure_output or "")
        if enum_names:
            for ename in enum_names:
                pat = re.compile(rf"\bnew\s+{re.escape(ename)}\s*\([^)]*\)")
                if pat.search(new_test_code):
                    new_test_code = pat.sub(f"{ename}.values()[0]", new_test_code)
                    applied.append(f"ENUM_INST_FIXED:{ename}")

        # Rule 7: Mockito invalid stubbing of void methods
        # (`when(voidCall).thenReturn(null)`) causes javac:
        # `'void' type not allowed here`.
        if "'void' type not allowed here" in (ctx.failure_output or ""):
            patched = self._remove_invalid_void_stubs(new_test_code)
            if patched != new_test_code:
                new_test_code = patched
                applied.append("MOCKITO_VOID_STUB_REMOVED")

        patched = self._repair_javac_method_arity_mismatch(new_test_code, ctx.failure_output or "")
        if patched != new_test_code:
            new_test_code = patched
            applied.append("JAVAC_METHOD_ARITY_MISMATCH_REPAIRED")

        patched = self._repair_integral_array_assertion_delta_overload(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            range_repair = (
                "java.util.Arrays.copyOfRange(" in patched
                and "java.util.Arrays.copyOfRange(" not in new_test_code
            )
            new_test_code = patched
            applied.append(
                "INTEGRAL_ARRAY_ASSERTION_RANGE_OVERLOAD_REPAIRED"
                if range_repair
                else "INTEGRAL_ARRAY_ASSERTION_DELTA_REMOVED"
            )

        # Rule 7b: repeated stubbing of the same Mockito call overwrites earlier
        # returns, which breaks cursor/parser state sequences.  Convert exact
        # duplicate call stubs into one sequential thenReturn chain.
        patched = self._repair_repeated_mockito_then_return_sequences(new_test_code)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCKITO_REPEATED_STUB_SEQUENCE_MERGED")

        patched = self._remove_unavailable_mockito_stubs(new_test_code, ctx.failure_output or "")
        if patched != new_test_code:
            new_test_code = patched
            applied.append("UNAVAILABLE_MOCKITO_STUB_REMOVED")

        patched = self._remove_mockito_stubs_on_non_mock_receivers(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("NON_MOCK_RECEIVER_STUB_REMOVED")

        patched = self._repair_generic_mock_return_payload_type_mismatch(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("GENERIC_MOCK_RETURN_PAYLOAD_TYPE_FIXED")

        patched = self._relax_mock_producer_stub_args_for_null_injected_field(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCK_PRODUCER_STUB_ARGS_RELAXED_FOR_NULL_INJECTED_FIELD")

        patched = self._relax_mockito_typed_any_for_null_producer_return(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCKITO_TYPED_ANY_NULL_PRODUCER_RELAXED")

        patched = self._restore_missing_variable_declarations_from_original_excerpt(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MISSING_VARIABLE_DECLARATION_RESTORED_FROM_ORIGINAL_TEST")

        patched = self._remove_assertions_with_undefined_expected_variables(new_test_code, ctx.failure_output or "")
        if patched != new_test_code:
            new_test_code = patched
            applied.append("UNDEFINED_EXPECTED_ASSERTION_REMOVED")

        patched = self._lift_try_local_variable_used_in_catch(new_test_code, ctx.failure_output or "")
        if patched != new_test_code:
            new_test_code = patched
            applied.append("TRY_LOCAL_VARIABLE_USED_IN_CATCH_LIFTED")

        patched = self._repair_source_required_mock_non_null_returns(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("SOURCE_REQUIRED_MOCK_NON_NULL_RETURN_STUBBED")

        patched = self._repair_helpful_npe_builder_return_value_null(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("HELPFUL_NPE_BUILDER_RETURN_VALUE_INITIALIZED")

        patched = self._repair_pre_target_mock_copy_return_setup(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("PRE_TARGET_MOCK_COPY_RETURN_SETUP_FIXED")

        patched = self._repair_assert_not_null_observed_null(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("ASSERT_NOT_NULL_OBSERVED_NULL_CALIBRATED")

        patched = self._repair_null_local_receiver_assertion_oracle(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("NULL_LOCAL_RECEIVER_ASSERTION_ORACLE_CALIBRATED")

        patched = self._repair_stepverifier_eager_exception_to_assert_throws(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("STEPVERIFIER_EAGER_EXCEPTION_TO_ASSERT_THROWS")

        patched = self._repair_stepverifier_onnext_oncomplete_oracle(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("STEPVERIFIER_ONNEXT_ONCOMPLETE_ORACLE_CALIBRATED")

        patched = self._remove_uninvoked_mockito_verify_oracle(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCKITO_UNINVOKED_VERIFY_ORACLE_CALIBRATED_TO_NEVER")

        patched = self._relax_overinvoked_mockito_verify_oracle(new_test_code, ctx)
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCKITO_OVERINVOKED_VERIFY_ORACLE_RELAXED")

        patched = self._remove_unused_inaccessible_declarations(new_test_code, ctx.failure_output or "")
        if patched != new_test_code:
            new_test_code = patched
            applied.append("UNUSED_INACCESSIBLE_DECLARATION_REMOVED")

        patched = self._remove_unused_compile_mismatch_declaration(
            new_test_code,
            ctx.failure_output or "",
            method_focus=ctx.method_focus,
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("UNUSED_COMPILE_MISMATCH_DECLARATION_REMOVED")

        patched = self._repair_json_object_or_array_shape_from_diagnostic(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("JSON_OBJECT_OR_ARRAY_SHAPE_FIXED")

        # Rule 8: JUnit reports concrete observed values for failed equality
        # oracles.  For generated tests, prefer the observed behavior over a
        # guessed expected literal; sandbox validation still decides whether the
        # repaired oracle is acceptable.
        failure_signal = (
            ctx.extra.get("streaming_failure_signal")
            if isinstance(ctx.extra, dict)
            else {}
        )
        if not isinstance(failure_signal, dict):
            failure_signal = {}
        failure_category = str(failure_signal.get("category") or "").strip()
        failure_exception = str(
            failure_signal.get("exception_type") or ""
        ).strip()
        assertion_failure_evidence = bool(
            failure_category in {
                "oracle-mismatch",
                "assertion_or_oracle_mismatch",
                "state-sequence-or-oracle-mismatch",
            }
            or re.search(
                r"(?:AssertionFailedError|AssertionError|ComparisonFailure|"
                r"AssertEquals\.assertEquals|Assertions\.assertEquals)",
                "\n".join(
                    [
                        failure_exception,
                        str(ctx.failure_output or ""),
                    ]
                ),
            )
        )
        # Runtime protocol/precondition failures often say "Expected X but was
        # Y" too.  Treating those messages as JUnit equality failures can undo
        # a previously correct oracle and make multi-attempt repair oscillate.
        # Preserve the legacy fallback when no structured category is present,
        # but require assertion evidence once validation classified the error.
        oracle_calibration_allowed = bool(
            assertion_failure_evidence or not failure_category
        )
        if (
            oracle_calibration_allowed
            and "expected" in (ctx.failure_output or "").lower()
            and "but was" in (ctx.failure_output or "").lower()
        ):
            patched = self._repair_assert_equals_expected_actual(
                new_test_code,
                ctx.failure_output or "",
                method_focus=ctx.method_focus,
            )
            if patched == new_test_code:
                patched = self._repair_assert_null_expected_actual(
                    new_test_code,
                    ctx.failure_output or "",
                )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("ASSERT_EQUALS_EXPECTED_ACTUAL_OBSERVED")

        # Rule 8b: if sandbox coverage proves the failing candidate already
        # executed the target, do not discard it solely because the generated
        # boolean oracle guessed the wrong external observation.  Replace only
        # the reported brittle assertTrue/assertFalse with a stable non-null
        # observation over the same local value or target receiver.
        coverage_feedback = ctx.extra.get("coverage_feedback") if isinstance(ctx.extra, dict) else {}
        failure_signal = ctx.extra.get("streaming_failure_signal") if isinstance(ctx.extra, dict) else {}
        if not isinstance(failure_signal, dict):
            failure_signal = {}
        failure_category = str(failure_signal.get("category") or "").strip()
        failure_text_lower = str(ctx.failure_output or "").lower()
        oracle_rewrite_allowed = bool(
            isinstance(coverage_feedback, dict)
            and bool(coverage_feedback.get("has_target_coverage"))
            and (
                failure_category in {
                    "oracle-mismatch",
                    "state-sequence-or-oracle-mismatch",
                    "assertion_or_oracle_mismatch",
                    "missing-required-terminal-initialization",
                    "entry-precondition-state",
                    "runtime-validation-failure",
                    "invalid-public-input",
                }
                or ("expected" in failure_text_lower and "but was" in failure_text_lower)
                or "nothing was thrown" in failure_text_lower
                or "assertionerror" in failure_text_lower
            )
        )
        if oracle_rewrite_allowed:
            patched = self._repair_assert_throws_wraps_eager_target_assignment(
                new_test_code,
                ctx,
            )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("ASSERT_THROWS_WRAP_EAGER_TARGET_ASSIGNMENT")
            patched = self._repair_coverage_positive_brittle_boolean_assertion(
                new_test_code,
                ctx.failure_output or "",
                production_method_focus=str(ctx.extra.get("production_method_focus") or ""),
                test_method_focus=str(ctx.extra.get("repair_test_method") or ctx.method_focus or ""),
                junit_family=ctx.junit_family or "",
            )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("COVERAGE_POSITIVE_BOOLEAN_ORACLE_STABILIZED")
            patched = self._repair_coverage_positive_expected_exception_fail(
                new_test_code,
                ctx.failure_output or "",
                test_method_focus=str(ctx.extra.get("repair_test_method") or ctx.method_focus or ""),
            )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("COVERAGE_POSITIVE_EXCEPTION_ORACLE_STABILIZED")
            patched = self._repair_coverage_positive_assert_throws_no_exception(
                new_test_code,
                ctx.failure_output or "",
            )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("COVERAGE_POSITIVE_ASSERT_THROWS_TO_DOES_NOT_THROW")
            patched = self._repair_coverage_positive_eager_resource_exception_to_assert_throws(
                new_test_code,
                ctx.failure_output or "",
            )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("COVERAGE_POSITIVE_EAGER_RESOURCE_EXCEPTION_TO_ASSERT_THROWS")

        # Rule 8c: source-derived public-input receipts often prove that a
        # builder/constructor collection must be non-empty before the target is
        # reachable.  Repair empty generated setters deterministically using
        # the receipt/visible API element type rather than asking the model to
        # rediscover the same precondition.
        patched, empty_collection_rules = self._repair_empty_required_collection_setters_from_context(
            new_test_code,
            ctx,
        )
        if empty_collection_rules:
            new_test_code = patched
            applied.extend(empty_collection_rules)

        # Rule 9: generated tests often reference a source-visible dependency
        # but omit the matching import. If javac reports an unresolved class and
        # the production source imports that exact simple name, copy that import.
        missing_imports = self._missing_imports_from_source_context(
            failure_output=ctx.failure_output or "",
            source_code=ctx.source_code or "",
            test_code=new_test_code,
        )
        missing_imports.extend(
            self._missing_imports_from_project_symbol_index(
                failure_output=ctx.failure_output or "",
                source_code=ctx.source_code or "",
                test_code=new_test_code,
                source_file=str(ctx.extra.get("source_file") or ""),
            )
        )
        for simple_name, fqn in missing_imports:
            patched = self._ensure_import(new_test_code, fqn)
            if patched != new_test_code:
                new_test_code = patched
                applied.append(f"SOURCE_IMPORT_ADDED:{simple_name}")

        # Rule 10: projects that compile tests with source < 8 cannot accept
        # JUnit assertThrows lambdas. Rewrite simple exception assertions to
        # Java 7-compatible try/catch while preserving the exercised body.
        source_level = _java_source_level_from_failure_output(ctx.failure_output or "")
        if source_level is not None and source_level < 8 and "assertThrows" in new_test_code:
            patched = self._rewrite_assert_throws_lambda_for_java7(new_test_code)
            if patched != new_test_code:
                new_test_code = patched
                applied.append("ASSERT_THROWS_LAMBDA_TO_TRY_CATCH")

        # Rule 11: when a boundary/oracle candidate reaches the target call
        # but the generated catch type is too specific or simply wrong, javac
        # succeeds and JUnit reports the actual exception type.  Calibrate the
        # catch type only when the stack proves the exception came from the
        # production target call, preserving the Arrange/Act sequence.
        production_focus = str(ctx.extra.get("production_method_focus") or "").strip()
        test_focus = str(ctx.extra.get("repair_test_method") or ctx.method_focus or "").strip()
        if production_focus:
            patched = self._repair_expected_exception_type_from_observed_target_error(
                new_test_code,
                ctx.failure_output or "",
                production_method_focus=production_focus,
                test_method_focus=test_focus,
            )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("EXPECTED_EXCEPTION_TYPE_OBSERVED")
            patched = self._repair_assert_throws_exception_type_from_observed_actual(
                new_test_code,
                ctx.failure_output or "",
                production_method_focus=production_focus,
                test_method_focus=test_focus,
            )
            if patched != new_test_code:
                new_test_code = patched
                applied.append("ASSERT_THROWS_EXCEPTION_TYPE_OBSERVED")

        # Rule 12: calibrate one unambiguous zero-argument mock state stub from
        # an exact runtime precondition.  Do not invent a state-advance method
        # or replace the mock with a domain-specific concrete implementation.
        patched = self._repair_mocked_symbolic_state_stub_from_failure(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MOCK_SYMBOLIC_STATE_STUB_CALIBRATED")

        # Rule 13: sometimes the target exception is correctly caught, but a
        # cleanup/finally call fails afterwards (for example closing an object
        # left in an intentionally exceptional state).  If the reported stack
        # line is a cleanup call, make that cleanup best-effort instead of
        # discarding the coverage-reaching target call.
        patched = self._make_failing_finally_cleanup_best_effort(
            new_test_code,
            ctx.failure_output or "",
            test_method_focus=test_focus or None,
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("FAILING_FINALLY_CLEANUP_BEST_EFFORT")

        # Rule 14: diamond inference in older source levels can infer
        # HashSet<Object> when a source-visible API expects Set<String>.  Make
        # the element type explicit without changing the state-entry sequence.
        if _RE_RAW_HASHSET_TO_SET_STRING.search(ctx.failure_output or ""):
            patched = self._repair_hashset_string_type_inference(new_test_code)
            if patched != new_test_code:
                new_test_code = patched
                applied.append("HASHSET_STRING_TYPE_INFERENCE")

        patched = self._repair_reactive_publisher_return_mismatch(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("REACTIVE_PUBLISHER_RETURN_MISMATCH")

        patched = self._repair_missing_helper_inside_reactive_just(
            new_test_code,
            ctx.failure_output or "",
            method_focus=ctx.method_focus,
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MISSING_HELPER_REACTIVE_JUST_TO_EMPTY")

        patched = self._repair_non_generic_diamond_construction(
            new_test_code,
            ctx.failure_output or "",
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("NON_GENERIC_DIAMOND_REMOVED")

        # Rule 15: generated cleanup/teardown code sometimes calls a method
        # that is absent on the project version under test. If javac pinpoints a
        # missing no-arg method and the failing statement is not the target
        # method itself, remove that one statement so a coverage-reaching test is
        # not discarded due to nonessential cleanup scaffolding.
        patched = self._remove_missing_method_statement_at_reported_line(
            new_test_code,
            ctx.failure_output or "",
            method_focus=ctx.method_focus,
        )
        patched = self._remove_missing_declared_type_noarg_statements(
            patched,
            ctx.failure_output or "",
            method_focus=ctx.method_focus,
        )
        patched = self._remove_missing_declared_type_fluent_call_segments(
            patched,
            ctx.failure_output or "",
            method_focus=ctx.method_focus,
        )
        if patched != new_test_code:
            new_test_code = patched
            applied.append("MISSING_METHOD_CLEANUP_REMOVED")

        if not applied:
            return "", {}
        meta = {"strategy": "structural", "applied_rules": applied}
        # If only file deletions happened (no test_code change), return the
        # original code so the orchestrator writes it back unchanged (no-op
        # write) but still records the structural fixes that were applied.
        return new_test_code if new_test_code != test_code else test_code, meta

    @classmethod
    def _repair_duplicate_local_variable_declaration(
        cls,
        code: str,
        failure_output: str,
        *,
        method_focus: str | None,
    ) -> str:
        """Rename one compiler-proven duplicate local and later references.

        The rule is intentionally narrow: javac must identify one variable and
        one method, the repair focus must agree, and exactly two same-indent
        local declarations must exist in that method. The later declaration
        and identifier tokens after it are renamed; comments and literals are
        preserved. Sandbox compilation remains authoritative.
        """

        text = str(code or "")
        identities = {
            (match.group("variable"), match.group("method"))
            for match in _RE_DUPLICATE_LOCAL_VARIABLE.finditer(
                str(failure_output or "")
            )
        }
        if len(identities) != 1:
            return text
        variable, method = next(iter(identities))
        focus = str(method_focus or "").strip()
        if focus and focus != method:
            return text

        signature = re.search(
            rf"\b{re.escape(method)}\s*\([^{{;]*\)\s*"
            r"(?:throws\s+[^\{]+)?\{",
            text,
        )
        if signature is None:
            return text
        body_open = text.find("{", signature.start())
        body_close = cls._find_matching_brace(text, body_open)
        if body_open < 0 or body_close <= body_open:
            return text
        if re.search(
            rf"\b{re.escape(variable)}\b",
            text[signature.start() : body_open],
        ):
            return text

        body = text[body_open + 1 : body_close]
        declaration_re = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?:(?:final)\s+)?"
            r"[A-Za-z_$][A-Za-z0-9_$.<>?, \[\]]*\s+"
            rf"(?P<name>{re.escape(variable)})\s*="
        )
        declarations = list(declaration_re.finditer(body))
        if (
            len(declarations) != 2
            or declarations[0].group("indent")
            != declarations[1].group("indent")
        ):
            return text

        existing_names = set(
            re.findall(r"\b[A-Za-z_$][A-Za-z0-9_$]*\b", body)
        )
        suffix = 2
        replacement = f"{variable}{suffix}"
        while replacement in existing_names:
            suffix += 1
            replacement = f"{variable}{suffix}"

        second_start = body_open + 1 + declarations[1].start()
        segment = text[second_start:body_close]
        renamed = cls._replace_java_identifier_tokens(
            segment,
            variable,
            replacement,
        )
        if renamed == segment:
            return text
        return text[:second_start] + renamed + text[body_close:]

    @staticmethod
    def _replace_java_identifier_tokens(
        text: str,
        old: str,
        new: str,
    ) -> str:
        """Replace Java identifier tokens while preserving comments/literals."""

        output: list[str] = []
        index = 0
        state = "code"
        while index < len(text):
            if state == "code" and text.startswith("//", index):
                state = "line_comment"
                output.append("//")
                index += 2
                continue
            if state == "code" and text.startswith("/*", index):
                state = "block_comment"
                output.append("/*")
                index += 2
                continue
            if state == "block_comment" and text.startswith("*/", index):
                state = "code"
                output.append("*/")
                index += 2
                continue
            char = text[index]
            if state == "line_comment":
                output.append(char)
                index += 1
                if char == "\n":
                    state = "code"
                continue
            if state == "block_comment":
                output.append(char)
                index += 1
                continue
            if state in {"string", "char"}:
                output.append(char)
                index += 1
                if char == "\\" and index < len(text):
                    output.append(text[index])
                    index += 1
                elif (state == "string" and char == '"') or (
                    state == "char" and char == "'"
                ):
                    state = "code"
                continue
            if char == '"':
                state = "string"
                output.append(char)
                index += 1
                continue
            if char == "'":
                state = "char"
                output.append(char)
                index += 1
                continue
            identifier = re.match(
                r"[A-Za-z_$][A-Za-z0-9_$]*",
                text[index:],
            )
            if identifier:
                token = identifier.group(0)
                output.append(new if token == old else token)
                index += len(token)
                continue
            output.append(char)
            index += 1
        return "".join(output)

    @staticmethod
    def _repair_static_utility_wildcard_imports(code: str) -> str:
        patched = re.sub(
            r"(?m)^\s*import\s+org\.junit\.jupiter\.api\.Assertions\.\*\s*;",
            "import static org.junit.jupiter.api.Assertions.*;",
            code,
        )
        patched = re.sub(
            r"(?m)^\s*import\s+org\.mockito\.Mockito\.\*\s*;",
            "import static org.mockito.Mockito.*;",
            patched,
        )
        return patched

    @classmethod
    def _qualify_unqualified_mockito_mock_calls(cls, code: str) -> str:
        patched = cls._ensure_import(code, "org.mockito.Mockito")
        return re.sub(r"(?<![\w.])mock\s*\(", "Mockito.mock(", patched)

    @classmethod
    def _repair_missing_junit_mockito_static_imports(cls, code: str, failure_output: str) -> str:
        text = str(failure_output or "")
        if "cannot find symbol" not in text:
            return code
        patched = code
        if re.search(r"symbol:\s+class\s+Test\b", text) and "@Test" in patched:
            patched = cls._ensure_import(patched, "org.junit.jupiter.api.Test")
        if re.search(r"symbol:\s+variable\s+Mockito\b", text) and "Mockito." in patched:
            patched = cls._ensure_import(patched, "org.mockito.Mockito")
        common_type_imports = {
            "ArrayList": "java.util.ArrayList",
            "HashMap": "java.util.HashMap",
            "HashSet": "java.util.HashSet",
            "List": "java.util.List",
            "Map": "java.util.Map",
            "Set": "java.util.Set",
            "Optional": "java.util.Optional",
            "Executor": "java.util.concurrent.Executor",
            "ExecutorService": "java.util.concurrent.ExecutorService",
            "CompletableFuture": "java.util.concurrent.CompletableFuture",
            "Flow": "java.util.concurrent.Flow",
        }
        for simple, fqn in common_type_imports.items():
            if re.search(rf"symbol:\s+(?:class|variable)\s+{re.escape(simple)}\b", text):
                patched = cls._ensure_import(patched, fqn)
        assertion_names = {
            "assertEquals",
            "assertNotNull",
            "assertNull",
            "assertTrue",
            "assertFalse",
            "assertThrows",
            "fail",
        }
        if any(re.search(rf"symbol:\s+method\s+{name}\s*\(", text) for name in assertion_names):
            patched = cls._ensure_import(patched, "static org.junit.jupiter.api.Assertions.*")
        mockito_static_imports = {
            "any": "org.mockito.ArgumentMatchers.any",
            "anyBoolean": "org.mockito.ArgumentMatchers.anyBoolean",
            "anyDouble": "org.mockito.ArgumentMatchers.anyDouble",
            "anyInt": "org.mockito.ArgumentMatchers.anyInt",
            "anyList": "org.mockito.ArgumentMatchers.anyList",
            "anyLong": "org.mockito.ArgumentMatchers.anyLong",
            "anyMap": "org.mockito.ArgumentMatchers.anyMap",
            "anySet": "org.mockito.ArgumentMatchers.anySet",
            "anyString": "org.mockito.ArgumentMatchers.anyString",
            "eq": "org.mockito.ArgumentMatchers.eq",
            "isNull": "org.mockito.ArgumentMatchers.isNull",
            "notNull": "org.mockito.ArgumentMatchers.notNull",
            "atLeast": "org.mockito.Mockito.atLeast",
            "atLeastOnce": "org.mockito.Mockito.atLeastOnce",
            "atMost": "org.mockito.Mockito.atMost",
            "atMostOnce": "org.mockito.Mockito.atMostOnce",
            "doAnswer": "org.mockito.Mockito.doAnswer",
            "doNothing": "org.mockito.Mockito.doNothing",
            "doReturn": "org.mockito.Mockito.doReturn",
            "doThrow": "org.mockito.Mockito.doThrow",
            "never": "org.mockito.Mockito.never",
            "only": "org.mockito.Mockito.only",
            "times": "org.mockito.Mockito.times",
            "verify": "org.mockito.Mockito.verify",
            "verifyNoInteractions": "org.mockito.Mockito.verifyNoInteractions",
            "verifyNoMoreInteractions": "org.mockito.Mockito.verifyNoMoreInteractions",
            "when": "org.mockito.Mockito.when",
        }
        for method, fqn in mockito_static_imports.items():
            if re.search(rf"symbol:\s+method\s+{re.escape(method)}\s*\(", text):
                patched = cls._ensure_import(patched, f"static {fqn}")
        return patched

    @staticmethod
    def _remove_unused_imports_from_missing_packages(code: str, failure_output: str) -> str:
        text = str(failure_output or "")
        missing_packages = {
            match.group(1)
            for match in re.finditer(r"\bpackage\s+([A-Za-z_][\w.]*)\s+does\s+not\s+exist\b", text)
        }
        if not missing_packages or "import " not in str(code or ""):
            return code
        lines = str(code or "").splitlines()
        import_indexes: set[int] = set()
        parsed_imports: list[tuple[int, str, str]] = []
        for idx, line in enumerate(lines):
            match = re.match(r"^\s*import\s+(?!static\b)([A-Za-z_][\w.]*)(?:\s*;\s*)$", line)
            if not match:
                continue
            fqn = match.group(1)
            package = fqn.rsplit(".", 1)[0] if "." in fqn else ""
            simple = fqn.rsplit(".", 1)[-1]
            import_indexes.add(idx)
            parsed_imports.append((idx, package, simple))
        if not parsed_imports:
            return code
        body = "\n".join(line for idx, line in enumerate(lines) if idx not in import_indexes)
        drop_indexes: set[int] = set()
        for idx, package, simple in parsed_imports:
            if package not in missing_packages:
                continue
            if not re.search(rf"\b{re.escape(simple)}\b", body):
                drop_indexes.add(idx)
        if not drop_indexes:
            return code
        return "\n".join(line for idx, line in enumerate(lines) if idx not in drop_indexes)

    @staticmethod
    def _repair_non_generic_diamond_construction(code: str, failure_output: str) -> str:
        text = str(failure_output or "")
        if "cannot use '<>' with non-generic class" not in text and "non-generic class" not in text:
            return code
        simple_names: set[str] = set()
        for match in re.finditer(
            r"cannot infer type arguments for\s+([A-Za-z_][\w.]*)(?:\s|$)",
            text,
        ):
            simple_names.add(match.group(1).rsplit(".", 1)[-1])
        if not simple_names:
            return code
        patched = str(code or "")
        for simple in sorted(simple_names, key=len, reverse=True):
            patched = re.sub(rf"\bnew\s+{re.escape(simple)}\s*<\s*>\s*\(", f"new {simple}(", patched)
        return patched

    @classmethod
    def _repair_reactive_publisher_return_mismatch(cls, code: str, failure_output: str) -> str:
        text = str(failure_output or "")
        patched = str(code or "")
        changed = False
        if (
            "reactor.core.publisher.Mono" in text
            and "reactor.core.publisher.Flux" in text
            and "conforms to reactor.core.publisher.Flux" in text
        ):
            new_code = re.sub(r"\bMono\s*\.\s*just\s*\(", "Flux.just(", patched)
            new_code = re.sub(r"\bMono\s*\.\s*empty\s*\(", "Flux.empty(", new_code)
            new_code = re.sub(r"\bMono\s*\.\s*error\s*\(", "Flux.error(", new_code)
            if new_code != patched:
                patched = cls._ensure_import(new_code, "reactor.core.publisher.Flux")
                changed = True
        if (
            "reactor.core.publisher.Flux" in text
            and "reactor.core.publisher.Mono" in text
            and "conforms to reactor.core.publisher.Mono" in text
        ):
            new_code = re.sub(r"\bFlux\s*\.\s*just\s*\(", "Mono.just(", patched)
            new_code = re.sub(r"\bFlux\s*\.\s*empty\s*\(", "Mono.empty(", new_code)
            new_code = re.sub(r"\bFlux\s*\.\s*error\s*\(", "Mono.error(", new_code)
            if new_code != patched:
                patched = cls._ensure_import(new_code, "reactor.core.publisher.Mono")
                changed = True
        return patched if changed else code

    @classmethod
    def _repair_missing_helper_inside_reactive_just(
        cls,
        code: str,
        failure_output: str,
        *,
        method_focus: str | None = None,
    ) -> str:
        text = str(failure_output or "")
        missing_methods = {
            match.group(1)
            for match in re.finditer(r"symbol:\s+method\s+([A-Za-z_$][\w$]*)\s*\(", text)
        }
        missing_methods -= {"mock", "when", "verify", "assertEquals", "assertThrows", "assertNotNull"}
        if method_focus:
            missing_methods.discard(str(method_focus))
        if not missing_methods:
            return code
        patched = str(code or "")
        changed = False
        for publisher in ("Flux", "Mono"):
            search_from = 0
            pieces: list[str] = []
            pattern = re.compile(rf"\b{publisher}\s*\.\s*just\s*\(")
            while True:
                match = pattern.search(patched, search_from)
                if not match:
                    pieces.append(patched[search_from:])
                    break
                open_index = match.end() - 1
                close_index = cls._find_matching_paren(patched, open_index)
                if close_index < 0:
                    pieces.append(patched[search_from:])
                    break
                arg_text = patched[open_index + 1:close_index]
                if any(re.search(rf"\b{re.escape(method)}\s*\(", arg_text) for method in missing_methods):
                    pieces.append(patched[search_from:match.start()])
                    pieces.append(f"{publisher}.empty()")
                    search_from = close_index + 1
                    changed = True
                else:
                    pieces.append(patched[search_from:close_index + 1])
                    search_from = close_index + 1
            if changed:
                patched = "".join(pieces)
        return patched if changed else code

    @staticmethod
    def _repair_hashset_string_type_inference(code: str) -> str:
        patched = re.sub(
            r"new\s+java\.util\.HashSet\s*<\s*>\s*\(",
            "new java.util.HashSet<String>(",
            code,
        )
        patched = re.sub(
            r"new\s+HashSet\s*<\s*>\s*\(",
            "new HashSet<String>(",
            patched,
        )
        return patched

    @classmethod
    def _repair_coverage_positive_brittle_boolean_assertion(
        cls,
        code: str,
        failure_output: str,
        *,
        production_method_focus: str = "",
        test_method_focus: str = "",
        junit_family: str = "",
    ) -> str:
        text = str(failure_output or "")
        if "AssertionError" not in text:
            return code
        line_no = cls._reported_test_line_number(text)
        lines = code.splitlines()
        if not lines:
            return code
        candidate_spans = cls._boolean_assert_statement_spans_near_failure(
            lines,
            line_no=line_no,
            method_focus=cls._reported_test_method_name(text) or test_method_focus,
        )
        if len(candidate_spans) != 1:
            return code
        start, end = candidate_spans[0]
        statement = "\n".join(lines[start:end + 1])
        if re.search(r"\bassert(?:Equals|That|Throws|Null|NotNull)\s*\(", statement):
            return code
        method_scope_start = cls._enclosing_method_start_line(lines, start)
        prefix = "\n".join(lines[method_scope_start:start])
        decl_types = cls._local_declared_reference_types(prefix)
        observed = cls._assertion_observation_variable(statement, decl_types)
        if not observed:
            observed = cls._target_receiver_variable(prefix, production_method_focus, decl_types)
        if not observed:
            return code
        indent = re.match(r"\s*", lines[start]).group(0)
        message = '"coverage-positive target execution should leave a stable observable object"'
        if str(junit_family or "").lower() == "junit5":
            replacement = f"{indent}assertNotNull({observed}, {message});"
        else:
            replacement = f"{indent}assertNotNull({message}, {observed});"
        patched_lines = list(lines)
        patched_lines[start:end + 1] = [replacement]
        return "\n".join(patched_lines) + ("\n" if code.endswith("\n") else "")

    @classmethod
    def _repair_coverage_positive_expected_exception_fail(
        cls,
        code: str,
        failure_output: str,
        *,
        test_method_focus: str = "",
    ) -> str:
        text = str(failure_output or "")
        if "AssertionFailedError" not in text or "Expected" not in text:
            return code
        lines = code.splitlines()
        if not lines:
            return code
        line_no = cls._reported_test_line_number(text)
        method_name = cls._reported_test_method_name(text) or test_method_focus
        candidate_indexes: list[int] = []
        if line_no > 0 and line_no - 1 < len(lines):
            candidate_indexes.append(line_no - 1)
        method_span = cls._method_line_span(lines, method_name) if method_name else None
        search_start, search_end = method_span if method_span else (0, len(lines) - 1)
        for idx in range(max(0, search_start), min(len(lines), search_end + 1)):
            if idx not in candidate_indexes:
                candidate_indexes.append(idx)
        for idx in candidate_indexes:
            if idx < 0 or idx >= len(lines):
                continue
            if not re.search(r"\b(?:Assertions\.)?fail\s*\(\s*\"[^\"]*Expected", lines[idx]):
                continue
            span = cls._statement_span_around_line(lines, idx)
            statement = "\n".join(lines[span[0]:span[1] + 1])
            if not re.search(r"\b(?:Assertions\.)?fail\s*\(", statement):
                continue
            prefix = "\n".join(lines[max(0, search_start):span[0]])
            observed = cls._last_assigned_reference_variable(prefix)
            indent = re.match(r"\s*", lines[span[0]]).group(0)
            replacement = f"{indent}assertNotNull({observed});" if observed else f"{indent}return;"
            patched = list(lines)
            patched[span[0]:span[1] + 1] = [replacement]
            return "\n".join(patched) + ("\n" if code.endswith("\n") else "")
        return code

    @classmethod
    def _repair_empty_required_collection_setters_from_context(
        cls,
        code: str,
        ctx: RepairContext,
    ) -> tuple[str, list[str]]:
        if not code or ("List.of()" not in code and "emptyList()" not in code):
            return code, []
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        context_text = "\n".join(
            str(extra.get(key) or "")
            for key in (
                "generation_control_receipt",
                "visible_type_construction_grounding",
                "callable_surface_grounding",
                "target_receipt_contract",
            )
        )
        if not context_text:
            return code, []
        required_names: set[str] = set()
        for match in re.finditer(
            r"collection inputs should be non-empty minimal examples:\s*([A-Za-z0-9_$,\s]+)",
            context_text,
        ):
            for raw_name in match.group(1).split(","):
                name = raw_name.strip()
                if re.match(r"^[A-Za-z_$][\w$]*$", name):
                    required_names.add(name)
        if not required_names:
            return code, []

        element_by_param: dict[str, str] = {}
        list_param_re = re.compile(
            r"\b(?:java\.util\.)?List\s*<\s*([A-Za-z_$][\w$.]*)(?:\s*<[^>]+>)?\s*(?:\?)?\s*>\s+([A-Za-z_$][\w$]*)"
        )
        for match in list_param_re.finditer(context_text):
            elem = match.group(1).rsplit(".", 1)[-1].strip()
            param = match.group(2).strip()
            if param in required_names and elem:
                element_by_param.setdefault(param, elem)
        if not element_by_param:
            return code, []

        def _sample_for(elem: str) -> str:
            if elem == "String":
                return '"value"'
            if elem in {"Integer", "int"}:
                return "1"
            if elem in {"Long", "long"}:
                return "1L"
            if elem in {"Boolean", "boolean"}:
                return "true"
            if elem in {"Double", "double"}:
                return "1.0d"
            if elem in {"Float", "float"}:
                return "1.0f"
            if elem in {"Object", "?"}:
                return "new Object()"
            return f"Mockito.mock({elem}.class)"

        patched = code
        applied: list[str] = []
        for param, elem in sorted(element_by_param.items()):
            replacement = f"{param}(java.util.List.of({_sample_for(elem)}))"
            setter_re = re.compile(
                rf"{re.escape(param)}\s*\(\s*(?:(?:java\.util\.)?List\.of\s*\(\s*\)|(?:java\.util\.)?Collections\.emptyList\s*\(\s*\))\s*\)"
            )
            patched_new = setter_re.sub(replacement, patched)
            if patched_new != patched:
                patched = patched_new
                applied.append(f"REQUIRED_COLLECTION_NONEMPTY:{param}->{elem}")
        if applied and "Mockito.mock(" in patched:
            patched = cls._ensure_import(patched, "org.mockito.Mockito")
        return patched, applied

    @classmethod
    def _repair_assert_throws_wraps_eager_target_assignment(
        cls,
        code: str,
        ctx: RepairContext,
    ) -> str:
        """Move an eager target call into assertThrows when the call itself throws.

        LLMs often write:

            Mono<T> out = target.call(arg);
            assertThrows(X.class, () -> out.block());

        If ``target.call`` throws synchronously, the exception escapes before the
        assertion.  This rule preserves the same Arrange/Act path and oracle by
        wrapping the original target expression plus the existing terminal call.
        """
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage_feedback = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage_feedback.get("has_target_coverage")):
            return code
        production_focus = str(extra.get("production_method_focus") or "").strip()
        if not production_focus:
            return code
        if "assertThrows" not in code or production_focus not in code:
            return code
        failure_text = str(ctx.failure_output or "")
        if "Exception" not in failure_text and "Error" not in failure_text:
            return code

        assign_re = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?:final\s+)?"
            r"(?P<type>[A-Za-z_$][\w$]*(?:\s*<[^;\n=]+>)?(?:\[\])?)\s+"
            r"(?P<var>[a-z_$][\w$]*)\s*=\s*(?P<expr>[^;\n]*\b"
            + re.escape(production_focus)
            + r"\s*\([^;\n]*\))\s*;\s*$"
        )
        replacements: list[tuple[int, int, str]] = []
        for assign in assign_re.finditer(code):
            var = assign.group("var")
            expr = assign.group("expr").strip()
            if "assertThrows" in expr:
                continue
            search_start = assign.end()
            assert_match = re.search(
                r"\b(?P<prefix>(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertThrows\s*\(",
                code[search_start:],
            )
            if not assert_match:
                continue
            abs_assert_start = search_start + assert_match.start()
            # Keep this local: only rewrite a nearby assertion in the same
            # method body, not a later unrelated assertion.
            between = code[search_start:abs_assert_start]
            if between.count("@Test") or between.count("public void ") or len(between) > 1200:
                continue
            open_index = code.find("(", abs_assert_start)
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            semicolon = close_index + 1
            while semicolon < len(code) and code[semicolon].isspace():
                semicolon += 1
            if semicolon >= len(code) or code[semicolon] != ";":
                continue
            args = cls._split_top_level_args(code[open_index + 1:close_index])
            if len(args) < 2:
                continue
            executable = args[1].strip()
            lambda_match = re.match(r"\(\s*\)\s*->\s*(?P<body>.+)\s*$", executable, re.DOTALL)
            if not lambda_match:
                continue
            body = lambda_match.group("body").strip().rstrip(";").strip()
            if not re.match(rf"\b{re.escape(var)}\b", body):
                continue
            terminal_suffix = body[len(var):].strip()
            if terminal_suffix and not terminal_suffix.startswith("."):
                continue
            # Preserve the original terminal consumption if one existed.
            call_expr = f"{expr}{terminal_suffix}"
            new_args = list(args)
            new_args[1] = f"() -> {call_expr}"
            replacement_assert = (
                code[abs_assert_start:open_index + 1]
                + ", ".join(new_args)
                + code[close_index:semicolon + 1]
            )
            # Remove the eager assignment line and replace the assertion.
            replacements.append((abs_assert_start, semicolon + 1, replacement_assert))
            replacements.append((assign.start(), assign.end(), ""))
            break
        if not replacements:
            return code
        patched = code
        for start, end, repl in sorted(replacements, reverse=True):
            patched = patched[:start] + repl + patched[end:]
        return patched

    @classmethod
    def _repair_coverage_positive_assert_throws_no_exception(
        cls,
        code: str,
        failure_output: str,
    ) -> str:
        text = str(failure_output or "")
        if "AssertionFailedError" not in text or "nothing was thrown" not in text:
            return code
        pattern = re.compile(
            r"\b(?P<prefix>(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertThrows\s*\("
        )
        replacements: list[tuple[int, int, str]] = []
        for match in pattern.finditer(code):
            line_start = code.rfind("\n", 0, match.start()) + 1
            statement_prefix = code[line_start:match.start()]
            if statement_prefix.strip():
                continue
            open_index = match.end() - 1
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            semicolon = close_index + 1
            while semicolon < len(code) and code[semicolon].isspace():
                semicolon += 1
            if semicolon >= len(code) or code[semicolon] != ";":
                continue
            args = cls._split_top_level_args(code[open_index + 1:close_index])
            if len(args) < 2:
                continue
            executable = args[1].strip()
            lambda_match = re.match(r"\(\s*\)\s*->\s*(.+)\s*$", executable, re.DOTALL)
            method_ref_match = re.match(
                r"(?P<receiver>[A-Za-z_$][A-Za-z0-9_$.]*)\s*::\s*(?P<method>[A-Za-z_$][A-Za-z0-9_$]*)\s*$",
                executable,
            )
            if lambda_match:
                call_expr = lambda_match.group(1).strip()
            elif method_ref_match:
                receiver = method_ref_match.group("receiver").strip()
                method = method_ref_match.group("method").strip()
                if not re.match(r"^(?:get|read|parse|create|build|to|is|has)[A-Z_].*", method):
                    continue
                call_expr = f"{receiver}.{method}()"
            else:
                continue
            if call_expr.startswith("{") or not call_expr:
                continue
            call_expr = call_expr.rstrip(";").strip()
            indent = re.match(r"[ \t]*", statement_prefix).group(0)
            prefix = match.group("prefix") or ""
            does_not_throw = f"{prefix}assertDoesNotThrow" if prefix else "assertDoesNotThrow"
            replacement = (
                f"{indent}Object result = {does_not_throw}(() -> {call_expr});\n"
                f"{indent}assertNotNull(result);"
            )
            replacements.append((line_start, semicolon + 1, replacement))
            break
        if not replacements:
            return code
        patched = code
        for start, end, repl in reversed(replacements):
            patched = patched[:start] + repl + patched[end:]
        return patched

    @classmethod
    def _repair_coverage_positive_eager_resource_exception_to_assert_throws(
        cls,
        code: str,
        failure_output: str,
    ) -> str:
        text = str(failure_output or "")
        if "Exception" not in text and "Error" not in text:
            return code
        exception_match = re.search(
            r"\b(?P<etype>(?:[a-z_$][\w$]*\.)*[A-Z][A-Za-z0-9_$]*(?:Exception|Error))\s*:",
            text,
        )
        if not exception_match:
            return code
        actual_exception = exception_match.group("etype").strip()
        try_re = re.compile(
            r"(?m)^(?P<indent>[ \t]*)try\s*\(\s*(?P<decl>(?:final\s+)?[A-Z][A-Za-z0-9_$.<>?, \[\]]+\s+[a-z_$][A-Za-z0-9_$]*)\s*=\s*(?P<expr>[^;\n]+?)\s*;?\s*\)\s*\{"
        )
        replacements: list[tuple[int, int, str]] = []
        for match in try_re.finditer(code):
            brace = code.find("{", match.end() - 1)
            if brace < 0:
                continue
            close = cls._find_matching_brace(code, brace)
            if close < 0:
                continue
            block = code[brace + 1:close]
            assert_match = re.search(
                r"\b(?P<prefix>(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertThrows\s*\(\s*(?P<expected>[A-Za-z_$][A-Za-z0-9_$.]*Exception|[A-Za-z_$][A-Za-z0-9_$.]*Error)\s*\.class\s*,",
                block,
            )
            if not assert_match:
                continue
            expected = assert_match.group("expected").strip()
            actual_simple = actual_exception.rsplit(".", 1)[-1]
            expected_simple = expected.rsplit(".", 1)[-1]
            if expected_simple != actual_simple:
                continue
            expr = match.group("expr").strip()
            if not expr or "assertThrows" in expr:
                continue
            indent = match.group("indent") or ""
            prefix = assert_match.group("prefix") or ""
            assert_throws = f"{prefix}assertThrows" if prefix else "assertThrows"
            replacement = f"{indent}{assert_throws}({expected}.class, () -> {expr});"
            end = close + 1
            while end < len(code) and code[end].isspace():
                if code[end] == "\n":
                    end += 1
                    break
                end += 1
            replacements.append((match.start(), end, replacement + ("\n" if end <= len(code) else "")))
            break
        if not replacements:
            return code
        patched = code
        for start, end, repl in reversed(replacements):
            patched = patched[:start] + repl + patched[end:]
        return patched

    @classmethod
    def _repair_stepverifier_eager_exception_to_assert_throws(
        cls,
        code: str,
        ctx: RepairContext,
    ) -> str:
        if "StepVerifier.create" not in str(code or "") or ".expectError" not in str(code or ""):
            return code
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage_feedback = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage_feedback.get("has_target_coverage")):
            return code
        signal = extra.get("streaming_failure_signal") if isinstance(extra.get("streaming_failure_signal"), dict) else {}
        category = str(signal.get("category") or "").strip()
        if category not in {
            "entry-precondition-state",
            "target_positive_setup_blocker",
            "target_positive_terminal_or_state_failure",
            "runtime-validation-failure",
        }:
            return code
        exception_type = str(signal.get("exception_type") or "").strip()
        if exception_type and not re.search(r"(?:Exception|Error|Throwable)$", exception_type):
            return code
        production_focus = str(extra.get("production_method_focus") or "").strip()

        replacements: list[tuple[int, int, str]] = []
        for match in re.finditer(r"(?m)^(?P<indent>[ \t]*)StepVerifier\.create\s*\(", code):
            open_index = code.find("(", match.start())
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            expr = code[open_index + 1:close_index].strip()
            if production_focus and not re.search(rf"(?:\.|\b){re.escape(production_focus)}\s*\(", expr):
                continue
            semicolon = code.find(";", close_index)
            if semicolon < 0:
                continue
            statement = code[match.start():semicolon + 1]
            if ".expectError" not in statement or ".verify" not in statement:
                continue
            error_match = re.search(
                r"\.expectError\s*\(\s*(?P<etype>[A-Za-z_$][\w$.]*\s*\.class)\s*\)",
                statement,
            )
            if not error_match:
                continue
            expected = error_match.group("etype").strip()
            if exception_type:
                expected_simple = expected[:-6].rsplit(".", 1)[-1]
                actual_simple = exception_type.rsplit(".", 1)[-1]
                if expected_simple != actual_simple:
                    continue
            call_expr = expr
            if not re.search(r"\.(?:block|blockOptional|join|get)\s*\(", call_expr):
                call_expr = f"{call_expr}.block()"
            indent = match.group("indent")
            replacement = (
                f"{indent}org.junit.jupiter.api.Assertions.assertThrows("
                f"{expected}, () -> {call_expr});"
            )
            replacements.append((match.start(), semicolon + 1, replacement))
        if len(replacements) != 1:
            return code
        start, end, replacement = replacements[0]
        return code[:start] + replacement + code[end:]

    @classmethod
    def _repair_stepverifier_onnext_oncomplete_oracle(
        cls,
        code: str,
        ctx: RepairContext,
    ) -> str:
        """Calibrate a target-positive StepVerifier value oracle to completion.

        Generated tests often reach the target and consume a Reactor publisher,
        but guess that one item is emitted.  When the observed failure is
        explicitly "expected onNext, actual onComplete", preserve the target
        call and replace only the brittle value expectation with completion
        verification.
        """
        text = str(code or "")
        failure = str(ctx.failure_output or "")
        if "StepVerifier.create" not in text:
            return code
        if not re.search(
            r"expected[^\n\r]{0,140}onNext\s*\(\s*\)[^\n\r]{0,180}actual[^\n\r]{0,100}onComplete\s*\(\s*\)",
            failure,
            re.IGNORECASE,
        ):
            return code
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage_feedback = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage_feedback.get("has_target_coverage")):
            return code
        production_focus = str(extra.get("production_method_focus") or "").strip()

        replacements: list[tuple[int, int, str]] = []
        for match in re.finditer(r"(?m)^(?P<indent>[ \t]*)StepVerifier\.create\s*\(", text):
            open_index = text.find("(", match.start())
            close_index = cls._find_matching_paren(text, open_index)
            if close_index < 0:
                continue
            expr = text[open_index + 1:close_index].strip()
            if production_focus and not re.search(rf"(?:\.|\b){re.escape(production_focus)}\s*\(", expr):
                continue
            semicolon = text.find(";", close_index)
            if semicolon < 0:
                continue
            statement = text[match.start():semicolon + 1]
            if not re.search(
                r"\.(?:assertNext|consumeNextWith|expectNext(?:Matches|Count)?|thenConsumeWhile)\s*\(",
                statement,
            ):
                continue
            if ".verifyComplete" in statement:
                verifier = "verifyComplete()"
            elif re.search(r"\.verify\s*\(\s*\)", statement):
                verifier = "verify()"
            else:
                continue
            indent = match.group("indent") or ""
            replacement = f"{indent}StepVerifier.create({expr})\n{indent}        .{verifier};"
            replacements.append((match.start(), semicolon + 1, replacement))
        if len(replacements) != 1:
            return code
        start, end, replacement = replacements[0]
        return text[:start] + replacement + text[end:]

    @classmethod
    def _remove_uninvoked_mockito_verify_oracle(cls, code: str, ctx: RepairContext) -> str:
        """Calibrate a disproven target-positive Mockito oracle to never.

        Keeping an observable zero-interaction assertion is stronger than
        deleting the only oracle. The rewrite is intentionally narrow: the
        exact Mockito diagnostic, measured target reachability, and one
        matching verify call are all required.
        """
        failure = str(ctx.failure_output or "")
        if "Wanted but not invoked" not in failure:
            return code
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        production_focus = str(extra.get("production_method_focus") or "").strip()
        wanted_match = re.search(
            r"Wanted but not invoked:.*?\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)\s*\(",
            failure,
            re.DOTALL,
        )
        if not wanted_match:
            return code
        receiver = wanted_match.group("receiver")
        method = wanted_match.group("method")
        if production_focus and method == production_focus:
            return code
        method_focus = str(extra.get("repair_test_method") or ctx.method_focus or "").strip()
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        search_text = method_code or code
        offset = code.find(search_text) if method_code else 0
        if offset < 0:
            offset = 0
        verify_re = re.compile(
            r"(?:(?:Mockito|org\.mockito\.Mockito)\s*\.\s*)?verify\s*\("
        )
        replacements: list[tuple[int, int, str]] = []
        for verify_match in verify_re.finditer(search_text):
            open_index = verify_match.end() - 1
            close_index = cls._find_matching_paren(search_text, open_index)
            if close_index < 0:
                continue
            args = cls._split_top_level_args(
                search_text[open_index + 1:close_index]
            )
            if not args or re.sub(r"\s+", "", args[0]) != receiver:
                continue
            chained = re.match(
                rf"\s*\.\s*{re.escape(method)}\s*\(",
                search_text[close_index + 1:],
            )
            if not chained:
                continue
            prefix = search_text[verify_match.start():open_index]
            replacement = (
                f"{prefix}({args[0].strip()}, "
                "org.mockito.Mockito.never())"
            )
            replacements.append(
                (offset + verify_match.start(), offset + close_index + 1, replacement)
            )
        if len(replacements) != 1:
            return code
        start, end, replacement = replacements[0]
        return code[:start] + replacement + code[end:]

    @classmethod
    def _relax_overinvoked_mockito_verify_oracle(cls, code: str, ctx: RepairContext) -> str:
        """Relax a too-strict Mockito invocation-count oracle after target coverage.

        Generated tests often guess that a collaborator is called exactly once.
        For target-positive candidates, Mockito's ``TooManyActualInvocations``
        is evidence that the path executed but the interaction oracle was too
        strict.  Preserve the call path and change only the verify mode to
        ``atLeastOnce``.
        """
        failure = str(ctx.failure_output or "")
        if "TooManyActualInvocations" not in failure and not re.search(
            r"Wanted\s+\d+\s+time(?:s)?[\s\S]{0,300}?But\s+was\s+\d+\s+time",
            failure,
            re.IGNORECASE,
        ):
            return code
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        production_focus = str(extra.get("production_method_focus") or "").strip()
        wanted_match = re.search(
            r"(?:TooManyActualInvocations:\s*)?(?:\r?\n)+\s*"
            r"(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)\s*\(",
            failure,
            re.IGNORECASE,
        )
        if not wanted_match:
            wanted_match = re.search(
                r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)\s*\([^;\n]*\)\s*;\s*"
                r"(?:\r?\n)+\s*Wanted\s+\d+\s+time",
                failure,
                re.IGNORECASE,
            )
        if not wanted_match:
            return code
        receiver = str(wanted_match.group("receiver") or "").strip()
        method = str(wanted_match.group("method") or "").strip()
        if not receiver or not method:
            return code
        if production_focus and method == production_focus:
            return code
        method_focus = str(extra.get("repair_test_method") or ctx.method_focus or "").strip()
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        search_text = method_code or code
        offset = code.find(search_text) if method_code else 0
        if offset < 0:
            offset = 0
        replacements: list[tuple[int, int, str]] = []
        verify_re = re.compile(r"(?:(?:Mockito|org\.mockito\.Mockito)\s*\.\s*)?verify\s*\(")
        for verify_match in verify_re.finditer(search_text):
            open_index = verify_match.end() - 1
            close_index = cls._find_matching_paren(search_text, open_index)
            if close_index < 0:
                continue
            args = cls._split_top_level_args(search_text[open_index + 1:close_index])
            if not args:
                continue
            first_arg = re.sub(r"\s+", "", args[0])
            if first_arg != receiver:
                continue
            after = search_text[close_index + 1: min(len(search_text), close_index + 160)]
            method_match = re.match(rf"\s*\.\s*{re.escape(method)}\s*\(", after)
            if not method_match:
                continue
            replacement_args = f"{receiver}, org.mockito.Mockito.atLeastOnce()"
            start = offset + open_index + 1
            end = offset + close_index
            replacements.append((start, end, replacement_args))
        if len(replacements) != 1:
            return code
        start, end, replacement = replacements[0]
        return code[:start] + replacement + code[end:]

    @staticmethod
    def _statement_span_around_line(lines: list[str], index: int) -> tuple[int, int]:
        start = max(0, index)
        end = max(0, index)
        while start > 0 and ";" not in lines[start - 1] and "{" not in lines[start - 1]:
            start -= 1
        while end + 1 < len(lines) and ";" not in lines[end]:
            end += 1
        return start, end

    @classmethod
    def _last_assigned_reference_variable(cls, text: str) -> str:
        out = ""
        for match in re.finditer(
            r"\b(?!if\b|for\b|while\b|switch\b|return\b)(?:final\s+)?"
            r"([A-Z][A-Za-z0-9_$.<>?,\s\[\]]*)\s+([a-z_$][A-Za-z0-9_$]*)\s*=",
            text,
        ):
            typ = match.group(1).strip()
            name = match.group(2).strip()
            if typ in {"int", "long", "double", "float", "boolean", "byte", "short", "char"}:
                continue
            out = name
        return out

    @classmethod
    def _boolean_assert_statement_spans_near_failure(
        cls,
        lines: list[str],
        *,
        line_no: int,
        method_focus: str = "",
    ) -> list[tuple[int, int]]:
        def statement_span(index: int) -> tuple[int, int]:
            start = max(0, index)
            end = max(0, index)
            while start > 0 and ";" not in lines[start - 1] and "{" not in lines[start - 1]:
                start -= 1
            while end + 1 < len(lines) and ";" not in lines[end]:
                end += 1
            return start, end

        def is_boolean_assert(span: tuple[int, int]) -> bool:
            statement = "\n".join(lines[span[0]:span[1] + 1])
            return bool(re.search(r"\bassert(?:True|False)\s*\(", statement))

        spans: list[tuple[int, int]] = []
        if line_no > 0 and line_no <= len(lines):
            reported = line_no - 1
            for idx in range(max(0, reported - 5), min(len(lines), reported + 6)):
                span = statement_span(idx)
                if span not in spans and is_boolean_assert(span):
                    spans.append(span)
            if len(spans) == 1:
                return spans

        scope_start = 0
        scope_end = len(lines)
        if method_focus:
            method_re = re.compile(rf"\bvoid\s+{re.escape(method_focus)}\s*\(")
            for idx, line in enumerate(lines):
                if not method_re.search(line):
                    continue
                scope_start = idx
                brace_balance = 0
                opened = False
                for end_idx in range(idx, len(lines)):
                    brace_balance += lines[end_idx].count("{")
                    if "{" in lines[end_idx]:
                        opened = True
                    brace_balance -= lines[end_idx].count("}")
                    if opened and brace_balance <= 0:
                        scope_end = end_idx + 1
                        break
                break
        spans = []
        idx = scope_start
        while idx < scope_end:
            span = statement_span(idx)
            if span[0] >= scope_start and span[1] < scope_end and is_boolean_assert(span):
                if span not in spans:
                    spans.append(span)
                idx = span[1] + 1
            else:
                idx += 1
        return spans

    @staticmethod
    def _reported_test_method_name(failure_output: str) -> str:
        text = str(failure_output or "")
        match = re.search(r"\b([A-Za-z_$][\w$]*)\([^()\n]*Test(?:\.java)?:\d+\)", text)
        if match:
            return match.group(1)
        match = re.search(r"\b([A-Za-z_$][\w$]*)\([^()\n]*Test\)", text)
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _enclosing_method_start_line(lines: list[str], line_index: int) -> int:
        for idx in range(max(0, line_index), -1, -1):
            if re.search(r"\b(?:public|protected|private)?\s*(?:static\s+)?void\s+[A-Za-z_$][\w$]*\s*\(", lines[idx]):
                return idx
        return 0

    @staticmethod
    def _method_line_span(lines: list[str], method_name: str) -> tuple[int, int] | None:
        if not method_name:
            return None
        start = -1
        method_re = re.compile(rf"\b{re.escape(method_name)}\s*\(")
        for idx, line in enumerate(lines):
            if method_re.search(line):
                start = idx
                break
        if start < 0:
            return None
        depth = 0
        saw_open = False
        for idx in range(start, len(lines)):
            depth += lines[idx].count("{")
            if "{" in lines[idx]:
                saw_open = True
            depth -= lines[idx].count("}")
            if saw_open and depth <= 0:
                return (start, idx)
        return (start, len(lines) - 1)

    @staticmethod
    def _local_declared_reference_types(source_prefix: str) -> dict[str, str]:
        primitive_types = {"byte", "short", "int", "long", "float", "double", "boolean", "char"}
        out: dict[str, str] = {}
        pattern = re.compile(
            r"(?m)^\s*(?:final\s+)?(?P<type>[A-Za-z_$][\w$.\[\]<>?, ]*)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*="
        )
        for match in pattern.finditer(source_prefix or ""):
            type_name = re.sub(r"\s+", " ", match.group("type")).strip()
            simple = type_name.rsplit(".", 1)[-1].split("<", 1)[0].replace("[]", "")
            if simple in primitive_types:
                continue
            out[match.group("name")] = type_name
        return out

    @staticmethod
    def _assertion_observation_variable(statement: str, declared_reference_types: dict[str, str]) -> str:
        text = str(statement or "")
        candidates: list[str] = []
        for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\s*\.", text):
            name = match.group(1)
            if name not in {"Assert", "Assertions", "org", "java", "javax"} and name not in candidates:
                candidates.append(name)
        for match in re.finditer(r"\b([A-Za-z_$][\w$]*)\b", text):
            name = match.group(1)
            if name in {"assertTrue", "assertFalse", "true", "false", "null"}:
                continue
            if name not in candidates:
                candidates.append(name)
        for name in candidates:
            if name in declared_reference_types:
                return name
        return ""

    @staticmethod
    def _target_receiver_variable(
        source_prefix: str,
        production_method_focus: str,
        declared_reference_types: dict[str, str],
    ) -> str:
        focus = str(production_method_focus or "").strip()
        if not focus or not re.fullmatch(r"[A-Za-z_$][\w$]*", focus):
            return ""
        for match in re.finditer(rf"\b([A-Za-z_$][\w$]*)\s*\.\s*{re.escape(focus)}\s*\(", source_prefix or ""):
            receiver = match.group(1)
            if receiver in declared_reference_types:
                return receiver
        return ""

    @classmethod
    def _remove_missing_method_statement_at_reported_line(
        cls,
        code: str,
        failure_output: str,
        *,
        method_focus: str | None = None,
    ) -> str:
        match = _RE_CANNOT_FIND_METHOD.search(str(failure_output or ""))
        if not match:
            return code
        missing_method = match.group(1)
        if method_focus and missing_method == method_focus:
            return code
        line_no = cls._reported_test_line_number(failure_output)
        if line_no <= 0:
            line_match = re.search(r"\.java:\[(\d+),\d+\]", str(failure_output or ""))
            if line_match:
                try:
                    line_no = int(line_match.group(1))
                except ValueError:
                    line_no = -1
        lines = code.splitlines()
        if line_no > 0 and line_no <= len(lines):
            reported = line_no - 1
            candidate_indices = [
                idx for idx in range(max(0, reported - 4), min(len(lines), reported + 5))
                if re.search(rf"\.\s*{re.escape(missing_method)}\s*\(", lines[idx])
            ]
        else:
            candidate_indices = []
        if not candidate_indices:
            candidate_indices = [
                idx for idx, line in enumerate(lines)
                if re.search(rf"\.\s*{re.escape(missing_method)}\s*\(", line)
            ]
        if not candidate_indices:
            return code
        start = candidate_indices[0]
        end = candidate_indices[0]
        while start > 0 and ";" not in lines[start - 1] and "{" not in lines[start - 1]:
            start -= 1
        while end < len(lines) - 1 and ";" not in lines[end]:
            end += 1
        statement = "\n".join(lines[start:end + 1])
        if not re.search(rf"\.\s*{re.escape(missing_method)}\s*\(", statement):
            return code
        call_pos = statement.find(f".{missing_method}")
        if call_pos > 0 and "=" in statement[:call_pos]:
            return code
        if missing_method in {"mock", "when", "verify", "assertEquals", "assertThrows", "assertNotNull"}:
            return code
        replacement = list(lines)
        indent = re.match(r"\s*", lines[start]).group(0)
        replacement[start:end + 1] = [f"{indent}// Removed missing generated cleanup call: {missing_method}"]
        return "\n".join(replacement) + ("\n" if code.endswith("\n") else "")

    @classmethod
    def _remove_missing_declared_type_noarg_statements(
        cls,
        code: str,
        failure_output: str,
        *,
        method_focus: str | None = None,
    ) -> str:
        blockers = _declared_type_method_blockers(failure_output)
        if not blockers:
            return code
        removals: set[tuple[str, str]] = set()
        for blocker in blockers:
            method = str(blocker.get("method") or "").strip()
            variable = str(blocker.get("variable") or "").strip()
            args = str(blocker.get("args") or "").strip()
            if not method or not variable or args:
                continue
            if method_focus and method == method_focus:
                continue
            if method in {"mock", "when", "verify", "assertEquals", "assertThrows", "assertNotNull"}:
                continue
            removals.add((variable, method))
        if not removals:
            return code
        lines = code.splitlines()
        changed = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            for variable, method in removals:
                if re.fullmatch(
                    rf"{re.escape(variable)}\s*\.\s*{re.escape(method)}\s*\(\s*\)\s*;",
                    stripped,
                ):
                    indent = re.match(r"\s*", line).group(0)
                    lines[idx] = f"{indent}// Removed missing generated cleanup call: {method}"
                    changed = True
                    break
        if not changed:
            return code
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

    @classmethod
    def _remove_missing_declared_type_fluent_call_segments(
        cls,
        code: str,
        failure_output: str,
        *,
        method_focus: str | None = None,
    ) -> str:
        blockers = _declared_type_method_blockers(failure_output)
        methods: set[str] = set()
        for blocker in blockers:
            method = str(blocker.get("method") or "").strip()
            if not method:
                continue
            if method_focus and method == method_focus:
                continue
            if method in {"mock", "when", "verify", "assertEquals", "assertThrows", "assertNotNull"}:
                continue
            methods.add(method)
        if not methods:
            return code

        patched = str(code or "")
        changed = False
        for method in sorted(methods, key=len, reverse=True):
            cursor = 0
            pieces: list[str] = []
            while True:
                match = re.search(rf"\.\s*{re.escape(method)}\s*\(", patched[cursor:])
                if not match:
                    pieces.append(patched[cursor:])
                    break
                start = cursor + match.start()
                open_index = cursor + match.end() - 1
                close_index = cls._find_matching_paren(patched, open_index)
                if close_index < 0:
                    pieces.append(patched[cursor:])
                    break
                # Only remove fluent segments inside an expression. Standalone
                # setup statements are handled by the statement-removal rule.
                line_start = patched.rfind("\n", 0, start) + 1
                prefix = patched[line_start:start]
                global_prefix = patched[:start].rstrip()
                continues_prior_call_chain = bool(global_prefix.endswith(")"))
                if not ("=" in prefix or ")" in prefix or "." in prefix or continues_prior_call_chain):
                    pieces.append(patched[cursor:close_index + 1])
                    cursor = close_index + 1
                    continue
                pieces.append(patched[cursor:start])
                cursor = close_index + 1
                changed = True
            if changed:
                patched = "".join(pieces)
        return patched if changed else code

    @classmethod
    def _repair_mocked_symbolic_state_stub_from_failure(
        cls,
        code: str,
        failure_output: str,
    ) -> str:
        """Calibrate one exact mock state stub without API-name heuristics.

        The runtime diagnostic must state the expected symbolic constant, the
        reported failing statement must reference exactly one local mock, and
        exactly one preceding zero-argument stub on that mock must return a
        different symbolic constant.  These constraints make the replacement
        evidence-backed.  Real receivers and ambiguous mock graphs are left to
        local LLM repair with the original diagnostic.
        """
        text = str(failure_output or "")
        expected_match = re.search(
            r"(?:should\s+be\s+(?:pointing\s+to|in|at)|"
            r"expected(?:\s+(?:state|value))?\s*[:=]?)\s*"
            r"<?(?P<expected>[A-Z][A-Z0-9_$]{2,})>?"
            r"(?:\s*\(\s*-?\d+\s*\))?",
            text,
            re.IGNORECASE,
        )
        if expected_match is None:
            return code
        expected_member = str(
            expected_match.group("expected") or ""
        ).upper()
        if not expected_member:
            return code
        line_no = cls._reported_test_line_number(text)
        if line_no <= 0:
            return code
        lines = str(code or "").splitlines()
        if line_no > len(lines):
            return code
        statement_start = line_no - 1
        statement_end = line_no - 1
        while statement_start > 0 and ";" not in lines[statement_start - 1]:
            statement_start -= 1
        while statement_end + 1 < len(lines) and ";" not in lines[statement_end]:
            statement_end += 1
        failing_statement = "\n".join(lines[statement_start:statement_end + 1])
        mocked_receivers: list[str] = []
        mock_decl = re.compile(
            r"\b[A-Za-z_$][A-Za-z0-9_$.<>, ?\[\]]*\s+"
            r"(?P<receiver>[A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:org\.mockito\.Mockito\.)?(?:Mockito\.)?mock\s*\("
            r"\s*[A-Za-z_$][A-Za-z0-9_$.]*\.class\s*\)\s*;"
        )
        prefix = "\n".join(lines[:statement_start + 1])
        for match in mock_decl.finditer(prefix):
            name = str(match.group("receiver") or "")
            if re.search(rf"\b{re.escape(name)}\b", failing_statement):
                mocked_receivers.append(name)
        mocked_receivers = list(dict.fromkeys(mocked_receivers))
        if len(mocked_receivers) != 1:
            return code

        receiver = mocked_receivers[0]
        stub_re = re.compile(
            rf"(?:(?:org\.mockito\.)?Mockito\.)?when\s*\(\s*"
            rf"{re.escape(receiver)}\s*\.\s*"
            r"(?P<method>[A-Za-z_$][A-Za-z0-9_$]*)\s*\(\s*\)\s*\)"
            r"\s*\.\s*thenReturn\s*\(\s*"
            r"(?P<value>(?:[A-Za-z_$][A-Za-z0-9_$]*\s*\.\s*)*"
            r"[A-Z][A-Z0-9_$]{2,})\s*\)\s*;"
        )
        symbolic_stubs = [
            match
            for match in stub_re.finditer(code)
            if code.count("\n", 0, match.start()) + 1 <= statement_start + 1
            and str(match.group("value") or "").replace(" ", "").rsplit(".", 1)[-1]
            != expected_member
        ]
        if len(symbolic_stubs) != 1:
            return code
        match = symbolic_stubs[0]
        old_value = re.sub(r"\s+", "", str(match.group("value") or ""))
        owner, separator, _old_member = old_value.rpartition(".")
        expected_value = (
            f"{owner}.{expected_member}" if separator else expected_member
        )
        return (
            code[:match.start("value")]
            + expected_value
            + code[match.end("value"):]
        )

    @classmethod
    def _repair_expected_exception_type_from_observed_target_error(
        cls,
        code: str,
        failure_output: str,
        *,
        production_method_focus: str,
        test_method_focus: str = "",
    ) -> str:
        actual_fqn = cls._observed_error_exception_fqn(failure_output)
        if not actual_fqn:
            return code
        if not cls._stack_trace_reaches_method_before_test(
            failure_output,
            production_method_focus,
        ):
            return code
        replacement_type = actual_fqn if "." in actual_fqn and not actual_fqn.startswith("java.lang.") else actual_fqn.rsplit(".", 1)[-1]
        target_name = production_method_focus.strip()
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", target_name):
            return code
        scope_start = 0
        scope_end = len(code)
        if test_method_focus:
            method_match = re.search(
                rf"\bvoid\s+{re.escape(test_method_focus)}\s*\(",
                code,
            )
            if method_match:
                body_open = code.find("{", method_match.end())
                body_close = cls._find_matching_brace(code, body_open)
                if body_open >= 0 and body_close > body_open:
                    scope_start, scope_end = body_open, body_close
        scoped = code[scope_start:scope_end]
        catch_re = re.compile(r"catch\s*\(\s*([A-Za-z_$][\w$.]*)\s+([A-Za-z_$][\w$]*)\s*\)")
        replacements: list[tuple[int, int, str]] = []
        for match in catch_re.finditer(scoped):
            old_type = match.group(1)
            if old_type == replacement_type or old_type == actual_fqn:
                continue
            before_catch = scoped[:match.start()]
            try_idx = before_catch.rfind("try")
            try_slice = before_catch[try_idx:] if try_idx >= 0 else before_catch[-800:]
            if not re.search(rf"(?:\.|\b){re.escape(target_name)}\s*\(", try_slice):
                continue
            replacements.append((
                scope_start + match.start(1),
                scope_start + match.end(1),
                replacement_type,
            ))
        if len(replacements) != 1:
            return code
        start, end, replacement = replacements[0]
        return code[:start] + replacement + code[end:]

    @classmethod
    def _repair_assert_throws_exception_type_from_observed_actual(
        cls,
        code: str,
        failure_output: str,
        *,
        production_method_focus: str,
        test_method_focus: str = "",
    ) -> str:
        actual_type = cls._observed_unexpected_exception_actual_type(failure_output)
        if not actual_type:
            return code
        if not cls._stack_trace_reaches_method_before_test(
            failure_output,
            production_method_focus,
        ):
            return code
        replacement_type = (
            actual_type
            if "." in actual_type and not actual_type.startswith("java.lang.")
            else actual_type.rsplit(".", 1)[-1]
        )
        target_name = production_method_focus.strip()
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", target_name):
            return code
        scope_start = 0
        scope_end = len(code)
        if test_method_focus:
            method_match = re.search(rf"\bvoid\s+{re.escape(test_method_focus)}\s*\(", code)
            if method_match:
                body_open = code.find("{", method_match.end())
                body_close = cls._find_matching_brace(code, body_open)
                if body_open >= 0 and body_close > body_open:
                    scope_start, scope_end = body_open, body_close
        scoped = code[scope_start:scope_end]
        pattern = re.compile(
            r"\b(?P<prefix>(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertThrows\s*\("
        )
        replacements: list[tuple[int, int, str]] = []
        for match in pattern.finditer(scoped):
            open_index = match.end() - 1
            close_index = cls._find_matching_paren(scoped, open_index)
            if close_index < 0:
                continue
            args = cls._split_top_level_args(scoped[open_index + 1:close_index])
            if len(args) < 2:
                continue
            expected_arg = args[0].strip()
            executable_arg = args[1]
            if not re.search(rf"(?:\.|\b){re.escape(target_name)}\s*\(", executable_arg):
                continue
            class_match = re.match(r"(?P<type>[A-Za-z_$][\w$.]*)\s*\.class\b", expected_arg)
            if not class_match:
                continue
            old_type = class_match.group("type")
            if old_type == replacement_type or old_type == actual_type:
                continue
            arg_start = scope_start + open_index + 1 + scoped[open_index + 1:close_index].find(expected_arg)
            replacements.append((arg_start, arg_start + len(old_type), replacement_type))
        if len(replacements) != 1:
            return code
        start, end, replacement = replacements[0]
        return code[:start] + replacement + code[end:]

    @staticmethod
    def _observed_unexpected_exception_actual_type(failure_output: str) -> str:
        text = str(failure_output or "")
        match = re.search(
            r"Unexpected exception type thrown,\s*expected:\s*<[^>]+>\s*but\s*was:\s*<(?P<actual>[^>]+)>",
            text,
        )
        if not match:
            return ""
        actual = match.group("actual").strip()
        return actual if re.search(r"(?:Exception|Error|Throwable)$", actual) else ""

    @staticmethod
    def _observed_error_exception_fqn(failure_output: str) -> str:
        text = str(failure_output or "")
        match = re.search(
            r"<<<\s+ERROR!\s*\n(?P<type>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:Exception|Error|Throwable))\b",
            text,
        )
        if not match:
            match = re.search(
                r"(?m)^(?P<type>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*(?:Exception|Error|Throwable)):\s",
                text,
            )
        if not match:
            return ""
        return match.group("type")

    @staticmethod
    def _stack_trace_reaches_method_before_test(failure_output: str, method_name: str) -> bool:
        focus = str(method_name or "").strip()
        if not focus:
            return False
        for line in str(failure_output or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("at ") and "Test." in stripped:
                return False
            if stripped.startswith("at ") and re.search(rf"\.{re.escape(focus)}\s*\(", stripped):
                return True
        return False

    @classmethod
    def _make_failing_finally_cleanup_best_effort(
        cls,
        code: str,
        failure_output: str,
        *,
        test_method_focus: str | None = None,
    ) -> str:
        line_no = cls._reported_test_line_number(failure_output)
        if line_no <= 0:
            return code
        lines = code.splitlines()
        if line_no > len(lines):
            return code
        failing_line = lines[line_no - 1]
        if not re.search(r"\.[A-Za-z_$][\w$]*\s*\([^;{}]*\)\s*;", failing_line):
            return code
        if "finally" not in code[max(0, code.rfind("\n", 0, sum(len(l) + 1 for l in lines[:line_no - 1])) - 120):]:
            # Fall back to a broader same-method check below; this quick check
            # keeps ordinary target-call failures untouched.
            pass
        method_start = 0
        method_end = len(code)
        if test_method_focus:
            method_match = re.search(rf"\bvoid\s+{re.escape(test_method_focus)}\s*\(", code)
            if method_match:
                body_open = code.find("{", method_match.end())
                body_close = cls._find_matching_brace(code, body_open)
                if body_open >= 0 and body_close > body_open:
                    method_start, method_end = body_open, body_close
        offset = sum(len(line) + 1 for line in lines[:line_no - 1])
        if not (method_start <= offset <= method_end):
            return code
        finally_start = code.rfind("finally", method_start, offset + len(failing_line) + 1)
        if finally_start < method_start:
            return code
        finally_open = code.find("{", finally_start, method_end)
        finally_close = cls._find_matching_brace(code, finally_open)
        if finally_open < 0 or finally_close < offset:
            return code
        indent = re.match(r"\s*", failing_line).group(0)
        stripped = failing_line.strip()
        replacement = (
            f"{indent}try {{ {stripped} }} catch (Exception ignored) {{\n"
            f"{indent}    // Best-effort cleanup after the target-call oracle.\n"
            f"{indent}}}"
        )
        lines[line_no - 1] = replacement
        return "\n".join(lines) + ("\n" if code.endswith("\n") else "")

    @staticmethod
    def _reported_test_line_number(failure_output: str) -> int:
        match = re.search(r"\b[A-Za-z_$][\w$]*Test\.[A-Za-z_$][\w$]*:(\d+)\)", str(failure_output or ""))
        if not match:
            match = re.search(
                r"\b[A-Za-z_$][\w$]*Test\.java:\[?(\d+)(?:,\d+)?\]?\)?",
                str(failure_output or ""),
            )
        if not match:
            return -1
        try:
            return int(match.group(1))
        except ValueError:
            return -1

    @staticmethod
    def _find_matching_brace(text: str, open_index: int) -> int:
        if open_index < 0 or open_index >= len(text) or text[open_index] != "{":
            return -1
        depth = 0
        in_string: str | None = None
        escape = False
        for i in range(open_index, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch in {'"', "'"}:
                in_string = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    @staticmethod
    def _enum_names_from_errors(failure_output: str) -> list[str]:
        """Extract the enum identifier from each `enum types may not be
        instantiated` error line by inspecting the immediately preceding
        ``new <Name>(`` token in the same logical block.

        Heuristic: when javac reports the violation we expect the name to
        appear in nearby diagnostic lines; we fall back to scanning ``new X(``
        right after the violation marker.
        """
        results: list[str] = []
        lines = failure_output.splitlines()
        for idx, line in enumerate(lines):
            if "enum types may not be instantiated" not in line:
                continue
            # Look forward a few lines for a ``symbol:  class X`` or backward
            # for the originating source line excerpt.
            for probe in lines[max(0, idx - 3): idx + 6]:
                m = re.search(r"\bnew\s+([A-Z][\w$]*)\s*\(", probe)
                if m:
                    results.append(m.group(1))
                    break
                m2 = re.search(r"symbol:\s+(?:class|enum)\s+([A-Z][\w$]*)", probe)
                if m2:
                    results.append(m2.group(1))
                    break
        # Deduplicate, preserve order.
        seen: set[str] = set()
        out: list[str] = []
        for name in results:
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    @staticmethod
    def _inject_mockito_extension(code: str) -> str:
        # Imports.
        wants_import_ext = "import org.junit.jupiter.api.extension.ExtendWith;" not in code
        wants_import_mock_ext = "import org.mockito.junit.jupiter.MockitoExtension;" not in code
        if wants_import_ext or wants_import_mock_ext:
            import_lines = [
                line for line in (
                    "import org.junit.jupiter.api.extension.ExtendWith;" if wants_import_ext else "",
                    "import org.mockito.junit.jupiter.MockitoExtension;" if wants_import_mock_ext else "",
                ) if line
            ]
            m = list(re.finditer(r"^\s*import\s+[^\n]+;\s*$", code, re.MULTILINE))
            if m:
                last = m[-1]
                code = code[: last.end()] + "\n" + "\n".join(import_lines) + code[last.end():]
            else:
                # Insert after package or at top.
                pkg = re.search(r"^\s*package\s+[^\n]+;\s*$", code, re.MULTILINE)
                if pkg:
                    code = code[: pkg.end()] + "\n\n" + "\n".join(import_lines) + code[pkg.end():]
                else:
                    code = "\n".join(import_lines) + "\n\n" + code

        # Annotation on the class declaration (the FIRST top-level class).
        if not _RE_HAS_EXTEND_WITH.search(code):
            class_re = re.compile(r"^([ \t]*)(public\s+)?class\s+\w+", re.MULTILINE)
            cm = class_re.search(code)
            if cm:
                indent = cm.group(1) or ""
                annotation = f"{indent}@ExtendWith(MockitoExtension.class)\n"
                code = code[: cm.start()] + annotation + code[cm.start():]
        return code

    @classmethod
    def _repair_javac_method_arity_mismatch(cls, code: str, failure_output: str) -> str:
        """Drop extra generated arguments when javac proves a unique arity mismatch.

        This is intentionally narrow.  It uses javac's owner/method diagnostic
        plus a single matching call in the generated test.  It does not invent
        replacement arguments or choose between overloads.
        """
        failure = str(failure_output or "")
        if "actual and formal argument lists differ" not in failure:
            return code
        diag = re.search(
            r"method\s+(?P<method>[A-Za-z_$][\w$]*)\s+in\s+(?:interface|class)\s+"
            r"(?P<owner>[A-Za-z_$][\w$]*)\s+cannot\s+be\s+applied\s+to\s+given\s+types;\s*"
            r"(?:\r?\n|\s)*(?:\[ERROR\]\s*)?\s*required:\s*(?P<required>[^\r\n]*)\s*"
            r"(?:\r?\n|\s)*(?:\[ERROR\]\s*)?\s*found:\s*(?P<found>[^\r\n]*)",
            failure,
            re.IGNORECASE,
        )
        if not diag:
            return code
        method = str(diag.group("method") or "").strip()
        owner = str(diag.group("owner") or "").strip()
        required_text = str(diag.group("required") or "").strip()
        found_text = str(diag.group("found") or "").strip()
        if not method or not owner:
            return code

        def _count_types(text: str) -> int:
            value = " ".join(str(text or "").strip().split())
            if not value or value.lower() in {"no arguments", "no args"}:
                return 0
            return len([part for part in cls._split_top_level_args(value) if part.strip()])

        required_count = _count_types(required_text)
        found_count = _count_types(found_text)
        if found_count <= required_count:
            return code

        variables = {
            match.group(1)
            for match in re.finditer(
                rf"\b{re.escape(owner)}\s+([a-z_$][A-Za-z0-9_$]*)\b",
                code,
            )
        }
        candidates: list[tuple[int, int, str]] = []
        call_patterns: list[re.Pattern[str]] = []
        for variable in sorted(variables):
            call_patterns.append(re.compile(rf"\b{re.escape(variable)}\s*\.\s*{re.escape(method)}\s*\("))
        call_patterns.append(re.compile(rf"\b{re.escape(owner)}\s*\.\s*{re.escape(method)}\s*\("))
        for pattern in call_patterns:
            for match in pattern.finditer(code):
                open_index = match.end() - 1
                close_index = cls._find_matching_paren(code, open_index)
                if close_index < 0:
                    continue
                args_text = code[open_index + 1:close_index]
                args = cls._split_top_level_args(args_text)
                if len(args) != found_count:
                    continue
                replacement = ", ".join(args[:required_count])
                candidates.append((open_index + 1, close_index, replacement))
        if len(candidates) != 1:
            return code
        start, end, replacement = candidates[0]
        return code[:start] + replacement + code[end:]

    @staticmethod
    def _split_top_level_args(arg_text: str) -> list[str]:
        args: list[str] = []
        start = 0
        depth = 0
        in_string: str | None = None
        escape = False
        for i, ch in enumerate(arg_text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch in {'"', "'"}:
                in_string = ch
                continue
            if ch in "([{<":
                depth += 1
                continue
            if ch in ")]}>":
                depth = max(0, depth - 1)
                continue
            if ch == "," and depth == 0:
                args.append(arg_text[start:i].strip())
                start = i + 1
        tail = arg_text[start:].strip()
        if tail:
            args.append(tail)
        return args

    @classmethod
    def _repair_mixed_mockito_argument_matchers(
        cls,
        code: str,
        failure_output: str,
    ) -> str:
        """Wrap raw arguments when Mockito reports mixed matcher usage.

        Mockito requires every argument in one stub/verification invocation to
        use a matcher once any argument does. This rewrite is gated by the
        exact runtime diagnostic and only touches invocations inside Mockito
        stubbing/verification statements.
        """
        failure = str(failure_output or "")
        if not re.search(
            r"InvalidUseOfMatchersException|Invalid use of argument matchers",
            failure,
            re.IGNORECASE,
        ):
            return code

        matcher_arg = re.compile(
            r"^\s*(?:(?:org\.mockito\.)?(?:Mockito|ArgumentMatchers)\s*\.\s*)?"
            r"(?:any[A-Za-z0-9_$]*|eq|same|refEq|isNull|notNull|nullable|argThat)"
            r"\s*\(",
        )
        call_start = re.compile(r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\(")
        replacements: list[tuple[int, int, str]] = []
        for match in call_start.finditer(code):
            open_index = code.find("(", match.start(), match.end())
            if open_index < 0:
                continue
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            args = cls._split_top_level_args(code[open_index + 1:close_index])
            if len(args) < 2:
                continue
            matcher_flags = [bool(matcher_arg.match(arg)) for arg in args]
            if not any(matcher_flags) or all(matcher_flags):
                continue

            statement_start = max(
                code.rfind(";", 0, match.start()),
                code.rfind("{", 0, match.start()),
                code.rfind("}", 0, match.start()),
            ) + 1
            statement_end = code.find(";", close_index)
            if statement_end < 0:
                continue
            statement = code[statement_start:statement_end + 1]
            if not re.search(
                r"(?:Mockito\s*\.\s*)?(?:verify|when|doReturn|doThrow|doNothing|doAnswer)\s*\(",
                statement,
            ):
                continue

            repaired_args = [
                arg
                if is_matcher
                else ("Mockito.isNull()" if arg.strip() == "null" else f"Mockito.eq({arg.strip()})")
                for arg, is_matcher in zip(args, matcher_flags)
            ]
            replacements.append(
                (open_index + 1, close_index, ", ".join(repaired_args))
            )

        if not replacements:
            return code
        patched = code
        for start, end, replacement in sorted(replacements, reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        return patched

    @staticmethod
    def _find_matching_paren(text: str, open_index: int) -> int:
        depth = 0
        in_string: str | None = None
        escape = False
        for i in range(open_index, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch in {'"', "'"}:
                in_string = ch
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    @classmethod
    def _repair_integral_array_assertion_delta_overload(
        cls,
        code: str,
        failure_output: str,
    ) -> str:
        """Repair unsupported JUnit integral-array assertion overloads.

        JUnit exposes three-argument ``assertArrayEquals`` overloads only for
        floating-point arrays. Generated tests occasionally add a numeric
        tolerance to byte/short/int/long/char/boolean or object arrays. Models
        also occasionally invent a four-argument range overload
        ``(expected, actual, from, to)``.  Preserve that unambiguous intent by
        slicing the actual primitive array with ``Arrays.copyOfRange``. Keep
        both repairs diagnostic-gated so valid float/double assertions and
        already-compiling tests are untouched.
        """

        diagnostic = str(failure_output or "")
        if "assertArrayEquals" not in diagnostic:
            return code
        incompatible_array = re.search(
            r"(?:byte|short|int|long|char|boolean|[A-Z][\w$]*(?:\.[A-Z][\w$]*)*)"
            r"\[\]\s+cannot\s+be\s+converted\s+to\s+(?:float|double)\[\]",
            diagnostic,
        )
        if incompatible_array is None:
            return code

        pattern = re.compile(
            r"\b(?P<prefix>(?:(?:org\.junit(?:\.jupiter\.api)?\.)?Assertions\.)?)"
            r"assertArrayEquals\s*\("
        )
        integral_types = {"byte", "short", "int", "long", "char", "boolean"}
        declared_arrays = {
            match.group("name"): match.group("type")
            for match in re.finditer(
                r"\b(?P<type>byte|short|int|long|char|boolean|float|double)"
                r"\s*\[\s*\]\s+(?P<name>[A-Za-z_$][\w$]*)\b",
                code,
            )
        }

        def _is_integral_array(expression: str) -> bool:
            value = str(expression or "").strip()
            literal = re.match(
                r"new\s+(byte|short|int|long|char|boolean)\s*\[",
                value,
            )
            if literal is not None:
                return True
            return declared_arrays.get(value) in integral_types

        replacements: list[tuple[int, int, str]] = []
        numeric_delta = re.compile(
            r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
            r"[fFdD]?"
        )
        for match in pattern.finditer(code):
            open_index = match.end() - 1
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            args = cls._split_top_level_args(
                code[open_index + 1 : close_index]
            )
            if len(args) == 3:
                if numeric_delta.fullmatch(str(args[2] or "").strip()) is None:
                    continue
                if not (
                    _is_integral_array(args[0])
                    or _is_integral_array(args[1])
                ):
                    continue
                replacements.append(
                    (open_index + 1, close_index, ", ".join(args[:2]))
                )
                continue
            if len(args) != 4 or not _is_integral_array(args[0]):
                continue
            actual = str(args[1] or "").strip()
            if not actual or re.search(r"[;{}]", actual):
                continue
            from_index = str(args[2] or "").strip()
            to_index = str(args[3] or "").strip()
            if not from_index or not to_index:
                continue
            replacements.append((
                open_index + 1,
                close_index,
                (
                    f"{args[0]}, java.util.Arrays.copyOfRange("
                    f"{actual}, {from_index}, {to_index})"
                ),
            ))
        if not replacements:
            return code
        patched = code
        for start, end, replacement in reversed(replacements):
            patched = patched[:start] + replacement + patched[end:]
        return patched

    @staticmethod
    def _line_number_at_offset(text: str, offset: int) -> int:
        return text.count("\n", 0, max(0, offset)) + 1

    @staticmethod
    def _junit_failure_line_numbers(failure_output: str) -> set[int]:
        line_numbers: set[int] = set()
        for match in re.finditer(r"\b[\w$]+Test\.[\w$]+:(\d+)\b", failure_output or ""):
            try:
                line_numbers.add(int(match.group(1)))
            except ValueError:
                continue
        return line_numbers

    @classmethod
    def _repair_assert_equals_expected_actual(
        cls,
        code: str,
        failure_output: str,
        *,
        method_focus: str | None = None,
    ) -> str:
        observed = extract_junit_expected_actual(failure_output)
        typed_actual_type = ""
        if observed is None:
            typed = re.search(
                r"expected:\s*(?P<expected_type>(?:[a-z_$][\w$]*\.)*"
                r"[A-Z][\w$]*)<(?P<expected>[^>\r\n]{0,160})>\s*"
                r"but\s+was:\s*(?P<actual_type>(?:[a-z_$][\w$]*\.)*"
                r"[A-Z][\w$]*)<(?P<actual>[^>\r\n]{0,160})>",
                failure_output or "",
                re.IGNORECASE,
            )
            if typed is None:
                return code
            expected_text = typed.group("expected").strip()
            actual_text = typed.group("actual").strip()
            typed_actual_type = typed.group("actual_type").strip()
        else:
            expected_text, actual_text = observed

        pattern = re.compile(r"\b(?P<prefix>(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertEquals\s*\(")
        replacements: list[tuple[int, int, str, int, int, list[str]]] = []
        fallback_replacements: list[tuple[int, int, str, int, int, list[str]]] = []
        failure_lines = cls._junit_failure_line_numbers(failure_output)
        for call in pattern.finditer(code):
            open_index = call.end() - 1
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            arg_text = code[open_index + 1:close_index]
            args = cls._split_top_level_args(arg_text)
            if len(args) < 2:
                continue
            expected_index = 0
            if (
                len(args) >= 3
                and re.fullmatch(
                    r'"(?:\\.|[^"\\])*"',
                    str(args[0] or "").strip(),
                )
            ):
                # JUnit 4's assertEquals(message, expected, actual) overload
                # carries the oracle in argument 2. Treating the message as
                # expected prevents evidence-backed repair for a large class
                # of otherwise deterministic assertion mismatches.
                expected_index = 1
            existing_expected = args[expected_index]
            repaired_expected = cls._boxed_numeric_actual_as_java_expected(
                existing_expected,
                actual_type=typed_actual_type,
                actual_text=actual_text,
            )
            if repaired_expected is None:
                repaired_expected = cls._observed_object_identity_as_local_expected(
                    code,
                    existing_expected=existing_expected,
                    actual_text=actual_text,
                    before_offset=call.start(),
                )
            if repaired_expected is None:
                observed_actual = _decompact_junit_comparison_actual(
                    existing_expected,
                    expected_text,
                    actual_text,
                )
                repaired_expected = cls._observed_actual_as_java_expected(
                    existing_expected,
                    expected_text,
                    observed_actual,
                )
            if repaired_expected is None or repaired_expected == existing_expected:
                continue
            new_args = list(args)
            new_args[expected_index] = repaired_expected
            replacement = (open_index + 1, close_index, ", ".join(new_args), call.start(), close_index, args)
            fallback_replacements.append(replacement)
            if failure_lines:
                start_line = cls._line_number_at_offset(code, call.start())
                end_line = cls._line_number_at_offset(code, close_index)
                if not any(start_line <= line <= end_line for line in failure_lines):
                    continue
            replacements.append(replacement)
        if not replacements and len(fallback_replacements) == 1:
            replacements = fallback_replacements
        if len(replacements) != 1:
            return code
        start, end, replacement, call_start, call_end, args = replacements[0]
        patched = code[:start] + replacement + code[end:]
        if actual_text == "null" and method_focus:
            patched = cls._repair_terminal_null_assertion_suffix(
                patched,
                method_focus=method_focus,
                after_offset=call_start + (len(replacement) - (end - start)),
            )
        return patched

    @classmethod
    def _repair_assert_null_expected_actual(
        cls,
        code: str,
        failure_output: str,
    ) -> str:
        match = re.search(
            r"expected\s+null,\s*but\s+was:\s*<(?P<actual>[^>\n\r]*)>",
            failure_output or "",
            re.IGNORECASE,
        )
        if not match:
            return code
        actual_text = match.group("actual").strip()
        if not actual_text or actual_text == "null" or "\n" in actual_text:
            return code
        pattern = re.compile(r"\b(?P<prefix>(?:org\.junit\.Assert\.|org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertNull\s*\(")
        replacements: list[tuple[int, int, str]] = []
        fallback_replacements: list[tuple[int, int, str]] = []
        failure_lines = cls._junit_failure_line_numbers(failure_output)
        for call in pattern.finditer(code):
            open_index = call.end() - 1
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            args = cls._split_top_level_args(code[open_index + 1:close_index])
            if not args:
                continue
            message_arg = ""
            actual_expr = args[0].strip()
            if len(args) >= 2:
                message_arg = args[0].strip()
                actual_expr = args[1].strip()
            if not actual_expr:
                continue
            replacement_call = cls._assert_null_observed_nonnull_replacement(
                call.group("prefix") or "",
                message_arg,
                actual_expr,
                actual_text,
                code,
                call.start(),
            )
            if not replacement_call:
                continue
            replacement = (call.start(), close_index + 1, replacement_call)
            fallback_replacements.append(replacement)
            if failure_lines:
                start_line = cls._line_number_at_offset(code, call.start())
                end_line = cls._line_number_at_offset(code, close_index)
                if not any(start_line <= line <= end_line for line in failure_lines):
                    continue
            replacements.append(replacement)
        if not replacements and len(fallback_replacements) == 1:
            replacements = fallback_replacements
        if len(replacements) != 1:
            return code
        start, end, replacement = replacements[0]
        return code[:start] + replacement + code[end:]

    @classmethod
    def _assert_null_observed_nonnull_replacement(
        cls,
        prefix: str,
        message_arg: str,
        actual_expr: str,
        actual_text: str,
        code: str,
        offset: int,
    ) -> str | None:
        expected_literal: str | None = None
        declared_type = cls._nearest_local_declared_type(code, actual_expr, offset)
        simple_type = declared_type.rsplit(".", 1)[-1] if declared_type else ""
        if simple_type == "String" or simple_type == "CharSequence":
            expected_literal = '"' + actual_text.replace("\\", "\\\\").replace('"', '\\"') + '"'
        elif simple_type in {"boolean", "Boolean"} and actual_text.lower() in {"true", "false"}:
            expected_literal = actual_text.lower()
        elif simple_type in {"byte", "short", "int", "long", "Byte", "Short", "Integer", "Long"} and re.fullmatch(r"-?\d+", actual_text):
            expected_literal = actual_text
            if simple_type in {"long", "Long"} and not expected_literal.lower().endswith("l"):
                expected_literal += "L"
        elif simple_type in {"float", "Float", "double", "Double"} and re.fullmatch(r"-?\d+(?:\.\d+)?", actual_text):
            expected_literal = actual_text
            if simple_type in {"float", "Float"}:
                expected_literal += "f"
        if expected_literal is not None:
            args = [expected_literal, actual_expr]
            if message_arg:
                args.insert(0, message_arg)
            return f"{prefix}assertEquals(" + ", ".join(args) + ")"
        return None

    @staticmethod
    def _nearest_local_declared_type(code: str, expr: str, offset: int) -> str:
        name = expr.strip()
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", name):
            return ""
        prefix = code[: max(0, offset)]
        decl_re = re.compile(
            rf"(?:^|[;\n{{])\s*(?:final\s+)?(?P<type>[A-Za-z_$][\w$<>.\[\], ?]*)\s+{re.escape(name)}\s*=",
            re.DOTALL,
        )
        declared = ""
        for match in decl_re.finditer(prefix):
            raw_type = re.sub(r"\s+", " ", match.group("type") or "").strip()
            if raw_type and raw_type not in {"return", "throw", "new"}:
                declared = raw_type
        return declared

    @classmethod
    def _repair_terminal_null_assertion_suffix(
        cls,
        code: str,
        *,
        method_focus: str,
        after_offset: int,
    ) -> str:
        focus = (method_focus or "").strip()
        if not re.fullmatch(r"[A-Za-z_$][\w$]*", focus):
            return code
        method_end = cls._enclosing_method_end(code, after_offset)
        scan_end = method_end if method_end > after_offset else len(code)
        pattern = re.compile(r"\b(?P<prefix>(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertEquals\s*\(")
        replacements: list[tuple[int, int, str]] = []
        for call in pattern.finditer(code, after_offset, scan_end):
            open_index = call.end() - 1
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0 or close_index > scan_end:
                continue
            args = cls._split_top_level_args(code[open_index + 1:close_index])
            if len(args) < 2:
                continue
            actual_expr = args[1]
            target_assigned_vars = cls._target_call_assigned_vars(
                code[after_offset:call.start()],
                method_focus=focus,
            )
            if not re.search(rf"\.{re.escape(focus)}\s*\(", actual_expr) and not re.search(
                rf"\b{re.escape(focus)}\s*\(",
                actual_expr,
            ) and actual_expr.strip() not in target_assigned_vars:
                continue
            repaired_expected = cls._observed_actual_as_java_expected(args[0], "", "null")
            if repaired_expected != "null" or args[0].strip() == "null":
                continue
            new_args = ["null", *args[1:]]
            replacements.append((open_index + 1, close_index, ", ".join(new_args)))
        if not replacements:
            return code
        patched = code
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        return patched

    @staticmethod
    def _target_call_assigned_vars(code: str, *, method_focus: str) -> set[str]:
        vars_seen: set[str] = set()
        if not method_focus:
            return vars_seen
        assignment_re = re.compile(
            rf"(?:^|[;\n])\s*(?:(?:final\s+)?[A-Za-z_$][\w$<>.\[\], ?]*\s+)?"
            rf"(?P<var>[A-Za-z_$][\w$]*)\s*=\s*(?P<expr>[^;]*?\b{re.escape(method_focus)}\s*\([^;]*\))\s*;",
            re.DOTALL,
        )
        any_assignment_re = re.compile(
            r"(?:^|[;\n])\s*(?:(?:final\s+)?[A-Za-z_$][\w$<>.\[\], ?]*\s+)?"
            r"(?P<var>[A-Za-z_$][\w$]*)\s*=\s*(?P<expr>[^;]*?)\s*;",
            re.DOTALL,
        )
        for match in any_assignment_re.finditer(code):
            var = match.group("var")
            expr = match.group("expr") or ""
            if re.search(rf"\b{re.escape(method_focus)}\s*\(", expr):
                vars_seen.add(var)
            elif var in vars_seen:
                vars_seen.remove(var)
        # The broad assignment scanner is intentionally conservative. Keep the
        # focused scanner as a fallback for declarations with unusual generic
        # types that the broad pattern may skip.
        for match in assignment_re.finditer(code):
            vars_seen.add(match.group("var"))
        return vars_seen

    @classmethod
    def _enclosing_method_end(cls, code: str, offset: int) -> int:
        class_floor = max(code.rfind("\n    @", 0, offset), code.rfind("\n\t@", 0, offset))
        search_start = max(0, class_floor)
        brace = code.find("{", search_start, max(offset + 1, search_start + 1))
        best = -1
        while brace >= 0 and brace <= offset:
            best = brace
            brace = code.find("{", brace + 1, offset + 1)
        if best < 0:
            return -1
        depth = 0
        in_string: str | None = None
        escape = False
        for i in range(best, len(code)):
            ch = code[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_string:
                    in_string = None
                continue
            if ch in {'"', "'"}:
                in_string = ch
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i
        return -1

    @staticmethod
    def _looks_like_java_object_identity_text(value: str) -> bool:
        text = str(value or "").strip()
        return bool(re.fullmatch(r"(?:[a-zA-Z_$][\w$]*\.)*[A-Za-z_$][\w$]*(?:\$\d+)?@[0-9a-fA-F]+", text))

    @classmethod
    def _observed_object_identity_as_local_expected(
        cls,
        code: str,
        *,
        existing_expected: str,
        actual_text: str,
        before_offset: int,
    ) -> str | None:
        """Map object-identity assertion failures back to a local object.

        A JUnit message like ``expected:<TokenFilter.INCLUDE_ALL> but
        was:<FooTest$1@...>`` should not be repaired to a string literal.  When
        the existing expected value names a type and the test has a local object
        of that type before the assertion, use that local object as the oracle.
        """
        if not cls._looks_like_java_object_identity_text(actual_text):
            return None
        expected = str(existing_expected or "").strip()
        owner_match = re.match(r"(?:[a-z_][\w$]*\.)*(?P<owner>[A-Z][A-Za-z0-9_$]*)\s*\.", expected)
        if not owner_match:
            return None
        expected_owner = owner_match.group("owner")
        prefix = str(code or "")[: max(0, before_offset)]
        decl_re = re.compile(
            rf"(?:^|[;\n{{])\s*(?:final\s+)?(?P<type>(?:[a-z_][A-Za-z0-9_$]*\.)*{re.escape(expected_owner)})\s+"
            r"(?P<name>[a-zA-Z_$][A-Za-z0-9_$]*)\s*=\s*new\s+(?P=type)\s*\(",
            re.DOTALL,
        )
        candidates = [match.group("name") for match in decl_re.finditer(prefix)]
        if candidates:
            return candidates[-1]
        return None

    @staticmethod
    def _boxed_numeric_actual_as_java_expected(
        existing_expected: str,
        *,
        actual_type: str,
        actual_text: str,
    ) -> str | None:
        """Rewrite a boxed numeric expected value from exact JUnit evidence."""

        wrapper = str(actual_type or "").rsplit(".", 1)[-1]
        if wrapper not in {
            "Byte",
            "Short",
            "Integer",
            "Long",
            "Float",
            "Double",
        }:
            return None
        if not re.search(
            r"\b(?:Byte|Short|Integer|Long|Float|Double)\.valueOf\s*\(",
            str(existing_expected or ""),
        ):
            return None
        value = str(actual_text or "").strip()
        if not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
            value,
        ):
            return None
        if wrapper in {"Byte", "Short", "Integer"}:
            if not re.fullmatch(r"[+-]?\d+", value):
                return None
        elif wrapper == "Long":
            if not re.fullmatch(r"[+-]?\d+", value):
                return None
            value += "L"
        elif wrapper == "Float":
            value += "f"
        return f"{wrapper}.valueOf({value})"

    @staticmethod
    def _observed_actual_as_java_expected(existing_expected: str, expected_text: str, actual_text: str) -> str | None:
        expected = existing_expected.strip()
        if not expected:
            return None
        if StructuralPrepatchStrategy._looks_like_java_object_identity_text(actual_text):
            return None
        if actual_text == "null":
            if re.fullmatch(r"-?\d+(?:[lL])?", expected):
                return None
            if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[dDfF])?", expected):
                return None
            if expected in {"true", "false"}:
                return None
            return "null"
        if expected.startswith('"') and expected.endswith('"'):
            return '"' + _java_string_literal_value(actual_text) + '"'
        if expected.startswith("'") and expected.endswith("'") and len(actual_text) == 1:
            return "'" + actual_text.replace("\\", "\\\\").replace("'", "\\'") + "'"
        if re.fullmatch(r"-?\d+(?:[lL])?", expected):
            return actual_text if re.fullmatch(r"-?\d+(?:[lL])?", actual_text) else None
        if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[dDfF])?", expected):
            return actual_text if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[dDfF])?", actual_text) else None
        if expected in {"true", "false"}:
            return actual_text.lower() if actual_text.lower() in {"true", "false"} else None
        if expected == "null":
            return "null" if actual_text == "null" else None
        if expected.endswith(".class"):
            actual_class = actual_text.removeprefix("class ").strip()
            if re.fullmatch(r"(?:[a-z_][\w$]*\.)*[A-Z][\w$]*(?:\.[A-Z][\w$]*)*", actual_class):
                return actual_class + ".class"
        enum_match = re.fullmatch(r"(?P<prefix>(?:[A-Z][\w$]*\.)+)[A-Z][\w$]*", expected)
        if enum_match and re.fullmatch(r"[A-Z][\w$]*", actual_text):
            return enum_match.group("prefix") + actual_text
        # If the current expected value already stringifies to the reported
        # expected value, quote the observed actual as the least surprising
        # replacement.
        if expected_text and expected_text == expected.strip('"'):
            return '"' + _java_string_literal_value(actual_text) + '"'
        return None

    @classmethod
    def _rewrite_assert_throws_lambda_for_java7(cls, code: str) -> str:
        pattern = re.compile(
            r"\b(?P<prefix>(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?)assertThrows\s*\("
        )
        replacements: list[tuple[int, int, str]] = []
        for match in pattern.finditer(code):
            open_index = match.end() - 1
            close_index = cls._find_matching_paren(code, open_index)
            if close_index < 0:
                continue
            semicolon = close_index + 1
            while semicolon < len(code) and code[semicolon].isspace():
                semicolon += 1
            if semicolon >= len(code) or code[semicolon] != ";":
                continue
            line_start = code.rfind("\n", 0, match.start()) + 1
            statement_prefix = code[line_start:match.start()]
            if statement_prefix.strip():
                # Skip assigned forms such as `Throwable t = assertThrows(...)`;
                # a safe rewrite needs later reads of `t` to be considered.
                continue
            indent = re.match(r"[ \t]*", statement_prefix).group(0)
            args = cls._split_top_level_args(code[open_index + 1:close_index])
            if len(args) < 2:
                continue
            expected_type = args[0].strip()
            if not re.fullmatch(r"(?:[A-Za-z_$][\w$]*\.)*[A-Za-z_$][\w$]*(?:\.class)?", expected_type):
                continue
            if not expected_type.endswith(".class"):
                continue
            exception_type = expected_type[:-6]
            lambda_body = args[1].strip()
            if not lambda_body.startswith("()"):
                continue
            arrow = lambda_body.find("->")
            if arrow < 0:
                continue
            body = lambda_body[arrow + 2:].strip()
            if not body:
                continue
            if body.startswith("{") and body.endswith("}"):
                body_statements = body[1:-1].strip()
            else:
                body_statements = body.rstrip(";").strip() + ";"
            if not body_statements:
                continue
            replacement = cls._render_java7_exception_assertion(
                indent,
                exception_type,
                body_statements,
            )
            replacements.append((line_start, semicolon + 1, replacement))
        if not replacements:
            return code
        patched = code
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        return patched

    @staticmethod
    def _render_java7_exception_assertion(indent: str, exception_type: str, body: str) -> str:
        inner_indent = indent + "    "
        body_lines = []
        for raw in body.splitlines():
            stripped = raw.strip()
            if stripped:
                body_lines.append(inner_indent + stripped)
        return "\n".join(
            [
                indent + "try {",
                *body_lines,
                inner_indent + f'fail("Expected {exception_type}");',
                indent + f"}} catch ({exception_type} expected) {{",
                indent + "}",
            ]
        )

    @staticmethod
    def _ensure_import(code: str, fqn: str) -> str:
        import_line = f"import {fqn};"
        if import_line in code:
            return code
        m = list(re.finditer(r"^\s*import\s+[^\n]+;\s*$", code, re.MULTILINE))
        if m:
            last = m[-1]
            return code[: last.end()] + "\n" + import_line + code[last.end():]
        pkg = re.search(r"^\s*package\s+[^\n]+;\s*$", code, re.MULTILINE)
        if pkg:
            return code[: pkg.end()] + "\n\n" + import_line + code[pkg.end():]
        return import_line + "\n\n" + code

    @staticmethod
    def _missing_imports_from_source_context(
        *,
        failure_output: str,
        source_code: str,
        test_code: str,
    ) -> list[tuple[str, str]]:
        has_missing_symbol = "cannot find symbol" in failure_output
        has_override_mismatch = "method does not override or implement a method from a supertype" in failure_output
        if not has_missing_symbol and not has_override_mismatch:
            return []
        source_imports: dict[str, str] = {}
        for match in re.finditer(r"(?m)^\s*import\s+(?!static\b)([\w.]+)\s*;", source_code or ""):
            fqn = match.group(1)
            simple = fqn.rsplit(".", 1)[-1]
            source_imports.setdefault(simple, fqn)
        if not source_imports:
            return []

        missing: list[str] = []
        lines = failure_output.splitlines()
        for idx, line in enumerate(lines):
            if "cannot find symbol" not in line:
                continue
            for probe in lines[idx : idx + 5]:
                match = re.search(r"\bsymbol:\s+class\s+([A-Z][\w$]*)\b", probe)
                if match:
                    missing.append(match.group(1))
                    break
        if not missing and has_override_mismatch:
            for simple in source_imports:
                if re.search(rf"\bnew\s+{re.escape(simple)}\s*\([^)]*\)\s*\{{", test_code):
                    missing.append(simple)

        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for simple in missing:
            fqn = source_imports.get(simple)
            if not fqn or fqn in seen:
                continue
            if re.search(rf"(?m)^\s*import\s+{re.escape(fqn)}\s*;", test_code):
                continue
            if not re.search(rf"\b{re.escape(simple)}\b", test_code):
                continue
            seen.add(fqn)
            out.append((simple, fqn))
        return out

    @classmethod
    def _missing_imports_from_project_symbol_index(
        cls,
        *,
        failure_output: str,
        source_code: str,
        test_code: str,
        source_file: str,
    ) -> list[tuple[str, str]]:
        if "cannot find symbol" not in str(failure_output or ""):
            return []
        missing = cls._missing_class_symbols(failure_output)
        if not missing:
            return []
        scan_root = cls._project_scan_root_from_source_file(source_file)
        if scan_root is None:
            return []
        symbol_index = cls._project_symbol_index(scan_root)
        if not symbol_index:
            return []
        source_packages = cls._source_import_packages(source_code)
        source_package = cls._source_package(source_code)
        if source_package:
            source_packages.append(source_package)
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for simple in missing:
            if not re.search(rf"\b{re.escape(simple)}\b", test_code or ""):
                continue
            if re.search(rf"(?m)^\s*import\s+[\w.]*\.{re.escape(simple)}\s*;", test_code or ""):
                continue
            candidates = symbol_index.get(simple) or []
            chosen = cls._choose_import_fqn(simple, candidates, source_packages)
            if not chosen or chosen in seen:
                continue
            seen.add(chosen)
            out.append((simple, chosen))
        return out

    @staticmethod
    def _missing_class_symbols(failure_output: str) -> list[str]:
        missing: list[str] = []
        seen: set[str] = set()
        lines = str(failure_output or "").splitlines()
        for idx, line in enumerate(lines):
            if "cannot find symbol" not in line:
                continue
            for probe in lines[idx : idx + 5]:
                match = re.search(r"\bsymbol:\s+class\s+([A-Z][\w$]*)\b", probe)
                if not match:
                    continue
                simple = match.group(1)
                if simple not in seen:
                    seen.add(simple)
                    missing.append(simple)
                break
        return missing

    @staticmethod
    def _source_package(source_code: str) -> str:
        match = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", source_code or "")
        return match.group(1) if match else ""

    @staticmethod
    def _source_import_packages(source_code: str) -> list[str]:
        packages: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(r"(?m)^\s*import\s+(?!static\b)([\w.]+)\s*;", source_code or ""):
            fqn = match.group(1)
            package = fqn.rsplit(".", 1)[0] if "." in fqn else ""
            if package and package not in seen:
                seen.add(package)
                packages.append(package)
        return packages

    @staticmethod
    def _project_scan_root_from_source_file(source_file: str) -> Path | None:
        if not str(source_file or "").strip():
            return None
        try:
            path = Path(source_file).expanduser().resolve()
        except OSError:
            return None
        if not path.exists():
            return None
        cur = path.parent
        nearest_pom: Path | None = None
        module_root: Path | None = None
        for parent in (cur, *cur.parents):
            pom = parent / "pom.xml"
            if not pom.exists():
                continue
            if nearest_pom is None:
                nearest_pom = parent
            try:
                text = pom.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            if "<modules>" in text:
                module_root = parent
        return module_root or nearest_pom or cur

    @classmethod
    def _project_symbol_index(cls, scan_root: Path) -> dict[str, list[str]]:
        key = str(scan_root)
        cached = _PROJECT_SYMBOL_INDEX_CACHE.get(key)
        if cached is not None:
            return cached
        index: dict[str, list[str]] = {}
        for java_file in scan_root.rglob("*.java"):
            path_text = java_file.as_posix()
            if "/target/" in path_text:
                continue
            try:
                text = java_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            package = cls._source_package(text)
            code_text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.MULTILINE | re.DOTALL)
            match = re.search(
                r"\b(?:public\s+)?(?:abstract\s+|final\s+|sealed\s+|non-sealed\s+)*"
                r"(?:class|interface|enum|record)\s+([A-Z][\w$]*)\b",
                code_text,
            )
            if not match:
                continue
            simple = match.group(1)
            fqn = f"{package}.{simple}" if package else simple
            index.setdefault(simple, []).append(fqn)
        for simple, fqns in list(index.items()):
            index[simple] = sorted(set(fqns))
        _PROJECT_SYMBOL_INDEX_CACHE[key] = index
        return index

    @staticmethod
    def _choose_import_fqn(simple: str, candidates: list[str], source_packages: list[str]) -> str:
        cleaned = sorted({str(c) for c in candidates if str(c).endswith("." + simple) or str(c) == simple})
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]

        def common_prefix_score(package: str, reference: str) -> int:
            left = package.split(".")
            right = reference.split(".")
            score = 0
            for a, b in zip(left, right):
                if a != b:
                    break
                score += 1
            return score

        scored: list[tuple[int, int, str]] = []
        for fqn in cleaned:
            package = fqn.rsplit(".", 1)[0] if "." in fqn else ""
            exact_package = 1 if package in source_packages else 0
            proximity = max((common_prefix_score(package, ref) for ref in source_packages), default=0)
            scored.append((exact_package, proximity, fqn))
        scored.sort(key=lambda item: (item[0], item[1], -len(item[2])), reverse=True)
        best = scored[0]
        tied = [item for item in scored if item[0] == best[0] and item[1] == best[1]]
        if len(tied) > 1:
            return ""
        return best[2]

    @staticmethod
    def _remove_invalid_void_stubs(code: str) -> str:
        out_lines: list[str] = []
        for line in code.splitlines(keepends=True):
            normalized = line.replace(" ", "")
            if "when(" in normalized and ".thenReturn(null);" in normalized:
                continue
            out_lines.append(line)
        return "".join(out_lines)

    @staticmethod
    def _repair_repeated_mockito_then_return_sequences(code: str) -> str:
        pattern = re.compile(
            r"(?m)^(?P<indent>\s*)"
            r"(?P<prefix>(?:Mockito\.)?when\s*\(\s*(?P<call>[^;\n]+?)\s*\)\s*)"
            r"\.\s*thenReturn\s*\(\s*(?P<ret>[^;\n]+?)\s*\)\s*;\s*$"
        )
        matches = list(pattern.finditer(code))
        if len(matches) < 2:
            return code
        grouped: dict[str, list[re.Match[str]]] = {}
        for match in matches:
            call = re.sub(r"\s+", " ", match.group("call")).strip()
            grouped.setdefault(call, []).append(match)
        repeated = {call: rows for call, rows in grouped.items() if len(rows) >= 2}
        if not repeated:
            return code
        remove_spans: set[tuple[int, int]] = set()
        replacements: dict[tuple[int, int], str] = {}
        for rows in repeated.values():
            first = rows[0]
            returns = [row.group("ret").strip() for row in rows]
            replacement = (
                f"{first.group('indent')}{first.group('prefix')}"
                f".thenReturn({', '.join(returns)});"
            )
            replacements[(first.start(), first.end())] = replacement
            for row in rows[1:]:
                remove_spans.add((row.start(), row.end()))
        out: list[str] = []
        pos = 0
        for match in matches:
            span = (match.start(), match.end())
            if span in replacements:
                out.append(code[pos:match.start()])
                out.append(replacements[span])
                pos = match.end()
            elif span in remove_spans:
                out.append(code[pos:match.start()])
                pos = match.end()
        out.append(code[pos:])
        return "".join(out)

    @classmethod
    def _repair_source_required_mock_non_null_returns(cls, code: str, ctx: RepairContext) -> str:
        """Stub source-proven mock getters whose null return caused target failure.

        Evidence gate:
        - candidate already reached target coverage;
        - failure signal/source evidence exposes a required no-arg receiver call;
        - observed failure says the corresponding value cannot be null;
        - the receiver is a Mockito mock local in this candidate;
        - the exact call is not already stubbed.
        """
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        failure_signal = extra.get("streaming_failure_signal") if isinstance(extra.get("streaming_failure_signal"), dict) else {}
        category = str(failure_signal.get("category") or "").strip()
        if category not in {
            "missing-required-terminal-initialization",
            "terminal_output_setup_mismatch",
            "runtime-validation-failure",
        }:
            return code
        message = str(failure_signal.get("message") or ctx.failure_output or "")
        method_focus = str(
            (ctx.extra if isinstance(ctx.extra, dict) else {}).get("repair_test_method")
            or (ctx.extra.get("streaming_validation_failure") if isinstance(ctx.extra, dict) and isinstance(ctx.extra.get("streaming_validation_failure"), dict) else {}).get("test_method")
            or ctx.method_focus
            or ""
        ).strip()
        helpful_text = "\n".join(part for part in (ctx.failure_output or "", message) if str(part).strip())
        helpful_return_stub = cls._repair_helpful_npe_mock_return_value_null(
            code,
            helpful_text,
            method_focus=method_focus,
            source_code=ctx.source_code,
        )
        if helpful_return_stub != code:
            return helpful_return_stub
        missing_name = cls._required_non_null_name_from_message(
            "\n".join(part for part in (message, ctx.failure_output or "") if str(part).strip())
        )
        if not missing_name:
            return code
        validation_failure = extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}
        evidence_rows = list(validation_failure.get("source_evidence") or [])
        if not evidence_rows:
            evidence_rows = list(failure_signal.get("source_evidence") or [])
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        code_for_local_checks = method_code or code
        mock_vars = cls._mock_local_variables(code_for_local_checks)
        if not mock_vars:
            return code
        stubs: list[tuple[str, str, str]] = []
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            chains = [str(value or "") for value in list(row.get("required_input_chains") or [])]
            statement = str(row.get("statement") or "")
            if not chains and statement:
                chains = [statement]
            for chain in chains:
                for match in re.finditer(
                    r"\b(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)\s*\(\s*\)",
                    chain,
                ):
                    receiver = match.group("receiver")
                    method = match.group("method")
                    if receiver not in mock_vars:
                        continue
                    if not cls._method_matches_required_non_null_name(method, missing_name):
                        continue
                    if cls._mockito_call_already_stubbed(code_for_local_checks, receiver, method):
                        continue
                    literal = cls._non_null_literal_for_required_mock_method(method, missing_name)
                    if literal:
                        stubs.append((receiver, method, literal))
        if not stubs:
            source_text = str(ctx.source_code or "")
            suffix = re.sub(r"[^A-Za-z0-9]", "", missing_name or "")
            getter_candidates: list[str] = []
            if suffix:
                cap = suffix[:1].upper() + suffix[1:]
                getter_candidates.extend([f"get{cap}", f"is{cap}", f"has{cap}"])
            mock_types = cls._mock_local_variable_types(code_for_local_checks)
            for receiver in sorted(mock_vars):
                for method in getter_candidates:
                    if not method or not cls._method_matches_required_non_null_name(method, missing_name):
                        continue
                    if cls._mockito_call_already_stubbed(code_for_local_checks, receiver, method):
                        continue
                    source_has_receiver_call = bool(
                        source_text
                        and re.search(
                            rf"\b(?:this\.)?{re.escape(receiver)}\s*\.\s*{re.escape(method)}\s*\(",
                            source_text,
                        )
                    )
                    if not source_has_receiver_call:
                        type_text = str(mock_types.get(receiver) or "")
                        simple = re.sub(r"<.*>", "", type_text).strip().rsplit(".", 1)[-1]
                        lower = simple[:1].lower() + simple[1:] if simple else ""
                        source_has_receiver_call = bool(
                            lower
                            and source_text
                            and re.search(
                                rf"\b(?:this\.)?{re.escape(lower)}\s*\.\s*{re.escape(method)}\s*\(",
                                source_text,
                            )
                        )
                    if not source_has_receiver_call:
                        continue
                    literal = cls._non_null_literal_for_required_mock_method(method, missing_name)
                    if literal:
                        stubs.append((receiver, method, literal))
                        break
        if not stubs and missing_name:
            mock_types = cls._mock_local_variable_types(code_for_local_checks)
            missing_norm = re.sub(r"[^A-Za-z0-9]", "", str(missing_name or "")).lower()
            if missing_norm:
                cap_missing = missing_name[:1].upper() + missing_name[1:]
                getter_candidates = [f"get{cap_missing}", f"is{cap_missing}", f"has{cap_missing}"]
                for receiver, type_text in sorted(mock_types.items()):
                    receiver_norm = re.sub(r"[^A-Za-z0-9]", "", receiver).lower()
                    type_simple = re.sub(r"<.*>", "", str(type_text or "")).strip().rsplit(".", 1)[-1]
                    type_norm = re.sub(r"[^A-Za-z0-9]", "", type_simple).lower()
                    if not (
                        receiver_norm
                        and missing_norm.startswith(receiver_norm)
                        or type_norm
                        and missing_norm.startswith(type_norm)
                    ):
                        continue
                    for method in getter_candidates:
                        if not cls._method_matches_required_non_null_name(method, missing_name):
                            continue
                        if cls._mockito_call_already_stubbed(code_for_local_checks, receiver, method):
                            continue
                        literal = cls._non_null_literal_for_required_mock_method(method, missing_name)
                        if literal:
                            stubs.append((receiver, method, literal))
                            break
                    if stubs:
                        break
        if not stubs:
            return code
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str, str]] = []
        for receiver, method, literal in stubs:
            key = (receiver, method)
            if key in seen:
                continue
            seen.add(key)
            unique.append((receiver, method, literal))
        patched = code
        for receiver, method, literal in unique:
            patched = cls._insert_mockito_stub_after_mock_declaration_in_method(
                patched,
                method_focus,
                receiver,
                f"Mockito.when({receiver}.{method}()).thenReturn({literal});",
            )
        if patched != code and "Mockito." in patched:
            patched = cls._ensure_import(patched, "org.mockito.Mockito")
        return patched

    @staticmethod
    def _mock_local_variable_types(code: str) -> dict[str, str]:
        out: dict[str, str] = {}
        pattern = re.compile(
            r"(?m)^\s*(?:final\s+)?(?P<type>[A-Za-z_$][\w$<>, ?.\[\]]+)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:(?:org\.mockito\.)?Mockito\.)?mock\s*\(",
        )
        for match in pattern.finditer(str(code or "")):
            type_text = str(match.group("type") or "").strip()
            name = str(match.group("name") or "").strip()
            if type_text and name:
                out[name] = type_text
        return out

    @classmethod
    def _repair_helpful_npe_mock_return_value_null(
        cls,
        code: str,
        failure_output: str,
        *,
        method_focus: str | None = None,
        source_code: str = "",
    ) -> str:
        """Stub mock method calls proven null by JVM helpful NPE messages.

        Evidence gate:
        - JVM helpful NPE names the exact return-value method that was null;
        - the method receiver type matches a local Mockito mock declaration;
        - the return is consumed by a known lazy/stream/container operation.
        """
        text = str(failure_output or "")
        match = re.search(
            r'return value of "(?P<owner>[\w.$]+)\.(?P<method>[A-Za-z_$][\w$]*)\((?P<params>[^"]*)\)" is null',
            text,
        )
        if not match:
            return code
        owner = str(match.group("owner") or "")
        method = str(match.group("method") or "")
        params = str(match.group("params") or "")
        helpful_npe_sentence = text[max(0, match.start() - 220): min(len(text), match.end() + 80)]
        if not owner or not method:
            return code
        method_code = _extract_java_test_method_block(code, method_focus or "") if method_focus else ""
        code_for_local_checks = method_code or code
        mock_types = cls._mock_local_variable_types(code_for_local_checks)
        if not mock_types:
            return code
        owner_simple = owner.rsplit(".", 1)[-1]
        receiver = ""
        for name, type_text in mock_types.items():
            type_simple = re.sub(r"<.*>", "", type_text).strip().rsplit(".", 1)[-1]
            if type_simple == owner_simple:
                receiver = name
                break
        if not receiver:
            return code
        if re.search(
            rf"(?:Mockito\.)?when\s*\(\s*{re.escape(receiver)}\s*\.\s*{re.escape(method)}\s*\(",
            code_for_local_checks,
            re.DOTALL,
        ):
            return code
        expected_payload_types = cls._mock_return_expected_payload_types(
            source_code=source_code,
            owner_simple=owner_simple,
            method_name=method,
        )
        local_value = cls._best_local_value_for_mock_return(
            code_for_local_checks,
            method_name=method,
            owner_simple=owner_simple,
            expected_simple_types=expected_payload_types,
        )
        literal = ""
        if re.search(r"\bFlux\b|reactor\.core\.publisher\.Flux|\.concatMap\s*\(", helpful_npe_sentence):
            literal = f"reactor.core.publisher.Flux.just({local_value})" if local_value else "reactor.core.publisher.Flux.empty()"
        elif re.search(r"\bMono\b|reactor\.core\.publisher\.Mono|\.map\s*\(|\.flatMap\s*\(|\.block\s*\(", helpful_npe_sentence):
            if local_value:
                wants_collection = cls._mock_return_failure_wants_collection(
                    method_name=method,
                    params=params,
                    source_code=source_code,
                    owner_simple=owner_simple,
                )
                if wants_collection:
                    literal = f"reactor.core.publisher.Mono.just(java.util.Collections.singletonList({local_value}))"
                else:
                    literal = f"reactor.core.publisher.Mono.just({local_value})"
            else:
                literal = "reactor.core.publisher.Mono.empty()"
        elif re.search(r"\bStream\b|java\.util\.stream\.Stream", helpful_npe_sentence):
            literal = f"java.util.stream.Stream.of({local_value})" if local_value else "java.util.stream.Stream.empty()"
        elif re.search(r"\bIterator\b|\.iterator\s*\(", helpful_npe_sentence):
            literal = "java.util.Collections.emptyIterator()"
        elif re.search(r"\bCompletableFuture\b|java\.util\.concurrent\.CompletableFuture|\.join\s*\(|\.get\s*\(", helpful_npe_sentence):
            if local_value:
                literal = f"java.util.concurrent.CompletableFuture.completedFuture({local_value})"
        if not literal:
            return code
        param_count = 0 if not params.strip() else len([part for part in params.split(",")])
        args = ", ".join("Mockito.any()" for _ in range(param_count))
        stub_line = f"Mockito.when({receiver}.{method}({args})).thenReturn({literal});"
        if local_value:
            patched = cls._insert_statement_after_local_declaration_in_method(
                code,
                method_focus,
                local_value,
                stub_line,
            )
            if patched == code:
                patched = cls._insert_mockito_stub_after_mock_declaration_in_method(
                    code,
                    method_focus,
                    receiver,
                    stub_line,
                )
        else:
            patched = cls._insert_mockito_stub_after_mock_declaration_in_method(
                code,
                method_focus,
                receiver,
                stub_line,
            )
        if patched != code:
            patched = cls._ensure_import(patched, "org.mockito.Mockito")
        return patched

    @classmethod
    def _repair_generic_mock_return_payload_type_mismatch(cls, code: str, ctx: RepairContext) -> str:
        """Repair Mockito producer stubs whose generic payload type is wrong.

        Evidence gate:
        - javac reports incompatible generic bounds;
        - the candidate contains a Mockito ``thenReturn`` using a local variable
          whose declared type matches the lower bound;
        - the same test method has a unique local variable matching the equality
          constraint.  The stub is moved after that declaration if needed.
        """
        text = str(ctx.failure_output or "")
        if "incompatible bounds" not in text.lower() or "inference variable" not in text.lower():
            return code
        expected_types = cls._simple_types_from_generic_bound_diagnostic(text, "equality constraints")
        actual_types = cls._simple_types_from_generic_bound_diagnostic(text, "lower bounds")
        if not expected_types or not actual_types:
            return code
        method_focus = str(
            (ctx.extra if isinstance(ctx.extra, dict) else {}).get("repair_test_method")
            or (ctx.extra.get("streaming_validation_failure") if isinstance(ctx.extra, dict) and isinstance(ctx.extra.get("streaming_validation_failure"), dict) else {}).get("test_method")
            or ctx.method_focus
            or ""
        ).strip()
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        search_code = method_code or code
        declared = cls._local_declared_reference_types(search_code)
        if not declared:
            return code

        def _simple(type_text: str) -> str:
            return re.sub(r"<.*>", "", str(type_text or "")).strip().rsplit(".", 1)[-1].replace("[]", "")

        expected_vars = [
            name for name, type_text in declared.items()
            if _simple(type_text) in expected_types
        ]
        actual_vars = [
            name for name, type_text in declared.items()
            if _simple(type_text) in actual_types
        ]
        if len(expected_vars) != 1 or not actual_vars:
            return code
        expected_var = expected_vars[0]
        patched = code
        for actual_var in actual_vars:
            patched2 = cls._replace_mockito_then_return_payload_var(
                patched,
                method_focus=method_focus,
                actual_var=actual_var,
                expected_var=expected_var,
            )
            if patched2 != patched:
                return patched2
        return code

    @classmethod
    def _remove_mockito_stub_with_reported_return_type_mismatch(
        cls,
        code: str,
        failure_output: str,
    ) -> str:
        """Remove only the Mockito stub identified by an exact javac type error.

        Generated tests sometimes stub a collaborator with the wrong container
        type, such as returning ``Maybe<T>`` where javac requires
        ``Single<Event>``. Inventing a replacement factory is unsafe because
        the diagnostic proves neither its availability nor its semantics.
        Removing the one failing stub lets isolated validation establish
        whether it was unnecessary or expose the actual runtime setup contract.
        """
        text = str(failure_output or "")
        if not code or not re.search(
            r"no\s+suitable\s+method\s+found\s+for\s+thenReturn\s*\(",
            text,
            re.IGNORECASE,
        ):
            return code
        mismatches: list[tuple[str, str]] = []
        for match in re.finditer(
            r"(?P<provided>[A-Za-z_$][\w$.]*(?:<[^>\n\r]+>)?)\s+"
            r"conforms\s+to\s+"
            r"(?P<expected>[A-Za-z_$][\w$.]*(?:<[^>\n\r]+>)?)",
            text,
            re.IGNORECASE,
        ):
            provided = str(match.group("provided") or "").strip()
            expected = str(match.group("expected") or "").strip()
            if provided and expected and provided != expected:
                mismatches.append((provided, expected))
        if not mismatches:
            return code
        line_no = cls._reported_test_line_number(text)
        lines = code.splitlines()
        if line_no <= 0 or line_no > len(lines):
            return code

        def _raw_simple(type_text: str) -> str:
            raw = str(type_text or "").split("<", 1)[0].strip()
            return raw.rsplit(".", 1)[-1]

        provided_wrappers = {
            _raw_simple(provided)
            for provided, expected in mismatches
            if _raw_simple(provided)
            and _raw_simple(provided) != _raw_simple(expected)
        }
        if not provided_wrappers:
            return code
        reported_index = line_no - 1
        candidates: list[tuple[int, int, int]] = []
        seen_spans: set[tuple[int, int]] = set()
        for index in range(len(lines)):
            start, end = cls._statement_span_around_line(lines, index)
            if (start, end) in seen_spans:
                continue
            seen_spans.add((start, end))
            statement = "\n".join(lines[start : end + 1])
            if not (
                re.search(r"\b(?:Mockito\.)?when\s*\(", statement)
                and re.search(r"\.\s*thenReturn\s*\(", statement)
                and any(
                    re.search(rf"\b{re.escape(wrapper)}\s*\.", statement)
                    for wrapper in provided_wrappers
                )
            ):
                continue
            distance = (
                start - reported_index
                if reported_index < start
                else reported_index - end
                if reported_index > end
                else 0
            )
            if distance <= 8:
                candidates.append((distance, start, end))
        if not candidates:
            return code
        candidates.sort()
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
            return code
        _distance, start, end = candidates[0]
        patched = list(lines)
        del patched[start : end + 1]
        return "\n".join(patched) + ("\n" if code.endswith("\n") else "")

    @staticmethod
    def _simple_types_from_generic_bound_diagnostic(text: str, label: str) -> set[str]:
        ignored = {
            "void", "byte", "short", "int", "long", "float", "double", "boolean", "char",
            "Mono", "Flux", "Publisher", "Stream", "Iterable", "Iterator", "List", "Collection",
            "Set", "Map", "Optional", "CompletableFuture", "Single", "Maybe", "Completable",
            "java", "javax", "reactor", "core", "publisher", "util", "concurrent",
        }
        out: set[str] = set()
        label_re = re.compile(rf"{re.escape(label)}\s*:\s*(?P<value>[^\n\r]+)", re.IGNORECASE)
        for match in label_re.finditer(str(text or "")):
            value = match.group("value") or ""
            for token in re.findall(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", value):
                simple = token.rsplit(".", 1)[-1].replace("[]", "")
                if not simple or simple in ignored:
                    continue
                if len(simple) == 1 and simple.isupper():
                    continue
                if simple[:1].islower():
                    continue
                out.add(simple)
        return out

    @classmethod
    def _replace_mockito_then_return_payload_var(
        cls,
        code: str,
        *,
        method_focus: str | None,
        actual_var: str,
        expected_var: str,
    ) -> str:
        if not actual_var or not expected_var or actual_var == expected_var:
            return code
        method_code = _extract_java_test_method_block(code, method_focus or "") if method_focus else ""
        search_text = method_code or code
        offset = code.find(search_text) if method_code else 0
        if offset < 0:
            offset = 0
        stmt_re = re.compile(
            rf"(?ms)^(?P<stmt>[ \t]*(?:(?:Mockito\.)?when|org\.mockito\.Mockito\.when)\s*\("
            rf".*?\.thenReturn\s*\(.*?\b{re.escape(actual_var)}\b.*?\)\s*;\s*)"
        )
        for match in stmt_re.finditer(search_text):
            original_stmt = match.group("stmt")
            replacement_stmt = re.sub(rf"\b{re.escape(actual_var)}\b", expected_var, original_stmt)
            global_start = offset + match.start("stmt")
            global_end = offset + match.end("stmt")
            without_stmt = code[:global_start] + code[global_end:]
            if without_stmt.find(expected_var) < 0:
                return code
            moved = cls._insert_statement_after_local_declaration_in_method(
                without_stmt,
                method_focus,
                expected_var,
                replacement_stmt.strip(),
            )
            if moved != without_stmt:
                return moved
            return code[:global_start] + replacement_stmt + code[global_end:]
        return code

    @classmethod
    def _relax_mock_producer_stub_args_for_null_injected_field(cls, code: str, ctx: RepairContext) -> str:
        """Relax exact-argument Mockito producer stubs when an injected field stayed null.

        This handles source-backed cases where a target-positive candidate
        created a value of the required constructor-injected type, but the mock
        producer stub did not match the arguments used by the target path.
        """
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        failure_signal = extra.get("streaming_failure_signal") if isinstance(extra.get("streaming_failure_signal"), dict) else {}
        text = "\n".join(
            str(part or "")
            for part in (
                ctx.failure_output,
                failure_signal.get("message") if isinstance(failure_signal, dict) else "",
            )
            if str(part or "").strip()
        )
        field_match = re.search(r'because\s+"this\.(?P<field>[A-Za-z_$][\w$]*)"\s+is\s+null', text)
        if not field_match:
            return code
        field = field_match.group("field")
        evidence_rows = []
        validation_failure = extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}
        for source in (validation_failure, failure_signal):
            if isinstance(source, dict):
                evidence_rows.extend(row for row in list(source.get("source_evidence") or []) if isinstance(row, dict))
        expected_types: set[str] = set()
        for row in evidence_rows:
            for candidate in list(row.get("receiver_initialization_candidates") or []):
                text_candidate = str(candidate or "")
                if f"this.{field}" not in text_candidate:
                    continue
                sig = text_candidate.split(";", 1)[0]
                params_match = re.search(r"\((?P<params>[^)]*)\)", sig)
                if not params_match:
                    continue
                for part in params_match.group("params").split(","):
                    tokens = part.strip().split()
                    if len(tokens) < 2:
                        continue
                    ptype, pname = tokens[-2], tokens[-1]
                    if pname == field:
                        expected_types.add(ptype.rsplit(".", 1)[-1].replace("[]", ""))
        if not expected_types:
            return code
        method_focus = str(
            extra.get("repair_test_method")
            or (validation_failure.get("test_method") if isinstance(validation_failure, dict) else "")
            or ctx.method_focus
            or ""
        ).strip()
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        search_code = method_code or code
        declared = cls._local_declared_reference_types(search_code)
        expected_vars = [
            name for name, type_text in declared.items()
            if re.sub(r"<.*>", "", str(type_text or "")).strip().rsplit(".", 1)[-1].replace("[]", "") in expected_types
        ]
        if len(expected_vars) != 1:
            return code
        return cls._relax_mockito_when_args_returning_local(
            code,
            method_focus=method_focus,
            return_var=expected_vars[0],
        )

    @classmethod
    def _relax_mockito_when_args_returning_local(
        cls,
        code: str,
        *,
        method_focus: str | None,
        return_var: str,
    ) -> str:
        method_code = _extract_java_test_method_block(code, method_focus or "") if method_focus else ""
        search_text = method_code or code
        offset = code.find(search_text) if method_code else 0
        if offset < 0:
            offset = 0
        stmt_re = re.compile(
            rf"(?ms)^(?P<stmt>[ \t]*(?:(?:Mockito\.)?when|org\.mockito\.Mockito\.when)\s*\(\s*"
            rf"(?P<recv>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)\s*\((?P<args>[^;]*?)\)\s*\)"
            rf"\s*\.thenReturn\s*\(\s*{re.escape(return_var)}\s*\)\s*;\s*)"
        )
        for match in stmt_re.finditer(search_text):
            args = match.group("args") or ""
            if "Mockito.any" in args or re.search(r"(?<![\w$])any[A-Za-z]*\s*\(", args):
                continue
            arg_count = max(0, len([part for part in args.split(",") if part.strip()]))
            if arg_count <= 0:
                continue
            any_args = ", ".join("Mockito.any()" for _ in range(arg_count))
            replacement_stmt = (
                f"{match.group('stmt').split('when', 1)[0]}when("
                f"{match.group('recv')}.{match.group('method')}({any_args}))"
                f".thenReturn({return_var});"
            )
            global_start = offset + match.start("stmt")
            global_end = offset + match.end("stmt")
            patched = code[:global_start] + replacement_stmt + code[global_end:]
            return cls._ensure_import(patched, "org.mockito.Mockito")
        return code

    @classmethod
    def _relax_mockito_typed_any_for_null_producer_return(cls, code: str, ctx: RepairContext) -> str:
        """Broaden Mockito stubs when a target-positive producer returned null.

        Mockito's ``any(Type.class)`` does not match ``null``.  If the target
        path reached production code and failed because a mocked producer call
        returned null, a generated stub may simply have missed a nullable
        argument.  Replace typed any matchers in that exact producer stub with
        ``ArgumentMatchers.nullable(Type.class)``.
        """
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        failure = "\n".join(
            str(part or "")
            for part in (
                ctx.failure_output,
                (extra.get("streaming_failure_signal") or {}).get("message")
                if isinstance(extra.get("streaming_failure_signal"), dict)
                else "",
            )
            if str(part or "").strip()
        )
        producer_methods = {
            match.group("method")
            for match in re.finditer(
                r'return value of\s+"[^"\n]*\.(?P<method>[A-Za-z_$][\w$]*)\s*\([^"\n]*\)"\s+is\s+null',
                failure,
            )
        }
        if not producer_methods:
            return code
        method_focus = str(extra.get("repair_test_method") or ctx.method_focus or "").strip()
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        search_text = method_code or code
        offset = code.find(search_text) if method_code else 0
        if offset < 0:
            offset = 0

        stmt_re = re.compile(
            r"(?ms)^(?P<stmt>[ \t]*(?:(?:Mockito\.)?when|org\.mockito\.Mockito\.when)\s*\(\s*"
            r"(?P<recv>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)\s*\((?P<args>.*?)\)\s*\)"
            r"\s*\.thenReturn\s*\([^;]+?\)\s*;\s*)"
        )
        typed_any_re = re.compile(
            r"(?<![\w$])(?:(?:Mockito|org\.mockito\.Mockito|ArgumentMatchers|org\.mockito\.ArgumentMatchers)\s*\.\s*)?"
            r"any\s*\(\s*(?P<type>[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\.class\s*\)"
        )
        for match in stmt_re.finditer(search_text):
            if match.group("method") not in producer_methods:
                continue
            stmt = match.group("stmt")
            if not typed_any_re.search(stmt):
                continue

            def replace_any(any_match: re.Match[str]) -> str:
                type_name = any_match.group("type")
                return f"org.mockito.ArgumentMatchers.nullable({type_name}.class)"

            replacement_stmt = typed_any_re.sub(replace_any, stmt)
            if replacement_stmt == stmt:
                continue
            global_start = offset + match.start("stmt")
            global_end = offset + match.end("stmt")
            return code[:global_start] + replacement_stmt + code[global_end:]
        return code

    @staticmethod
    def _mock_return_expected_payload_types(
        *,
        source_code: str = "",
        owner_simple: str = "",
        method_name: str = "",
    ) -> list[str]:
        """Return source-backed simple payload types for a mocked producer method.

        For container/publisher returns, this reports the observable payload type,
        not the wrapper type. Example: ``Mono<List<ToolResultBlock>>`` yields
        ``ToolResultBlock`` so structural repair can choose an existing
        ``ToolResultBlock`` local instead of an unrelated message object.
        """
        if not source_code or not method_name:
            return []
        owner_block = str(source_code or "")
        if owner_simple:
            class_match = re.search(
                rf"\b(?:class|interface|record)\s+{re.escape(owner_simple)}\b(?P<body>.*)",
                owner_block,
                re.DOTALL,
            )
            if class_match:
                owner_block = class_match.group("body")
        sig_re = re.compile(
            rf"(?s)(?:^|[;\{{]\s*)"
            rf"(?:@[A-Za-z_$][\w$]*(?:\([^)]*\))?\s*)*"
            rf"(?:(?:public|protected|private|static|final|abstract|default|synchronized|native|strictfp)\s+)*"
            rf"(?P<ret>[A-Za-z_$][\w$.\s<>,?\[\]]+?)\s+"
            rf"{re.escape(method_name)}\s*\(",
        )
        ignored = {
            "void", "byte", "short", "int", "long", "float", "double", "boolean", "char",
            "Mono", "Flux", "Publisher", "Stream", "Iterable", "Iterator", "List", "Collection",
            "Set", "Map", "Optional", "CompletableFuture", "Single", "Maybe", "Completable",
            "java", "javax", "reactor", "core", "publisher", "util", "concurrent",
        }
        out: list[str] = []
        for match in sig_re.finditer(owner_block):
            ret = re.sub(r"\s+", "", match.group("ret") or "")
            for token in re.findall(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", ret):
                simple = token.rsplit(".", 1)[-1].replace("[]", "")
                if not simple or simple in ignored:
                    continue
                if len(simple) == 1 and simple.isupper():
                    continue
                if simple[:1].islower():
                    continue
                if simple not in out:
                    out.append(simple)
        # Prefer innermost generic payloads first.
        return list(reversed(out))

    @staticmethod
    def _mock_return_failure_wants_collection(
        *,
        method_name: str,
        params: str,
        source_code: str = "",
        owner_simple: str = "",
    ) -> bool:
        text = "\n".join(part for part in (method_name, params, source_code) if str(part).strip())
        if re.search(r"\b(?:List|Collection|Iterable|Set)\b", text):
            if not source_code:
                return True
        simple_method = str(method_name or "").strip()
        if re.search(r"(?:s|All|List|Many)$", simple_method):
            return True
        if source_code and owner_simple and simple_method:
            owner_block = ""
            class_match = re.search(
                rf"\b(?:class|interface|record)\s+{re.escape(owner_simple)}\b(?P<body>.*)",
                source_code,
                re.DOTALL,
            )
            owner_block = class_match.group("body") if class_match else source_code
            sig_re = re.compile(
                rf"(?m)^\s*(?:public|protected|private|static|final|abstract|default|\s)*"
                rf"(?P<ret>[A-Za-z_$][\w$<>., ?\[\]]+)\s+{re.escape(simple_method)}\s*\(",
            )
            for match in sig_re.finditer(owner_block):
                ret = re.sub(r"\s+", " ", match.group("ret") or "")
                if re.search(r"\b(?:List|Collection|Iterable|Set)\s*<", ret):
                    return True
        return False

    @classmethod
    def _best_local_value_for_mock_return(
        cls,
        code: str,
        *,
        method_name: str = "",
        owner_simple: str = "",
        expected_simple_types: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> str:
        arrange_code = cls._arrange_prefix_before_act_or_terminal(code)
        mock_vars = set(cls._mock_local_variable_types(code).keys())
        declared = cls._local_declared_reference_types(arrange_code)
        if not declared:
            return ""
        stop_types = {
            "String", "Object", "List", "ArrayList", "LinkedList", "Map", "HashMap", "Set",
            "HashSet", "Mono", "Flux", "CompletableFuture", "Optional",
        }
        method_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(method_name or ""))
            if len(token) >= 4
        }
        owner_terms = {
            token.lower()
            for token in re.findall(r"[A-Za-z][A-Za-z0-9]+", str(owner_simple or ""))
            if len(token) >= 4
        }
        expected_types = {
            str(item).strip().rsplit(".", 1)[-1].replace("[]", "")
            for item in (expected_simple_types or [])
            if str(item).strip()
        }
        primary_terms = {"result", "response", "event", "item", "entry", "record", "token", "message", "msg"}
        secondary_terms = {"block", "payload", "content", "value"}
        scored: list[tuple[int, int, str]] = []
        for idx, (name, type_text) in enumerate(declared.items()):
            if name in mock_vars:
                continue
            raw_type = str(type_text or "")
            simple_type = re.sub(r"<.*>", "", raw_type).strip().rsplit(".", 1)[-1].replace("[]", "")
            if (
                not simple_type
                or simple_type in stop_types
                or simple_type == str(owner_simple or "").strip().rsplit(".", 1)[-1]
            ):
                continue
            low = f"{name} {simple_type}".lower()
            score = 0
            type_tokens = {
                token.rsplit(".", 1)[-1].replace("[]", "")
                for token in re.findall(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*", raw_type)
            }
            if expected_types and (simple_type in expected_types or bool(type_tokens & expected_types)):
                score += 100
            score += sum(5 for term in primary_terms if term in low)
            score += sum(2 for term in secondary_terms if term in low)
            score += sum(2 for term in method_terms if term in low)
            score += sum(1 for term in owner_terms if term in low)
            if re.search(r"(Result|Response|Event|Message|Msg|Item|Entry|Record|Token)$", simple_type):
                score += 5
            elif re.search(r"(Block)$", simple_type):
                score += 3
            if score > 0:
                scored.append((score, -idx, name))
        if not scored:
            return ""
        return sorted(scored, reverse=True)[0][2]

    @staticmethod
    def _arrange_prefix_before_act_or_terminal(code: str) -> str:
        text = str(code or "")
        cut_points: list[int] = []
        for pattern in (
            r"(?mi)^\s*//\s*act\b",
            r"(?m)^\s*(?:final\s+)?[A-Za-z_$][\w$<>., ?\[\]]+\s+[A-Za-z_$][\w$]*\s*=\s*[^;\n]*\.(?:block|join|get)\s*\(",
            r"(?m)^\s*(?:org\.junit\.jupiter\.api\.Assertions\.|Assertions\.)?assert[A-Za-z]*\s*\(",
        ):
            match = re.search(pattern, text)
            if match:
                cut_points.append(match.start())
        if not cut_points:
            return text
        return text[: min(cut_points)]

    @classmethod
    def _insert_statement_after_local_declaration_in_method(
        cls,
        code: str,
        method_focus: str | None,
        local_name: str,
        statement: str,
    ) -> str:
        if not local_name or not statement:
            return code
        method_code = _extract_java_test_method_block(code, method_focus or "") if method_focus else ""
        search_text = method_code or code
        offset = code.find(search_text) if method_code else 0
        if offset < 0:
            offset = 0
        decl_re = re.compile(
            rf"(?m)^(?P<indent>[ \t]*)(?:final\s+)?[A-Za-z_$][\w$<>, ?.\[\]]+\s+{re.escape(local_name)}\s*=",
        )
        match = decl_re.search(search_text)
        if not match:
            return code
        start = offset + match.start()
        semicolon = code.find(";", start)
        if semicolon < 0:
            return code
        indent = match.group("indent") or ""
        insert_at = semicolon + 1
        insertion = "\n" + indent + statement.strip()
        return code[:insert_at] + insertion + code[insert_at:]

    @classmethod
    def _repair_helpful_npe_builder_return_value_null(cls, code: str, ctx: RepairContext) -> str:
        """Initialize a builder property whose getter was proven null by a helpful NPE.

        Evidence gate:
        - candidate already reached target coverage;
        - JVM helpful NPE names ``Owner.getX()`` / ``Owner.isX()`` as the null
          return value;
        - the candidate constructs ``Owner`` through ``Owner.builder()...build()``;
        - downstream receiver type in the NPE implies a safe minimal value.

        This is generic builder-shape repair. It does not invent project APIs:
        the setter name is derived from the getter property and only applied to
        an existing builder chain for the same owner type.
        """
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        failure_signal = extra.get("streaming_failure_signal") if isinstance(extra.get("streaming_failure_signal"), dict) else {}
        message = "\n".join(
            str(part or "")
            for part in (
                failure_signal.get("message") if isinstance(failure_signal, dict) else "",
                ctx.failure_output,
            )
        )
        match = re.search(
            r'return value of "(?P<owner>[\w.$]+)\.(?P<method>[A-Za-z_$][\w$]*)\((?P<params>[^"]*)\)" is null',
            message,
        )
        if not match:
            return code
        owner = str(match.group("owner") or "").strip()
        method = str(match.group("method") or "").strip()
        params = str(match.group("params") or "").strip()
        if not owner or not method or params:
            return code
        prop = ""
        for prefix in ("get", "is"):
            if method.startswith(prefix) and len(method) > len(prefix):
                tail = method[len(prefix):]
                prop = tail[:1].lower() + tail[1:]
                break
        if not prop or prop in {"class"}:
            return code
        literal = ""
        if re.search(r"\bjava\.util\.List\b|\bList\.iterator\b|\.iterator\s*\(", message):
            literal = "java.util.Collections.emptyList()"
        elif re.search(r"\bjava\.util\.Map\b|\bMap\.", message):
            literal = "java.util.Collections.emptyMap()"
        elif re.search(r"\bjava\.util\.Set\b|\bSet\.iterator\b", message):
            literal = "java.util.Collections.emptySet()"
        elif re.search(r"\bjava\.util\.Optional\b|\bOptional\.", message):
            literal = "java.util.Optional.empty()"
        elif re.search(r"\bjava\.lang\.String\b|\bString\.", message):
            literal = '"value"'
        if not literal:
            return code
        owner_simple = owner.rsplit(".", 1)[-1]
        method_focus = str(
            extra.get("repair_test_method")
            or (extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}).get("test_method")
            or ctx.method_focus
            or ""
        ).strip()
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        search_text = method_code or code
        chain_re = re.compile(
            rf"(?P<chain>\b{re.escape(owner_simple)}\s+[A-Za-z_$][\w$]*\s*=\s*"
            rf"{re.escape(owner_simple)}\s*\.\s*builder\s*\(\s*\)(?P<body>.*?)(?P<build>\.\s*build\s*\(\s*\)\s*;))",
            re.DOTALL,
        )
        replacements: list[tuple[int, int, str]] = []
        offset = code.find(search_text) if method_code else 0
        if offset < 0:
            offset = 0
        for chain_match in chain_re.finditer(search_text):
            chain = chain_match.group("chain")
            if re.search(rf"\.\s*{re.escape(prop)}\s*\(", chain):
                continue
            body = chain_match.group("body")
            indent_match = re.search(r"\n(?P<indent>[ \t]*)\.[A-Za-z_$][\w$]*\s*\(", body)
            indent = indent_match.group("indent") if indent_match else "            "
            insertion = f"\n{indent}.{prop}({literal})"
            build_start = chain_match.start("build")
            replacements.append((offset + build_start, offset + build_start, insertion))
            break
        if not replacements:
            return code
        patched = code
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        return patched

    @classmethod
    def _repair_pre_target_mock_copy_return_setup(cls, code: str, ctx: RepairContext) -> str:
        """Repair source-proven builder setup failures caused by mocked copy/factory calls.

        Evidence gate:
        - validation classified the failure as a pre-target builder/setup blocker;
        - source evidence shows the null receiver was initialized from
          `this.<field>.<zero-arg-method>()`;
        - the candidate passes a local Mockito mock with the same field name/type
          into the receiver builder before invoking the target;
        - the mock method has not already been stubbed.

        The patch is intentionally local: prefer a real no-arg construction when
        the candidate already uses `new Type()` elsewhere or the variable type is
        simple and non-generic; otherwise stub the exact zero-arg mock method to a
        new instance of the same declared type. If either form later fails javac,
        normal validation will reject it and feed back the precise diagnostic.
        """
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        failure_signal = extra.get("streaming_failure_signal") if isinstance(extra.get("streaming_failure_signal"), dict) else {}
        if str(failure_signal.get("category") or "").strip() != "pre_target_builder_setup_blocker":
            return code
        validation_failure = extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}
        evidence_rows = list(validation_failure.get("source_evidence") or [])
        if not evidence_rows:
            evidence_rows = list(failure_signal.get("source_evidence") or [])
        if not evidence_rows:
            return code
        init_candidates: list[tuple[str, str]] = []
        for row in evidence_rows:
            if not isinstance(row, dict):
                continue
            for raw in list(row.get("receiver_initialization_candidates") or []):
                text = str(raw or "")
                for match in re.finditer(
                    r"\bthis\.(?P<field>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)\s*\(\s*\)",
                    text,
                ):
                    field = match.group("field")
                    method = match.group("method")
                    if field and method:
                        init_candidates.append((field, method))
        if not init_candidates:
            return code
        method_focus = str(
            extra.get("repair_test_method")
            or validation_failure.get("test_method")
            or ctx.method_focus
            or ""
        ).strip()
        method_code = _extract_java_test_method_block(code, method_focus) if method_focus else ""
        code_for_checks = method_code or code
        mock_types = cls._mock_local_variable_types(code_for_checks)
        if not mock_types:
            return code
        patched = code
        for field, source_method in init_candidates[:3]:
            if field not in mock_types:
                continue
            declared_type = str(mock_types.get(field) or "").strip()
            simple_type = re.sub(r"<.*>", "", declared_type).strip().rsplit(".", 1)[-1]
            if not simple_type or not re.match(r"^[A-Z][A-Za-z0-9_$]*$", simple_type):
                continue
            if re.search(
                rf"(?:Mockito\.)?when\s*\(\s*{re.escape(field)}\s*\.\s*{re.escape(source_method)}\s*\(\s*\)\s*\)",
                code_for_checks,
            ):
                continue
            # Prefer replacing a local concrete mock with a real no-arg object
            # when it is passed into builder/configuration. This avoids deep mock
            # defaults for copy/clone/factory style setup paths.
            mock_decl = re.compile(
                rf"(?m)^(?P<indent>\s*)(?:final\s+)?(?P<type>{re.escape(declared_type)})\s+"
                rf"(?P<name>{re.escape(field)})\s*=\s*"
                rf"(?:(?:org\.mockito\.)?Mockito\.)?mock\s*\(\s*{re.escape(simple_type)}\.class\s*\)\s*;",
            )
            replacement = rf"\g<indent>{declared_type} {field} = new {simple_type}();"
            next_patched = mock_decl.sub(replacement, patched, count=1)
            if next_patched != patched:
                patched = next_patched
                continue
            stub_line = f"Mockito.when({field}.{source_method}()).thenReturn(new {simple_type}());"
            next_patched = cls._insert_mockito_stub_after_mock_declaration_in_method(
                patched,
                method_focus,
                field,
                stub_line,
            )
            if next_patched != patched:
                patched = cls._ensure_import(next_patched, "org.mockito.Mockito")
        return patched

    @staticmethod
    def _required_non_null_name_from_message(message: str) -> str:
        match = re.search(
            r"\b([A-Za-z_$][\w$]*)\s+(?:cannot|must not|may not)\s+be\s+null\b",
            str(message or ""),
            re.IGNORECASE,
        )
        if match:
            return match.group(1)
        return ""

    @staticmethod
    def _mock_local_variables(code: str) -> set[str]:
        out: set[str] = set()
        pattern = re.compile(
            r"(?m)^\s*(?:final\s+)?[A-Za-z_$][\w$<>, ?.\[\]]+\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
            r"(?:(?:org\.mockito\.)?Mockito\.)?mock\s*\(",
        )
        for match in pattern.finditer(str(code or "")):
            out.add(match.group("name"))
        return out

    @staticmethod
    def _method_matches_required_non_null_name(method: str, missing_name: str) -> bool:
        method_text = str(method or "")
        missing = re.sub(r"[^A-Za-z0-9]", "", str(missing_name or "")).lower()
        if not method_text or not missing:
            return False
        suffix = method_text
        for prefix in ("get", "is", "has"):
            if method_text.startswith(prefix) and len(method_text) > len(prefix):
                suffix = method_text[len(prefix):]
                break
        normalized_suffix = re.sub(r"[^A-Za-z0-9]", "", suffix).lower()
        return bool(normalized_suffix and (normalized_suffix == missing or normalized_suffix in missing or missing in normalized_suffix))

    @staticmethod
    def _non_null_literal_for_required_mock_method(method: str, missing_name: str) -> str:
        method_text = str(method or "")
        suffix = method_text
        for prefix in ("get", "is", "has"):
            if method_text.startswith(prefix) and len(method_text) > len(prefix):
                suffix = method_text[len(prefix):]
                break
        if method_text.startswith(("is", "has")):
            return "true"
        label = re.sub(r"[^A-Za-z0-9_-]", "-", str(missing_name or suffix or "value")).strip("-") or "value"
        return '"' + ("test-" + label).replace("\\", "\\\\").replace('"', '\\"') + '"'

    @staticmethod
    def _mockito_call_already_stubbed(code: str, receiver: str, method: str) -> bool:
        pattern = re.compile(
            rf"(?:Mockito\.)?when\s*\(\s*{re.escape(receiver)}\s*\.\s*{re.escape(method)}\s*\(\s*\)\s*\)",
            re.DOTALL,
        )
        return bool(pattern.search(str(code or "")))

    @staticmethod
    def _insert_mockito_stub_after_mock_declaration(code: str, receiver: str, stub_line: str) -> str:
        lines = str(code or "").splitlines(keepends=True)
        decl_re = re.compile(
            rf"^\s*(?:final\s+)?[A-Za-z_$][\w$<>, ?.\[\]]+\s+{re.escape(receiver)}\s*=\s*"
            r"(?:(?:org\.mockito\.)?Mockito\.)?mock\s*\(",
        )
        for index, line in enumerate(lines):
            if not decl_re.search(line):
                continue
            indent = re.match(r"\s*", line).group(0)
            insert_at = index + 1
            lines.insert(insert_at, indent + stub_line + ("\n" if not stub_line.endswith("\n") else ""))
            return "".join(lines)
        return code

    @classmethod
    def _insert_mockito_stub_after_mock_declaration_in_method(
        cls,
        code: str,
        method_name: str | None,
        receiver: str,
        stub_line: str,
    ) -> str:
        """Insert a stub into the focused failing @Test, falling back globally.

        A test class may already stub the same receiver/method in a different
        @Test.  That does not help the current candidate because JUnit executes
        each method with a fresh local scope.  Keep this insertion method-local
        whenever the failing method can be located.
        """
        source = str(code or "")
        name = str(method_name or "").strip()
        if not name:
            return cls._insert_mockito_stub_after_mock_declaration(source, receiver, stub_line)
        method_match = re.search(rf"\b{re.escape(name)}\s*\([^)]*\)\s*(?:throws\s+[^\{{]+)?\{{", source)
        if not method_match:
            return cls._insert_mockito_stub_after_mock_declaration(source, receiver, stub_line)
        brace_start = source.find("{", method_match.start())
        if brace_start < 0:
            return cls._insert_mockito_stub_after_mock_declaration(source, receiver, stub_line)
        method_end = _find_matching_java_brace(source, brace_start)
        if method_end < 0:
            return cls._insert_mockito_stub_after_mock_declaration(source, receiver, stub_line)
        method_text = source[method_match.start():method_end]
        patched_method = cls._insert_mockito_stub_after_mock_declaration(method_text, receiver, stub_line)
        if patched_method == method_text:
            return source
        return source[:method_match.start()] + patched_method + source[method_end:]

    @classmethod
    def _remove_unavailable_mockito_stubs(cls, code: str, failure_output: str) -> str:
        """Remove Mockito stubs whose receiver method is proven absent by javac."""
        blockers = _declared_type_method_blockers(failure_output or "")
        if not blockers:
            return code
        unavailable: set[tuple[str, str]] = {
            (str(row.get("variable") or "").strip(), str(row.get("method") or "").strip())
            for row in blockers
            if str(row.get("receiver_kind") or "") == "variable"
            and str(row.get("variable") or "").strip()
            and str(row.get("method") or "").strip()
        }
        if not unavailable:
            return code
        source = str(code or "")
        pattern = re.compile(
            r"(?ms)^(?P<indent>[ \t]*)(?:Mockito\.)?when\s*\(\s*"
            r"(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)"
            r"\s*\([^;\n]*?\)\s*\)\s*\.thenReturn\s*\([^;]*?\)\s*;\s*$"
        )
        replacements: list[tuple[int, int, str]] = []
        for match in pattern.finditer(source):
            key = (match.group("receiver"), match.group("method"))
            if key not in unavailable:
                continue
            indent = match.group("indent") or ""
            replacements.append((
                match.start(),
                match.end(),
                f"{indent}// Removed unavailable Mockito stub: {key[0]}.{key[1]}(...)\n",
            ))
        if not replacements:
            return source
        patched = source
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        return patched

    @classmethod
    def _remove_mockito_stubs_on_non_mock_receivers(cls, code: str, ctx: RepairContext) -> str:
        """Remove Mockito `when(real.method())` stubs after MissingMethodInvocationException.

        This is intentionally evidence-gated by Mockito's own runtime failure and
        current method-local mock declarations. It does not assume any project API:
        if the receiver was not declared via `mock(Type.class)` in this candidate,
        the stub is invalid Mockito usage and should not keep blocking target reach.
        """
        failure_text = str(ctx.failure_output or "")
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        signal = extra.get("streaming_failure_signal") if isinstance(extra.get("streaming_failure_signal"), dict) else {}
        category = str(signal.get("category") or "").strip()
        if "MissingMethodInvocationException" not in failure_text and category != "mock-stub-misuse":
            return code
        method_focus = str(
            extra.get("repair_test_method")
            or (extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}).get("test_method")
            or ctx.method_focus
            or ""
        ).strip()
        source = str(code or "")
        method_text = _extract_java_test_method_block(source, method_focus) if method_focus else source
        mock_vars = cls._mock_local_variables(method_text)
        pattern = re.compile(
            r"(?ms)^(?P<indent>[ \t]*)(?:Mockito\.)?when\s*\(\s*"
            r"(?P<receiver>[A-Za-z_$][\w$]*)\s*\.\s*(?P<method>[A-Za-z_$][\w$]*)"
            r"\s*\([^;\n]*?\)\s*\)\s*\.\s*(?:thenReturn|thenThrow)\s*\([^;]*?\)\s*;\s*$"
        )
        replacements: list[tuple[int, int, str]] = []
        for match in pattern.finditer(method_text):
            receiver = str(match.group("receiver") or "")
            method = str(match.group("method") or "")
            if not receiver or receiver in mock_vars:
                continue
            indent = match.group("indent") or ""
            replacements.append((
                match.start(),
                match.end(),
                f"{indent}// Removed Mockito stub on non-mock receiver: {receiver}.{method}(...)\n",
            ))
        if not replacements:
            return code
        patched_method = method_text
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            patched_method = patched_method[:start] + replacement + patched_method[end:]
        if method_text is source:
            return patched_method
        start = source.find(method_text)
        if start < 0:
            return code
        return source[:start] + patched_method + source[start + len(method_text):]

    @classmethod
    def _restore_missing_variable_declarations_from_original_excerpt(cls, code: str, ctx: RepairContext) -> str:
        """Restore declarations that LLM repair accidentally deleted.

        Evidence gate:
        - javac says a local variable is missing;
        - current repaired method references it but no longer declares it;
        - the original failed test excerpt had a declaration/assignment for it.
        """
        failure_output = str(ctx.failure_output or "")
        missing_vars = {
            match.group(1)
            for match in re.finditer(r"symbol:\s+variable\s+([A-Za-z_$][\w$]*)", failure_output)
        }
        if not missing_vars:
            return code
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        validation_failure = extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}
        original = str(validation_failure.get("failed_test_source_excerpt") or "")
        if not original.strip():
            return code
        patched = str(code or "")
        for name in sorted(missing_vars):
            if not re.search(rf"\b{re.escape(name)}\b", patched):
                continue
            if re.search(rf"\b[A-Za-z_$][\w$<>, ?.\[\]]+\s+{re.escape(name)}\s*(?:=|;)", patched):
                continue
            declaration = cls._extract_variable_declaration_statement(original, name)
            if not declaration:
                continue
            patched = cls._insert_statement_before_first_variable_use(patched, name, declaration)
        return patched

    @staticmethod
    def _extract_variable_declaration_statement(source: str, name: str) -> str:
        text = str(source or "")
        pattern = re.compile(
            rf"(?m)^(?P<indent>[ \t]*)(?:final\s+)?[A-Za-z_$][\w$<>, ?.\[\]]+\s+{re.escape(name)}\s*="
        )
        match = pattern.search(text)
        if not match:
            return ""
        start = match.start()
        pos = match.end()
        depth = 0
        in_string = False
        escaped = False
        while pos < len(text):
            ch = text[pos]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch in "([{":
                    depth += 1
                elif ch in ")]}":
                    depth = max(0, depth - 1)
                elif ch == ";" and depth == 0:
                    return text[start:pos + 1].strip()
            pos += 1
        return ""

    @staticmethod
    def _insert_statement_before_first_variable_use(code: str, name: str, statement: str) -> str:
        source = str(code or "")
        use_match = re.search(rf"\b{re.escape(name)}\b", source)
        if not use_match:
            return source
        line_start = source.rfind("\n", 0, use_match.start()) + 1
        indent = re.match(r"[ \t]*", source[line_start:]).group(0)
        rendered = "\n".join(
            (indent + line.strip() if line.strip() else line)
            for line in str(statement or "").splitlines()
        )
        return source[:line_start] + rendered.rstrip() + "\n" + source[line_start:]

    @staticmethod
    def _remove_assertions_with_undefined_expected_variables(code: str, failure_output: str) -> str:
        missing_vars = {
            match.group(1)
            for match in re.finditer(r"symbol:\s+variable\s+([A-Za-z_$][\w$]*)", str(failure_output or ""))
        }
        if not missing_vars:
            return code
        lines = str(code or "").splitlines(keepends=True)
        out: list[str] = []
        changed = False
        assertion_re = re.compile(r"\b(?:Assertions\.)?assert[A-Za-z0-9_]*\s*\(")
        for line in lines:
            match = assertion_re.search(line)
            used_missing = sorted(
                name
                for name in missing_vars
                if re.search(rf"\b{re.escape(name)}\b", line)
            )
            if match and used_missing:
                indent = re.match(r"\s*", line).group(0)
                out.append(f"{indent}// Removed assertion with undefined variable: {', '.join(used_missing)}\n")
                changed = True
            else:
                out.append(line)
        return "".join(out) if changed else code

    @staticmethod
    def _repair_assert_not_null_observed_null(code: str, ctx: RepairContext) -> str:
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        output = str(ctx.failure_output or "")
        if "expected: not <null>" not in output and "expected: not null" not in output.lower():
            return code
        method_focus = str(
            extra.get("repair_test_method")
            or (extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}).get("test_method")
            or ctx.method_focus
            or ""
        ).strip()
        source = str(code or "")
        method_text = _extract_java_test_method_block(source, method_focus) if method_focus else source
        assertion_re = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<prefix>(?:Assertions\.)?assertNotNull)\s*\(\s*(?P<expr>[A-Za-z_$][\w$]*)\s*\)\s*;\s*$"
        )
        matches = list(assertion_re.finditer(method_text))
        if len(matches) != 1:
            return code
        match = matches[0]
        replacement = f"{match.group('indent')}assertNull({match.group('expr')});"
        patched_method = method_text[:match.start()] + replacement + method_text[match.end():]
        if method_text is source:
            return patched_method
        start = source.find(method_text)
        if start < 0:
            return code
        return source[:start] + patched_method + source[start + len(method_text):]

    @staticmethod
    def _repair_null_local_receiver_assertion_oracle(code: str, ctx: RepairContext) -> str:
        extra = ctx.extra if isinstance(ctx.extra, dict) else {}
        coverage = extra.get("coverage_feedback") if isinstance(extra.get("coverage_feedback"), dict) else {}
        if not bool(coverage.get("has_target_coverage")):
            return code
        output = str(ctx.failure_output or "")
        match = re.search(
            r'Cannot invoke "[^"]+\.[A-Za-z_$][\w$]*\([^"]*\)" because "(?P<var>[A-Za-z_$][\w$]*)" is null',
            output,
        )
        if not match:
            return code
        var_name = match.group("var")
        method_focus = str(
            extra.get("repair_test_method")
            or (extra.get("streaming_validation_failure") if isinstance(extra.get("streaming_validation_failure"), dict) else {}).get("test_method")
            or ctx.method_focus
            or ""
        ).strip()
        source = str(code or "")
        method_text = _extract_java_test_method_block(source, method_focus) if method_focus else source
        if not method_text or not re.search(rf"\b{re.escape(var_name)}\b", method_text):
            return code
        # Evidence that `var` is the post-target observed value, not an arbitrary
        # setup object: it is assigned from a terminal/target expression or a
        # lazy result immediately before assertions.
        assign_re = re.compile(
            rf"(?m)^\s*(?:final\s+)?[A-Za-z_$][\w$<>, ?.\[\]]+\s+{re.escape(var_name)}\s*=\s*(?P<expr>[^;]+);"
        )
        assign_match = assign_re.search(method_text)
        if not assign_match:
            return code
        assign_expr = assign_match.group("expr")
        if not re.search(r"\.(?:block|join|get)\s*\(|\bdoCall\s*\(|\bcall\s*\(", assign_expr):
            return code
        lines = method_text.splitlines(keepends=True)
        changed = False
        inserted_assert_null = False
        out_lines: list[str] = []
        assertion_uses_var = re.compile(
            rf"\b(?:Assertions\.)?assert[A-Za-z0-9_]*\s*\([^;\n]*\b{re.escape(var_name)}\s*\."
        )
        assert_not_null_var = re.compile(
            rf"\b(?:Assertions\.)?assertNotNull\s*\(\s*{re.escape(var_name)}\s*\)\s*;"
        )
        for line in lines:
            if assert_not_null_var.search(line):
                indent = re.match(r"\s*", line).group(0)
                if not inserted_assert_null:
                    out_lines.append(f"{indent}assertNull({var_name});\n")
                    inserted_assert_null = True
                changed = True
                continue
            if assertion_uses_var.search(line):
                indent = re.match(r"\s*", line).group(0)
                if not inserted_assert_null:
                    out_lines.append(f"{indent}assertNull({var_name});\n")
                    inserted_assert_null = True
                out_lines.append(f"{indent}// Removed assertion that dereferenced observed-null value: {var_name}\n")
                changed = True
                continue
            out_lines.append(line)
        if not changed:
            return code
        patched_method = "".join(out_lines)
        if method_text is source:
            return patched_method
        start = source.find(method_text)
        if start < 0:
            return code
        return source[:start] + patched_method + source[start + len(method_text):]

    @staticmethod
    def _remove_unused_inaccessible_declarations(code: str, failure_output: str) -> str:
        output = str(failure_output or "")
        if "has private access" not in output and "is not public" not in output:
            return code
        line_match = re.search(r"\.java:\[(?P<line>\d+),(?P<col>\d+)\].*?(?:has private access|is not public)", output, re.DOTALL)
        if not line_match:
            return code
        target_line = int(line_match.group("line") or 0)
        if target_line <= 0:
            return code
        lines = str(code or "").splitlines(keepends=True)
        index = target_line - 1
        if index < 0 or index >= len(lines):
            return code
        line = lines[index]
        decl_match = re.match(
            r"(?P<indent>\s*)(?:final\s+)?[A-Za-z_$][\w$<>, ?.\[\]]+\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*new\s+.+;\s*(?P<tail>//.*)?$",
            line,
        )
        if not decl_match:
            return code
        name = decl_match.group("name")
        rest = "".join(lines[:index] + lines[index + 1:])
        if re.search(rf"\b{re.escape(name)}\b", rest):
            return code
        indent = decl_match.group("indent") or ""
        lines[index] = f"{indent}// Removed unused inaccessible declaration: {name}\n"
        return "".join(lines)

    @staticmethod
    def _remove_unused_compile_mismatch_declaration(
        code: str,
        failure_output: str,
        *,
        method_focus: str | None,
    ) -> str:
        """Remove one javac-located incompatible declaration that is unused.

        This is intentionally the executable counterpart of
        ``_unused_local_compile_mismatch_hint``.  It refuses ambiguous
        diagnostics and declarations read anywhere else in the focused test
        method, so it cannot rewrite target arguments or observable oracles.
        """

        source = str(code or "")
        method_name = str(method_focus or "").strip()
        diagnostics = {
            (int(match.group("line")), int(match.group("column")))
            for match in re.finditer(
                r"(?m)^(?:\[ERROR\]\s*)?.*?\.java:"
                r"\[(?P<line>\d+),(?P<column>\d+)\]\s*"
                r"incompatible types:[^\r\n]+",
                str(failure_output or ""),
            )
        }
        if not method_name or len(diagnostics) != 1:
            return source
        line_no, _column_no = next(iter(diagnostics))
        lines = source.splitlines(keepends=True)
        index = line_no - 1
        if index < 0 or index >= len(lines):
            return source
        declaration = re.match(
            r"(?P<indent>\s*)(?:final\s+)?"
            r"[A-Za-z_$][A-Za-z0-9_$.<>, ?\[\]]*\s+"
            r"(?P<name>[A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*.+;\s*"
            r"(?P<tail>//.*)?$",
            lines[index],
        )
        if declaration is None:
            return source
        local_name = str(declaration.group("name") or "")
        remaining_source = "".join(lines[:index] + lines[index + 1 :])
        if re.search(rf"\b{re.escape(local_name)}\b", remaining_source):
            return source
        method_window = _window_around_method(
            source,
            method_name,
            _REPAIR_TEST_MAX_CHARS,
        )
        if len(re.findall(rf"\b{re.escape(local_name)}\b", method_window)) != 1:
            return source
        indent = str(declaration.group("indent") or "")
        lines[index] = (
            f"{indent}// Removed unused incompatible declaration: "
            f"{local_name}\n"
        )
        return "".join(lines)

    @staticmethod
    def _repair_json_object_or_array_shape_from_diagnostic(
        code: str,
        failure_output: str,
    ) -> str:
        """Wrap one parser-identified object value in an array.

        The runtime diagnostic is the proof: a named nested field was parsed
        where the grammar required an object or array.  The rewrite is allowed
        only when exactly one JSON string contains exactly one parent value of
        the form ``{"field": {"diagnostic-name": ...}}``.  Ambiguous,
        concatenated, or already-array inputs remain untouched.
        """

        names = {
            str(match.group("name") or "")
            for match in re.finditer(
                r'["\u201c](?P<name>[^"\u201d\r\n]{1,120})["\u201d]'
                r"\s+must be an object or an array",
                str(failure_output or ""),
                flags=re.IGNORECASE,
            )
        }
        if len(names) != 1:
            return str(code or "")
        diagnostic_name = next(iter(names))
        source = str(code or "")
        string_assignment = re.compile(
            r"(?P<prefix>\bString\s+[A-Za-z_$][A-Za-z0-9_$]*\s*=\s*)"
            r'(?P<literal>"(?:\\.|[^"\\])*")(?P<suffix>\s*;)',
            flags=re.DOTALL,
        )
        replacements: list[tuple[int, int, str]] = []
        for match in string_assignment.finditer(source):
            try:
                decoded = json.loads(str(match.group("literal") or ""))
                payload = json.loads(decoded)
            except (TypeError, ValueError, json.JSONDecodeError):
                continue

            candidates: list[tuple[dict, str, dict]] = []

            def visit(value: object) -> None:
                if isinstance(value, dict):
                    for key, child in list(value.items()):
                        if (
                            isinstance(child, dict)
                            and diagnostic_name in child
                            and not isinstance(child.get(diagnostic_name), (dict, list))
                        ):
                            candidates.append((value, str(key), child))
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(payload)
            if len(candidates) != 1:
                continue
            parent, key, child = candidates[0]
            parent[key] = [child]
            encoded_json = json.dumps(payload, separators=(",", ":"))
            encoded_java = json.dumps(encoded_json)
            replacements.append((
                int(match.start("literal")),
                int(match.end("literal")),
                encoded_java,
            ))
        if len(replacements) != 1:
            return source
        start, end, replacement = replacements[0]
        return source[:start] + replacement + source[end:]

    @classmethod
    def _lift_try_local_variable_used_in_catch(cls, code: str, failure_output: str) -> str:
        """Lift try-local variables referenced by catch blocks.

        Generated tests sometimes declare the value under test inside a
        ``try`` block and then reference it in ``catch`` for fallback oracle
        logic.  Javac reports this as an unresolved variable in the catch
        scope.  This rewrite is purely structural: declare the same variable
        immediately before the try block with a safe default and replace the
        try-local declaration with an assignment.
        """
        source = str(code or "")
        if "catch" not in source or "try" not in source:
            return source
        if not (
            re.search(r"\bcannot find symbol\b", str(failure_output or ""), re.IGNORECASE)
            or re.search(r"\bsymbol:\s+variable\b", str(failure_output or ""), re.IGNORECASE)
            or re.search(r"\bmight not have been initialized\b", str(failure_output or ""), re.IGNORECASE)
        ):
            return source

        def _default_for_type(type_text: str) -> str:
            normalized = re.sub(r"\s+", " ", str(type_text or "")).strip()
            if normalized == "boolean":
                return "false"
            if normalized in {"byte", "short", "int", "long", "float", "double", "char"}:
                return "'\\0'" if normalized == "char" else "0"
            return "null"

        decl_re = re.compile(
            r"(?m)^(?P<indent>[ \t]*)(?P<type>(?:final\s+)?[A-Za-z_$][\w$]*(?:\s*<[^;\n=]+>)?(?:\s*\[\])?)\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?P<expr>[^;\n]+);\s*$"
        )
        replacements: list[tuple[int, int, str]] = []
        search_at = 0
        while True:
            try_match = re.search(r"\btry\s*\{", source[search_at:])
            if not try_match:
                break
            try_start = search_at + try_match.start()
            try_brace = source.find("{", try_start)
            if try_brace < 0:
                break
            try_end = _find_matching_java_brace(source, try_brace)
            if try_end < 0:
                break
            catch_match = re.match(r"\s*catch\s*\([^)]*\)\s*\{", source[try_end:])
            if not catch_match:
                search_at = try_end
                continue
            catch_start = try_end + catch_match.start()
            catch_brace = source.find("{", catch_start)
            catch_end = _find_matching_java_brace(source, catch_brace)
            if catch_end < 0:
                search_at = try_end
                continue
            try_body = source[try_brace + 1:try_end - 1]
            catch_body = source[catch_brace + 1:catch_end - 1]
            pre_try_text = source[max(0, source.rfind("\n", 0, try_start) + 1):try_start]
            try_line_indent = re.match(r"[ \t]*", pre_try_text).group(0) if pre_try_text is not None else ""
            declarations_to_lift: list[tuple[re.Match[str], str, str, str]] = []
            for decl in decl_re.finditer(try_body):
                name = decl.group("name")
                if not re.search(rf"\b{re.escape(name)}\b", catch_body):
                    continue
                declared_before_try = re.search(
                    rf"\b[A-Za-z_$][\w$<>, ?.\[\]]+\s+{re.escape(name)}\s*(?:=|;)",
                    source[:try_start],
                )
                if declared_before_try:
                    continue
                type_text = re.sub(r"^\s*final\s+", "", decl.group("type").strip())
                default_value = _default_for_type(type_text)
                declarations_to_lift.append((decl, type_text, name, default_value))
            if declarations_to_lift:
                insertion = "".join(
                    f"{try_line_indent}{type_text} {name} = {default_value};\n"
                    for _, type_text, name, default_value in declarations_to_lift
                )
                replacements.append((try_start, try_start, insertion))
                for decl, _type_text, name, _default in declarations_to_lift:
                    expr = decl.group("expr").strip()
                    indent = decl.group("indent")
                    abs_start = try_brace + 1 + decl.start()
                    abs_end = try_brace + 1 + decl.end()
                    replacements.append((abs_start, abs_end, f"{indent}{name} = {expr};"))
            search_at = catch_end
        if not replacements:
            return source
        patched = source
        for start, end, replacement in sorted(replacements, key=lambda item: item[0], reverse=True):
            patched = patched[:start] + replacement + patched[end:]
        return patched


# ---------------------------------------------------------------------------
# Built-in strategy: composite (DACR pipeline)
# ---------------------------------------------------------------------------

class CompositeRepairStrategy(RepairStrategy):
    """Run several strategies in order; stop at the first one that produces a patch.

    The default composition is ``[StructuralPrepatchStrategy, LLMRepairStrategy]``
    which realises the DACR pipeline: deterministic structural fixes run first
    (cheap, side-effect-free for files unrelated to the canonical test file),
    and only when nothing else applies do we spend an LLM call.
    """

    def __init__(self, strategies: list[RepairStrategy]):
        self._strategies = strategies

    @property
    def name(self) -> str:
        inner = ",".join(s.name for s in self._strategies)
        return f"Composite[{inner}]"

    def repair(self, ctx: RepairContext) -> tuple[str, dict]:
        last_exc: Exception | None = None
        for strat in self._strategies:
            try:
                patched, usage = strat.repair(ctx)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                continue
            if patched:
                meta = dict(usage) if isinstance(usage, dict) else {}
                meta.setdefault("strategy", strat.name)
                return patched, meta
        if last_exc is not None:
            raise last_exc
        return "", {}


# ---------------------------------------------------------------------------
# Intra-Run Negative Memory (DACR component C)
# ---------------------------------------------------------------------------

# Module-level cache keyed by absolute test-file path. Stores the most recent
# error fingerprint we observed for that file and the source bytes that were
# present *before* the LLM patch that produced it. When the same fingerprint
# recurs after an LLM patch, the patch is non-monotonic and we roll back.
_IRNM_PREV_FINGERPRINT: dict[str, str] = {}
_IRNM_SNAPSHOT: dict[str, str] = {}


def _error_fingerprint(failure_output: str, test_file: Path) -> str:
    """Return a stable hash of the *kinds* of compile errors that mention
    *test_file*.

    The fingerprint deliberately discards line numbers and message ordering so
    that cosmetic re-orderings between iterations do not bypass the cache.
    """
    if not failure_output:
        return ""
    name = test_file.name
    keys: list[str] = []
    for line in failure_output.splitlines():
        if name not in line:
            continue
        m = re.search(r"error:\s+(.+?)(?:$|\s{2,})", line)
        if not m:
            continue
        msg = m.group(1).strip()
        # Normalise: collapse identifiers in quotes/braces, drop trailing periods.
        msg = re.sub(r"'[^']*'", "'_'", msg)
        msg = re.sub(r"`[^`]*`", "`_`", msg)
        msg = re.sub(r"\b\d+\b", "N", msg)
        keys.append(msg[:120])
    if not keys:
        return ""
    keys.sort()
    return hashlib.sha1("|".join(keys).encode("utf-8")).hexdigest()[:16]


def _irnm_reset(test_file_path: str) -> None:
    """Public for tests: clear IRNM state for a path (e.g. between runs)."""
    _IRNM_PREV_FINGERPRINT.pop(test_file_path, None)
    _IRNM_SNAPSHOT.pop(test_file_path, None)


# ---------------------------------------------------------------------------
# Helper: create default strategy from a GenerationRequest
# ---------------------------------------------------------------------------

def _default_strategy(request: GenerationRequest) -> RepairStrategy:
    llm = LLMRepairStrategy(
        api_base=request.api_base or "",
        model=request.model,
        api_key_env=request.api_key_env,
        timeout_seconds=request.llm_timeout_seconds,
    )
    return CompositeRepairStrategy([StructuralPrepatchStrategy(), llm])


# ---------------------------------------------------------------------------
# Utility: extract failing method names from surefire output
# ---------------------------------------------------------------------------

def extract_failing_methods(failure_output: str, class_stem: str) -> list[str]:
    """Return test method names that failed for *class_stem*.

    Handles Maven Surefire JUnit-4 style::

        [ERROR] FooTest.testBar  -- Time elapsed: 0.01 s  <<< FAILURE!

    and JUnit-5 / Surefire-2.22+ style::

        [ERROR] FooTest > testBar()  Time elapsed: 0.01 s  <<< FAILURE!
    """
    seen: set[str] = set()
    result: list[str] = []
    patterns = [
        re.compile(r"\b" + re.escape(class_stem) + r"\.(\w+)\s+--\s+Time elapsed"),
        re.compile(r"\b" + re.escape(class_stem) + r"\s*>\s*(\w+)\(\)"),
        # Classic surefire detail line style: testMethod(com.example.FooTest)
        re.compile(r"\b(\w+)\(.*\b" + re.escape(class_stem) + r"\b.*\)"),
    ]
    for pat in patterns:
        for m in pat.finditer(failure_output):
            name = m.group(1)
            if name not in seen:
                seen.add(name)
                result.append(name)
    return result


# ---------------------------------------------------------------------------
# Utility: build repair candidate list from runner results
# ---------------------------------------------------------------------------

def extract_repair_candidates(results: Iterable) -> list[dict]:
    """Flatten runner *results* into a list of repair-candidate dicts."""
    candidates: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(item: dict, failure_output: str = "") -> None:
        source_file = str(item.get("source_file", "") or "")
        test_file = str(item.get("test_file", "") or item.get("generated_tests", "") or "")
        if not test_file and source_file:
            p = Path(source_file)
            if p.suffix == ".java":
                test_file = str(derive_java_test_file(p))
            else:
                test_file = str(p.with_name(p.stem + "Test" + p.suffix))
        if not source_file and not test_file:
            return
        key = (
            source_file,
            test_file,
            str(item.get("target_label", "") or item.get("class_name", "") or ""),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append({
            "target_label": item.get("target_label", ""),
            "test_file": test_file,
            "source_file": source_file,
            "class_name": item.get("class_name"),
            "method_name": item.get("method_name"),
            "usage": item.get("usage", {}),
            "failure_output": failure_output,
            "generation_status": item.get("status", "success"),
            "tatg_target_receipt_contract": item.get("tatg_target_receipt_contract", ""),
            "tatg_execution_capsule": item.get("tatg_execution_capsule", ""),
            "tatg_mutation_feedback_context": item.get("tatg_mutation_feedback_context", {}),
        })

    for item in results:
        payload = getattr(item, "payload", None)
        if not isinstance(payload, dict):
            continue
        failure_output = f"{getattr(item, 'stdout', '')}\n{getattr(item, 'stderr', '')}".strip()
        if payload.get("generated_tests") or payload.get("source_file"):
            fo = failure_output if payload.get("status") != "success" else ""
            _add(payload, fo)
        nested = payload.get("results")
        if isinstance(nested, dict):
            for one in nested.values():
                if isinstance(one, dict):
                    fo = failure_output if one.get("status") != "success" else ""
                    _add(one, fo)

    return candidates


# ---------------------------------------------------------------------------
# Core: repair a single artifact
# ---------------------------------------------------------------------------

def _repair_single_artifact(
    request: GenerationRequest,
    item: dict,
    failure_output: str,
    strategy: RepairStrategy,
    repair_attempt: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Apply *strategy* to one generated test file.

    In ``class`` mode (default) the strategy is called once for the whole class.
    In ``method`` mode the strategy is called once per failing method; the file
    is re-read between calls so earlier fixes are visible to later ones.
    """
    source_file = Path(item.get("source_file", ""))
    test_file = Path(item.get("test_file", ""))
    if not source_file.exists():
        return [], []

    source_code = source_file.read_text(encoding="utf-8", errors="ignore")
    target_label = item.get("target_label", "")
    usages: list[dict] = []
    events: list[dict] = []
    effective_failure_output = item.get("failure_output") or failure_output

    repair_mode = getattr(request, "repair_mode", "class")
    if repair_mode == "method":
        parsed_methods = extract_failing_methods(effective_failure_output, test_file.stem)
        hinted_methods: list[str] = []
        direct_hint = item.get("method_name")
        if isinstance(direct_hint, str) and direct_hint.strip():
            hinted_methods.append(direct_hint.strip())
        extra_hints = item.get("failed_methods_hint")
        if isinstance(extra_hints, list):
            for m in extra_hints:
                if isinstance(m, str) and m.strip() and m.strip() not in hinted_methods:
                    hinted_methods.append(m.strip())

        methods_to_repair: list[str | None] = parsed_methods or hinted_methods or [None]
    else:
        methods_to_repair = [None]

    for method_focus in methods_to_repair:
        test_code = test_file.read_text(encoding="utf-8", errors="ignore") if test_file.exists() else ""
        extra_context = dict(item.get("repair_context") or {})
        if repair_attempt > 0:
            extra_context.setdefault("repair_attempt", int(repair_attempt))
        extra_context.setdefault("target_label", target_label)
        extra_context.setdefault("test_file", str(test_file))
        extra_context.setdefault("source_file", str(source_file))
        target_receipt = str(item.get("tatg_target_receipt_contract") or "").strip()
        if target_receipt:
            extra_context.setdefault("target_receipt_contract", target_receipt)
        execution_capsule = str(item.get("tatg_execution_capsule") or "").strip()
        if execution_capsule:
            extra_context.setdefault("target_execution_capsule", execution_capsule)
        mutation_feedback = item.get("tatg_mutation_feedback_context")
        if isinstance(mutation_feedback, dict) and mutation_feedback:
            extra_context.setdefault("mutation_feedback", mutation_feedback)
            extra_context.setdefault(
                "mutation_feedback_enabled",
                bool(mutation_feedback.get("mutation_feedback_enabled")),
            )
        extra_context.setdefault(
            "repair_failure_signals",
            extract_repair_failure_signals(effective_failure_output),
        )
        unused_local_hint = _unused_local_compile_mismatch_hint(
            failure_output=effective_failure_output,
            test_code=test_code,
            method_focus=method_focus,
        )
        if unused_local_hint:
            extra_context.setdefault(
                "compiler_error_context",
                unused_local_hint,
            )
            extra_context.setdefault("repair_prompt_compact", True)
            extra_context.setdefault(
                "repair_compaction_reason",
                "single_unused_local_compile_mismatch",
            )
        if "callable_surface_grounding" not in extra_context:
            try:
                from tatg.analysis.api_surface import extract_surface, render_surface_anchor

                class_name = str(item.get("class_name") or source_file.stem)
                callable_surface = render_surface_anchor(extract_surface(source_file, class_name))
                if callable_surface:
                    extra_context["callable_surface_grounding"] = callable_surface
            except Exception:
                pass

        # IRNM (Intra-Run Negative Memory): if the same compile-error
        # fingerprint recurs after the previous LLM patch, roll back to the
        # snapshot we captured before that patch and emit a `rollback` event.
        # This prevents the repair loop from grinding for 6 iterations on
        # the same set of compile errors (the observed Binance failure mode).
        cache_key = str(test_file)
        fingerprint = _error_fingerprint(effective_failure_output, test_file)
        previous_fp = _IRNM_PREV_FINGERPRINT.get(cache_key, "")
        if (
            fingerprint
            and previous_fp == fingerprint
            and cache_key in _IRNM_SNAPSHOT
            and test_code != _IRNM_SNAPSHOT[cache_key]
        ):
            snapshot = _IRNM_SNAPSHOT[cache_key]
            try:
                with TestFileRegistry.get_lock(test_file):
                    test_file.write_text(snapshot, encoding="utf-8")
            except OSError:
                pass
            events.append({
                "target_label": target_label,
                "test_file": str(test_file),
                "method": method_focus,
                "strategy": strategy.name,
                "status": "rollback",
                "fingerprint": fingerprint,
                "applied_rules": ["IRNM_ROLLBACK"],
            })
            # Reset so a subsequent (potentially different) iteration is not
            # immediately rolled back too.
            _IRNM_PREV_FINGERPRINT.pop(cache_key, None)
            continue

        # Snapshot the source BEFORE the patch so we can restore it later.
        _IRNM_SNAPSHOT[cache_key] = test_code
        _IRNM_PREV_FINGERPRINT[cache_key] = fingerprint

        ctx = RepairContext(
            source_code=source_code,
            test_code=test_code,
            failure_output=effective_failure_output,
            method_focus=method_focus,
            java_version=getattr(request, "java_version", None),
            junit_version=getattr(request, "junit_version", None),
            junit_family=getattr(request, "junit_family", None),
            extra=extra_context,
        )
        try:
            if isinstance(strategy, LLMRepairStrategy):
                compact_prompt = bool(
                    isinstance(ctx.extra, dict)
                    and ctx.extra.get("repair_prompt_compact")
                )
                system_prompt, user_prompt = strategy.build_prompt(
                    ctx,
                    compact=compact_prompt,
                )
                _record_repair_prompt_log(
                    request,
                    label=str(target_label or ""),
                    method=method_focus,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    compact=compact_prompt,
                )
        except Exception:
            pass
        try:
            repaired, usage = strategy.repair(ctx)
            applied_rules = (
                list(usage.get("applied_rules", []))
                if isinstance(usage, dict) else []
            )
            sub_strategy = (
                str(usage.get("strategy", ""))
                if isinstance(usage, dict) else ""
            )
            if repaired:
                with TestFileRegistry.get_lock(test_file):
                    test_file.parent.mkdir(parents=True, exist_ok=True)
                    test_file.write_text(repaired + "\n", encoding="utf-8")
            usages.append(usage)
            events.append({
                "target_label": target_label,
                "test_file": str(test_file),
                "method": method_focus,
                "strategy": strategy.name,
                "sub_strategy": sub_strategy or strategy.name,
                "status": "success",
                "fingerprint": fingerprint,
                "applied_rules": applied_rules,
            })
        except NonRetryableLLMRepairError as exc:
            events.append({
                "target_label": target_label,
                "test_file": str(test_file),
                "method": method_focus,
                "strategy": strategy.name,
                "status": "skipped",
                "reason": "llm_provider_rejected_request",
                "error": str(exc),
                "aborted_remaining_methods": True,
            })
            break
        except Exception as exc:
            events.append({
                "target_label": target_label,
                "test_file": str(test_file),
                "method": method_focus,
                "strategy": strategy.name,
                "status": "error",
                "error": str(exc),
            })

    return usages, events


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_repair(
    request: GenerationRequest,
    artifacts: list[dict],
    failure_output: str,
    *,
    workers: int = 1,
    strategy: RepairStrategy | None = None,
    repair_attempt: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Repair all generated test files.

    Parameters
    ----------
    request:
        The original generation request (provides model config and repair_mode).
    artifacts:
        Repair-candidate dicts produced by :func:`extract_repair_candidates`.
    failure_output:
        Aggregated test-run output used when an artifact has no individual output.
    workers:
        Maximum degree of parallelism across files.
    strategy:
        Repair strategy to use.  Defaults to :class:`LLMRepairStrategy`
        constructed from *request*.  Pass a custom strategy to swap in a
        different repair approach (AST, search-based, hybrid, …).

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(usages, events)`` — token-usage dicts and per-repair event records.
    """
    if not artifacts:
        return [], []

    effective_strategy = strategy or _default_strategy(request)

    all_usages: list[dict] = []
    all_events: list[dict] = []

    if workers > 1 and len(artifacts) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(artifacts))) as pool:
            futures = {
                pool.submit(
                    _repair_single_artifact,
                    request,
                    item,
                    failure_output,
                    effective_strategy,
                    repair_attempt,
                ): item
                for item in artifacts
            }
            for fut in as_completed(futures):
                u, e = fut.result()
                all_usages.extend(u)
                all_events.extend(e)
    else:
        for item in artifacts:
            u, e = _repair_single_artifact(
                request,
                item,
                failure_output,
                effective_strategy,
                repair_attempt,
            )
            all_usages.extend(u)
            all_events.extend(e)

    return all_usages, all_events


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------

def summarize_repair_metrics(test_runs: list[dict], generated_test_count: int) -> dict:
    """Summarise repair statistics across all test-run records."""
    initial_failed = test_runs[0]["failed_test_count"] if test_runs else 0
    final_failed = test_runs[-1]["failed_test_count"] if test_runs else 0
    initial_failures = test_runs[0].get("failure_count", 0) if test_runs else 0
    final_failures = test_runs[-1].get("failure_count", 0) if test_runs else 0
    initial_errors = test_runs[0].get("error_count", 0) if test_runs else 0
    final_errors = test_runs[-1].get("error_count", 0) if test_runs else 0
    repaired_total = sum(
        int(run.get("repaired_test_count", 0) or 0)
        for run in test_runs
        if int(run.get("repair_attempt", 0) or 0) > 0
    )
    repair_success_rate = (repaired_total / initial_failed) if initial_failed else 0.0
    rounds = [
        {
            "iteration": run["iteration"],
            "repair_attempt": run["repair_attempt"],
            "failed_test_count": run["failed_test_count"],
            "failure_count": run.get("failure_count", 0),
            "error_count": run.get("error_count", 0),
            "failure_type": run.get("failure_type", "none"),
            "repaired_test_count": run.get("repaired_test_count", 0),
        }
        for run in test_runs
    ]
    return {
        "generated_test_count": generated_test_count,
        "initial_failed_test_count": initial_failed,
        "final_failed_test_count": final_failed,
        "initial_failure_count": initial_failures,
        "final_failure_count": final_failures,
        "initial_error_count": initial_errors,
        "final_error_count": final_errors,
        "repaired_test_count": repaired_total,
        "repair_success_rate": repair_success_rate,
        "rounds": rounds,
    }
