#!/usr/bin/env python3
"""Run resumable mutation-only validation from an immutable TATG report.

The source report and its embedded final green test suites are read-only. Each
target test file is backed up, replaced for the duration of one invocation,
and restored in a finally block.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


API_FAILURE_PATTERNS = (
    "bad gateway",
    "connection reset",
    "connection refused",
    "llm_api_bad_gateway",
    "provider rejection",
    "timed out",
)

API_HTTP_STATUS_RE = re.compile(
    r"(?i)(?:http(?:\s+status)?|status(?:\s+code)?|response)"
    r"\s*[:=]?\s*(?:code\s*[:=]?\s*)?(502|503)\b"
)
API_TIMEOUT_RE = re.compile(
    r"(?i)\b(?:llm|model|provider|api|connection|request)\b"
    r"[^\n]{0,80}\b(?:timed out|timeout)\b"
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_tests(source: str) -> int:
    return len(
        re.findall(
            r"@\s*(?:(?:org\.junit\.jupiter\.api|org\.junit)\.)?Test\b",
            source,
        )
    )


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value or "")
    return value.strip("._") or "target"


def load_json(path: Path) -> dict[str, Any] | None:
    candidates = (path, Path(str(path) + ".partial"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def nested_dict(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return current if isinstance(current, dict) else {}


def final_test_file(payload: dict[str, Any]) -> tuple[str, str] | None:
    candidate_lists = (
        nested_dict(payload, "final_validation").get("candidate_test_files"),
        nested_dict(payload, "report_summary", "final_validation").get(
            "candidate_test_files"
        ),
    )
    for candidates in candidate_lists:
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            path = candidate.get("path") or candidate.get("test_file")
            content = candidate.get("content")
            if (
                isinstance(path, str)
                and isinstance(content, str)
                and count_tests(content) > 0
            ):
                return path, content
    return None


def method_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    coverage = payload.get("final_method_coverage")
    if not isinstance(coverage, dict):
        coverage = {}
    return {
        "line": coverage.get("line_coverage"),
        "branch": coverage.get("branch_coverage"),
    }


def mutation_score(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("target_mutation_score", "mutation_score", "score"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    coverage = payload.get("final_method_coverage")
    if isinstance(coverage, dict):
        return mutation_score(coverage)
    return None


def report_metrics(
    payload: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    coverage = method_coverage(payload)
    policy = (
        payload.get("validation_policy")
        if isinstance(payload.get("validation_policy"), dict)
        else {}
    )
    before = (
        policy.get("mutation_before_extra")
        if isinstance(policy.get("mutation_before_extra"), dict)
        else payload.get("mutation_before_extra")
    )
    # The iteration snapshot can precede deferred promotion/final suite
    # reconciliation. Prefer the descriptor-qualified PITest row measured
    # from the suite that is actually retained on disk.
    final_suite = (
        policy.get("mutation_final_suite")
        if isinstance(policy.get("mutation_final_suite"), dict)
        else payload.get("mutation_final_suite")
    )
    after_extra = (
        policy.get("mutation_after_extra")
        if isinstance(policy.get("mutation_after_extra"), dict)
        else payload.get("mutation_after_extra")
    )
    before_score = mutation_score(before)
    after_score = mutation_score(final_suite)
    if after_score is None:
        after_score = mutation_score(payload)
    if after_score is None:
        after_score = mutation_score(after_extra)
    if before_score is None:
        before_score = baseline.get("mutation")
    final_line = coverage.get("line")
    final_branch = coverage.get("branch")
    if not isinstance(final_line, (int, float)):
        final_line = baseline.get("line")
    if not isinstance(final_branch, (int, float)):
        final_branch = baseline.get("branch")

    def delta(final: Any, initial: Any) -> float | None:
        if isinstance(final, (int, float)) and isinstance(initial, (int, float)):
            return float(final) - float(initial)
        return None

    return {
        "archived_baseline_mutation_score": baseline.get("mutation"),
        "baseline_line_coverage": baseline.get("line"),
        "final_line_coverage": final_line,
        "line_coverage_delta": delta(final_line, baseline.get("line")),
        "baseline_branch_coverage": baseline.get("branch"),
        "final_branch_coverage": final_branch,
        "branch_coverage_delta": delta(final_branch, baseline.get("branch")),
        "baseline_mutation_score": before_score,
        "final_mutation_score": after_score,
        "mutation_score_delta": delta(after_score, before_score),
    }


def outcome_status(
    *,
    return_code: int,
    is_api_failure: bool,
    metrics: dict[str, Any],
) -> str:
    """Preserve exact target evidence even when the CLI reports a late error."""
    if is_api_failure:
        return "api_instability"
    if return_code == 0:
        return "completed"
    if isinstance(metrics.get("final_mutation_score"), (int, float)):
        return "completed_with_cli_error"
    return "failed"


def api_failure(log_text: str, payload: dict[str, Any] | None) -> bool:
    evidence = log_text
    if isinstance(payload, dict):
        evidence += "\n" + json.dumps(
            {
                "coverage_measurement_status": payload.get(
                    "coverage_measurement_status"
                ),
                "mutation_no_gain_reason": payload.get("mutation_no_gain_reason"),
                "mutation_extra_round_skipped_reason": payload.get(
                    "mutation_extra_round_skipped_reason"
                ),
            }
        )
    evidence = evidence.lower()
    return bool(
        any(pattern in evidence for pattern in API_FAILURE_PATTERNS)
        or API_HTTP_STATUS_RE.search(evidence)
        or API_TIMEOUT_RE.search(evidence)
    )


def model_is_ready(api_base: str, model: str, timeout: int = 30) -> bool:
    url = api_base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    entries = payload.get("data") if isinstance(payload, dict) else None
    return any(
        isinstance(entry, dict) and entry.get("id") == model
        for entry in entries or []
    )


def recover_lm_studio(api_base: str, model: str, audit_path: Path) -> bool:
    if model_is_ready(api_base, model):
        return True
    port_match = re.search(r":(\d+)(?:/|$)", api_base)
    port = port_match.group(1) if port_match else "1234"
    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    attempts: list[dict[str, Any]] = []
    if lms.exists():
        commands = (
            [str(lms), "server", "start", "--port", port],
            [
                str(lms),
                "load",
                model,
                "--identifier",
                model,
                "--context-length",
                "40000",
                "--parallel",
                "4",
                "--yes",
            ],
        )
        for command in commands:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
            attempts.append(
                {
                    "command": command,
                    "return_code": completed.returncode,
                    "output_tail": completed.stdout[-2000:],
                }
            )
    else:
        subprocess.run(["open", "-a", "LM Studio"], check=False)
        attempts.append({"command": ["open", "-a", "LM Studio"]})

    for _ in range(30):
        if model_is_ready(api_base, model):
            append_jsonl(
                audit_path,
                {
                    "event": "lm_studio_recovered",
                    "at": now_iso(),
                    "attempts": attempts,
                },
            )
            return True
        time.sleep(10)
    append_jsonl(
        audit_path,
        {
            "event": "lm_studio_recovery_failed",
            "at": now_iso(),
            "attempts": attempts,
        },
    )
    return False


def rewrite_archive_path(
    path: Path,
    rewrite_rules: list[tuple[Path, Path]] | None,
) -> Path:
    """Rewrite one expired archive prefix using explicit caller configuration."""

    value = path.expanduser()
    for source, destination in list(rewrite_rules or []):
        try:
            relative = value.relative_to(source.expanduser())
        except ValueError:
            continue
        return destination.expanduser() / relative
    return value


def artifact_target_coverage(payload: dict[str, Any]) -> tuple[float | None, float | None]:
    def number(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) else None

    line = number(payload.get("target_line_coverage"))
    branch = number(payload.get("target_branch_coverage"))
    coverage = payload.get("final_method_coverage")
    if isinstance(coverage, dict):
        if line is None:
            line = number(coverage.get("line_coverage"))
        if branch is None:
            branch = number(coverage.get("branch_coverage"))
    return line, branch


def find_matching_green_artifact(
    row: dict[str, Any],
    search_roots: list[Path] | None,
) -> tuple[Path, dict[str, Any]] | None:
    """Find a descriptor- and coverage-identical persistent green artifact."""

    class_name = str(row.get("class_name") or "").strip()
    method_name = str(row.get("method_name") or "").strip()
    descriptor = str(row.get("target_descriptor") or "").strip()
    if not class_name or not method_name:
        return None
    expected_line = row.get("target_line_coverage")
    expected_branch = row.get("target_branch_coverage")
    pattern = f"{class_name}.{method_name}*.json"
    matches: list[tuple[int, Path, dict[str, Any]]] = []
    for root in list(search_roots or []):
        if not root.is_dir():
            continue
        for candidate in root.rglob(pattern):
            if "candidate_salvage_audit" in candidate.name:
                continue
            payload = load_json(candidate)
            if payload is None or final_test_file(payload) is None:
                continue
            if str(payload.get("class_name") or class_name) != class_name:
                continue
            if str(payload.get("method_name") or method_name) != method_name:
                continue
            candidate_descriptor = str(
                payload.get("target_descriptor")
                or payload.get("target_method_descriptor")
                or ""
            ).strip()
            if descriptor and candidate_descriptor != descriptor:
                continue
            line, branch = artifact_target_coverage(payload)
            if isinstance(expected_line, (int, float)):
                if line is None or abs(line - float(expected_line)) > 1e-9:
                    continue
            if isinstance(expected_branch, (int, float)):
                if branch is None or abs(branch - float(expected_branch)) > 1e-9:
                    continue
            score = 0 if descriptor and candidate_descriptor == descriptor else 1
            matches.append((score, candidate, payload))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], str(item[1])))
    _score, path, payload = matches[0]
    return path, payload


def build_records(
    report_path: Path,
    out: Path,
    *,
    artifact_path_rewrites: list[tuple[Path, Path]] | None = None,
    artifact_search_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("baseline report must be a JSON list")
    records: list[dict[str, Any]] = []
    skipped = out / "skipped_targets.jsonl"
    for row in rows:
        if not isinstance(row, dict):
            continue
        original_artifact_path = Path(str(row.get("json_path") or ""))
        artifact_path = rewrite_archive_path(
            original_artifact_path,
            artifact_path_rewrites,
        )
        artifact = load_json(artifact_path)
        payload_source = "artifact"
        # A merged immutable report may itself contain the exact final green
        # suite even when the historical per-target artifact lived under
        # /tmp and no longer exists.  Prefer the artifact when available, but
        # do not discard checksum-preserved merged evidence solely because its
        # old path expired.
        if (artifact is None or final_test_file(artifact) is None) and final_test_file(row) is not None:
            artifact = row
            payload_source = "merged_report_row"
        if artifact is None or final_test_file(artifact) is None:
            matched = find_matching_green_artifact(row, artifact_search_roots)
            if matched is not None:
                artifact_path, artifact = matched
                payload_source = "coverage_identity_search"
        if artifact is None:
            append_jsonl(
                skipped,
                {
                    "reason": "missing_baseline_artifact",
                    "json_path": str(original_artifact_path),
                    "resolved_json_path": str(artifact_path),
                },
            )
            continue
        test_file = final_test_file(artifact)
        if test_file is None:
            append_jsonl(
                skipped,
                {
                    "reason": "missing_embedded_final_green_test",
                    "json_path": str(artifact_path),
                },
            )
            continue
        destination, content = test_file
        project_root = artifact.get("project_root")
        target_file = artifact.get("target_file")
        if not isinstance(project_root, str) or not isinstance(target_file, str):
            append_jsonl(
                skipped,
                {
                    "reason": "missing_required_target_metadata",
                    "json_path": str(artifact_path),
                },
            )
            continue
        target_descriptor = row.get("target_descriptor") or artifact.get(
            "target_descriptor"
        )
        identity = "|".join(
            str(value or "")
            for value in (
                row.get("project"),
                target_file,
                row.get("class_name"),
                row.get("method_name"),
                target_descriptor,
            )
        )
        short_hash = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
        display = (
            f"{row.get('class_name')}.{row.get('method_name')}"
            f"#{row.get('overload_short') or 'default'}"
        )
        records.append(
            {
                "target_id": f"{safe_component(display)}--{short_hash}",
                "display_target": display,
                "project": row.get("project"),
                "project_root": project_root,
                "target_file": target_file,
                "class_name": row.get("class_name") or artifact.get("class_name"),
                "method_name": row.get("method_name") or artifact.get("method_name"),
                "target_descriptor": target_descriptor,
                "module_path": artifact.get("module_path"),
                "test_command": sanitize_archived_test_command(
                    artifact.get("test_command"),
                    Path(project_root),
                ),
                "java_version": artifact.get("java_version"),
                "junit_version": artifact.get("junit_version"),
                "project_test_file": destination,
                "baseline_test_content": content,
                "baseline_test_count": count_tests(content),
                "baseline": {
                    "line": row.get("target_line_coverage"),
                    "branch": row.get("target_branch_coverage"),
                    "mutation": row.get("target_mutation_score"),
                },
                "baseline_artifact": str(artifact_path),
                "baseline_artifact_original": str(original_artifact_path),
                "baseline_payload_source": payload_source,
            }
        )
    return records


def sanitize_archived_test_command(
    command: Any,
    project_root: Path,
) -> str | None:
    """Rebase and deduplicate archived JaCoCo agent arguments."""

    if not isinstance(command, str) or not command.strip():
        return None
    current_dest = str(Path(project_root) / "target" / "jacoco.exec")
    agent_re = re.compile(
        r"-javaagent:(?P<body>[^\s'\"]*jacoco[^\s'\"]*)",
        flags=re.IGNORECASE,
    )

    def rebase(match: re.Match[str]) -> str:
        body = match.group("body")
        if "destfile=" in body:
            body = re.sub(
                r"destfile=[^,\s'\"]+",
                "destfile=" + current_dest,
                body,
                count=1,
            )
        return "-javaagent:" + body

    normalized = agent_re.sub(rebase, command)
    return re.sub(
        r"(?P<agent>-javaagent:[^\s'\"]*jacoco[^\s'\"]*)"
        r"(?:\s+(?P=agent))+",
        r"\g<agent>",
        normalized,
        flags=re.IGNORECASE,
    )


def command_for(
    args: argparse.Namespace, record: dict[str, Any], json_out: Path
) -> list[str]:
    command = [
        str(args.python),
        "-m",
        "tatg.cli",
        "generate",
        "--backend",
        "tatg",
        "--scope",
        "method",
        "--project-root",
        record["project_root"],
        "--target-file",
        record["target_file"],
        "--class-name",
        record["class_name"],
        "--method-name",
        record["method_name"],
        "--max-iterations",
        "0",
        "--repair-iterations",
        str(args.repair_iterations),
        "--retain-existing-tests",
        "--with-mutation",
        "--mutation-extra-iterations",
        str(args.mutation_iterations),
        "--model",
        args.model,
        "--api-base",
        args.api_base,
        "--api-key-env",
        args.api_key_env,
        "--llm-timeout-seconds",
        str(args.llm_timeout),
        "--tatg-preanalysis-timeout-seconds",
        str(args.llm_timeout),
        "--tatg-branch-analysis-timeout-seconds",
        str(args.llm_timeout),
        "--tatg-generation-timeout-seconds",
        str(args.llm_timeout),
        "--test-timeout-seconds",
        str(args.test_timeout),
        "--e2e-timeout-seconds",
        str(args.e2e_timeout),
        "--generation-parallelism",
        "1",
        "--repair-parallelism",
        "1",
        "--validation-parallelism",
        "4",
        "--log-prompts",
        "--json-out",
        str(json_out),
        "--no-progress",
    ]
    if record.get("target_descriptor"):
        command += ["--target-descriptor", str(record["target_descriptor"])]
    if record.get("module_path"):
        command += ["--module-path", str(record["module_path"])]
    if record.get("test_command"):
        command += ["--test-command", str(record["test_command"])]
    if record.get("java_version"):
        command += ["--java-version", str(record["java_version"])]
    if record.get("junit_version"):
        command += ["--junit-version", str(record["junit_version"])]
    return command


def restore_test(path: Path, backup: Path | None, existed: bool) -> None:
    if existed and backup and backup.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, path)
    elif not existed and path.exists():
        path.unlink()


def prioritize_records(
    records: list[dict[str, Any]],
    selectors: list[str],
) -> list[dict[str, Any]]:
    """Place exact display-target selectors before substring fallbacks.

    Descriptor-expanded targets commonly share a prefix, for example
    ``Owner.method#String`` and ``Owner.method#String,int``.  Treating both as
    the same substring priority can silently run the wrong overload when a
    focused validation uses ``--limit``.
    """

    if not selectors:
        return list(records)

    def priority(record: dict[str, Any]) -> tuple[int, int]:
        display = str(record.get("display_target") or "")
        for index, selector in enumerate(selectors):
            if display == selector:
                return 0, index
        for index, selector in enumerate(selectors):
            if selector in display:
                return 1, index
        return 2, 0

    indexed = list(enumerate(records))
    indexed.sort(key=lambda pair: (*priority(pair[1]), pair[0]))
    return [record for _, record in indexed]


def select_exact_records(
    records: list[dict[str, Any]],
    selectors: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Select exact descriptor-level targets without unrelated limit fill."""

    by_display = {
        str(record.get("display_target") or ""): record
        for record in records
        if str(record.get("display_target") or "")
    }
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    seen: set[str] = set()
    for selector in selectors:
        normalized = str(selector or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        record = by_display.get(normalized)
        if record is None:
            missing.append(normalized)
        else:
            selected.append(record)
    return selected, missing


def restore_unfinished_targets(out: Path) -> list[str]:
    events_path = out / "target_events.jsonl"
    if not events_path.exists():
        return []
    open_targets: dict[str, dict[str, Any]] = {}
    for line in events_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        target_id = event.get("target_id")
        if not isinstance(target_id, str):
            continue
        if event.get("event") == "target_started":
            open_targets[target_id] = event
        elif event.get("event") == "target_restored":
            open_targets.pop(target_id, None)
    restored: list[str] = []
    for target_id, event in open_targets.items():
        destination = Path(str(event.get("project_test_file") or ""))
        backup = out / "targets" / target_id / "project_test_file.before"
        if destination and backup.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, destination)
            restored.append(target_id)
            append_jsonl(
                events_path,
                {
                    **event,
                    "event": "target_restored_after_interruption",
                    "finished_at": now_iso(),
                },
            )
    return restored


def run_target(
    args: argparse.Namespace,
    record: dict[str, Any],
    out: Path,
    sequence: int,
) -> dict[str, Any]:
    target_dir = out / "targets" / record["target_id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = Path(record["project_test_file"])
    existed = destination.exists()
    backup = target_dir / "project_test_file.before" if existed else None
    if backup:
        shutil.copy2(destination, backup)
    start_event = {
        "event": "target_started",
        "target_id": record["target_id"],
        "display_target": record["display_target"],
        "sequence": sequence,
        "started_at": now_iso(),
        "project_test_file": str(destination),
    }
    append_jsonl(out / "target_events.jsonl", start_event)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(record["baseline_test_content"], encoding="utf-8")
        last_result: dict[str, Any] = {}
        attempts: list[dict[str, Any]] = []
        for attempt in (1, 2):
            if not recover_lm_studio(
                args.api_base, args.model, out / "lm_studio_recovery.jsonl"
            ):
                result = {
                    "target_id": record["target_id"],
                    "display_target": record["display_target"],
                    "attempt": attempt,
                    "status": "api_instability",
                    "api_failure": True,
                }
                attempts.append(result)
                last_result = result
                continue
            suffix = "" if attempt == 1 else f".retry{attempt}"
            json_out = target_dir / f"result{suffix}.json"
            log_out = target_dir / f"run{suffix}.log"
            command = command_for(args, record, json_out)
            write_json(target_dir / f"command{suffix}.json", command)
            started = time.monotonic()
            with log_out.open("wb") as log_handle:
                child_env = os.environ.copy()
                child_env[args.api_key_env] = args.api_key_value
                if args.tool_path:
                    child_env["PATH"] = os.pathsep.join(
                        [*args.tool_path, child_env.get("PATH", "")]
                    )
                process = subprocess.Popen(
                    command,
                    cwd=str(args.repo),
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=child_env,
                )
                try:
                    return_code = process.wait(timeout=args.e2e_timeout + 300)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    time.sleep(20)
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    return_code = process.wait()
            elapsed = time.monotonic() - started
            payload = load_json(json_out)
            log_text = log_out.read_text(encoding="utf-8", errors="ignore")
            is_api_failure = api_failure(log_text, payload)
            metrics = report_metrics(payload or {}, record["baseline"])
            final_count = (
                count_tests(destination.read_text(encoding="utf-8", errors="ignore"))
                if destination.exists()
                else None
            )
            final_suite_archive = target_dir / "project_test_file.after"
            if destination.is_file():
                shutil.copy2(destination, final_suite_archive)
            result = {
                "target_id": record["target_id"],
                "display_target": record["display_target"],
                "project": record.get("project"),
                "attempt": attempt,
                "status": outcome_status(
                    return_code=return_code,
                    is_api_failure=is_api_failure,
                    metrics=metrics,
                ),
                "return_code": return_code,
                "elapsed_seconds": round(elapsed, 3),
                "json_out": str(json_out),
                "log_out": str(log_out),
                "baseline_test_count": record["baseline_test_count"],
                "new_test_count": (
                    final_count - record["baseline_test_count"]
                    if isinstance(final_count, int)
                    else None
                ),
                "final_test_count": final_count,
                "final_test_suite_archive": (
                    str(final_suite_archive)
                    if final_suite_archive.is_file()
                    else None
                ),
                "api_failure": is_api_failure,
                "mutation_extra_iterations_run": (
                    payload or {}
                ).get("mutation_extra_iterations_run"),
                "mutation_no_gain_reason": (
                    payload or {}
                ).get("mutation_no_gain_reason"),
                "mutation_request_audit": (
                    (
                        (payload or {}).get("validation_policy")
                        if isinstance(
                            (payload or {}).get("validation_policy"),
                            dict,
                        )
                        else {}
                    ).get("mutation_request_audit")
                    or (payload or {}).get("mutation_request_audit")
                ),
                **metrics,
            }
            attempts.append(result)
            last_result = result
            append_jsonl(out / "attempts.jsonl", result)
            if not is_api_failure:
                break
        return {**last_result, "attempts": attempts}
    finally:
        restore_test(destination, backup, existed)
        append_jsonl(
            out / "target_events.jsonl",
            {
                **start_event,
                "event": "target_restored",
                "finished_at": now_iso(),
            },
        )


SUMMARY_FIELDS = (
    "target_id",
    "display_target",
    "project",
    "status",
    "return_code",
    "elapsed_seconds",
    "baseline_test_count",
    "new_test_count",
    "final_test_count",
    "baseline_line_coverage",
    "final_line_coverage",
    "line_coverage_delta",
    "baseline_branch_coverage",
    "final_branch_coverage",
    "branch_coverage_delta",
    "baseline_mutation_score",
    "final_mutation_score",
    "mutation_score_delta",
    "mutation_extra_iterations_run",
    "mutation_no_gain_reason",
    "api_failure",
    "json_out",
    "log_out",
)


def write_summary(out: Path, rows: list[dict[str, Any]]) -> None:
    write_json(out / "summary.json", rows)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable)
    )
    parser.add_argument("--model", default="qwen/qwen3.6-35b-a3b")
    parser.add_argument("--api-base", default="http://127.0.0.1:11435/v1")
    parser.add_argument("--api-key-env", default="TATG_LOCAL_API_KEY")
    parser.add_argument("--api-key-value", default="lm-studio")
    parser.add_argument(
        "--tool-path",
        action="append",
        default=[],
        help="Directory prepended to PATH for Maven/Java child commands.",
    )
    parser.add_argument("--mutation-iterations", type=int, default=2)
    parser.add_argument("--repair-iterations", type=int, default=2)
    parser.add_argument("--llm-timeout", type=int, default=3600)
    parser.add_argument("--test-timeout", type=int, default=1200)
    parser.add_argument("--e2e-timeout", type=int, default=14400)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--first-target",
        action="append",
        default=[],
        help="Display-target substring to schedule first; may be repeated.",
    )
    parser.add_argument(
        "--only-target",
        action="append",
        default=[],
        help=(
            "Run only this exact descriptor-level display target; may be "
            "repeated. Missing selectors fail before project files are touched."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--artifact-path-rewrite",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Explicitly map an expired artifact path prefix to a persistent "
            "archive prefix; may be repeated."
        ),
    )
    parser.add_argument(
        "--artifact-search-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Persistent archive root searched only for artifacts with exact "
            "class, method, descriptor, and merged coverage identity."
        ),
    )
    return parser.parse_args()


def parse_artifact_path_rewrites(values: list[str]) -> list[tuple[Path, Path]]:
    rewrites: list[tuple[Path, Path]] = []
    for value in values:
        old, separator, new = str(value or "").partition("=")
        if not separator or not old.strip() or not new.strip():
            raise ValueError(
                "--artifact-path-rewrite must use a non-empty OLD=NEW value"
            )
        rewrites.append((Path(old.strip()), Path(new.strip())))
    return rewrites


def main() -> int:
    args = parse_args()
    artifact_path_rewrites = parse_artifact_path_rewrites(
        args.artifact_path_rewrite
    )
    args.out.mkdir(parents=True, exist_ok=True)
    restored_after_interruption = restore_unfinished_targets(args.out)
    status_path = args.out / "driver_status.json"
    write_json(
        status_path,
        {
            "status": "starting",
            "pid": os.getpid(),
            "started_at": now_iso(),
            "restored_after_interruption": restored_after_interruption,
        },
    )
    records = build_records(
        args.baseline_report,
        args.out,
        artifact_path_rewrites=artifact_path_rewrites,
        artifact_search_roots=args.artifact_search_root,
    )
    if args.only_target:
        records, missing_selectors = select_exact_records(
            records,
            args.only_target,
        )
        if missing_selectors:
            raise ValueError(
                "exact --only-target selector(s) not found: "
                + ", ".join(missing_selectors)
            )
    if args.first_target:
        records = prioritize_records(records, args.first_target)
    if args.limit > 0:
        records = records[: args.limit]
    serializable_records = [
        {key: value for key, value in record.items() if key != "baseline_test_content"}
        for record in records
    ]
    write_json(args.out / "selected_targets.json", serializable_records)
    write_json(
        args.out / "archive_immutability_manifest.json",
        {
            "created_at": now_iso(),
            "baseline_report": str(args.baseline_report),
            "size": args.baseline_report.stat().st_size,
            "sha256": sha256_file(args.baseline_report),
            "artifact_path_rewrites": [
                {"old": str(old), "new": str(new)}
                for old, new in artifact_path_rewrites
            ],
            "artifact_search_roots": [
                str(path) for path in args.artifact_search_root
            ],
        },
    )
    if args.dry_run:
        write_json(
            status_path,
            {
                "status": "dry_run_completed",
                "selected_count": len(records),
                "finished_at": now_iso(),
            },
        )
        return 0
    if not records:
        write_json(
            status_path,
            {
                "status": "startup_failed",
                "reason": "no_targets_with_embedded_green_tests",
                "finished_at": now_iso(),
            },
        )
        return 2

    rows: list[dict[str, Any]] = []
    summary_path = args.out / "summary.json"
    if summary_path.exists():
        try:
            existing = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(existing, list):
                rows = [row for row in existing if isinstance(row, dict)]
        except (OSError, json.JSONDecodeError):
            rows = []
    completed = {
        row.get("target_id")
        for row in rows
        if row.get("status") == "completed"
    }
    pending = [record for record in records if record["target_id"] not in completed]
    write_summary(args.out, rows)
    for index, record in enumerate(pending, start=len(rows) + 1):
        write_json(
            status_path,
            {
                "status": "running",
                "pid": os.getpid(),
                "selected_count": len(records),
                "completed_count": len(rows),
                "current_index": index,
                "current_target": record["display_target"],
                "updated_at": now_iso(),
            },
        )
        result = run_target(args, record, args.out, index)
        rows.append(result)
        write_summary(args.out, rows)
    write_json(
        status_path,
        {
            "status": "completed",
            "pid": os.getpid(),
            "selected_count": len(records),
            "completed_count": len(rows),
            "finished_at": now_iso(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
