from __future__ import annotations

import os
from pathlib import Path
import re
import shutil


def augment_executable_path(
    env: dict[str, str],
    *,
    configured_paths: str | None = None,
) -> None:
    """Append configured executable directories without host-specific defaults."""

    existing = [
        item for item in str(env.get("PATH") or "").split(os.pathsep) if item
    ]
    seen = set(existing)
    raw_configured = configured_paths
    if raw_configured is None:
        raw_configured = str(
            env.get("TATG_EXTRA_PATH")
            or os.environ.get("TATG_EXTRA_PATH")
            or ""
        )
    additions: list[str] = []
    for candidate in str(raw_configured or "").split(os.pathsep):
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        if Path(candidate).is_dir():
            additions.append(candidate)
            seen.add(candidate)
    if additions:
        env["PATH"] = os.pathsep.join(existing + additions)


def java_tool_candidates(
    tool_name: str,
    *,
    env: dict[str, str] | None = None,
) -> list[str]:
    """Return configured Java tool candidates without host-specific paths."""

    values = dict(os.environ if env is None else env)
    home_keys = ["JAVA_HOME", "TATG_JAVA_HOME"]
    versioned_keys = sorted(
        (
            key
            for key in values
            if re.fullmatch(r"(?:TATG_)?JAVA_HOME_[0-9]+", key)
        ),
        key=lambda key: int(key.rsplit("_", 1)[-1]),
        reverse=True,
    )
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: Path | str) -> None:
        rendered = str(path)
        if rendered in seen:
            return
        candidate = Path(rendered)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            candidates.append(rendered)
            seen.add(rendered)

    for key in [*home_keys, *versioned_keys]:
        home = str(values.get(key) or "").strip()
        if home:
            add(Path(home) / "bin" / tool_name)

    configured_env = dict(values)
    augment_executable_path(configured_env)
    resolved = shutil.which(tool_name, path=configured_env.get("PATH"))
    if resolved:
        add(resolved)

    return candidates


def java_test_class_stem(source_stem: str) -> str:
    """Return a Surefire-friendly top-level test class stem for a Java type.

    Production Java binary/source names may contain ``$`` for synthetic or
    nested-looking classes, but Maven Surefire treats ``$`` specially in test
    selectors and may skip such top-level generated tests.  Generated test
    classes use a plain identifier while the target class name/descriptor remain
    unchanged for source analysis and coverage matching.
    """
    stem = re.sub(r"[^A-Za-z0-9_]+", "", str(source_stem or ""))
    if not stem:
        stem = "Generated"
    if not re.match(r"[A-Za-z_]", stem[0]):
        stem = f"Generated{stem}"
    return stem


def derive_java_test_file(source_file: Path) -> Path:
    """Derive the canonical Java test file path for a source file.

    The mapping is:
      .../src/main/java/<pkg>/<Class>.java -> .../src/test/java/<pkg>/<Class>Test.java

    This implementation handles both absolute and relative input paths.
    """
    if source_file.suffix != ".java":
        return source_file.with_name(source_file.stem + "Test" + source_file.suffix)

    parts = list(source_file.parts)
    for idx in range(len(parts) - 2):
        if parts[idx : idx + 3] == ["src", "main", "java"]:
            mapped = (
                parts[:idx]
                + ["src", "test", "java"]
                + parts[idx + 3 : -1]
                + [java_test_class_stem(source_file.stem) + "Test.java"]
            )
            return Path(*mapped)

    return source_file.with_name(java_test_class_stem(source_file.stem) + "Test.java")
