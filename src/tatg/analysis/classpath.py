"""Classpath extraction utilities for SootUp and other static analysis tools.

Supports multiple build systems:

* **Maven** — runs ``mvn dependency:build-classpath`` and includes the compiled
  class directories (``target/classes``).
* **Gradle** — injects a one-shot ``printCompileClasspath`` task into a temporary
  init script, then runs ``./gradlew`` (or ``gradle``) to print it; also includes
  ``build/classes`` dirs.
* **Direct Java** — collects pre-compiled ``.class`` directories and any ``.jar``
  files found under a user-specified library root.  No build tool is invoked.
* **Auto-detect** — delegates to :func:`~tatg.build_system.detect_build_system`
  and picks the appropriate strategy.

The main entry-point is :func:`build_classpath_for_project`.  It returns a
``os.pathsep``-separated string of JARs and class-directory paths, ready to
pass directly to the SootUp JAR via ``--cp`` or to :class:`AnalysisRequest`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from tatg.build_system import BuildSystem, detect_build_system

__all__ = [
    "build_classpath_for_project",
    "ClasspathError",
]


class ClasspathError(RuntimeError):
    """Raised when classpath extraction fails."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_classpath_for_project(
    project_root: str | Path,
    *,
    build_system: BuildSystem | str | None = None,
    module: str | None = None,
    lib_dir: str | Path | None = None,
    timeout: int = 120,
    include_test_classes: bool = False,
) -> str:
    """Return a colon-separated classpath for *project_root*.

    Parameters
    ----------
    project_root:
        Root directory of the project.
    build_system:
        Force a particular build system.  Auto-detected when *None*.
    module:
        Maven/Gradle sub-module path relative to *project_root*.  When set,
        the classpath is built for that module only.  Example: ``"mall-admin"``.
    lib_dir:
        For :data:`BuildSystem.DIRECT_JAVA` only — directory containing
        ``*.jar`` dependency files.  Ignored for Maven/Gradle.
    timeout:
        Maximum seconds for the build-tool subprocess.
    include_test_classes:
        When *True*, also include ``target/test-classes`` /
        ``build/classes/java/test`` on the classpath.

    Returns
    -------
    str
        ``os.pathsep``-separated classpath string.

    Raises
    ------
    ClasspathError
        When the extraction subprocess fails or returns no entries.
    """
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ClasspathError(f"project_root does not exist: {root}")

    if build_system is None:
        target_dir = root / module if module else root
        bs = detect_build_system(target_dir if target_dir.is_dir() else root)
    elif isinstance(build_system, str):
        bs = BuildSystem(build_system.lower())
    else:
        bs = build_system

    if bs == BuildSystem.MAVEN:
        return _maven_classpath(root, module=module, timeout=timeout, include_test=include_test_classes)
    if bs == BuildSystem.GRADLE:
        return _gradle_classpath(root, module=module, timeout=timeout, include_test=include_test_classes)
    if bs in (BuildSystem.DIRECT_JAVA, BuildSystem.UNKNOWN):
        return _direct_java_classpath(root, lib_dir=lib_dir, include_test=include_test_classes)

    raise ClasspathError(
        f"Classpath extraction is not supported for build system: {bs.value!r}. "
        "Supported systems: maven, gradle, direct_java."
    )


# ---------------------------------------------------------------------------
# Maven
# ---------------------------------------------------------------------------


def _maven_classpath(
    project_root: Path,
    *,
    module: str | None,
    timeout: int,
    include_test: bool,
) -> str:
    """Extract classpath via ``mvn dependency:build-classpath``."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        cp_file = Path(f.name)

    try:
        mvn_bin = (
            str(os.environ.get("TATG_MAVEN_BIN") or "").strip()
            or shutil.which("mvn")
            or "mvn"
        )
        mvn_args = [mvn_bin, "--batch-mode", "-q"]
        if module:
            mvn_args += ["-pl", module, "-am"]
        mvn_args += [
            "dependency:build-classpath",
            f"-Dmdep.outputFile={cp_file}",
            "-DincludeScope=compile",
            "-DexcludeArtifactIds=",
        ]

        result = subprocess.run(
            mvn_args,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise ClasspathError(
                f"mvn dependency:build-classpath failed (rc={result.returncode}):\n"
                f"stdout: {result.stdout[-2000:]}\n"
                f"stderr: {result.stderr[-2000:]}"
            )

        dep_cp = cp_file.read_text(encoding="utf-8").strip()
    finally:
        cp_file.unlink(missing_ok=True)

    # Collect compiled class directories from this module (and sub-modules)
    target_root = project_root / module if module else project_root
    class_dirs = _collect_maven_class_dirs(target_root, include_test=include_test)

    entries = [p for p in dep_cp.split(os.pathsep) if p.strip()]
    entries = class_dirs + entries

    if not entries:
        raise ClasspathError("Maven classpath extraction produced no entries.")

    return os.pathsep.join(entries)


def _collect_maven_class_dirs(root: Path, *, include_test: bool) -> list[str]:
    """Walk *root* for Maven ``target/classes`` (and optionally ``target/test-classes``)."""
    dirs: list[str] = []
    for classes_dir in root.rglob("target/classes"):
        if classes_dir.is_dir():
            dirs.append(str(classes_dir))
    if include_test:
        for test_dir in root.rglob("target/test-classes"):
            if test_dir.is_dir():
                dirs.append(str(test_dir))
    return dirs


# ---------------------------------------------------------------------------
# Gradle
# ---------------------------------------------------------------------------


def _gradle_classpath(
    project_root: Path,
    *,
    module: str | None,
    timeout: int,
    include_test: bool,
) -> str:
    """Extract classpath via a temporary Gradle init script that prints it."""

    # Gradle init script that adds a ``printCompileClasspath`` task to every project
    _init_script = textwrap.dedent("""\
        allprojects {
            task printCompileClasspath {
                doLast {
                    try {
                        def cp = configurations.compileClasspath.resolvedConfiguration.resolvedArtifacts
                                    .collect { it.file.absolutePath }
                        if (cp) println("TATG_CP:" + cp.join(File.pathSeparator))
                    } catch (ignored) {}
                    try {
                        def runtimeJars = configurations.runtimeClasspath.resolvedConfiguration.resolvedArtifacts
                                    .collect { it.file.absolutePath }
                        if (runtimeJars) println("TATG_RT:" + runtimeJars.join(File.pathSeparator))
                    } catch (ignored) {}
                }
            }
        }
    """)

    with tempfile.NamedTemporaryFile(
        suffix=".gradle", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(_init_script)
        init_file = Path(f.name)

    try:
        gradle_exe = _find_gradle_exe(project_root)
        task_prefix = f":{module}:" if module else ""

        cmd = [gradle_exe, "--init-script", str(init_file), "--quiet", "--no-daemon",
               f"{task_prefix}printCompileClasspath"]
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        init_file.unlink(missing_ok=True)

    if result.returncode != 0:
        raise ClasspathError(
            f"Gradle classpath task failed (rc={result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )

    entries: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith("TATG_CP:") or line.startswith("TATG_RT:"):
            for part in line.split(":", 1)[1].split(os.pathsep):
                part = part.strip()
                if part and part not in entries:
                    entries.append(part)

    # Gradle compiled output dirs
    target_root = project_root / module if module else project_root
    class_dirs = _collect_gradle_class_dirs(target_root, include_test=include_test)
    # Prepend class dirs so project classes take priority
    entries = class_dirs + [e for e in entries if e not in class_dirs]

    if not entries:
        raise ClasspathError(
            "Gradle classpath extraction produced no entries. "
            "Make sure the project has been compiled (./gradlew compileJava) "
            "and that 'compileClasspath' configuration exists."
        )

    return os.pathsep.join(entries)


def _find_gradle_exe(project_root: Path) -> str:
    """Return ``./gradlew`` if present, otherwise fall back to ``gradle``."""
    wrapper = project_root / "gradlew"
    if wrapper.exists() and os.access(wrapper, os.X_OK):
        return str(wrapper)
    return "gradle"


def _collect_gradle_class_dirs(root: Path, *, include_test: bool) -> list[str]:
    """Walk *root* for Gradle ``build/classes/java/main`` dirs."""
    dirs: list[str] = []
    for d in root.rglob("build/classes/java/main"):
        if d.is_dir():
            dirs.append(str(d))
    for d in root.rglob("build/classes/kotlin/main"):
        if d.is_dir():
            dirs.append(str(d))
    if include_test:
        for d in root.rglob("build/classes/java/test"):
            if d.is_dir():
                dirs.append(str(d))
    return dirs


# ---------------------------------------------------------------------------
# Direct Java (pre-compiled)
# ---------------------------------------------------------------------------


def _direct_java_classpath(
    project_root: Path,
    *,
    lib_dir: str | Path | None,
    include_test: bool,
) -> str:
    """Collect class dirs and JARs for a plain-compiled Java project.

    Searches for:
    * Any ``*.class``-containing directory under *project_root* (e.g. ``out/``,
      ``bin/``, ``build/``, ``target/classes``).
    * All ``*.jar`` files under *lib_dir* (or ``lib/`` / ``libs/`` if present).
    """
    entries: list[str] = []

    # 1. Class directories (common output folder patterns)
    for pattern in ("target/classes", "build/classes", "out/production", "bin", "classes"):
        for d in project_root.rglob(pattern):
            if d.is_dir() and any(d.rglob("*.class")):
                s = str(d)
                if s not in entries:
                    entries.append(s)

    if include_test:
        for pattern in ("target/test-classes", "build/test-classes", "out/test", "test-classes"):
            for d in project_root.rglob(pattern):
                if d.is_dir():
                    s = str(d)
                    if s not in entries:
                        entries.append(s)

    # 2. JAR files from the lib directory
    lib_candidates: list[Path] = []
    if lib_dir is not None:
        lib_candidates.append(Path(lib_dir).resolve())
    else:
        for name in ("lib", "libs", "dependencies", "vendor"):
            candidate = project_root / name
            if candidate.is_dir():
                lib_candidates.append(candidate)

    for lib in lib_candidates:
        for jar in sorted(lib.rglob("*.jar")):
            s = str(jar)
            if s not in entries:
                entries.append(s)

    if not entries:
        raise ClasspathError(
            f"Direct-Java classpath extraction found no class directories or JARs "
            f"under {project_root}. "
            "Make sure the project has been compiled and specify lib_dir if dependencies "
            "live outside the project root."
        )

    return os.pathsep.join(entries)
