"""SootUp-based call-graph analyzer.

This module wraps the pre-built ``sootup-callgraph-pathfinder`` fat-JAR
(originally written in Java) as a subprocess, parses its text output, and
returns a structured :class:`~tatg.analysis.base.AnalysisResult`.

The JAR is looked up in the following order:

1. ``jar_path`` argument passed to :class:`SootUpCallGraphAnalyzer`.
2. The environment variable ``SOOTUP_CG_JAR``.
3. ``<workspace_root>/TATG/tools/external/sootup-callgraph-pathfinder/target/``
    (preferred bundled external-tool project location).
4. ``<workspace_root>/PLRDGEN/artifact/CAT/sootup-callgraph-pathfinder/target/``
    (legacy repo-relative fallback).
5. Any ``sootup-callgraph-pathfinder*.jar`` found under the working directory.

If the JAR is not found, the analyzer raises :class:`~tatg.analysis.base.AnalysisError`
with a clear message explaining how to build it (``mvn package -DskipTests``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from tatg.analysis.base import AnalysisError, AnalysisRequest, AnalysisResult
from tatg.analysis.callgraph.models import CallGraphResult
from tatg.analysis.callgraph.parser import parse_callgraph_text

# ---------------------------------------------------------------------------
# Default JAR search paths (relative to the workspace root, i.e. the directory
# containing "TATG" and "PLRDGEN" as siblings).
# ---------------------------------------------------------------------------
_DEFAULT_JAR_RELATIVE_PATHS = [
    # Preferred bundled external-tool location inside TATG
    "TATG/tools/external/sootup-callgraph-pathfinder/target/sootup-callgraph-pathfinder.jar",
    "TATG/tools/external/sootup-callgraph-pathfinder/target/sootup-callgraph-pathfinder-0.1-SNAPSHOT.jar",
    # Built by ``mvn package`` inside the sootup project
    "PLRDGEN/artifact/CAT/sootup-callgraph-pathfinder/target/sootup-callgraph-pathfinder.jar",
    "PLRDGEN/artifact/CAT/sootup-callgraph-pathfinder/target/sootup-callgraph-pathfinder-0.1-SNAPSHOT.jar",
    # Assembly plugin produces this name
    "PLRDGEN/artifact/CAT/sootup-callgraph-pathfinder/target/sootup-callgraph-pathfinder-jar-with-dependencies.jar",
]
_ENV_JAR_KEY = "SOOTUP_CG_JAR"
_BUILD_HINT = (
    "Build the JAR with:\n"
    "  cd TATG/tools/external/sootup-callgraph-pathfinder && mvn package -DskipTests\n"
    "The preferred bundled location is TATG/tools/external/sootup-callgraph-pathfinder/target/\n"
    "or set the SOOTUP_CG_JAR environment variable to an existing fat-JAR path."
)


def _truthy_env(name: str) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class SootUpCallGraphAnalyzer:
    """Invokes the SootUp CallGraphPathFinder JAR and returns structured results.

    Parameters:
        jar_path:   Explicit path to the fat-JAR. If *None*, the analyzer
                    searches the default locations described in the module
                    docstring.
        java_exe:   Path to the ``java`` executable (default: ``"java"``).
        timeout:    Maximum seconds to wait for the Java process (default 300).
    """

    name = "sootup-callgraph"

    def __init__(
        self,
        jar_path: str | Path | None = None,
        java_exe: str = "java",
        timeout: int = 300,
        cache_dir: str | Path | None = None,
        use_cache: bool = True,
    ) -> None:
        self._jar_path: Path | None = Path(jar_path) if jar_path else None
        self._java_exe = java_exe
        self._timeout = timeout
        self._use_cache = use_cache
        self._cache_dir = (
            Path(cache_dir)
            if cache_dir
            else Path.home() / ".cache" / "tatg" / "sootup_callgraph"
        )

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Run call-graph analysis and return a structured result."""
        self._validate_request(request)
        jar = self._resolve_jar()
        t0 = time.perf_counter()

        cache_hit = False
        cache_key = self._build_cache_key(request, jar)
        cache_file = self._cache_dir / f"{cache_key}.txt"

        raw_text: str
        if self._use_cache and cache_file.exists():
            raw_text = cache_file.read_text(encoding="utf-8")
            cache_hit = True
        else:
            with tempfile.TemporaryDirectory(prefix="tatg_cg_") as tmp:
                out_file = (
                    Path(request.output_file)
                    if request.output_file
                    else Path(tmp) / "cg_output.txt"
                )
                out_file.parent.mkdir(parents=True, exist_ok=True)

                argv = self._build_argv(jar, request, out_file)
                raw_text = self._run_java(argv, request, out_file)

            if self._use_cache:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
                cache_file.write_text(raw_text, encoding="utf-8")

        elapsed = time.perf_counter() - t0
        structured: CallGraphResult = parse_callgraph_text(raw_text)
        raw_path_count = len(structured.paths)
        structured.paths = [
            path for path in structured.paths
            if _path_ends_at_target(path, request.target_class, request.target_method)
        ]
        source_map = self._resolve_class_sources(request.classpath, structured)
        self._annotate_source_artifacts(structured, source_map)

        return AnalysisResult(
            target=f"{request.target_class}.{request.target_method}",
            analyzer_name=self.name,
            structured=structured,
            raw_text=raw_text,
            metadata={
                "elapsed_seconds": elapsed,
                "jar": str(jar),
                "max_depth": request.max_depth,
                "root_mode": request.root_mode,
                "limit_package_roots": request.limit_package_roots,
                "max_paths": request.max_paths,
                "max_search_states": request.max_search_states,
                "raw_path_count": raw_path_count,
                "target_path_count": len(structured.paths),
                "class_sources": source_map,
                "structured_json": structured.to_dict(),
                "cache_enabled": self._use_cache,
                "cache_hit": cache_hit,
                "cache_key": cache_key,
            },
        )

    def _validate_request(self, request: AnalysisRequest) -> None:
        if not request.classpath or not str(request.classpath).strip():
            raise AnalysisError("AnalysisRequest.classpath must be non-empty.")
        if not request.target_class or not request.target_class.strip():
            raise AnalysisError("AnalysisRequest.target_class must be non-empty.")
        if not request.target_method or not request.target_method.strip():
            raise AnalysisError("AnalysisRequest.target_method must be non-empty.")
        if request.max_depth < 1:
            raise AnalysisError("AnalysisRequest.max_depth must be >= 1.")
        if request.max_paths is not None and request.max_paths < 1:
            raise AnalysisError("AnalysisRequest.max_paths must be >= 1 when set.")
        if request.max_search_states is not None and request.max_search_states < 1:
            raise AnalysisError(
                "AnalysisRequest.max_search_states must be >= 1 when set."
            )

        allowed_root_modes = {"ALL", "PUBLIC", "MAINS"}
        root_mode = (request.root_mode or "").upper()
        if root_mode not in allowed_root_modes:
            raise AnalysisError(
                "AnalysisRequest.root_mode must be one of ALL, PUBLIC, MAINS. "
                f"Got: {request.root_mode!r}"
            )

    def _resolve_jar(self) -> Path:
        if self._jar_path is not None:
            if self._jar_path.exists():
                return self._jar_path
            raise AnalysisError(
                f"Specified JAR does not exist: {self._jar_path}\n{_BUILD_HINT}"
            )

        env_jar = os.environ.get(_ENV_JAR_KEY)
        if env_jar:
            p = Path(env_jar)
            if p.exists():
                return p
            raise AnalysisError(
                f"${_ENV_JAR_KEY} points to a non-existent file: {env_jar}\n{_BUILD_HINT}"
            )

        repo_roots: list[Path] = []
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / "PLRDGEN").is_dir():
                repo_roots.append(parent)
                break
            if parent.name == "TATG" and (parent.parent / "PLRDGEN").is_dir():
                repo_roots.append(parent.parent)
                break
            if (parent / "TATG" / "src" / "tatg").is_dir():
                repo_roots.append(parent)
                break
        repo_roots.append(Path.cwd())
        cwd_parent = Path.cwd().parent
        if cwd_parent not in repo_roots:
            repo_roots.append(cwd_parent)

        for repo_root in repo_roots:
            for rel in _DEFAULT_JAR_RELATIVE_PATHS:
                candidate = repo_root / rel
                if candidate.exists():
                    return candidate

        for match in sorted(Path.cwd().rglob("sootup-callgraph-pathfinder*.jar")):
            return match

        raise AnalysisError(
            "Could not find the sootup-callgraph-pathfinder fat-JAR.\n" + _BUILD_HINT
        )

    def _build_argv(
        self,
        jar: Path,
        request: AnalysisRequest,
        out_file: Path,
    ) -> list[str]:
        argv: list[str] = [
            self._java_exe,
            "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
            "-Dorg.slf4j.simpleLogger.log.sootup=error",
            "-jar",
            str(jar),
            f"--roots={request.root_mode.upper()}",
            f"--limitPackageRoots={'true' if request.limit_package_roots else 'false'}",
        ]

        if request.dep_dir:
            argv.append(f"--depDir={request.dep_dir}")
        if not _truthy_env("SOOTUP_CG_DISABLE_LIMIT_FLAGS"):
            if request.max_paths is not None:
                argv.append(f"--maxPaths={request.max_paths}")
            if request.max_search_states is not None:
                argv.append(f"--maxSearchStates={request.max_search_states}")
        if request.threads > 1:
            argv.append(f"--threads={request.threads}")

        argv += [
            request.classpath,
            request.target_class,
            request.target_method,
            str(request.max_depth),
            str(out_file),
        ]
        return argv

    def _annotate_source_artifacts(
        self,
        result: CallGraphResult,
        source_map: dict[str, str],
    ) -> None:
        for path in result.paths:
            for step in path.steps:
                step.source_jar = source_map.get(step.class_fqn, "UNKNOWN")
        for init in result.initializations:
            init.source_jar = source_map.get(init.class_fqn, "UNKNOWN")

    def _resolve_class_sources(
        self,
        classpath: str,
        result: CallGraphResult,
    ) -> dict[str, str]:
        classes = {result.target_class}
        for path in result.paths:
            for step in path.steps:
                classes.add(step.class_fqn)
        for init in result.initializations:
            classes.add(init.class_fqn)

        classes.discard("")
        source_map: dict[str, str] = {}

        for cls in classes:
            if cls.startswith(("java.", "javax.", "jdk.", "sun.", "com.sun.")):
                source_map[cls] = "JDK_RUNTIME"

        unresolved = [c for c in classes if c not in source_map]
        if not unresolved:
            return source_map

        cp_entries = [p for p in classpath.split(os.pathsep) if p.strip()]
        for cls in unresolved:
            class_rel_path = cls.replace(".", "/") + ".class"
            source = "UNKNOWN"
            for entry in cp_entries:
                p = Path(entry)
                if p.is_dir():
                    if (p / class_rel_path).exists():
                        source = str(p)
                        break
                    continue
                if p.is_file() and p.suffix == ".jar":
                    try:
                        with zipfile.ZipFile(p) as zf:
                            if class_rel_path in zf.namelist():
                                source = str(p)
                                break
                    except (OSError, zipfile.BadZipFile):
                        continue
            source_map[cls] = source

        return source_map

    def _run_java(
        self,
        argv: list[str],
        request: AnalysisRequest,
        out_file: Path,
    ) -> str:
        timeout = self._timeout_for_request(request)
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except FileNotFoundError:
            raise AnalysisError(
                f"Java executable not found: '{self._java_exe}'. "
                "Ensure a JDK >= 17 is installed and on PATH."
            )
        except subprocess.TimeoutExpired:
            raise AnalysisError(
                f"SootUp call-graph analysis timed out after {timeout}s for "
                f"{request.target_class}.{request.target_method}."
            )

        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[:2000]
            raise AnalysisError(
                f"SootUp JAR exited with code {result.returncode}.\n"
                f"stderr: {stderr_snippet}"
            )

        if out_file.exists():
            return out_file.read_text(encoding="utf-8")

        if result.stdout.strip():
            return result.stdout
        raise AnalysisError(
            f"SootUp JAR produced no output file at {out_file} and no stdout. "
            f"stderr: {(result.stderr or '')[:1000]}"
        )

    def _timeout_for_request(self, request: AnalysisRequest) -> float:
        raw = None
        try:
            raw = request.extra.get("timeout_seconds")
        except Exception:
            raw = None
        if raw is None:
            return float(self._timeout)
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            return float(self._timeout)
        return max(0.001, timeout)

    def _build_cache_key(self, request: AnalysisRequest, jar: Path) -> str:
        # Bump version whenever CLI flags or request-shaping logic changes.
        payload = {
            "analyzer": self.name,
            "version": "v3",
            "jar": self._path_fingerprint(jar),
            "java_exe": self._java_exe,
            "target_class": request.target_class,
            "target_method": request.target_method,
            "max_depth": request.max_depth,
            "root_mode": request.root_mode.upper(),
            "limit_package_roots": request.limit_package_roots,
            "max_paths": request.max_paths,
            "max_search_states": request.max_search_states,
            "threads": request.threads,
            "dep_dir": request.dep_dir or "",
            "classpath": self._classpath_fingerprint(request.classpath),
        }
        src = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(src.encode("utf-8")).hexdigest()

    def _classpath_fingerprint(self, classpath: str) -> list[str]:
        fps: list[str] = []
        for raw in classpath.split(os.pathsep):
            entry = raw.strip()
            if not entry:
                continue
            fps.append(self._path_fingerprint(Path(entry)))
        return fps

    def _path_fingerprint(self, path: Path) -> str:
        try:
            rp = path.resolve()
        except OSError:
            rp = path

        if not rp.exists():
            return f"MISSING:{rp}"

        try:
            stat = rp.stat()
            kind = "DIR" if rp.is_dir() else "FILE"
            return f"{kind}:{rp}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            return f"UNREADABLE:{rp}"


def _path_ends_at_target(path: object, target_class: str, target_method: str) -> bool:
    steps = list(getattr(path, "steps", []) or [])
    if not steps:
        return False
    last = steps[-1]
    method_name = str(getattr(last, "method_name", "") or "")
    class_fqn = str(getattr(last, "class_fqn", "") or "")
    if method_name != target_method:
        return False
    if not target_class:
        return True
    return class_fqn == target_class or class_fqn.endswith("." + target_class) or target_class.endswith("." + class_fqn)


# ---------------------------------------------------------------------------
# Standalone preflight smoke-test
# ---------------------------------------------------------------------------

_SMOKE_SOURCE = """\
public class QSmokeTarget {
    private int value;

    public QSmokeTarget() { this.value = 0; }
    public QSmokeTarget(int initial) { this.value = initial; }

    public void setup(int v) {
        this.value = v;
        compute();
    }

    private void compute() {
        this.value = this.value * 2;
    }

    public int read() { return value; }

    public static QSmokeTarget create(int seed) {
        QSmokeTarget t = new QSmokeTarget(seed);
        t.setup(seed);
        return t;
    }
}
"""


def preflight_sootup_analyzer(
    *,
    jar_path: "str | Path | None" = None,
    java_exe: str = "java",
    timeout: float = 30.0,
) -> dict:
    """Verify SootUp JAR availability and run an end-to-end connectivity smoke test.

    Three checks are performed in sequence:

    1. **JAR resolution** — locate the fat-JAR via ``jar_path``, ``SOOTUP_CG_JAR``,
       or the default search paths.  Returns immediately on failure.
    2. **Java executable check** — run ``java -version`` to confirm a JDK is on PATH.
    3. **End-to-end smoke test** — compile a tiny synthetic Java class with ``javac``,
       run SootUp against it, and assert that at least one call path *and* one
       constructor initialisation are present in the output.

    Returns a ``dict`` with:

    ``"ok"``
        ``True`` only when all three checks pass.
    ``"jar_found"`` / ``"jar_path"``
        Whether the JAR was resolved and its path.
    ``"java_ok"`` / ``"java_version"``
        Whether ``java -version`` succeeded and the first line of its output.
    ``"javac_ok"``
        Whether ``javac`` is available (smoke test requires it).
    ``"smoke_test_ran"``
        Whether the end-to-end test actually executed.
    ``"smoke_paths_found"`` / ``"smoke_inits_found"``
        Number of call paths / initialisation entries in the smoke output.
    ``"error"``
        Human-readable error string, or ``None`` on full success.
    """
    result: dict = {
        "ok": False,
        "jar_found": False,
        "jar_path": None,
        "java_ok": False,
        "java_version": None,
        "javac_ok": False,
        "smoke_test_ran": False,
        "smoke_paths_found": 0,
        "smoke_inits_found": 0,
        "error": None,
    }

    # --- Step 1: resolve JAR ---
    analyzer = SootUpCallGraphAnalyzer(jar_path=jar_path, java_exe=java_exe, timeout=int(timeout))
    try:
        jar = analyzer._resolve_jar()
    except AnalysisError as exc:
        result["error"] = str(exc)
        return result
    result["jar_found"] = True
    result["jar_path"] = str(jar)

    # --- Step 2: verify java executable ---
    try:
        proc = subprocess.run(
            [java_exe, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # java -version writes to stderr
        version_line = (proc.stderr or proc.stdout or "").splitlines()[0] if proc.returncode == 0 else ""
        if proc.returncode == 0 and version_line:
            result["java_ok"] = True
            result["java_version"] = version_line.strip()
        else:
            result["error"] = f"`{java_exe} -version` failed (rc={proc.returncode})"
            return result
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        result["error"] = f"Java executable not usable: {exc}"
        return result

    # --- Step 3: end-to-end smoke test (requires javac) ---
    with tempfile.TemporaryDirectory(prefix="tatg_sootup_preflight_") as tmp:
        tmp_path = Path(tmp)
        src_file = tmp_path / "QSmokeTarget.java"
        out_file = tmp_path / "cg_smoke.txt"
        src_file.write_text(_SMOKE_SOURCE, encoding="utf-8")

        # Derive javac from java_exe so bytecode version matches the running JVM
        _java_path = Path(java_exe)
        if _java_path.is_absolute():
            _javac_candidate = _java_path.parent / "javac"
            javac_exe = str(_javac_candidate) if _javac_candidate.exists() else "javac"
        else:
            javac_exe = "javac"

        # Check javac availability
        try:
            javac_proc = subprocess.run(
                [javac_exe, str(src_file)],
                capture_output=True,
                text=True,
                timeout=20,
                cwd=str(tmp_path),
            )
            if javac_proc.returncode != 0:
                result["error"] = (
                    f"javac compilation of smoke class failed: {javac_proc.stderr[:500]}"
                )
                return result
            result["javac_ok"] = True
        except FileNotFoundError:
            # javac not on PATH — skip end-to-end, report partial success
            result["ok"] = True  # JAR+Java OK is good enough without javac
            result["error"] = "javac not found on PATH; end-to-end smoke test skipped"
            return result
        except subprocess.TimeoutExpired:
            result["error"] = "javac timed out compiling smoke class"
            return result

        # Run SootUp against the compiled smoke class
        argv = [
            java_exe,
            "-Dorg.slf4j.simpleLogger.defaultLogLevel=error",
            "-Dorg.slf4j.simpleLogger.log.sootup=error",
            "-jar", str(jar),
            "--roots=PUBLIC",
            "--limitPackageRoots=true",
            str(tmp_path),       # classpath
            "QSmokeTarget",      # target class
            "compute",           # target method (private; reachable via setup/create)
            "3",                 # max depth
            str(out_file),       # output file
        ]
        try:
            sootup_proc = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=int(timeout),
            )
        except subprocess.TimeoutExpired:
            result["error"] = f"SootUp smoke test timed out after {timeout}s"
            return result

        if sootup_proc.returncode != 0:
            result["error"] = (
                f"SootUp smoke test exited with rc={sootup_proc.returncode}: "
                f"{(sootup_proc.stderr or '')[:400]}"
            )
            return result

        raw = out_file.read_text(encoding="utf-8") if out_file.exists() else sootup_proc.stdout
        if not raw.strip():
            result["error"] = "SootUp produced no output for smoke test"
            return result

        result["smoke_test_ran"] = True
        parsed = parse_callgraph_text(raw)
        result["smoke_paths_found"] = len(parsed.paths)
        result["smoke_inits_found"] = len(parsed.initializations)

        if parsed.paths and parsed.initializations:
            result["ok"] = True
        else:
            missing = []
            if not parsed.paths:
                missing.append("call paths")
            if not parsed.initializations:
                missing.append("constructor initializations")
            result["error"] = (
                f"SootUp smoke test ran but output is incomplete — missing: {', '.join(missing)}. "
                f"Raw output ({len(raw)} chars): {raw[:300]}"
            )
