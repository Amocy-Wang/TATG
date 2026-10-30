from __future__ import annotations

from enum import Enum
from pathlib import Path
import re
import shlex
import xml.etree.ElementTree as ET


class BuildSystem(str, Enum):
    MAVEN = "maven"
    GRADLE = "gradle"
    BAZEL = "bazel"
    DIRECT_JAVA = "direct_java"
    PYTHON = "python"
    UNKNOWN = "unknown"


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _resolve_property(value: str, properties: dict[str, str]) -> str:
    if not value:
        return ""
    match = re.fullmatch(r"\$\{([^}]+)\}", value.strip())
    if not match:
        return value.strip()
    return properties.get(match.group(1), "").strip()


def _resolve_embedded_properties(value: str, properties: dict[str, str]) -> str:
    if not value:
        return ""

    def repl(match: re.Match[str]) -> str:
        return properties.get(match.group(1), match.group(0))

    return re.sub(r"\$\{([^}]+)\}", repl, value).strip()


def _iter_effective_maven_plugins(root: ET.Element):
    def walk(node: ET.Element, *, under_plugin_management: bool = False):
        name = _local_name(node.tag)
        next_under_plugin_management = under_plugin_management or name == "pluginManagement"
        if name == "plugin" and not under_plugin_management:
            yield node
            return
        for child in list(node):
            yield from walk(child, under_plugin_management=next_under_plugin_management)

    yield from walk(root)


def infer_maven_surefire_argline(project_root: Path) -> str:
    """Infer the effective Maven Surefire argLine configured in pom.xml."""
    pom_path = Path(project_root) / "pom.xml"
    if not pom_path.exists():
        return ""
    try:
        root = ET.parse(pom_path).getroot()
    except ET.ParseError:
        return ""

    properties: dict[str, str] = {}
    for node in root.iter():
        if _local_name(node.tag) != "properties":
            continue
        for child in list(node):
            properties[_local_name(child.tag)] = (child.text or "").strip()

    for plugin in root.iter():
        if _local_name(plugin.tag) != "plugin":
            continue
        artifact_id = ""
        configuration = None
        for child in list(plugin):
            if _local_name(child.tag) == "artifactId":
                artifact_id = (child.text or "").strip()
            elif _local_name(child.tag) == "configuration":
                configuration = child
        if artifact_id != "maven-surefire-plugin" or configuration is None:
            continue
        for child in list(configuration):
            if _local_name(child.tag) == "argLine":
                return _resolve_embedded_properties((child.text or "").strip(), properties)

    for key in ("surefire.argLine", "surefire.argline", "argLine"):
        value = _resolve_embedded_properties(properties.get(key, ""), properties)
        if value:
            return value
    return ""


def maven_jacoco_prepare_agent_args(
    project_root: Path,
    jacoco_version: str | None = None,
    *,
    destfile: Path | None = None,
) -> tuple[str, str]:
    """Return a prepare-agent goal plus Surefire argLine override that preserves existing args."""
    jacoco_version = jacoco_version or DEFAULT_JACOCO_VERSION
    property_name = "tatg.jacocoArgLine"
    prepare_goal = (
        f"org.jacoco:jacoco-maven-plugin:{jacoco_version}:prepare-agent "
        f"-Djacoco.propertyName={property_name}"
    )
    agent_jar = (
        Path.home()
        / ".m2"
        / "repository"
        / "org"
        / "jacoco"
        / "org.jacoco.agent"
        / jacoco_version
        / f"org.jacoco.agent-{jacoco_version}-runtime.jar"
    )
    destfile = Path(destfile) if destfile is not None else Path(project_root) / "target" / "jacoco.exec"
    explicit_agent = f"-javaagent:{agent_jar}=destfile={destfile}"
    existing_argline = infer_maven_surefire_argline(Path(project_root))
    merged_argline = existing_argline.replace("@{argLine}", explicit_agent)
    merged_argline = merged_argline.replace("${argLine}", explicit_agent)
    if explicit_agent not in merged_argline:
        merged_argline = explicit_agent + (f" {merged_argline}" if merged_argline else "")
    quoted_argline = shlex.quote(merged_argline.strip())
    argline_property = (
        f"-DargLine={quoted_argline} -Dsurefire.argline={quoted_argline} "
        "-DforkCount=1 -DreuseForks=false"
    )
    return prepare_goal, argline_property


def _normalize_java_version(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    if value.startswith("1.") and len(value) > 2:
        return value.split(".", 1)[1]
    return value


def _infer_junit_family_from_version(version: str) -> str:
    version = (version or "").strip()
    if not version:
        return ""
    if version.startswith("4"):
        return "junit4"
    if version.startswith("5"):
        return "junit5"
    return ""


def _infer_maven_runtime_versions(pom_path: Path) -> tuple[str, str, str]:
    if not pom_path.exists():
        return "", "", ""
    try:
        root = ET.parse(pom_path).getroot()
    except ET.ParseError:
        return "", "", ""

    properties: dict[str, str] = {}
    for node in root.iter():
        if _local_name(node.tag) != "properties":
            continue
        for child in list(node):
            key = _local_name(child.tag)
            properties[key] = (child.text or "").strip()

    java_version = ""
    for key in (
        "maven.compiler.release",
        "maven.compiler.target",
        "maven.compiler.source",
    ):
        resolved = _resolve_property(properties.get(key, ""), properties)
        if resolved:
            java_version = _normalize_java_version(resolved)
            break
    if not java_version:
        for plugin in (node for node in root.iter() if _local_name(node.tag) == "plugin"):
            artifact_id = ""
            configuration = None
            for child in list(plugin):
                if _local_name(child.tag) == "artifactId":
                    artifact_id = (child.text or "").strip()
                elif _local_name(child.tag) == "configuration":
                    configuration = child
            if artifact_id != "maven-compiler-plugin" or configuration is None:
                continue
            config_values: dict[str, str] = {}
            for child in list(configuration):
                name = _local_name(child.tag)
                if name in {"release", "target", "source"}:
                    config_values[name] = _resolve_property((child.text or "").strip(), properties)
            for key in ("release", "target", "source"):
                if config_values.get(key):
                    java_version = _normalize_java_version(config_values[key])
                    break
            if java_version:
                break
    if not java_version:
        for key in ("java.version", "jdk.version"):
            resolved = _resolve_property(properties.get(key, ""), properties)
            if resolved:
                java_version = _normalize_java_version(resolved)
                break

    junit_version = ""
    junit_family = ""
    for dep in (node for node in root.iter() if _local_name(node.tag) == "dependency"):
        group_id = ""
        artifact_id = ""
        version = ""
        for child in list(dep):
            name = _local_name(child.tag)
            text = (child.text or "").strip()
            if name == "groupId":
                group_id = text
            elif name == "artifactId":
                artifact_id = text
            elif name == "version":
                version = _resolve_property(text, properties)

        if group_id == "junit" and artifact_id == "junit":
            junit_family = "junit4"
            junit_version = version or junit_version
        elif group_id.startswith("org.junit.jupiter") or "junit-jupiter" in artifact_id:
            junit_family = "junit5"
            junit_version = version or junit_version

        if junit_family == "junit5":
            break

    if not junit_family:
        junit_family = _infer_junit_family_from_version(junit_version)
    return java_version, junit_version, junit_family


def _infer_gradle_runtime_versions(build_file: Path) -> tuple[str, str, str]:
    if not build_file.exists():
        return "", "", ""
    text = build_file.read_text(encoding="utf-8", errors="ignore")

    java_patterns = [
        r"JavaLanguageVersion\.of\((\d+)\)",
        r"sourceCompatibility\s*=\s*['\"]?([0-9][0-9._]*)",
        r"targetCompatibility\s*=\s*['\"]?([0-9][0-9._]*)",
        r"JavaVersion\.VERSION_([0-9_]+)",
    ]
    java_version = ""
    for pattern in java_patterns:
        match = re.search(pattern, text)
        if match:
            java_version = _normalize_java_version(match.group(1).replace("_", "."))
            break

    junit_version = ""
    junit_family = ""

    m = re.search(r"org\.junit\.jupiter:[^:'\"]+:([0-9][^'\"\s)]*)", text)
    if m:
        junit_family = "junit5"
        junit_version = m.group(1)
    elif "useJUnitPlatform" in text:
        junit_family = "junit5"

    if not junit_family:
        m = re.search(r"junit:junit:([0-9][^'\"\s)]*)", text)
        if m:
            junit_family = "junit4"
            junit_version = m.group(1)
        elif "junit:junit" in text:
            junit_family = "junit4"

    if not junit_family:
        junit_family = _infer_junit_family_from_version(junit_version)
    return java_version, junit_version, junit_family


def infer_runtime_versions(
    project_root: Path,
    build_system: BuildSystem | None = None,
) -> tuple[str, str, str]:
    """Infer Java version, JUnit version and JUnit family from project config."""
    project_root = Path(project_root).resolve()
    if build_system is None:
        build_system = detect_build_system(project_root)

    if build_system == BuildSystem.MAVEN:
        return _infer_maven_runtime_versions(project_root / "pom.xml")
    if build_system == BuildSystem.GRADLE:
        gradle_file = project_root / "build.gradle"
        if not gradle_file.exists():
            gradle_file = project_root / "build.gradle.kts"
        return _infer_gradle_runtime_versions(gradle_file)
    return "", "", ""


def detect_build_system(project_root: Path) -> BuildSystem:
    """
    Auto-detect the build system used by the project.
    Checks for presence of build configuration files.
    """
    project_root = Path(project_root).resolve()

    if (project_root / "pom.xml").exists():
        return BuildSystem.MAVEN
    if (project_root / "build.gradle").exists() or (project_root / "build.gradle.kts").exists():
        return BuildSystem.GRADLE
    if (project_root / "WORKSPACE").exists() or (project_root / "BUILD").exists():
        return BuildSystem.BAZEL
    if (project_root / "setup.py").exists() or (project_root / "pyproject.toml").exists():
        return BuildSystem.PYTHON

    has_java_src = any((project_root / "src").rglob("*.java")) if (project_root / "src").exists() else False
    if has_java_src:
        return BuildSystem.DIRECT_JAVA

    return BuildSystem.UNKNOWN


#: Default plugin versions used when the project pom/gradle config does not
#: configure JaCoCo / PITest itself. These are full GAVs invoked via
#: ``mvn <plugin>:<goal>`` so the goals run regardless of pom <build> config.
DEFAULT_JACOCO_VERSION = "0.8.12"
DEFAULT_PITEST_MAVEN_VERSION = "1.17.0"
DEFAULT_PITEST_JUNIT5_VERSION = "1.2.1"

#: Maven -D flags that skip read-only code-quality plugins (license checks,
#: style, static analysis).  These are no-ops when the plugin is absent, so
#: they are always safe to pass.  Skipping them avoids build failures caused
#: by generated test files that violate style/license rules.
_MAVEN_QUALITY_SKIP = (
    "-Drat.skip=true "
    "-Dcheckstyle.skip=true "
    "-Dspotbugs.skip=true "
    "-Dpmd.skip=true "
    "-Dspotless.skip=true "
    "-Dspotless.check.skip=true "
    "-Dfmt.skip=true"
)


def _pom_has_plugin(pom_path: Path, group_id: str, artifact_id: str) -> bool:
    """Return True when ``pom.xml`` declares ``<plugin>`` with the given GA."""
    if not pom_path.exists():
        return False
    try:
        root = ET.parse(pom_path).getroot()
    except ET.ParseError:
        return False
    for plugin in _iter_effective_maven_plugins(root):
        g = ""
        a = ""
        for child in list(plugin):
            name = _local_name(child.tag)
            if name == "groupId":
                g = (child.text or "").strip()
            elif name == "artifactId":
                a = (child.text or "").strip()
        if a == artifact_id and (not group_id or g == group_id or g == ""):
            return True
    return False


def _pom_plugin_has_goal(pom_path: Path, group_id: str, artifact_id: str, goal: str) -> bool:
    if not pom_path.exists():
        return False
    try:
        root = ET.parse(pom_path).getroot()
    except ET.ParseError:
        return False
    for plugin in _iter_effective_maven_plugins(root):
        g = ""
        a = ""
        for child in list(plugin):
            name = _local_name(child.tag)
            if name == "groupId":
                g = (child.text or "").strip()
            elif name == "artifactId":
                a = (child.text or "").strip()
        if a != artifact_id or (group_id and g not in {group_id, ""}):
            continue
        for node in plugin.iter():
            if _local_name(node.tag) == "goal" and (node.text or "").strip() == goal:
                return True
    return False


def _pom_get_modules(pom_path: Path) -> list[str]:
    """Return the module directory names declared in ``<modules>`` of a pom.xml."""
    if not pom_path.exists():
        return []
    try:
        root = ET.parse(pom_path).getroot()
    except ET.ParseError:
        return []
    modules: list[str] = []
    for node in root.iter():
        if _local_name(node.tag) == "module":
            mod = (node.text or "").strip()
            if mod:
                modules.append(mod)
    return modules


def _pom_has_plugin_in_tree(
    project_root: Path,
    group_id: str,
    artifact_id: str,
) -> bool:
    """Return True when the plugin is configured in the root pom, any direct
    child module pom, or the immediate parent pom (one level up).

    The parent check matters for multi-module projects where the target module
    is nested below an aggregator/root pom that declares the plugin.
    """
    project_root = Path(project_root)
    root_pom = project_root / "pom.xml"
    if _pom_has_plugin(root_pom, group_id, artifact_id):
        return True
    for mod in _pom_get_modules(root_pom):
        if _pom_has_plugin(project_root / mod / "pom.xml", group_id, artifact_id):
            return True
    # Also check the immediate parent directory pom (one level up).
    parent_pom = project_root.parent / "pom.xml"
    if _pom_has_plugin(parent_pom, group_id, artifact_id):
        return True
    return False


def _pom_plugin_has_goal_in_tree(
    project_root: Path,
    group_id: str,
    artifact_id: str,
    goal: str,
) -> bool:
    project_root = Path(project_root)
    root_pom = project_root / "pom.xml"
    if _pom_plugin_has_goal(root_pom, group_id, artifact_id, goal):
        return True
    for mod in _pom_get_modules(root_pom):
        if _pom_plugin_has_goal(project_root / mod / "pom.xml", group_id, artifact_id, goal):
            return True
    parent_pom = project_root.parent / "pom.xml"
    if _pom_plugin_has_goal(parent_pom, group_id, artifact_id, goal):
        return True
    return False


def _find_module_root(project_root: Path, source_file: Path) -> Path:
    """Walk upward from ``source_file`` to find the nearest directory that
    contains a ``pom.xml`` and is a sub-path of ``project_root``.  Returns
    that directory so callers can locate the module-level JaCoCo report.
    Falls back to ``project_root`` when no sub-module pom is found."""
    project_root = Path(project_root).resolve()
    candidate = Path(source_file).resolve().parent
    while True:
        if candidate == project_root:
            break
        # Stop if we've walked above project_root
        try:
            candidate.relative_to(project_root)
        except ValueError:
            break
        if (candidate / "pom.xml").exists():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return project_root


def pom_has_jacoco_plugin(project_root: Path) -> bool:
    """True when the Maven project already configures jacoco-maven-plugin
    (checked in root pom and all declared module poms)."""
    return _pom_has_plugin_in_tree(Path(project_root), "org.jacoco", "jacoco-maven-plugin")


def pom_has_jacoco_prepare_agent_binding(project_root: Path) -> bool:
    """True when JaCoCo prepare-agent is bound in the project/module pom tree."""
    return _pom_plugin_has_goal_in_tree(
        Path(project_root),
        "org.jacoco",
        "jacoco-maven-plugin",
        "prepare-agent",
    )


def pom_has_pitest_plugin(project_root: Path) -> bool:
    """True when the Maven project already configures pitest-maven
    (checked in root pom and all declared module poms)."""
    return _pom_has_plugin_in_tree(Path(project_root), "org.pitest", "pitest-maven")


def infer_test_command(
    project_root: Path,
    build_system: BuildSystem | None = None,
    test_filter: str | None = None,
    with_mutation: bool = False,
) -> str:
    """
    Generate a test command for the detected/specified build system.

    Args:
        project_root: Root directory of the project
        build_system: Build system to use (auto-detected if None)
        test_filter: Optional test class/method filter (e.g., "TestClassNameTest" for Maven)
        with_mutation: When True (Maven/Gradle only), append PITest's
            ``mutationCoverage`` goal so a ``mutations.xml`` is produced
            under ``target/pit-reports/`` for downstream parsing.

    Returns:
        A test command string
    """
    if build_system is None:
        build_system = detect_build_system(project_root)

    if build_system == BuildSystem.MAVEN:
        # Coverage strategy:
        #   - If the pom already binds JaCoCo ``prepare-agent``, rely on that
        #     binding and only invoke ``:report`` explicitly (many poms bind
        #     ``report`` to ``verify``, which ``mvn test`` would skip).
        #   - Otherwise, inject ``prepare-agent`` and ``report`` by full GAV so
        #     coverage works even when JaCoCo is absent or only partly declared.
        #
        # Injecting our own ``prepare-agent`` when one is already configured
        # would attach TWO ``-javaagent:jacoco`` flags (different versions ⇒
        # JVM ``java.lang.instrument ASSERTION FAILED`` crash on the forked
        # Surefire VM).
        #
        # ``-Dmaven.test.failure.ignore=true`` keeps the build going past
        # Surefire failures so the downstream ``jacoco:report`` (and the
        # post-pass pitest) still execute.
        jc_ver = DEFAULT_JACOCO_VERSION
        jacoco_report = f"org.jacoco:jacoco-maven-plugin:{jc_ver}:report"
        if pom_has_jacoco_prepare_agent_binding(Path(project_root)):
            base = (
                f"mvn -q -Dmaven.test.failure.ignore=true {_MAVEN_QUALITY_SKIP} "
                f"clean test {jacoco_report}"
            )
        else:
            jacoco_prepare, jacoco_argline_property = maven_jacoco_prepare_agent_args(Path(project_root), jc_ver)
            base = (
                f"mvn -q -Dmaven.test.failure.ignore=true {_MAVEN_QUALITY_SKIP} "
                f"{jacoco_prepare} clean test {jacoco_argline_property} {jacoco_report}"
            )
        if test_filter:
            base += f" -Dtest={test_filter}"
        # ``with_mutation`` is retained as a parameter for API stability but the
        # PITest goal is now executed in a separate post-evaluation pass (see
        # ``infer_mutation_command``) so we can first prune failing tests —
        # PITest aborts if the test suite is not green.
        return base

    if build_system == BuildSystem.GRADLE:
        cmd = "gradle clean test jacocoTestReport"
        if test_filter:
            cmd = f"gradle clean test --tests {test_filter} jacocoTestReport"
        # `with_mutation` intentionally ignored — mutation is now a separate
        # post-evaluation step. Gradle requires the pitest plugin to be applied
        # in build.gradle; if absent, ``infer_mutation_command`` returns None.
        return cmd

    if build_system == BuildSystem.PYTHON:
        cmd = "pytest -q"
        if test_filter:
            cmd = f"pytest -q {test_filter}"
        return cmd

    if build_system == BuildSystem.DIRECT_JAVA:
        return "javac -d target/classes src/**/*.java && java -cp target/classes org.junit.runner.JUnitCore"

    if build_system == BuildSystem.BAZEL:
        return "bazel test //..."

    return "echo 'Unable to infer test command'"


def infer_mutation_command(
    project_root: Path,
    build_system: BuildSystem | None = None,
    target_classes: str | None = None,
    target_tests: str | None = None,
    junit_family: str | None = None,
    maven_project_args: list[str] | None = None,
    extra_maven_properties: dict[str, str | None] | None = None,
    maven_executable: str | None = None,
) -> str | None:
    """
    Return a standalone command that runs mutation analysis and writes
    ``mutations.xml`` so downstream parsers can compute per-method
    ``mutation_score``. Returns ``None`` when mutation analysis is not
    supported for the detected build system in its current configuration.

    For Maven, the command is always emitted: PITest is invoked by full GAV
    so it works regardless of whether the pom configures pitest-maven. When
    the pom does NOT configure pitest, sane defaults for ``targetClasses`` /
    ``targetTests`` are required — callers should pass them explicitly when
    the pom is unconfigured.

    For Gradle, mutation requires the ``info.solidsoft.pitest`` plugin to be
    applied — we cannot inject it from the CLI, so ``None`` is returned when
    no ``pitest`` task is configured.
    """
    if build_system is None:
        build_system = detect_build_system(project_root)

    if build_system == BuildSystem.MAVEN:
        ver = DEFAULT_PITEST_MAVEN_VERSION
        maven_launcher = shlex.quote(str(maven_executable or "mvn"))
        project_args = ""
        if maven_project_args:
            project_args = " " + " ".join(shlex.quote(str(arg)) for arg in maven_project_args if str(arg).strip())
        cmd = (
            f"{maven_launcher} -q -Dmaven.test.failure.ignore=true "
            f"{_MAVEN_QUALITY_SKIP} "
            f"{project_args.strip()} "
            "test-compile "
            f"org.pitest:pitest-maven:{ver}:mutationCoverage "
            "-DoutputFormats=XML,HTML "
            "-DtimestampedReports=false "
            "-DexportLineCoverage=true"
        )
        # CLI properties intentionally override or narrow POM defaults. This is
        # important for benchmark runs: final validation may use a generated-test
        # subset even when the project POM configures a broad PITest suite.
        if target_classes:
            cmd += f" -DtargetClasses={shlex.quote(str(target_classes))}"
        if target_tests:
            cmd += f" -DtargetTests={shlex.quote(str(target_tests))}"
        if extra_maven_properties:
            for key, value in sorted(extra_maven_properties.items()):
                key_text = str(key).strip()
                if not key_text:
                    continue
                if value is None:
                    cmd += f" -D{key_text}"
                else:
                    cmd += f" -D{key_text}={shlex.quote(str(value))}"
        has_explicit_plugin_deps = bool(
            extra_maven_properties and "pitest.plugin.dependencies" in extra_maven_properties
        )
        if not pom_has_pitest_plugin(Path(project_root)) and not has_explicit_plugin_deps:
            # JUnit5 requires the pitest-junit5-plugin dependency. We attach it
            # via -Ppitest- profile is not possible, so we rely on pom config
            # for junit5 projects; document the limitation in README.
            if junit_family == "junit5":
                cmd += (
                    " -Dpitest.plugin.dependencies=org.pitest:pitest-junit5-plugin:"
                    f"{DEFAULT_PITEST_JUNIT5_VERSION}"
                )
        return cmd

    if build_system == BuildSystem.GRADLE:
        gradle_file = Path(project_root) / "build.gradle"
        if not gradle_file.exists():
            gradle_file = Path(project_root) / "build.gradle.kts"
        if not gradle_file.exists():
            return None
        try:
            text = gradle_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        if "info.solidsoft.pitest" not in text and "pitest" not in text.lower():
            return None
        return "gradle pitest"

    return None


def infer_coverage_report_path(
    project_root: Path,
    build_system: BuildSystem | None = None,
    *,
    source_file: Path | None = None,
) -> str:
    """Infer the JaCoCo CSV report path for the project.

    For multi-module Maven projects, pass ``source_file`` (the Java source
    being tested) so the correct sub-module report is returned instead of the
    (non-existent) root-level report.
    """
    if build_system is None:
        build_system = detect_build_system(project_root)

    if build_system == BuildSystem.MAVEN:
        base = (
            _find_module_root(project_root, source_file)
            if source_file is not None
            else Path(project_root)
        )
        return str(base / "target" / "site" / "jacoco" / "jacoco.csv")

    if build_system == BuildSystem.GRADLE:
        return str(project_root / "build" / "jacoco" / "jacoco.csv")

    if build_system == BuildSystem.PYTHON:
        return str(project_root / ".coverage")

    if build_system == BuildSystem.DIRECT_JAVA:
        return str(project_root / "coverage" / "jacoco.csv")

    return ""
