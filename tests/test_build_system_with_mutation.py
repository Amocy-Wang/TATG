"""
Regression tests for the split-pass mutation evaluation model.

``infer_test_command`` no longer appends PITest — it always emits a
jacoco-via-full-GAV command that works regardless of pom plugin config.
The separate ``infer_mutation_command`` returns the standalone PITest
goal invoked in the post-evaluation pass (after failing tests are pruned).
"""

from pathlib import Path

from tatg.build_system import (
    BuildSystem,
    DEFAULT_JACOCO_VERSION,
    DEFAULT_PITEST_MAVEN_VERSION,
    _find_module_root,
    _pom_get_modules,
    _pom_has_plugin_in_tree,
    pom_has_jacoco_prepare_agent_binding,
    infer_coverage_report_path,
    infer_maven_surefire_argline,
    infer_mutation_command,
    infer_test_command,
    maven_jacoco_prepare_agent_args,
    pom_has_jacoco_plugin,
)


def test_maven_test_command_uses_full_gav_jacoco_and_omits_pitest(tmp_path: Path) -> None:
    # Empty project (no pom.xml ⇒ no jacoco plugin configured) — full-GAV
    # prepare-agent + report goals are injected.
    cmd = infer_test_command(tmp_path, BuildSystem.MAVEN)
    assert f"org.jacoco:jacoco-maven-plugin:{DEFAULT_JACOCO_VERSION}:prepare-agent" in cmd
    assert f"org.jacoco:jacoco-maven-plugin:{DEFAULT_JACOCO_VERSION}:report" in cmd
    assert "clean test" in cmd
    assert "-Dmaven.test.failure.ignore=true" in cmd
    assert "-Djacoco.propertyName=tatg.jacocoArgLine" in cmd
    assert "-DargLine=" in cmd
    assert "-Dsurefire.argline=" in cmd
    assert "-javaagent:" in cmd
    assert "org.jacoco.agent" in cmd
    assert "target/jacoco.exec" in cmd
    assert "-DforkCount=1" in cmd
    assert "-DreuseForks=false" in cmd
    assert "pitest" not in cmd


def test_infer_maven_surefire_argline_resolves_project_property(tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
                """<?xml version="1.0"?>
<project>
    <properties>
        <surefire.argline>@{argLine} -Duser.language=TR</surefire.argline>
    </properties>
    <build><plugins>
        <plugin>
            <artifactId>maven-surefire-plugin</artifactId>
            <configuration><argLine>${surefire.argline}</argLine></configuration>
        </plugin>
    </plugins></build>
</project>
""",
                encoding="utf-8",
        )

        assert infer_maven_surefire_argline(tmp_path) == "@{argLine} -Duser.language=TR"


def test_jacoco_prepare_agent_args_preserve_hardcoded_surefire_argline(tmp_path: Path) -> None:
    (tmp_path / "pom.xml").write_text(
        """<?xml version="1.0"?>
<project><build><plugins>
  <plugin>
    <artifactId>maven-surefire-plugin</artifactId>
    <configuration><argLine>--illegal-access=deny</argLine></configuration>
  </plugin>
</plugins></build></project>
""",
        encoding="utf-8",
    )

    prepare_goal, argline_property = maven_jacoco_prepare_agent_args(tmp_path, DEFAULT_JACOCO_VERSION)

    assert "prepare-agent" in prepare_goal
    assert "-Djacoco.propertyName=tatg.jacocoArgLine" in prepare_goal
    assert "-javaagent:" in argline_property
    assert "org.jacoco.agent" in argline_property
    assert "target/jacoco.exec" in argline_property
    assert "-DargLine=" in argline_property
    assert "-Dsurefire.argline=" in argline_property
    assert "-DforkCount=1" in argline_property
    assert "-DreuseForks=false" in argline_property
    assert "--illegal-access=deny" in argline_property


def test_maven_test_command_skips_prepare_agent_when_pom_binds_jacoco_prepare_agent(
    tmp_path: Path,
) -> None:
    # When the pom already binds jacoco's ``prepare-agent`` execution,
    # injecting another ``-javaagent:jacoco`` of a different version causes
    # the forked Surefire JVM to crash with ``java.lang.instrument
    # ASSERTION FAILED``. The full-GAV ``prepare-agent`` goal must NOT be
    # appended in that case — only the explicit ``:report`` is safe.
    (tmp_path / "pom.xml").write_text(
        """<?xml version="1.0"?>
<project><build><plugins>
  <plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <version>0.8.7</version>
    <executions><execution><goals><goal>prepare-agent</goal></goals></execution></executions>
  </plugin>
</plugins></build></project>
""",
        encoding="utf-8",
    )
    assert pom_has_jacoco_prepare_agent_binding(tmp_path) is True
    cmd = infer_test_command(tmp_path, BuildSystem.MAVEN)
    assert "prepare-agent" not in cmd
    assert f"org.jacoco:jacoco-maven-plugin:{DEFAULT_JACOCO_VERSION}:report" in cmd


def test_maven_test_command_injects_prepare_agent_when_jacoco_plugin_has_no_binding(
    tmp_path: Path,
) -> None:
    (tmp_path / "pom.xml").write_text(
        """<?xml version="1.0"?>
<project><build><plugins>
  <plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <version>0.8.7</version>
  </plugin>
</plugins></build></project>
""",
        encoding="utf-8",
    )
    assert pom_has_jacoco_plugin(tmp_path) is True
    assert pom_has_jacoco_prepare_agent_binding(tmp_path) is False
    cmd = infer_test_command(tmp_path, BuildSystem.MAVEN)
    assert f"org.jacoco:jacoco-maven-plugin:{DEFAULT_JACOCO_VERSION}:prepare-agent" in cmd
    assert f"org.jacoco:jacoco-maven-plugin:{DEFAULT_JACOCO_VERSION}:report" in cmd


def test_maven_test_command_ignores_plugin_management_prepare_agent(
    tmp_path: Path,
) -> None:
    (tmp_path / "pom.xml").write_text(
        """<?xml version="1.0"?>
<project><build><pluginManagement><plugins>
  <plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <executions><execution><goals><goal>prepare-agent</goal></goals></execution></executions>
  </plugin>
</plugins></pluginManagement></build></project>
""",
        encoding="utf-8",
    )
    assert pom_has_jacoco_plugin(tmp_path) is False
    assert pom_has_jacoco_prepare_agent_binding(tmp_path) is False
    cmd = infer_test_command(tmp_path, BuildSystem.MAVEN)
    assert f"org.jacoco:jacoco-maven-plugin:{DEFAULT_JACOCO_VERSION}:prepare-agent" in cmd


def test_maven_test_command_with_mutation_flag_still_omits_pitest() -> None:
    cmd = infer_test_command(Path("."), BuildSystem.MAVEN, with_mutation=True)
    assert "pitest" not in cmd


def test_maven_test_command_with_test_filter() -> None:
    cmd = infer_test_command(Path("."), BuildSystem.MAVEN, test_filter="FooTest")
    assert "-Dtest=FooTest" in cmd


def test_gradle_test_command_uses_jacoco_report_task_and_omits_pitest() -> None:
    cmd = infer_test_command(Path("."), BuildSystem.GRADLE)
    assert "jacocoTestReport" in cmd
    assert "pitest" not in cmd


def test_infer_mutation_command_for_maven_emits_full_gav_pitest(tmp_path: Path) -> None:
    cmd = infer_mutation_command(
        tmp_path,
        BuildSystem.MAVEN,
        target_classes="com.example.*",
        target_tests="com.example.*",
    )
    assert cmd is not None
    assert f"org.pitest:pitest-maven:{DEFAULT_PITEST_MAVEN_VERSION}:mutationCoverage" in cmd
    assert "-DoutputFormats=XML,HTML" in cmd
    assert "-DexportLineCoverage=true" in cmd
    assert "-Dspotless.check.skip=true" in cmd
    assert "-Dfmt.skip=true" in cmd
    assert "-DtargetClasses='com.example.*'" in cmd
    assert "-DtargetTests='com.example.*'" in cmd


def test_infer_mutation_command_reuses_absolute_maven_launcher(tmp_path: Path) -> None:
    cmd = infer_mutation_command(
        tmp_path,
        BuildSystem.MAVEN,
        target_classes="com.example.Foo",
        maven_executable="/opt/tooling/bin/mvn",
    )

    assert cmd is not None
    assert cmd.startswith("/opt/tooling/bin/mvn -q ")


def test_infer_mutation_command_uses_pom_config_when_available(tmp_path: Path) -> None:
    pom = tmp_path / "pom.xml"
    pom.write_text(
        """<?xml version="1.0"?>
<project>
  <build><plugins>
    <plugin>
      <groupId>org.pitest</groupId>
      <artifactId>pitest-maven</artifactId>
      <version>1.17.0</version>
    </plugin>
  </plugins></build>
</project>
""",
        encoding="utf-8",
    )
    cmd = infer_mutation_command(
        tmp_path,
        BuildSystem.MAVEN,
        target_classes="ignored",
        target_tests="ignored",
    )
    assert cmd is not None
    assert "-DtargetClasses=ignored" in cmd
    assert "-DtargetTests=ignored" in cmd


def test_infer_mutation_command_for_junit5_attaches_pitest_junit5_plugin(
    tmp_path: Path,
) -> None:
    cmd = infer_mutation_command(
        tmp_path,
        BuildSystem.MAVEN,
        target_classes="x.*",
        target_tests="x.*",
        junit_family="junit5",
    )
    assert cmd is not None
    assert "pitest-junit5-plugin" in cmd


def test_infer_mutation_command_returns_none_for_gradle_without_pitest(
    tmp_path: Path,
) -> None:
    (tmp_path / "build.gradle").write_text(
        "plugins { id 'java' }\n", encoding="utf-8"
    )
    cmd = infer_mutation_command(tmp_path, BuildSystem.GRADLE)
    assert cmd is None


def test_infer_mutation_command_returns_gradle_command_when_pitest_applied(
    tmp_path: Path,
) -> None:
    (tmp_path / "build.gradle").write_text(
        "plugins { id 'info.solidsoft.pitest' version '1.15.0' }\n",
        encoding="utf-8",
    )
    cmd = infer_mutation_command(tmp_path, BuildSystem.GRADLE)
    assert cmd == "gradle pitest"


# ---------------------------------------------------------------------------
# Fix 1 — Quality-check skip flags
# ---------------------------------------------------------------------------

class TestQualitySkipFlags:
    def test_skip_flags_present_when_no_jacoco(self, tmp_path: Path) -> None:
        cmd = infer_test_command(tmp_path, BuildSystem.MAVEN)
        assert "-Drat.skip=true" in cmd
        assert "-Dcheckstyle.skip=true" in cmd
        assert "-Dspotbugs.skip=true" in cmd
        assert "-Dpmd.skip=true" in cmd

    def test_skip_flags_present_when_jacoco_configured(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            """<?xml version="1.0"?>
<project><build><plugins>
  <plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
  </plugin>
</plugins></build></project>
""",
            encoding="utf-8",
        )
        cmd = infer_test_command(tmp_path, BuildSystem.MAVEN)
        assert "-Drat.skip=true" in cmd
        assert "-Dcheckstyle.skip=true" in cmd

    def test_gradle_command_has_no_maven_skip_flags(self) -> None:
        cmd = infer_test_command(Path("."), BuildSystem.GRADLE)
        assert "-Drat.skip" not in cmd
        assert "-Dcheckstyle.skip" not in cmd

    def test_mutation_command_includes_quality_skip_flags(self, tmp_path: Path) -> None:
        cmd = infer_mutation_command(tmp_path, BuildSystem.MAVEN)
        assert cmd is not None
        assert "-Drat.skip=true" in cmd
        assert "-Dcheckstyle.skip=true" in cmd
        assert "-Dspotbugs.skip=true" in cmd
        assert "-Dpmd.skip=true" in cmd


# ---------------------------------------------------------------------------
# Fix 2 — Multi-module plugin detection
# ---------------------------------------------------------------------------

class TestMultiModulePluginDetection:
    def _root_pom(self, project_root: Path, modules: list[str]) -> None:
        mods_xml = "\n".join(f"    <module>{m}</module>" for m in modules)
        (project_root / "pom.xml").write_text(
            f"""<?xml version="1.0"?>
<project>
  <modules>
{mods_xml}
  </modules>
</project>
""",
            encoding="utf-8",
        )

    def test_pom_get_modules_returns_declared_modules(self, tmp_path: Path) -> None:
        self._root_pom(tmp_path, ["core", "api"])
        result = _pom_get_modules(tmp_path / "pom.xml")
        assert result == ["core", "api"]

    def test_pom_get_modules_empty_for_single_module(self, tmp_path: Path) -> None:
        (tmp_path / "pom.xml").write_text(
            "<project><build/></project>", encoding="utf-8"
        )
        assert _pom_get_modules(tmp_path / "pom.xml") == []

    def test_plugin_found_in_child_module_pom(self, tmp_path: Path) -> None:
        self._root_pom(tmp_path, ["core"])
        core = tmp_path / "core"
        core.mkdir()
        (core / "pom.xml").write_text(
            """<?xml version="1.0"?>
<project><build><plugins>
  <plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
  </plugin>
</plugins></build></project>
""",
            encoding="utf-8",
        )
        assert _pom_has_plugin_in_tree(tmp_path, "org.jacoco", "jacoco-maven-plugin") is True

    def test_no_double_injection_when_jacoco_prepare_agent_in_child_module(
        self, tmp_path: Path
    ) -> None:
        """A child-module prepare-agent binding prevents double-agent injection."""
        self._root_pom(tmp_path, ["core"])
        core = tmp_path / "core"
        core.mkdir()
        (core / "pom.xml").write_text(
            """<?xml version="1.0"?>
<project><build><plugins>
  <plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <executions><execution><goals><goal>prepare-agent</goal></goals></execution></executions>
  </plugin>
</plugins></build></project>
""",
            encoding="utf-8",
        )
        assert pom_has_jacoco_plugin(tmp_path) is True
        assert pom_has_jacoco_prepare_agent_binding(tmp_path) is True
        cmd = infer_test_command(tmp_path, BuildSystem.MAVEN)
        assert "prepare-agent" not in cmd

    def test_plugin_not_found_returns_false(self, tmp_path: Path) -> None:
        self._root_pom(tmp_path, ["core"])
        core = tmp_path / "core"
        core.mkdir()
        (core / "pom.xml").write_text(
            "<project><build/></project>", encoding="utf-8"
        )
        assert _pom_has_plugin_in_tree(tmp_path, "org.jacoco", "jacoco-maven-plugin") is False

    def test_prepare_agent_binding_found_in_parent_pom_above_project_root(
        self, tmp_path: Path
    ) -> None:
        """prepare-agent binding in the immediate parent directory pom is detected.

        This covers multi-module layouts where the root pom declares JaCoCo,
        but the target project_root is a child module.
        """
        # parent pom (tmp_path) has JaCoCo
        (tmp_path / "pom.xml").write_text(
            """<?xml version="1.0"?>
<project><build><plugins>
  <plugin>
    <groupId>org.jacoco</groupId>
    <artifactId>jacoco-maven-plugin</artifactId>
    <executions><execution><goals><goal>prepare-agent</goal></goals></execution></executions>
  </plugin>
</plugins></build></project>
""",
            encoding="utf-8",
        )
        # submodule has no JaCoCo of its own
        submod = tmp_path / "core-module"
        submod.mkdir()
        (submod / "pom.xml").write_text(
            "<project><build/></project>", encoding="utf-8"
        )
        # detecting from the submodule should still find it via parent lookup
        assert _pom_has_plugin_in_tree(submod, "org.jacoco", "jacoco-maven-plugin") is True
        # and pom_has_jacoco_plugin must agree (no injection = no double-agent)
        assert pom_has_jacoco_plugin(submod) is True
        assert pom_has_jacoco_prepare_agent_binding(submod) is True


# ---------------------------------------------------------------------------
# Fix 3 — Module-aware coverage report path
# ---------------------------------------------------------------------------

class TestModuleAwareCoverageReportPath:
    def test_single_module_returns_root_target_path(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java" / "Foo.java"
        src.parent.mkdir(parents=True)
        src.touch()
        path = infer_coverage_report_path(tmp_path, BuildSystem.MAVEN, source_file=src)
        assert path == str(tmp_path / "target" / "site" / "jacoco" / "jacoco.csv")

    def test_multi_module_returns_submodule_target_path(self, tmp_path: Path) -> None:
        # root pom declares a child module
        (tmp_path / "pom.xml").write_text(
            "<project><modules><module>core</module></modules></project>",
            encoding="utf-8",
        )
        core = tmp_path / "core"
        core.mkdir()
        (core / "pom.xml").write_text("<project/>", encoding="utf-8")
        src = core / "src" / "main" / "java" / "Foo.java"
        src.parent.mkdir(parents=True)
        src.touch()
        path = infer_coverage_report_path(tmp_path, BuildSystem.MAVEN, source_file=src)
        assert path == str(core / "target" / "site" / "jacoco" / "jacoco.csv")

    def test_no_source_file_falls_back_to_root(self, tmp_path: Path) -> None:
        path = infer_coverage_report_path(tmp_path, BuildSystem.MAVEN)
        assert path == str(tmp_path / "target" / "site" / "jacoco" / "jacoco.csv")

    def test_find_module_root_stops_at_project_root(self, tmp_path: Path) -> None:
        src = tmp_path / "src" / "main" / "java" / "Foo.java"
        src.parent.mkdir(parents=True)
        src.touch()
        result = _find_module_root(tmp_path, src)
        assert result == tmp_path

    def test_find_module_root_finds_nearest_pom(self, tmp_path: Path) -> None:
        sub = tmp_path / "core"
        sub.mkdir()
        (sub / "pom.xml").write_text("<project/>", encoding="utf-8")
        deep = sub / "src" / "main" / "java"
        deep.mkdir(parents=True)
        src = deep / "Foo.java"
        src.touch()
        result = _find_module_root(tmp_path, src)
        assert result == sub
