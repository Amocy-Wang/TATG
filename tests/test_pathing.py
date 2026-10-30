from pathlib import Path

import os

from tatg.pathing import (
    augment_executable_path,
    derive_java_test_file,
    java_tool_candidates,
)


def test_derive_java_test_file_absolute_main_path():
    src = Path("/repo/module/src/main/java/com/acme/Foo.java")
    out = derive_java_test_file(src)
    assert out == Path("/repo/module/src/test/java/com/acme/FooTest.java")


def test_derive_java_test_file_relative_main_path():
    src = Path("src/main/java/com/acme/Foo.java")
    out = derive_java_test_file(src)
    assert out == Path("src/test/java/com/acme/FooTest.java")


def test_derive_java_test_file_fallback_same_dir():
    src = Path("lib/com/acme/Foo.java")
    out = derive_java_test_file(src)
    assert out == Path("lib/com/acme/FooTest.java")


def test_augment_executable_path_uses_only_configured_directories(tmp_path):
    configured = tmp_path / "tools"
    configured.mkdir()
    env = {
        "PATH": os.pathsep.join(("system-bin",)),
        "TATG_EXTRA_PATH": str(configured),
    }

    augment_executable_path(env)

    assert env["PATH"].split(os.pathsep) == ["system-bin", str(configured)]


def test_augment_executable_path_has_no_implicit_host_directories():
    env = {"PATH": os.pathsep.join(("system-bin",))}

    augment_executable_path(env, configured_paths="")

    assert env["PATH"] == "system-bin"


def test_java_tool_candidates_prefers_configured_java_home(tmp_path):
    java_home = tmp_path / "jdk"
    javap = java_home / "bin" / "javap"
    javap.parent.mkdir(parents=True)
    javap.write_text("#!/bin/sh\n", encoding="utf-8")
    javap.chmod(0o755)

    assert java_tool_candidates(
        "javap",
        env={"JAVA_HOME": str(java_home), "PATH": ""},
    ) == [str(javap)]


def test_java_tool_candidates_uses_versioned_home_without_host_defaults(tmp_path):
    java_home = tmp_path / "jdk17"
    javap = java_home / "bin" / "javap"
    javap.parent.mkdir(parents=True)
    javap.write_text("#!/bin/sh\n", encoding="utf-8")
    javap.chmod(0o755)

    assert java_tool_candidates(
        "javap",
        env={"JAVA_HOME_17": str(java_home), "PATH": ""},
    ) == [str(javap)]
