#!/usr/bin/env python3
"""Run source-checkout TATG batches for all curated hard methods.

Covers two project groups:
  - defects4j subjects (SUBJECTS_ROOT)
  - new projects: jquick-curl, adk-java (NEW_SUBJECTS_ROOT)

Default mode runs the full HARD_METHODS list (23 methods) in method scope.
Use --methods to run a subset, or --list-hard-methods to enumerate selectors.

Usage
-----
# Run all hard methods with TATG (default):
python scripts/run_all_hard_methods_batch.py

# Subset by method, class, or class.method (comma-separated):
python scripts/run_all_hard_methods_batch.py --methods writeString,remove
python scripts/run_all_hard_methods_batch.py --methods FunctionCallingUtils
python scripts/run_all_hard_methods_batch.py --methods ToXmlGenerator.writeString,BaseAgent.findAgent

# Dry-run (print commands only):
TATG_DRY_RUN=1 python scripts/run_all_hard_methods_batch.py --methods writeString
"""

from __future__ import annotations

import argparse
import csv
import difflib
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

class _PlainProgress:
    """Small tqdm-compatible wrapper for dependency-light batch runs."""

    def __init__(self, iterable):
        self._iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def __iter__(self):
        return iter(self._iterable)

    def set_postfix_str(self, _value, **_kwargs):
        return None

    def write(self, value):
        print(value)


try:
    from tqdm import tqdm as _tqdm
except ImportError:  # graceful fallback: tqdm unavailable

    def _tqdm(iterable, **_kwargs):  # type: ignore[misc]
        return _PlainProgress(iterable)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tatg.api_config import (
    DEFAULT_API_KEY_ENV,
    apply_api_config_to_env,
    preflight_api_config,
    resolve_api_config,
)
from tatg.build_system import (
    DEFAULT_JACOCO_VERSION,
    detect_build_system,
    infer_runtime_versions,
    maven_jacoco_prepare_agent_args,
    pom_has_jacoco_plugin,
)
from tatg.pathing import augment_executable_path
from tatg.lmstudio import local_lmstudio_timeout_seconds
from tatg.targeting import (
    JAVA_TYPE_DECL_RE,
    _extract_classes,
    _extract_methods as _targeting_extract_methods,
    _extract_type_body as _targeting_extract_type_body,
    _strip_nested_type_bodies as _targeting_strip_nested_type_bodies,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_four_class_source_batch import (
    _backup_java_files,
    _copy_changed_java_files,
    _java_restore_mismatches,
    _json_outcome,
    _post_run_evaluate_and_merge,
    _repo_root,
    _restore_java_files,
    _run_command_with_timestamped_log,
    _safe_read_json,
)


# ──────────────────────────────────────────────────────────────────────────────
# Project roots
# ──────────────────────────────────────────────────────────────────────────────

SUBJECTS_ROOT = Path(
    os.environ.get("TATG_SUBJECTS_ROOT", "/path/to/defects4j-subjects-notests")
)
NEW_SUBJECTS_ROOT = Path(
    os.environ.get("TATG_NEW_SUBJECTS_ROOT", "/path/to/new_projects")
)

OUT_DIR = Path(
    os.environ.get(
        "TATG_HARD_METHOD_OUT_DIR",
        str(Path(tempfile.gettempdir()) / "tatg-all-hard-methods"),
    )
)
PYTHON = os.environ.get("TATG_PYTHON", sys.executable)

# Some older defects4j projects (JacksonXml-5f, Jsoup-93f) configure
# maven-compiler-plugin with <source>1.7</source> / <target>1.7</target> in
# their plugin execution blocks. Maven 3.9 + Java 17+ (including Java 24)
# dropped support for source/target 7, so compilation fails. Setting JAVA_HOME
# to Java 11 (which still supports source=7) fixes this for those projects.
JAVA11_HOME = os.environ.get("TATG_JAVA11_HOME") or None
# A recent JDK is needed for projects that compile with Java 21.
# Do not treat an arbitrary inherited JAVA_HOME as a recent JDK.  The batch
# resolves a compatible runtime from the target release before every method.
JAVA24_HOME = os.environ.get("TATG_JAVA24_HOME") or None

API_BASE = os.environ.get("TATG_API_BASE", os.environ.get("API_BASE", ""))
MODEL = os.environ.get("TATG_MODEL", os.environ.get("MODEL", ""))
API_KEY_ENV = os.environ.get("TATG_API_KEY_ENV", DEFAULT_API_KEY_ENV)
MAX_ITERATIONS = int(os.environ.get("TATG_MAX_ITERATIONS", "3") or "3")
REPAIR_ITERATIONS = int(os.environ.get("TATG_REPAIR_ITERATIONS", "2") or "2")
STREAMING_REPAIR_MODE = os.environ.get("TATG_STREAMING_REPAIR_MODE", "deferred")
LLM_TIMEOUT_SECONDS = int(os.environ.get("TATG_LLM_TIMEOUT_SECONDS", "180") or "180")
TATG_PREANALYSIS_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_PREANALYSIS_TIMEOUT_SECONDS", "600") or "600"
)
TATG_GENERATION_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_GENERATION_TIMEOUT_SECONDS", "240") or "240"
)
TATG_BRANCH_ANALYSIS_TIMEOUT_SECONDS = int(
    os.environ.get("TATG_BRANCH_ANALYSIS_TIMEOUT_SECONDS", "600") or "600"
)


def _augment_tool_path(env: dict[str, str]) -> None:
    """Add only caller-configured tool directories to a child environment."""

    augment_executable_path(env)


STATIC_ANALYSIS_DEFAULT_ENV = {
    # Bounded SootUp/CAT preanalysis defaults. These are applied with setdefault
    # in main(), so callers can override any knob from the shell.
    "SOOTUP_CG_MAX_DEPTH": "3",
    "SOOTUP_CG_START_DEPTH": "2",
    "TATG_STATIC_ANALYSIS_SOOTUP_BUDGET_SECONDS": "45",
    "SOOTUP_CG_TIMEOUT_SECONDS": "20",
    "SOOTUP_CG_MAX_PATHS": "16",
    "SOOTUP_CG_MAX_SEARCH_STATES": "30000",
    "SOOTUP_CG_THREADS": "1",
    "TATG_STATIC_ANALYSIS_PROMPT_PATHS": "6",
    "TATG_STATIC_ANALYSIS_PROMPT_INIT_TYPES": "12",
    "TATG_STATIC_ANALYSIS_PROMPT_CTORS_PER_TYPE": "3",
    "TATG_STATIC_ANALYSIS_PROMPT_FACTORIES_PER_TYPE": "3",
    "TATG_STATIC_ANALYSIS_PROMPT_IMPLS_PER_TYPE": "3",
}
if JAVA24_HOME:
    STATIC_ANALYSIS_DEFAULT_ENV["SOOTUP_JAVA_HOME"] = JAVA24_HOME
E2E_TIMEOUT_SECONDS = int(os.environ.get("TATG_E2E_TIMEOUT_SECONDS", "0") or "0")
TEST_TIMEOUT_SECONDS = int(os.environ.get("TATG_TEST_TIMEOUT_SECONDS", "900") or "900")
GENERATION_PARALLELISM = int(os.environ.get("TATG_GENERATION_PARALLELISM", "1") or "1")
REPAIR_PARALLELISM = int(os.environ.get("TATG_REPAIR_PARALLELISM", "1") or "1")
VALIDATION_PARALLELISM = int(os.environ.get("TATG_VALIDATION_PARALLELISM", "4") or "4")
WITH_MUTATION = os.environ.get("TATG_WITH_MUTATION", "0") not in {"0", "false", "False", "no", "NO", "off", "OFF"}
FORCE = False
DRY_RUN = os.environ.get("TATG_DRY_RUN", "0") == "1"
POST_RUN_EVALUATE = os.environ.get("TATG_POST_RUN_EVALUATE", "1") != "0"
SHOW_PROGRESS = os.environ.get("TATG_SHOW_PROGRESS", "1") != "0"
BACKEND = os.environ.get("TATG_BACKEND", "tatg").strip().lower() or "tatg"
BACKEND_CHOICES = ("tatg",)


# ──────────────────────────────────────────────────────────────────────────────
# Unified target type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HardMethodTarget:
    name: str              # unique key (used for output filenames / HARD_METHODS lookup)
    subjects_root: Path    # absolute root directory for this project group
    project: str           # path relative to subjects_root (may include module subdir)
    target_file: str       # source file path relative to the project (module) directory
    extra_mvn_flags: str = ""  # injected between "-pl ." and "-Dtest=..." in test command
    # Existing test files (relative to project root) that conflict with the generated
    # test class name and must be removed after backup so TATG's preflight passes.
    # _restore_java_files will put them back automatically from the backup.
    conflicting_tests: tuple[str, ...] = ()
    # Override JAVA_HOME for the TATG subprocess. Set this for projects whose
    # maven-compiler-plugin hardcodes <source>/<target> at a level that Java 17+
    # no longer supports (e.g. source=1.7 requires Java 11 or earlier).
    java_home: str | None = None
    # Java release passed to TATG. Keep this explicit when java_home is
    # overridden so Maven compiler and Surefire runtime use a consistent JDK.
    java_version: str | None = None
    # Maven goals/phases used in the test command. Default uses full lifecycle
    # `test`; some projects need a narrower path to skip problematic plugins.
    test_goals: str = "test"
    # JaCoCo handling for the generated Maven test command:
    #   auto   -> detect project-local JaCoCo and inject only when absent
    #   pom    -> trust an inherited/project JaCoCo prepare-agent binding
    #   inject -> force explicit prepare-agent injection
    jacoco_mode: str = "auto"
    # Whether to retry with class scope when method-target resolution fails.
    class_scope_fallback: bool = True
    # Override the Java class simple name passed as --class-name (and used for the
    # -Dtest=<ClassName>Test Maven selector). When empty, `name` is used. Set this
    # when `name` must differ from the actual class name to avoid dict-key collisions
    # (e.g. "Flat3MapNew" → actual class "Flat3Map").
    class_name: str = ""
    # Optional exact JVM descriptors for overloaded hard methods. Reporting and
    # triage use these descriptors instead of picking the best same-name overload,
    # which can hide failures on private helper overloads behind covered public
    # wrappers.
    target_descriptors: dict[str, str] = field(default_factory=dict)

    @property
    def java_class_name(self) -> str:
        """The actual Java class name to pass to TATG and Maven."""
        return self.class_name or self.name


@dataclass(frozen=True)
class HardMethodRunTask:
    """A single executable hard-method task.

    ``method_name`` remains the Java source method name passed to TATG.
    ``target_descriptor`` optionally narrows overloaded same-name methods to a
    JVM descriptor.  Descriptor-expanded tasks use ``artifact_method_name`` so
    sibling overloads do not overwrite each other's JSON/log/prompt artifacts.
    """

    class_name: str
    target: HardMethodTarget
    method_name: str
    target_descriptor: str | None = None

    @property
    def artifact_method_name(self) -> str:
        if not self.target_descriptor:
            return self.method_name
        short = _descriptor_artifact_suffix(self.target_descriptor)
        return f"{self.method_name}__{short or 'descriptor'}"

    @property
    def overload_short(self) -> str:
        return _jvm_desc_to_short(str(self.target_descriptor or ""))

    @property
    def label(self) -> str:
        if not self.target_descriptor:
            return f"{self.class_name}.{self.method_name}"
        return f"{self.class_name}.{self.method_name}({_jvm_desc_to_short(self.target_descriptor)})"


# Backward-compatible public name used by archive/replay tooling.
MethodOverloadTarget = HardMethodRunTask


# ──────────────────────────────────────────────────────────────────────────────
# Hard method registry (all 23 targets)
# ──────────────────────────────────────────────────────────────────────────────

# Maps class name → tuple of method names to test in method scope.
HARD_METHODS: dict[str, tuple[str, ...]] = {
    # ── defects4j ────────────────────────────────────────────────────────────
    "Rotation":                   ("getAngles",),
    "GaussNewtonOptimizer":       ("doOptimize",),
    "FromXmlParser":              ("nextTextValue",),
    "ToXmlGenerator":             ("writeString",),
    "Metaphone":                  ("metaphone",),
    "NumberUtils":                ("createNumber",),
    "FilteringParserDelegate":    ("nextToken",),
    "Flat3Map":                   ("remove",),
    "DateTimeFormat":             ("parsePatternTo",),
    "Entities":                   ("escape",),

    # ── jquick-curl ──────────────────────────────────────────────────────────
    "JQuickCurlCommonVisitor":    ("visitCurlCommand", "visitOption"),
    "JCurlCommandProcessor":      ("resolveCurlCommand",),

    # ── adk-java / core ──────────────────────────────────────────────────────
    "FunctionCallingUtils":       ("buildFunctionDeclaration", "buildSchemaFromType"),
    "SlidingWindowEventCompactor":("getCompactionEvents",),
    "SchemaUtils":                ("validateMapOnSchema", "validateOutputSchema"),
    "GeminiUtil":                 ("sanitizeRequestForGeminiApi", "stripThoughts"),
    "ChatCompletionsRequest":     ("fromLlmRequest",),
    "Contents":                   ("rearrangeEventsForLatestFunctionResponse",),
    "BaseAgent":                  ("findAgent",),

    # ── batch-processing-gateway ─────────────────────────────────────────────
    "SubmissionSummary":          ("copyFrom",),
    "ApplicationSubmissionHelper":("getDriverSpec", "getExecutorSpec", "getSparkConf", "populateEnv"),
    "SparkPodSpec":               ("copyFrom",),

    # ── commons-cli ──────────────────────────────────────────────────────────
    "HelpFormatter":              ("renderOptions",),
    "PatternOptionBuilder":       ("getValueType",),

    # ── commons-codec (Java 21; requires JAVA24_HOME) ────────────────────────
    "Base32":                     ("decode", "encode"),
    "Base64":                     ("decode", "encode"),
    "Md5Crypt":                   ("md5Crypt",),
    "MurmurHash3":                ("hash128x64Internal",),
    "Sha2Crypt":                  ("sha2Crypt",),
    "ColognePhonetic":            ("colognePhonetic",),
    "DaitchMokotoffSoundex":      ("soundex",),
    "DoubleMetaphone":            ("doubleMetaphone", "handleC", "handleG", "handleS"),
    "Nysiis":                     ("nysiis", "transcodeRemaining"),
    "Rule":                       ("pattern",),
    "QuotedPrintableCodec":       ("encodeQuotedPrintable",),

    # ── commons-collections ──────────────────────────────────────────────────
    "IteratorUtils":              ("getIterator",),
    "TreeBidiMap":                ("doRedBlackDelete", "swapPosition"),
    "Flat3MapNew":                ("containsKey", "containsValue", "equals", "get", "put", "remove"),
    "SequencesComparator":        ("getMiddleSnake",),
    "AbstractPatriciaTrie":       ("higherEntry", "nextEntryImpl", "put", "removeInternalEntry"),

    # ── commons-csv ──────────────────────────────────────────────────────────
    "CSVFormat":                  ("printWithQuotes", "toString"),
    "CSVParser":                  ("createHeaders", "nextRecord"),
    "Lexer":                      ("nextToken", "parseEncapsulatedToken"),

    # ── datafaker ────────────────────────────────────────────────────────────
    "FakeValues":                 ("toJavaNames",),
    "FakeValuesService":          ("bothify", "javaNameToYamlName", "resolveExpression"),
    "SqlTransformer":             ("handlePrimitivesInArray",),

    # ── gson / gson sub-module ───────────────────────────────────────────────
    "$Gson$Types":                ("equals", "getGenericSupertype", "resolve"),
    "ConstructorConstructor":     ("newDefaultImplementationConstructor",),
    "Excluder":                   ("excludeField",),
    "LinkedTreeMap":              ("find", "rebalance"),
    "Primitives":                 ("unwrap", "wrap"),
    "JsonTreeReader":             ("peek",),
    "ObjectTypeAdapter":          ("read",),
    "ReflectiveTypeAdapterFactory":("getBoundFields",),
    "TypeToken":                  ("isAssignableFrom",),
    "JsonReader":                 ("doPeek", "nextNonWhitespace", "nextUnquotedValue",
                                   "peek", "peekNumber", "readEscapeCharacter", "skipValue"),

    # ── jdom2 ────────────────────────────────────────────────────────────────
    "Element":                    ("getNamespace", "getNamespacesInScope"),
    "StringBin":                  ("locate",),
    "Verifier":                   ("isXMLExtender", "isXMLPublicIDCharacter"),
    "StAXStreamBuilder":          ("processPrunableElement",),
    "SAXHandler":                 ("startElement",),
    "DTDParser":                  ("formatInternal",),
    "Format":                     ("escapeAttribute", "escapeText"),
    "AbstractFormattedWalker":    ("next",),
    "AbstractSAXOutputProcessor": ("printContent",),
    "AbstractStAXEventProcessor": ("printContent", "printDocument"),
    "AbstractStAXStreamProcessor":("printContent", "printDocument"),
    "AbstractXMLOutputProcessor": ("printContent", "printDocument"),
    "WalkerNORMALIZE":            ("analyzeMultiText",),
    "WalkerTRIM":                 ("analyzeMultiText",),
    "XPathHelper":                ("getSingleStep",),

    # ── ruler ────────────────────────────────────────────────────────────────
    "ByteMachine":                ("addEndOfMatch", "addMatchReferences", "addRangePattern",
                                   "deleteRangePattern", "doTransitionOn", "findMatchPattern",
                                   "findRangePattern", "updateMatchReferences"),
    "GenericMachine":             ("deleteStep",),
    "JsonRuleCompiler":           ("processMatchExpression", "writeRules"),
    "MachineComplexityEvaluator": ("evaluate", "getMatchesAccessibleFromEachTransition"),
    "RuleCompiler":               ("processMatchExpression", "writeRules"),
    "Ruler":                      ("matches",),

    # ── windward ─────────────────────────────────────────────────────────────
    "AbstractRouterGroup":        ("matchRouter",),
    "AntPathMatcher":             ("isMatch",),

    # ── agentscope-java / agentscope-core ────────────────────────────────────
    "ReActAgent":                 ("doCall",),
    "OpenAIResponseParser":       ("parseCompletionResponse", "parseChunkResponse"),
    "PlanNotebook":               ("updateSubtaskState", "reviseCurrentPlan"),

    # ── a2a-java / http-client ───────────────────────────────────────────────
    "ServerSentEventParser":      ("processLine", "dispatchEvent"),

    # ── a2a-java / server-common ─────────────────────────────────────────────
    "AgentCardValidator":         ("validateTransportConfiguration",),
    "DefaultRequestHandler":      ("onMessageSend",),
}


ALL_TARGETS: tuple[HardMethodTarget, ...] = (
    # ── defects4j ────────────────────────────────────────────────────────────
    HardMethodTarget(
        name="Rotation",
        subjects_root=SUBJECTS_ROOT,
        project="Math-2f",
        target_file="src/main/java/org/apache/commons/math3/geometry/euclidean/threed/Rotation.java",
    ),
    HardMethodTarget(
        name="GaussNewtonOptimizer",
        subjects_root=SUBJECTS_ROOT,
        project="Math-2f",
        target_file="src/main/java/org/apache/commons/math3/fitting/leastsquares/GaussNewtonOptimizer.java",
    ),
    HardMethodTarget(
        name="FromXmlParser",
        subjects_root=SUBJECTS_ROOT,
        project="JacksonXml-5f",
        target_file="src/main/java/com/fasterxml/jackson/dataformat/xml/deser/FromXmlParser.java",
        # pom.xml hardcodes <source>1.7</source>/<target>1.7</target>; Java 17+ rejects this.
        java_home=JAVA11_HOME,
    ),
    HardMethodTarget(
        name="ToXmlGenerator",
        subjects_root=SUBJECTS_ROOT,
        project="JacksonXml-5f",
        target_file="src/main/java/com/fasterxml/jackson/dataformat/xml/ser/ToXmlGenerator.java",
        java_home=JAVA11_HOME,
    ),
    HardMethodTarget(
        name="Metaphone",
        subjects_root=SUBJECTS_ROOT,
        project="Codec-18f",
        target_file="src/main/java/org/apache/commons/codec/language/Metaphone.java",
    ),
    HardMethodTarget(
        name="NumberUtils",
        subjects_root=SUBJECTS_ROOT,
        project="Lang-4f",
        target_file="src/main/java/org/apache/commons/lang3/math/NumberUtils.java",
    ),
    HardMethodTarget(
        name="FilteringParserDelegate",
        subjects_root=SUBJECTS_ROOT,
        project="JacksonCore-26f",
        target_file="src/main/java/com/fasterxml/jackson/core/filter/FilteringParserDelegate.java",
    ),
    HardMethodTarget(
        name="Flat3Map",
        subjects_root=SUBJECTS_ROOT,
        project="Collections-28f",
        target_file="src/main/java/org/apache/commons/collections4/map/Flat3Map.java",
    ),
    HardMethodTarget(
        name="DateTimeFormat",
        subjects_root=SUBJECTS_ROOT,
        project="Time-13f",
        target_file="src/main/java/org/joda/time/format/DateTimeFormat.java",
    ),
    HardMethodTarget(
        name="Entities",
        subjects_root=SUBJECTS_ROOT,
        project="Jsoup-93f",
        target_file="src/main/java/org/jsoup/nodes/Entities.java",
        # pom.xml hardcodes <source>1.7</source>/<target>1.7</target>; Java 17+ rejects this.
        java_home=JAVA11_HOME,
    ),

    # ── jquick-curl ──────────────────────────────────────────────────────────
    # Single-module Maven project; project root == module root.
    HardMethodTarget(
        name="JQuickCurlCommonVisitor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jquick-curl",
        target_file="src/main/java/com/github/paohaijiao/visitor/JQuickCurlCommonVisitor.java",
        extra_mvn_flags="--no-transfer-progress",
    ),
    HardMethodTarget(
        name="JCurlCommandProcessor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jquick-curl",
        target_file="src/main/java/com/github/paohaijiao/support/JCurlCommandProcessor.java",
        extra_mvn_flags="--no-transfer-progress",
        # This existing test requires a live HTTP server — it will fail TATG's
        # preflight. Remove it temporarily (backup already saved it; restore brings it back).
        conflicting_tests=(
            "src/test/java/com/github/paohaijiao/test/JCurlCommandProcessorTest.java",
        ),
    ),

    # ── adk-java / core ──────────────────────────────────────────────────────
    # Multi-module project: use the parent (adk-java) as project root so that the
    # per-test sandbox can resolve google-adk-parent via the copied pom.xml.
    # target_file includes the "core/" prefix; -pl core -am builds only the core module.
    HardMethodTarget(
        name="FunctionCallingUtils",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="adk-java",
        target_file="core/src/main/java/com/google/adk/tools/FunctionCallingUtils.java",
        extra_mvn_flags="--no-transfer-progress -pl core -am",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="SlidingWindowEventCompactor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="adk-java",
        target_file="core/src/main/java/com/google/adk/summarizer/SlidingWindowEventCompactor.java",
        extra_mvn_flags="--no-transfer-progress -pl core -am",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="SchemaUtils",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="adk-java",
        target_file="core/src/main/java/com/google/adk/SchemaUtils.java",
        extra_mvn_flags="--no-transfer-progress -pl core -am",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="GeminiUtil",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="adk-java",
        target_file="core/src/main/java/com/google/adk/models/GeminiUtil.java",
        extra_mvn_flags="--no-transfer-progress -pl core -am",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="ChatCompletionsRequest",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="adk-java",
        target_file="core/src/main/java/com/google/adk/models/chat/ChatCompletionsRequest.java",
        extra_mvn_flags="--no-transfer-progress -pl core -am",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="Contents",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="adk-java",
        target_file="core/src/main/java/com/google/adk/flows/llmflows/Contents.java",
        extra_mvn_flags="--no-transfer-progress -pl core -am",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="BaseAgent",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="adk-java",
        target_file="core/src/main/java/com/google/adk/agents/BaseAgent.java",
        extra_mvn_flags="--no-transfer-progress -pl core -am",
        java_home=JAVA24_HOME,
    ),

    # ── batch-processing-gateway ─────────────────────────────────────────────
    HardMethodTarget(
        name="SubmissionSummary",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="batch-processing-gateway",
        target_file="src/main/java/com/apple/spark/api/SubmissionSummary.java",
        extra_mvn_flags="--no-transfer-progress",
    ),
    HardMethodTarget(
        name="ApplicationSubmissionHelper",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="batch-processing-gateway",
        target_file="src/main/java/com/apple/spark/core/ApplicationSubmissionHelper.java",
        extra_mvn_flags="--no-transfer-progress",
    ),
    HardMethodTarget(
        name="SparkPodSpec",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="batch-processing-gateway",
        target_file="src/main/java/com/apple/spark/operator/SparkPodSpec.java",
        extra_mvn_flags="--no-transfer-progress",
    ),

    # ── commons-cli ──────────────────────────────────────────────────────────
    HardMethodTarget(
        name="HelpFormatter",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-cli",
        target_file="src/main/java/org/apache/commons/cli/HelpFormatter.java",
        jacoco_mode="pom",
    ),
    HardMethodTarget(
        name="PatternOptionBuilder",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-cli",
        target_file="src/main/java/org/apache/commons/cli/PatternOptionBuilder.java",
        jacoco_mode="pom",
    ),

    # ── commons-codec (requires Java 24 for maven.compiler.source=21) ────────
    HardMethodTarget(
        name="Base32",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/binary/Base32.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="Base64",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/binary/Base64.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="Md5Crypt",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/digest/Md5Crypt.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="MurmurHash3",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/digest/MurmurHash3.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="Sha2Crypt",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/digest/Sha2Crypt.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="ColognePhonetic",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/language/ColognePhonetic.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="DaitchMokotoffSoundex",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/language/DaitchMokotoffSoundex.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="DoubleMetaphone",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/language/DoubleMetaphone.java",
        java_home=JAVA24_HOME,
        target_descriptors={
            "handleC": "(Ljava/lang/String;Lorg/apache/commons/codec/language/DoubleMetaphone$DoubleMetaphoneResult;I)I",
        },
    ),
    HardMethodTarget(
        name="Nysiis",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/language/Nysiis.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="Rule",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/language/bm/Rule.java",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="QuotedPrintableCodec",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-codec",
        target_file="src/main/java/org/apache/commons/codec/net/QuotedPrintableCodec.java",
        java_home=JAVA24_HOME,
    ),

    # ── commons-collections ──────────────────────────────────────────────────
    # JaCoCo is configured in pom.xml with package-scoped includes to avoid
    # the java.lang bootstrap-classloader LinkageError on Java 11+.
    HardMethodTarget(
        name="IteratorUtils",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-collections",
        target_file="src/main/java/org/apache/commons/collections4/IteratorUtils.java",
    ),
    HardMethodTarget(
        name="TreeBidiMap",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-collections",
        target_file="src/main/java/org/apache/commons/collections4/bidimap/TreeBidiMap.java",
    ),
    # Flat3Map from new_projects (all methods); "Flat3Map" is already taken by
    # the defects4j Collections-28f entry, so we use "Flat3MapNew" as the dict key
    # to keep HARD_METHODS keys unique. class_name overrides the actual Java class
    # name passed to TATG and Maven so they see the real "Flat3Map".
    HardMethodTarget(
        name="Flat3MapNew",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-collections",
        target_file="src/main/java/org/apache/commons/collections4/map/Flat3Map.java",
        class_name="Flat3Map",
    ),
    HardMethodTarget(
        name="SequencesComparator",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-collections",
        target_file="src/main/java/org/apache/commons/collections4/sequence/SequencesComparator.java",
    ),
    HardMethodTarget(
        name="AbstractPatriciaTrie",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-collections",
        target_file="src/main/java/org/apache/commons/collections4/trie/AbstractPatriciaTrie.java",
    ),

    # ── commons-csv ──────────────────────────────────────────────────────────
    HardMethodTarget(
        name="CSVFormat",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-csv",
        target_file="src/main/java/org/apache/commons/csv/CSVFormat.java",
        extra_mvn_flags="-Danimal.sniffer.skip=true",
        jacoco_mode="pom",
    ),
    HardMethodTarget(
        name="CSVParser",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-csv",
        target_file="src/main/java/org/apache/commons/csv/CSVParser.java",
        extra_mvn_flags="-Danimal.sniffer.skip=true",
        jacoco_mode="pom",
    ),
    HardMethodTarget(
        name="Lexer",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="commons-csv",
        target_file="src/main/java/org/apache/commons/csv/Lexer.java",
        extra_mvn_flags="-Danimal.sniffer.skip=true",
        jacoco_mode="pom",
    ),

    # ── datafaker ────────────────────────────────────────────────────────────
    HardMethodTarget(
        name="FakeValues",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="datafaker",
        target_file="src/main/java/net/datafaker/service/FakeValues.java",
        extra_mvn_flags="--no-transfer-progress -Dspotless.check.skip=true -Dspotless.skip=true",
    ),
    HardMethodTarget(
        name="FakeValuesService",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="datafaker",
        target_file="src/main/java/net/datafaker/service/FakeValuesService.java",
        extra_mvn_flags="--no-transfer-progress -Dspotless.check.skip=true -Dspotless.skip=true",
    ),
    HardMethodTarget(
        name="SqlTransformer",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="datafaker",
        target_file="src/main/java/net/datafaker/transformations/sql/SqlTransformer.java",
        extra_mvn_flags="--no-transfer-progress -Dspotless.check.skip=true -Dspotless.skip=true",
    ),

    # ── gson / gson sub-module ───────────────────────────────────────────────
    # Multi-module: point at the gson/gson sub-module so mvn -pl . resolves correctly.
    HardMethodTarget(
        name="$Gson$Types",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/$Gson$Types.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="ConstructorConstructor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/ConstructorConstructor.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="Excluder",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/Excluder.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="LinkedTreeMap",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/LinkedTreeMap.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="Primitives",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/Primitives.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="JsonTreeReader",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/bind/JsonTreeReader.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="ObjectTypeAdapter",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/bind/ObjectTypeAdapter.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="ReflectiveTypeAdapterFactory",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/internal/bind/ReflectiveTypeAdapterFactory.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="TypeToken",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/reflect/TypeToken.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),
    HardMethodTarget(
        name="JsonReader",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="gson/gson",
        target_file="src/main/java/com/google/gson/stream/JsonReader.java",
        extra_mvn_flags="--no-transfer-progress",
        test_goals="test-compile surefire:test",
    ),

    # ── jdom2 ────────────────────────────────────────────────────────────────
    HardMethodTarget(
        name="Element",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/Element.java",
    ),
    HardMethodTarget(
        name="StringBin",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/StringBin.java",
    ),
    HardMethodTarget(
        name="Verifier",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/Verifier.java",
    ),
    HardMethodTarget(
        name="StAXStreamBuilder",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/input/StAXStreamBuilder.java",
        target_descriptors={
            "processPrunableElement": "(Lorg/jdom2/JDOMFactory;Ljavax/xml/stream/XMLStreamReader;ILorg/jdom2/input/stax/StAXFilter;)Lorg/jdom2/Element;",
        },
    ),
    HardMethodTarget(
        name="SAXHandler",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/input/sax/SAXHandler.java",
    ),
    HardMethodTarget(
        name="DTDParser",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/input/stax/DTDParser.java",
    ),
    HardMethodTarget(
        name="Format",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/Format.java",
    ),
    HardMethodTarget(
        name="AbstractFormattedWalker",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/support/AbstractFormattedWalker.java",
    ),
    HardMethodTarget(
        name="AbstractSAXOutputProcessor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/support/AbstractSAXOutputProcessor.java",
    ),
    HardMethodTarget(
        name="AbstractStAXEventProcessor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/support/AbstractStAXEventProcessor.java",
    ),
    HardMethodTarget(
        name="AbstractStAXStreamProcessor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/support/AbstractStAXStreamProcessor.java",
    ),
    HardMethodTarget(
        name="AbstractXMLOutputProcessor",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/support/AbstractXMLOutputProcessor.java",
    ),
    HardMethodTarget(
        name="WalkerNORMALIZE",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/support/WalkerNORMALIZE.java",
    ),
    HardMethodTarget(
        name="WalkerTRIM",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/output/support/WalkerTRIM.java",
    ),
    HardMethodTarget(
        name="XPathHelper",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="jdom2",
        target_file="src/main/java/org/jdom2/xpath/XPathHelper.java",
    ),

    # ── ruler ────────────────────────────────────────────────────────────────
    HardMethodTarget(
        name="ByteMachine",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="ruler",
        target_file="src/main/java/software/amazon/event/ruler/ByteMachine.java",
    ),
    HardMethodTarget(
        name="GenericMachine",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="ruler",
        target_file="src/main/java/software/amazon/event/ruler/GenericMachine.java",
    ),
    HardMethodTarget(
        name="JsonRuleCompiler",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="ruler",
        target_file="src/main/java/software/amazon/event/ruler/JsonRuleCompiler.java",
    ),
    HardMethodTarget(
        name="MachineComplexityEvaluator",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="ruler",
        target_file="src/main/java/software/amazon/event/ruler/MachineComplexityEvaluator.java",
    ),
    HardMethodTarget(
        name="RuleCompiler",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="ruler",
        target_file="src/main/java/software/amazon/event/ruler/RuleCompiler.java",
    ),
    HardMethodTarget(
        name="Ruler",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="ruler",
        target_file="src/main/java/software/amazon/event/ruler/Ruler.java",
        target_descriptors={
            "matches": "(Lcom/fasterxml/jackson/databind/JsonNode;Lsoftware/amazon/event/ruler/Patterns;)Z",
        },
    ),

    # ── windward ─────────────────────────────────────────────────────────────
    HardMethodTarget(
        name="AbstractRouterGroup",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="windward",
        target_file="src/main/java/org/flmelody/core/AbstractRouterGroup.java",
    ),
    HardMethodTarget(
        name="AntPathMatcher",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="windward",
        target_file="src/main/java/org/flmelody/util/AntPathMatcher.java",
    ),

    # ── agentscope-java / agentscope-core ────────────────────────────────────
    # Root pom (agentscope-java/pom.xml) already declares JaCoCo — the parent-
    # pom check in _pom_has_plugin_in_tree prevents double-agent injection.
    HardMethodTarget(
        name="ReActAgent",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="agentscope-java",
        target_file="agentscope-core/src/main/java/io/agentscope/core/ReActAgent.java",
        extra_mvn_flags="--no-transfer-progress -pl agentscope-core -am -Dspotless.check.skip=true -Dspotless.skip=true",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="OpenAIResponseParser",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="agentscope-java",
        target_file="agentscope-core/src/main/java/io/agentscope/core/formatter/openai/OpenAIResponseParser.java",
        extra_mvn_flags="--no-transfer-progress -pl agentscope-core -am -Dspotless.check.skip=true -Dspotless.skip=true",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="PlanNotebook",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="agentscope-java",
        target_file="agentscope-core/src/main/java/io/agentscope/core/plan/PlanNotebook.java",
        extra_mvn_flags="--no-transfer-progress -pl agentscope-core -am -Dspotless.check.skip=true -Dspotless.skip=true",
        java_home=JAVA24_HOME,
    ),

    # ── a2a-java / http-client ───────────────────────────────────────────────
    HardMethodTarget(
        name="ServerSentEventParser",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="a2a-java/http-client",
        target_file="src/main/java/org/a2aproject/sdk/client/http/ServerSentEventParser.java",
        extra_mvn_flags="--no-transfer-progress",
        java_home=JAVA24_HOME,
    ),

    # ── a2a-java / server-common ─────────────────────────────────────────────
    HardMethodTarget(
        name="AgentCardValidator",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="a2a-java/server-common",
        target_file="src/main/java/org/a2aproject/sdk/server/AgentCardValidator.java",
        extra_mvn_flags="--no-transfer-progress",
        java_home=JAVA24_HOME,
    ),
    HardMethodTarget(
        name="DefaultRequestHandler",
        subjects_root=NEW_SUBJECTS_ROOT,
        project="a2a-java/server-common",
        target_file="src/main/java/org/a2aproject/sdk/server/requesthandlers/DefaultRequestHandler.java",
        extra_mvn_flags="--no-transfer-progress",
        java_home=JAVA24_HOME,
        java_version="17",
    ),
)

_TARGETS_BY_NAME: dict[str, HardMethodTarget] = {t.name: t for t in ALL_TARGETS}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _comma_values(value: str) -> list[str]:
    def clean_item(item: str) -> str:
        item = item.strip()
        if not item or item.startswith("#") or item == "selected_methods":
            return ""
        item = re.sub(r"\s+\[[^\]]+\]\s*$", "", item).strip()
        return item

    text = ",".join(clean_item(line) for line in str(value or "").splitlines())
    values: list[str] = []
    current: list[str] = []
    depth = 0

    def descriptor_object_return_in_progress(token: str) -> bool:
        """Return true only for the terminating ';' of a JVM object return type."""
        hash_index = token.find("#(")
        close_index = token.rfind(")")
        if hash_index < 0 or close_index < hash_index:
            return False
        return_descriptor = token[close_index + 1:].strip()
        return bool(re.fullmatch(r"\[*L[^;]*", return_descriptor))

    for idx, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        is_delim = ch in {",", ";"} and depth == 0
        if is_delim and ch == ";" and descriptor_object_return_in_progress("".join(current)):
            # Parameter object descriptors are protected by ``depth``. At depth
            # zero, this semicolon both terminates the object return descriptor
            # and separates it from the next selector.
            current.append(ch)
        if is_delim and ch == ",":
            current_text = "".join(current).strip()
            next_text = text[idx + 1:].split(",", 1)[0].split(";", 1)[0].strip()
            # Short overload selectors can contain comma-separated parameter
            # names without parentheses, e.g. Class.method#Type,ObjectMapper.
            # Keep those commas unless the next chunk clearly starts another
            # selector (Class.method).
            if "#" in current_text and "." not in next_text:
                current.append(ch)
                continue
        if is_delim:
            item = "".join(current).strip()
            item = clean_item(item)
            if item:
                values.append(item)
            current = []
            continue
        current.append(ch)
    item = "".join(current).strip()
    item = clean_item(item)
    if item:
        values.append(item)
    return values


def _shell_safe_class_name(name: str) -> str:
    """Escape $ in a Java class name so shell=True doesn't expand it as a variable."""
    return name.replace("$", r"\$")


def _java_test_class_name(class_name: str) -> str:
    simple = str(class_name or "").strip()
    simple = simple.replace("$", "")
    simple = re.sub(r"[^A-Za-z0-9_]", "", simple)
    return simple or "Generated"


def _make_test_command(
    class_name: str,
    extra_mvn_flags: str = "",
    test_goals: str = "test",
    project_root: Path | None = None,
    jacoco_mode: str = "auto",
    java_home: str | None = None,
) -> str:
    # Quality-check skip flags are safe no-ops when the plugin is absent; they
    # prevent checkstyle/spotbugs/rat from blocking streaming validation runs on
    # projects that configure these tools.
    skip = "-Drat.skip=true -Dcheckstyle.skip=true -Dspotbugs.skip=true -Dpmd.skip=true"
    flags = f" {extra_mvn_flags}" if extra_mvn_flags else ""
    javac = Path(java_home) / "bin" / "javac" if java_home else None
    if javac and javac.exists():
        flags = f"{flags} -Dmaven.compiler.executable={javac}"
    # Escape $ so the shell (shell=True in the runner) doesn't expand it as a variable.
    safe_name = _shell_safe_class_name(_java_test_class_name(class_name))

    # Inject JaCoCo when the project has no existing jacoco-maven-plugin binding.
    # This enables iter2+ coverage feedback for projects that don't configure JaCoCo
    # themselves.  When jacoco IS already configured we rely on its bound
    # prepare-agent goal and only append the :report goal.  Double-injecting
    # prepare-agent crashes the Surefire forked JVM (two -javaagent flags).
    # -Dmaven.test.failure.ignore=true ensures jacoco:report runs even when tests fail.
    jc_ver = DEFAULT_JACOCO_VERSION
    jacoco_prepare = f"org.jacoco:jacoco-maven-plugin:{jc_ver}:prepare-agent"
    jacoco_report = f"org.jacoco:jacoco-maven-plugin:{jc_ver}:report"
    normalized_jacoco_mode = (jacoco_mode or "auto").strip().lower()
    if normalized_jacoco_mode not in {"auto", "pom", "inject"}:
        raise ValueError(f"Unsupported jacoco_mode={jacoco_mode!r}")
    has_project_jacoco = project_root is not None and pom_has_jacoco_plugin(project_root)
    if normalized_jacoco_mode == "pom" or (
        normalized_jacoco_mode == "auto" and has_project_jacoco
    ):
        # JaCoCo already configured: only append :report so the XML/CSV is written
        # even when tests are invoked via `mvn test` (not `mvn verify`).
        goals_prefix = ""
        goals_suffix = f" {jacoco_report}"
    else:
        # No JaCoCo configured: inject prepare-agent before test + report after.
        jacoco_prepare, jacoco_argline_property = maven_jacoco_prepare_agent_args(project_root or Path.cwd(), jc_ver)
        goals_prefix = f"{jacoco_prepare} "
        flags = f"{flags} {jacoco_argline_property}"
        goals_suffix = f" {jacoco_report}"

    return (
        f"mvn {goals_prefix}{test_goals} -pl .{flags} {skip}"
        f" -Dmaven.test.failure.ignore=true"
        f" -Dtest={safe_name}Test"
        f" -Dsurefire.failIfNoSpecifiedTests=false"
        f"{goals_suffix}"
    )


def _java_release_major(value: str | None) -> int | None:
    text = str(value or "").strip().strip('"')
    match = re.match(r"^(?:1\.)?(\d+)", text)
    return int(match.group(1)) if match else None


def _java_home_major(java_home: str | None) -> int | None:
    if not java_home:
        return None
    release_file = Path(java_home) / "release"
    try:
        release_text = release_file.read_text(
            encoding="utf-8",
            errors="ignore",
        )
    except OSError:
        return None
    match = re.search(
        r'(?m)^\s*JAVA_VERSION\s*=\s*"(?P<version>[^"]+)"',
        release_text,
    )
    return _java_release_major(
        match.group("version") if match else ""
    )


def _compatible_target_java_home(
    java_home: str | None,
    required_release: str | None,
) -> str | None:
    """Return a configured JDK only when it can compile the target release."""

    if not java_home:
        return None
    required_major = _java_release_major(required_release)
    if required_major is None:
        return java_home
    actual_major = _java_home_major(java_home)
    if actual_major is None or actual_major < required_major:
        return None
    return java_home


def _versioned_java_home_env_keys(required_major: int) -> tuple[str, ...]:
    return (
        f"TATG_JAVA_HOME_{required_major}",
        f"JAVA_HOME_{required_major}",
        f"JDK_HOME_{required_major}",
        f"JAVA{required_major}_HOME",
    )


def _system_java_home_candidates(required_major: int) -> list[str]:
    """Discover JDK homes without embedding workstation-specific paths."""

    candidates: list[str] = []
    if sys.platform == "darwin":
        resolver = str(
            os.environ.get("TATG_JAVA_HOME_RESOLVER")
            or "/usr/libexec/java_home"
        ).strip()
        try:
            proc = subprocess.run(
                [resolver, "-v", str(required_major)],
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
        except Exception:  # noqa: BLE001
            proc = None
        if proc is not None and proc.returncode == 0:
            candidate = str(proc.stdout or "").strip()
            if candidate:
                candidates.append(candidate)

    # Homebrew JDKs are not always registered with /usr/libexec/java_home.
    # Querying brew keeps the discovery portable across Intel/Apple Silicon
    # prefixes and avoids embedding a user-local installation path.
    brew = shutil.which("brew")
    if brew:
        for formula in (f"openjdk@{required_major}", "openjdk"):
            try:
                proc = subprocess.run(
                    [brew, "--prefix", formula],
                    text=True,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
            except Exception:  # noqa: BLE001
                continue
            prefix = str(proc.stdout or "").strip()
            if proc.returncode != 0 or not prefix:
                continue
            root = Path(prefix)
            mac_home = root / "libexec" / "openjdk.jdk" / "Contents" / "Home"
            candidates.append(str(mac_home if mac_home.exists() else root))
    return candidates


def _resolve_compatible_target_java_home(
    preferred_java_home: str | None,
    required_release: str | None,
    *,
    env: dict[str, str] | None = None,
    system_discovery: bool = True,
) -> str | None:
    """Resolve a JDK that can compile the target language release.

    Explicit target configuration wins.  Otherwise use versioned environment
    variables and platform discovery.  An incompatible inherited JAVA_HOME is
    never returned.
    """

    values = os.environ if env is None else env
    required_major = _java_release_major(required_release)
    raw_candidates: list[str] = []

    if preferred_java_home:
        raw_candidates.append(str(preferred_java_home))
    if required_major is not None:
        for key in _versioned_java_home_env_keys(required_major):
            value = str(values.get(key) or "").strip()
            if value:
                raw_candidates.append(value)
        # Legacy configuration remains supported as a compatible newer-JDK
        # fallback, but it is validated rather than trusted by name.
        for key in ("TATG_JAVA24_HOME",):
            value = str(values.get(key) or "").strip()
            if value:
                raw_candidates.append(value)
        if system_discovery:
            raw_candidates.extend(_system_java_home_candidates(required_major))

    inherited = str(values.get("JAVA_HOME") or "").strip()
    if inherited:
        raw_candidates.append(inherited)

    seen: set[str] = set()
    compatible: list[tuple[int, str]] = []
    for raw in raw_candidates:
        candidate = str(Path(raw).expanduser())
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if not (Path(candidate) / "bin" / "java").exists():
            continue
        actual_major = _java_home_major(candidate)
        if required_major is None:
            return candidate
        if actual_major is not None and actual_major >= required_major:
            if preferred_java_home and candidate == str(Path(preferred_java_home).expanduser()):
                return candidate
            compatible.append((actual_major, candidate))

    if not compatible:
        return None
    # Prefer the closest compatible JDK to reduce tool/plugin incompatibility.
    compatible.sort(key=lambda item: (item[0], item[1]))
    return compatible[0][1]


def _configure_target_java_env(
    target_env: dict[str, str],
    *,
    preferred_java_home: str | None,
    required_release: str | None,
) -> str | None:
    """Apply a compatible JDK and remove a known-incompatible inheritance."""

    resolved = _resolve_compatible_target_java_home(
        preferred_java_home,
        required_release,
        env=target_env,
    )
    if resolved:
        target_env["JAVA_HOME"] = resolved
        java_bin = str(Path(resolved) / "bin")
        old_path = str(target_env.get("PATH") or "")
        path_parts = [part for part in old_path.split(os.pathsep) if part and part != java_bin]
        target_env["PATH"] = os.pathsep.join([java_bin, *path_parts])
        target_env["TATG_RESPECT_JAVA_HOME"] = "1"
        return resolved

    inherited = str(target_env.get("JAVA_HOME") or "").strip()
    required_major = _java_release_major(required_release)
    inherited_major = _java_home_major(inherited)
    if (
        inherited
        and required_major is not None
        and (inherited_major is None or inherited_major < required_major)
    ):
        target_env.pop("JAVA_HOME", None)
        target_env.pop("TATG_RESPECT_JAVA_HOME", None)
    return None


def _infer_target_java_version(project_root: Path, target: HardMethodTarget) -> str:
    """Infer the Java language/runtime version for a target module.

    Many multi-module projects declare compiler release/source/target only in
    the parent POM.  TATG's CLI defaults to Java 11 when no explicit
    ``--java-version`` is passed, so batch runs must surface the parent version
    instead of relying only on JAVA_HOME or maven.compiler.executable.
    """
    explicit = str(target.java_version or "").strip()
    if explicit:
        return explicit
    roots: list[Path] = []
    current = project_root.resolve()
    for _ in range(5):
        if current in roots:
            break
        roots.append(current)
        if current.parent == current:
            break
        current = current.parent
    for root in roots:
        try:
            build_system = detect_build_system(root)
            java_version, _junit_version, _junit_family = infer_runtime_versions(root, build_system)
        except Exception:  # noqa: BLE001
            java_version = ""
        java_version = str(java_version or "").strip()
        if java_version:
            return java_version
    return ""


def _default_test_file_for_target(target: HardMethodTarget) -> str | None:
    """Infer default generated test file path relative to project root.

    Converts `<...>/src/main/java/pkg/Class.java` to
    `<...>/src/test/java/pkg/<ClassName>Test.java`.
    """
    marker = "src/main/java/"
    if marker not in target.target_file:
        return None
    before, after = target.target_file.split(marker, 1)
    pkg_dir = Path(after).parent
    test_rel = Path(before) / "src/test/java" / pkg_dir / f"{_java_test_class_name(target.java_class_name)}Test.java"
    return str(test_rel)


def _pipeline_rc(return_codes: object) -> int | None:
    if not isinstance(return_codes, list) or not return_codes:
        return None
    values = [int(code) for code in return_codes if isinstance(code, (int, float))]
    return max(values) if values else None


def _has_nonzero_intermediate_rc(return_codes: object) -> bool:
    if not isinstance(return_codes, list):
        return False
    return any(int(code) != 0 for code in return_codes if isinstance(code, (int, float)))


def _pct(value: object) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100.0:.2f}%"
    except Exception:  # noqa: BLE001
        return ""


def _pct_with_total(value: object, total: int) -> str:
    if int(total or 0) <= 0:
        return "N/A"
    return _pct(value)


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _jvm_desc_to_short(descriptor: str) -> str:
    """Convert a JVM method descriptor like '([CII)V' to a readable param string like 'char[],int,int'."""
    if not descriptor or "(" not in descriptor:
        return descriptor or "?"
    try:
        inner = descriptor[1:descriptor.index(")")]
    except ValueError:
        return "?"
    parts: list[str] = []
    i = 0
    _primitives = {"I": "int", "J": "long", "D": "double", "F": "float",
                   "Z": "boolean", "B": "byte", "S": "short", "C": "char", "V": "void"}
    while i < len(inner):
        ch = inner[i]
        if ch == "L":
            try:
                end = inner.index(";", i)
            except ValueError:
                parts.append("?")
                break
            fqn = inner[i + 1:end].replace("/", ".")
            parts.append(fqn.rsplit(".", 1)[-1])
            i = end + 1
        elif ch == "[":
            # Count array dimensions
            dims = 0
            while i < len(inner) and inner[i] == "[":
                dims += 1
                i += 1
            if i >= len(inner):
                parts.append("?" + "[]" * dims)
                break
            if inner[i] == "L":
                try:
                    end = inner.index(";", i)
                except ValueError:
                    parts.append("?" + "[]" * dims)
                    break
                fqn = inner[i + 1:end].replace("/", ".")
                base = fqn.rsplit(".", 1)[-1]
                i = end + 1
            else:
                base = _primitives.get(inner[i], inner[i])
                i += 1
            parts.append(base + "[]" * dims)
        elif ch in _primitives:
            parts.append(_primitives[ch])
            i += 1
        else:
            parts.append(ch)
            i += 1
    return ",".join(parts) if parts else "()"


def _per_overload_coverage(data: dict, method_name: str, target_descriptor: str | None = None) -> list[dict]:
    """Return per-overload coverage metrics from the last test run's methods_coverage array.

    Each entry: {"descriptor": str, "short": str, "line_coverage": float|None,
                 "branch_coverage": float|None, "line_covered": int, "line_missed": int,
                 "branch_covered": int, "branch_missed": int}.
    """
    runs = data.get("test_runs") or []
    seen: dict[str, dict] = {}
    for run in runs:
        for m in run.get("methods_coverage") or []:
            if m.get("method_name") != method_name:
                continue
            desc = m.get("descriptor") or ""
            existing = seen.get(desc)
            new_score = float(m.get("line_coverage") or 0) + float(m.get("branch_coverage") or 0)
            if existing is None or new_score > float(existing.get("line_coverage") or 0) + float(existing.get("branch_coverage") or 0):
                seen[desc] = m
    # Some legacy/global JaCoCo merges leave a descriptorless aggregate row
    # beside the exact descriptor row.  Once the target descriptor is present,
    # the blank row is not a real overload and must not become a second metric.
    if target_descriptor and target_descriptor in seen:
        seen.pop("", None)
    result = []
    for desc, m in seen.items():
        lc = m.get("line_coverage")
        bc = m.get("branch_coverage")
        result.append({
            "descriptor": desc,
            "short": _jvm_desc_to_short(desc),
            "line_coverage": float(lc) if isinstance(lc, (int, float)) else None,
            "branch_coverage": float(bc) if isinstance(bc, (int, float)) else None,
            "line_covered": m.get("line_covered") or 0,
            "line_missed": m.get("line_missed") or 0,
            "branch_covered": m.get("branch_covered") or 0,
            "branch_missed": m.get("branch_missed") or 0,
            "is_target_descriptor": bool(target_descriptor and desc == target_descriptor),
        })
    result.sort(key=lambda x: x["descriptor"])
    return result


def _resolved_target_descriptor(
    data: dict,
    method_name: str,
    configured_descriptor: str | None = None,
) -> str | None:
    """Resolve the descriptor already established by the per-target run.

    Non-overloaded selectors do not always carry a descriptor in the batch
    task, while the CLI can still resolve the exact descriptor from JaCoCo.
    Prefer the explicit batch identity, then the final artifact identity, and
    finally a unique target-method coverage row.  Never guess between multiple
    descriptors.
    """

    configured = str(configured_descriptor or "").strip()
    if configured:
        return configured

    artifact_descriptor = str(
        data.get("target_descriptor")
        or data.get("target_method_descriptor")
        or ""
    ).strip()
    if artifact_descriptor:
        return artifact_descriptor

    report_target = (
        (((data.get("report_summary") or {}).get("coverage") or {}).get("target_method"))
        if isinstance(data.get("report_summary"), dict)
        else None
    )
    if isinstance(report_target, dict):
        descriptor = str(report_target.get("descriptor") or "").strip()
        if descriptor:
            return descriptor

    descriptors: set[str] = set()
    for run in data.get("test_runs") or []:
        for row in run.get("methods_coverage") or []:
            if str(row.get("method_name") or "") != method_name:
                continue
            descriptor = str(row.get("descriptor") or "").strip()
            if descriptor:
                descriptors.add(descriptor)
    return next(iter(descriptors)) if len(descriptors) == 1 else None


def _per_overload_tests(data: dict) -> dict[str, dict]:
    """Return per-overload_id test counts from streaming_validation details.

    Returns: {overload_id: {"submitted": int, "passed": int, "failed": int}}
    """
    vp = data.get("validation_policy") or {}
    counts: dict[str, dict] = {}
    for detail in (vp.get("streaming_validation_failed_details") or []):
        oid = detail.get("overload_id") or "?"
        counts.setdefault(oid, {"submitted": 0, "passed": 0, "failed": 0})
        counts[oid]["submitted"] += 1
        counts[oid]["failed"] += 1
    for detail in (vp.get("streaming_validation_passed_details") or []):
        oid = detail.get("overload_id") or "?"
        counts.setdefault(oid, {"submitted": 0, "passed": 0, "failed": 0})
        counts[oid]["submitted"] += 1
        counts[oid]["passed"] += 1
    return counts


def _best_method_cov_pair(
    data: dict, class_name: str, method_name: str, target_descriptor: str | None = None
) -> tuple[float | None, float | None]:
    best = _best_method_row(data, class_name, method_name, target_descriptor)
    if not best:
        return (None, None)
    line = best.get("line_coverage")
    branch_total = int(best.get("branch_covered") or 0) + int(best.get("branch_missed") or 0)
    branch = best.get("branch_coverage") if branch_total > 0 else None
    return (
        float(line) if isinstance(line, (int, float)) else None,
        float(branch) if isinstance(branch, (int, float)) else None,
    )


def _candidate_best_target_coverage(
    data: dict,
    method_name: str | None = None,
    target_descriptor: str | None = None,
) -> dict | None:
    """Return the best passed target-positive candidate-level coverage, if any.

    This is deliberately separate from final target-method JaCoCo coverage.  It is
    used as display/triage evidence when the final combined suite has no readable
    target row but streaming validation already measured a target-positive
    candidate in isolation.
    """
    roots: list[dict] = []
    if isinstance(data, dict):
        coverage = (data.get("report_summary") or {}).get("coverage") if isinstance(data.get("report_summary"), dict) else {}
        if isinstance(coverage, dict):
            direct = coverage.get("candidate_best_target_positive_passed")
            if isinstance(direct, dict):
                cov_method = str(direct.get("method_name") or method_name or "").strip()
                cov_descriptor = str(direct.get("descriptor") or "").strip()
                if (
                    (not method_name or not cov_method or cov_method == method_name)
                    and (not target_descriptor or not cov_descriptor or cov_descriptor == target_descriptor)
                    and (isinstance(direct.get("line_coverage"), (int, float)) or isinstance(direct.get("branch_coverage"), (int, float)))
                ):
                    return {
                        "source": direct.get("source") or "passed_candidate_best_snapshot",
                        "test_method": direct.get("test_method"),
                        "line_coverage": direct.get("line_coverage"),
                        "branch_coverage": direct.get("branch_coverage"),
                        "line_covered": direct.get("line_covered"),
                        "line_missed": direct.get("line_missed"),
                        "branch_covered": direct.get("branch_covered"),
                        "branch_missed": direct.get("branch_missed"),
                        "descriptor": direct.get("descriptor"),
                        "class_fqn": direct.get("class_fqn"),
                        "method_name": direct.get("method_name"),
                        "coverage_status": direct.get("status"),
                    }
        roots.append(data)
        report_summary = data.get("report_summary")
        if isinstance(report_summary, dict):
            validation_policy = report_summary.get("validation_policy")
            if isinstance(validation_policy, dict):
                roots.append(validation_policy)
        validation_policy = data.get("validation_policy")
        if isinstance(validation_policy, dict):
            roots.append(validation_policy)

    detail_keys = (
        "streaming_validation_passed_details",
        "coverage_positive_repair_passed_details",
        "streaming_candidate_coverage_audit_details",
        "candidate_coverage_audit_details",
        "candidate_audit",
    )
    candidates: list[dict] = []
    seen_ids: set[int] = set()
    passed_detail_ids: set[int] = set()
    for root in roots:
        for child in root.get("streaming_validation_passed_details") or []:
            if isinstance(child, dict):
                passed_detail_ids.add(id(child))
        for child in root.get("coverage_positive_repair_passed_details") or []:
            if isinstance(child, dict):
                passed_detail_ids.add(id(child))
        for key in detail_keys:
            values = root.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict) or id(item) in seen_ids:
                    continue
                seen_ids.add(id(item))
                candidates.append(item)

    best: dict | None = None
    best_score = -1.0
    for item in candidates:
        status = str(item.get("status") or "").strip().lower()
        gate = item.get("coverage_gate") if isinstance(item.get("coverage_gate"), dict) else {}
        # streaming_validation_passed_details often omits status; the list name is
        # already sufficient evidence that validation passed.
        from_passed_list = id(item) in passed_detail_ids
        if status and status not in {"passed", "success", "accepted", "repaired_passed"}:
            continue
        if not status and not from_passed_list and gate.get("accepted") is not True:
            continue
        cov = item.get("candidate_target_coverage")
        if not isinstance(cov, dict):
            cov = gate
        if not isinstance(cov, dict):
            continue
        cov_method = str(cov.get("method_name") or item.get("method_name") or "").strip()
        if method_name and cov_method and cov_method != method_name:
            continue
        cov_descriptor = str(cov.get("descriptor") or item.get("target_descriptor") or "").strip()
        if target_descriptor and cov_descriptor and cov_descriptor != target_descriptor:
            continue
        target_positive = any(
            bool(value)
            for value in (
                item.get("target_positive"),
                item.get("has_target_coverage"),
                cov.get("has_target_coverage"),
                gate.get("has_target_coverage"),
            )
        )
        if not target_positive:
            continue
        line = cov.get("line_coverage")
        branch = cov.get("branch_coverage")
        if not isinstance(line, (int, float)) and not isinstance(branch, (int, float)):
            continue
        line_covered = int(cov.get("line_covered") or 0)
        branch_covered = int(cov.get("branch_covered") or 0)
        score = float(line or 0.0) + float(branch or 0.0) + (line_covered + branch_covered) / 10000.0
        if score <= best_score:
            continue
        best_score = score
        best = {
            "source": "candidate_isolated_target_method_coverage",
            "test_method": item.get("test_method"),
            "line_coverage": float(line) if isinstance(line, (int, float)) else None,
            "branch_coverage": float(branch) if isinstance(branch, (int, float)) else None,
            "line_covered": line_covered,
            "line_missed": int(cov.get("line_missed") or 0),
            "branch_covered": branch_covered,
            "branch_missed": int(cov.get("branch_missed") or 0),
            "descriptor": cov.get("descriptor") or item.get("target_descriptor"),
            "class_fqn": cov.get("class_fqn"),
            "method_name": cov.get("method_name"),
            "coverage_status": cov.get("status") or gate.get("coverage_status"),
            "promotion_reject_reason": item.get("promotion_reject_reason"),
        }
    return best


def _candidate_best_failed_target_coverage(data: dict) -> dict | None:
    coverage = (data.get("report_summary") or {}).get("coverage") if isinstance(data.get("report_summary"), dict) else {}
    if isinstance(coverage, dict) and isinstance(coverage.get("candidate_best_target_positive_failed"), dict):
        direct = coverage["candidate_best_target_positive_failed"]
        if isinstance(direct.get("line_coverage"), (int, float)) or isinstance(direct.get("branch_coverage"), (int, float)):
            return dict(direct)
    roots: list[dict] = []
    if isinstance(data, dict):
        roots.append(data)
        if isinstance(data.get("validation_policy"), dict):
            roots.append(data["validation_policy"])
    best: dict | None = None
    best_score = -1.0
    for root in roots:
        for key in ("coverage_positive_failed_details", "streaming_validation_failed_details"):
            values = root.get(key)
            if not isinstance(values, list):
                continue
            for item in values:
                if not isinstance(item, dict):
                    continue
                cov = item.get("candidate_target_coverage")
                if not isinstance(cov, dict) or not bool(cov.get("has_target_coverage")):
                    continue
                line = cov.get("line_coverage")
                branch = cov.get("branch_coverage")
                if not isinstance(line, (int, float)) and not isinstance(branch, (int, float)):
                    continue
                score = float(line or 0.0) + float(branch or 0.0)
                if score <= best_score:
                    continue
                best_score = score
                best = dict(cov)
                best.setdefault("source", "failed_candidate_best_target_positive")
                if item.get("test_method"):
                    best.setdefault("test_method", item.get("test_method"))
                signal = item.get("failure_signal") if isinstance(item.get("failure_signal"), dict) else {}
                category = item.get("failure_category") or signal.get("category") or item.get("failure_type")
                if category:
                    best.setdefault("failure_category", category)
    return best


def _best_method_row(
    data: dict, class_name: str, method_name: str, target_descriptor: str | None = None
) -> dict | None:
    runs = data.get("test_runs") or []
    methods: list[dict] = []
    for run in runs:
        methods.extend(run.get("methods_coverage") or [])
    matches = [
        row for row in methods
        if row.get("method_name") == method_name and row.get("class_name") == class_name
    ]
    if target_descriptor:
        matches = [row for row in matches if row.get("descriptor") == target_descriptor]
    if not matches:
        return None
    return max(
        matches,
        key=lambda item: float(item.get("line_coverage") or 0.0) + float(item.get("branch_coverage") or 0.0),
    )


def _mutation_status_summary(data: dict, best_method: dict | None) -> dict:
    mutation_eval = data.get("mutation_evaluation") if isinstance(data.get("mutation_evaluation"), dict) else {}
    report_summary = data.get("report_summary") if isinstance(data.get("report_summary"), dict) else {}
    validation_policy = data.get("validation_policy") if isinstance(data.get("validation_policy"), dict) else {}
    if not validation_policy and isinstance(report_summary.get("validation_policy"), dict):
        validation_policy = report_summary.get("validation_policy") or {}
    method_score = None
    method_total = None
    method_killed = None
    method_survived = None
    method_no_coverage = None
    method_non_viable = None
    method_raw_total = None
    target_mutation = mutation_eval.get("target_method_mutation")
    if isinstance(target_mutation, dict):
        method_score = target_mutation.get("mutation_score")
        method_total = target_mutation.get("mutation_total")
        method_killed = target_mutation.get("mutation_killed")
        method_survived = target_mutation.get("mutation_survived")
        method_no_coverage = target_mutation.get("mutation_no_coverage")
        method_non_viable = target_mutation.get("mutation_non_viable")
        method_raw_total = target_mutation.get("mutation_raw_total")
    if not isinstance(method_score, (int, float)) and isinstance(best_method, dict):
        method_score = best_method.get("mutation_score")
        method_total = best_method.get("mutation_total")
        method_killed = best_method.get("mutation_killed")
        method_survived = best_method.get("mutation_survived")
        method_no_coverage = best_method.get("mutation_no_coverage")
        method_non_viable = best_method.get("mutation_non_viable")
        method_raw_total = best_method.get("mutation_raw_total")
    if not isinstance(method_score, (int, float)) and isinstance(best_method, dict):
        method_name = str(best_method.get("method_name") or "")
        descriptor = str(best_method.get("descriptor") or "")
        class_fqn = str(best_method.get("class_fqn") or "")
        class_name = str(best_method.get("class_name") or "")
        for item in mutation_eval.get("method_mutation_scores") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("method_name") or "") != method_name:
                continue
            if descriptor and str(item.get("descriptor") or "") != descriptor:
                continue
            item_class = str(item.get("class_fqn") or item.get("class_name") or "")
            if class_fqn and item_class not in {class_fqn, class_name}:
                continue
            method_score = item.get("mutation_score")
            method_total = item.get("mutation_total")
            method_killed = item.get("mutation_killed")
            method_survived = item.get("mutation_survived")
            method_no_coverage = item.get("mutation_no_coverage")
            method_non_viable = item.get("mutation_non_viable")
            method_raw_total = item.get("mutation_raw_total")
            break
    if not isinstance(method_score, (int, float)) and isinstance(best_method, dict):
        xml_mutation = _mutation_summary_from_pitest_xml(data, best_method)
        if xml_mutation:
            method_score = xml_mutation.get("mutation_score")
            method_total = xml_mutation.get("mutation_total")
            method_killed = xml_mutation.get("mutation_killed")
            method_survived = xml_mutation.get("mutation_survived")
            method_no_coverage = xml_mutation.get("mutation_no_coverage")
            method_non_viable = xml_mutation.get("mutation_non_viable")
            method_raw_total = xml_mutation.get("mutation_raw_total")
    mutation_after_extra = validation_policy.get("mutation_after_extra") if isinstance(validation_policy.get("mutation_after_extra"), dict) else {}
    if not isinstance(method_score, (int, float)) and mutation_after_extra:
        method_score = mutation_after_extra.get("mutation_score")
        method_total = mutation_after_extra.get("mutation_total")
        method_killed = mutation_after_extra.get("mutation_killed")
        method_survived = mutation_after_extra.get("mutation_survived")
        method_no_coverage = mutation_after_extra.get("mutation_no_coverage")
        method_non_viable = mutation_after_extra.get("mutation_non_viable")
        method_raw_total = mutation_after_extra.get("mutation_raw_total")
    mutation_status = mutation_eval.get("status")
    if not mutation_status and isinstance(method_score, (int, float)) and int(validation_policy.get("mutation_extra_iterations_run") or 0) > 0:
        mutation_status = "ok_extra_round"
    if isinstance(method_total, (int, float)) and float(method_total) <= 0.0:
        method_score = None
        mutation_applicability_status = (
            "no_viable_mutants"
            if isinstance(method_non_viable, (int, float))
            and float(method_non_viable) > 0.0
            else "no_applicable_mutants"
        )
        mutation_status = mutation_applicability_status
    elif isinstance(method_total, (int, float)) and float(method_total) > 0.0:
        mutation_applicability_status = (
            "applicable_measured"
            if isinstance(method_score, (int, float))
            else "applicable_score_missing"
        )
    else:
        mutation_applicability_status = "mutation_total_missing"
    return {
        "mutation_status": mutation_status,
        "mutation_return_code": mutation_eval.get("return_code"),
        "mutation_timed_out": mutation_eval.get("timed_out"),
        "mutation_junit5_plugin_injected": mutation_eval.get("temporary_pitest_junit5_plugin_injected"),
        "pitest_tests_run": mutation_eval.get("pitest_tests_run"),
        "pitest_retry_command_count": len(list(mutation_eval.get("retry_commands") or [])),
        "mutation_feedback_useful": mutation_eval.get("mutation_feedback_useful"),
        "mutation_extra_iterations_run": validation_policy.get("mutation_extra_iterations_run"),
        "mutation_extra_round_skipped_reason": validation_policy.get("mutation_extra_round_skipped_reason"),
        "mutation_extra_delta": (
            (validation_policy.get("mutation_extra_delta") or {}).get("delta_mutation_score")
            if isinstance(validation_policy.get("mutation_extra_delta"), dict)
            else None
        ),
        "mutation_no_gain_reason": validation_policy.get("mutation_no_gain_reason"),
        "target_mutation_score": float(method_score) if isinstance(method_score, (int, float)) else None,
        "target_mutation_total": method_total,
        "target_mutation_killed": method_killed,
        "target_mutation_survived": method_survived,
        "target_mutation_no_coverage": method_no_coverage,
        "target_mutation_non_viable": method_non_viable,
        "target_mutation_raw_total": method_raw_total,
        "target_mutation_applicability_status": mutation_applicability_status,
        "mutation_error_hint": _mutation_error_hint(mutation_eval),
    }


def _mutation_run_status(row: dict) -> str:
    """Classify mutation-extra outcome without collapsing everything to failed."""
    if row.get("target_mutation_applicability_status") in {
        "no_applicable_mutants",
        "no_viable_mutants",
    }:
        return str(row.get("target_mutation_applicability_status"))
    coverage_status = str(row.get("coverage_status") or row.get("coverage_measurement_status") or "").strip()
    if coverage_status == "residual_generated_test_compile_failure":
        return "residual_generated_test_compile_failure"
    skipped = str(row.get("mutation_extra_round_skipped_reason") or "").strip()
    if skipped:
        if skipped in {"pitest_report_missing", "no_green_tests_for_pitest", "target_mutation_summary_missing"}:
            return skipped
        if skipped.startswith("pitest_exception:"):
            return "pitest_exception"
        return f"mutation_skipped:{skipped}"
    score = row.get("target_mutation_score")
    iterations_run = int(row.get("mutation_extra_iterations_run") or 0)
    delta = row.get("mutation_extra_delta")
    no_gain = str(row.get("mutation_no_gain_reason") or "").strip()
    if isinstance(score, (int, float)):
        if iterations_run > 0:
            if isinstance(delta, (int, float)) and float(delta) > 0.0:
                return "completed_improved"
            if no_gain:
                return f"completed_no_gain:{no_gain}"
            return "completed_no_gain"
        return "mutation_measured"
    if str(row.get("coverage_measurement_status") or "") in {"no_retained_tests", "coverage_missing"}:
        return str(row.get("coverage_measurement_status"))
    if int(row.get("process_rc") or 0) != 0:
        return "process_failed"
    return "mutation_not_measured"


def _streaming_validation_summary(data: dict) -> dict:
    report_summary = data.get("report_summary") if isinstance(data.get("report_summary"), dict) else {}
    counts = report_summary.get("counts") if isinstance(report_summary.get("counts"), dict) else {}
    roots = [data.get("validation_policy"), report_summary.get("validation_policy")]
    validation_policy = next((root for root in roots if isinstance(root, dict)), {})
    failed_details = validation_policy.get("streaming_validation_failed_details") or []
    if not isinstance(failed_details, list):
        failed_details = []
    failure_types: dict[str, int] = {}
    gate_reasons: dict[str, int] = {}
    for detail in failed_details:
        if not isinstance(detail, dict):
            continue
        failure_type = str(detail.get("failure_type") or "unknown")
        failure_types[failure_type] = failure_types.get(failure_type, 0) + 1
        gate = detail.get("coverage_gate") if isinstance(detail.get("coverage_gate"), dict) else {}
        reason = str(detail.get("reason") or gate.get("reason") or "")
        if reason:
            gate_reasons[reason] = gate_reasons.get(reason, 0) + 1
    return {
        "generated_artifact_file_count": counts.get("generated_artifact_file_count"),
        "retained_test_file_count": counts.get("retained_test_file_count"),
        "retained_target_positive_count": counts.get("retained_target_positive_count"),
        "streaming_retained_target_positive_count": counts.get(
            "streaming_retained_target_positive_count"
        ),
        "final_green_test_method_count": counts.get(
            "final_green_test_method_count"
        ),
        "final_suite_target_positive": counts.get("final_suite_target_positive"),
        "final_suite_green_test_count": counts.get("final_suite_green_test_count"),
        "final_suite_reconciled_after_streaming": counts.get(
            "final_suite_reconciled_after_streaming"
        ),
        "passing_no_target_count": counts.get("passing_no_target_count"),
        "coverage_positive_failed_count": counts.get("coverage_positive_failed_count"),
        "coverage_positive_repair_passed": counts.get("coverage_positive_repair_passed"),
        # Mechanism counters for the oracle-recovery reach work.  These existed
        # in the run JSON but were never carried into the batch report, so every
        # subset evaluation read them back as None and looked inert.
        "coverage_positive_repair_submitted": counts.get("coverage_positive_repair_submitted"),
        "coverage_positive_repair_rescued": counts.get("coverage_positive_repair_rescued"),
        "size_gated_round_skipped": counts.get("size_gated_round_skipped"),
        "ladder_failed_items_seen": counts.get("ladder_failed_items_seen"),
        "candidate_extraction_empty_count": counts.get("candidate_extraction_empty_count"),
        "target_positive_passed_count": validation_policy.get("candidate_passed_target_positive_count"),
        "streaming_validation_submitted": validation_policy.get("streaming_validation_submitted"),
        "streaming_validation_completed": validation_policy.get("streaming_validation_completed"),
        "streaming_validation_written_files": validation_policy.get("streaming_validation_written_files"),
        "streaming_validation_applied_candidates": validation_policy.get("streaming_validation_applied_candidates"),
        "streaming_validation_passed": validation_policy.get("streaming_validation_passed"),
        "streaming_validation_failed": validation_policy.get("streaming_validation_failed"),
        "combination_accepted_count": validation_policy.get("combination_accepted_count"),
        "combination_rejected_count": validation_policy.get("combination_rejected_count"),
        "combination_conflict_failed_candidates": validation_policy.get("combination_conflict_failed_candidates"),
        "streaming_validation_target_coverage_gate_rejected": validation_policy.get("streaming_validation_target_coverage_gate_rejected"),
        "streaming_validation_failed_detail_count": len(failed_details),
        "streaming_validation_failure_types": json.dumps(failure_types, sort_keys=True) if failure_types else "",
        "streaming_validation_gate_reasons": json.dumps(gate_reasons, sort_keys=True) if gate_reasons else "",
    }


def _mutation_summary_from_pitest_xml(data: dict, best_method: dict) -> dict:
    project_root = Path(str(data.get("project_root") or ""))
    class_fqn = str(best_method.get("class_fqn") or "").strip()
    method_name = str(best_method.get("method_name") or "").strip()
    descriptor = str(best_method.get("descriptor") or "").strip()
    if not project_root.exists() or not class_fqn or not method_name:
        return {}
    try:
        from tatg.evaluation.mutation import find_latest_pitest_xml, parse_pitest_xml_per_method_descriptor, parse_pitest_xml_per_method

        pit_xml = find_latest_pitest_xml(project_root)
        if pit_xml is None:
            return {}
        method_row = {}
        if descriptor:
            method_row = parse_pitest_xml_per_method_descriptor(pit_xml, class_fqn).get((method_name, descriptor), {})
        if not method_row:
            method_row = parse_pitest_xml_per_method(pit_xml, class_fqn).get(method_name, {})
        if not method_row:
            return {}
        return {
            "mutation_score": method_row.get("mutation_score"),
            "mutation_total": method_row.get("total"),
            "mutation_raw_total": method_row.get("raw_total"),
            "mutation_non_viable": method_row.get("non_viable"),
            "mutation_killed": method_row.get("killed"),
            "mutation_survived": method_row.get("survived"),
            "mutation_no_coverage": method_row.get("no_coverage"),
        }
    except Exception:
        return {}


def _mutation_error_hint(mutation_eval: dict) -> str:
    if not isinstance(mutation_eval, dict) or mutation_eval.get("status") in {None, "ok"}:
        return ""
    classified = str(mutation_eval.get("error_hint") or "").strip()
    if classified:
        return classified
    text = "\n".join(str(mutation_eval.get(key) or "") for key in ("stderr_tail", "stdout_tail", "reason"))
    lower = text.lower()
    if "junit 5" in lower and "plugin" in lower and "not installed" in lower:
        return "pitest_junit5_plugin_missing"
    if "could not run any tests" in lower:
        return "pitest_no_tests_ran"
    if "no test classes identified" in lower or "ran 0 tests" in lower:
        return "pitest_no_tests_executed"
    if "no_coverage" in lower or "no coverage" in lower:
        return "target_mutants_no_coverage"
    if "timeout" in lower:
        return "pitest_timeout"
    if "could not resolve" in lower or "could not find artifact" in lower:
        return "pitest_dependency_resolution"
    return "pitest_failed"


def _latest_obligation_summary(data: dict) -> dict:
    streaming_declared = _streaming_obligation_ids(data)
    for run in reversed(data.get("test_runs") or []):
        summary = run.get("coverage_obligation_satisfaction")
        if isinstance(summary, dict) and int(summary.get("obligation_count") or 0) > 0:
            declared = run.get("declared_coverage_obligation_ids")
            if not declared and streaming_declared:
                declared = streaming_declared
            return {
                "coverage_obligation_satisfied": summary.get("satisfied_count"),
                "coverage_obligation_total": summary.get("obligation_count"),
                "coverage_obligation_rate": summary.get("satisfaction_rate"),
                "coverage_obligation_declared_ids": ",".join(str(item) for item in declared) if isinstance(declared, list) else "",
                "coverage_obligation_streaming_ids": ",".join(streaming_declared),
            }
    return {
        "coverage_obligation_satisfied": None,
        "coverage_obligation_total": None,
        "coverage_obligation_rate": None,
        "coverage_obligation_declared_ids": "",
        "coverage_obligation_streaming_ids": ",".join(streaming_declared),
    }


def _streaming_obligation_ids(data: dict) -> list[str]:
    ids: set[str] = set()
    for root in [data.get("validation_policy"), (data.get("report_summary") or {}).get("validation_policy")]:
        if not isinstance(root, dict):
            continue
        for item in root.get("streaming_validation_passed_details") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("coverage_obligation_id") or "").strip()
            if re.fullmatch(r"O\d{1,3}", text):
                ids.add(text)
    return sorted(ids, key=lambda item: int(item[1:]) if item.startswith("O") and item[1:].isdigit() else 10**9)


def _read_json(json_path: Path) -> dict:
    data, _source = _read_json_with_source(json_path)
    return data


def _partial_json_path(json_path: Path) -> Path:
    return json_path.with_name(f"{json_path.stem}.json.partial")


def _read_json_with_source(json_path: Path) -> tuple[dict, str]:
    def _load(path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        except Exception:  # noqa: BLE001
            return {}
        return data if isinstance(data, dict) else {}

    final = _load(json_path)
    partial = _load(_partial_json_path(json_path))
    if final:
        final_status = str(final.get("coverage_measurement_status") or "").strip()
        final_is_timeout_artifact = final_status in {
            "method_watchdog_timeout",
            "pre_generation_process_failed",
        }
        partial_has_progress = bool(
            isinstance(partial.get("report_summary"), dict)
            or isinstance(partial.get("test_runs"), list)
            or isinstance(partial.get("validation_policy"), dict)
        )
        if final_is_timeout_artifact and partial_has_progress:
            data = dict(partial)
            data["partial_artifact_used"] = True
            data["final_artifact_status"] = final_status
            data["final_artifact_return_code"] = (final.get("final_validation") or {}).get("return_code")
            data.setdefault("final_artifact_missing", False)
            return data, "partial"
        return final, "final"
    if partial:
        data = dict(partial)
        data["partial_artifact_used"] = True
        data.setdefault("final_artifact_missing", not json_path.exists())
        return data, "partial"
    return {}, "missing"


def _json_artifact_source(json_path: Path) -> str:
    _data, source = _read_json_with_source(json_path)
    return source


def _artifact_exists_for_summary(json_path: Path) -> bool:
    return json_path.exists() or _partial_json_path(json_path).exists()


def _java_major_from_class_file_version(version: float | int | str | None) -> str:
    try:
        major = int(float(str(version)))
    except Exception:  # noqa: BLE001
        return ""
    if major >= 45:
        return str(major - 44)
    return ""


def _class_version_failure_detail(log_text: str) -> dict[str, object] | None:
    text = str(log_text or "")
    match = re.search(
        r"class file version\s+(?P<offending>[0-9]+(?:\.[0-9]+)?)"
        r".{0,240}?recognizes class file versions up to\s+(?P<runtime>[0-9]+(?:\.[0-9]+)?)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    offending = match.group("offending")
    runtime = match.group("runtime")
    return {
        "offending_class_file_version": offending,
        "runtime_max_class_file_version": runtime,
        "recommended_java_major": _java_major_from_class_file_version(offending),
        "runtime_java_major": _java_major_from_class_file_version(runtime),
    }


def _class_version_failure_reason(log_text: str) -> str:
    return "java_runtime_version_mismatch" if _class_version_failure_detail(log_text) else ""


def _api_preflight_failure_reason(log_text: str) -> str:
    text = str(log_text or "")
    lower = text.lower()
    if "api configuration error" not in lower and "api preflight failed" not in lower:
        return ""
    if "bad gateway" in lower or "http error 502" in lower:
        return "llm_api_bad_gateway"
    if "timed out" in lower or "timeout" in lower:
        return "llm_api_preflight_timeout"
    if "connection refused" in lower or "failed to establish" in lower:
        return "llm_api_connection_failed"
    return "llm_api_preflight_failed"


def _preflight_blocking_test_sources_from_log(
    log_path: Path,
    project_root: Path,
    already_stashed: set[Path] | None = None,
) -> list[Path]:
    """Return project-local src/test Java files that appear in a preflight compile log."""

    if not log_path.exists():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []

    project = project_root.resolve()
    stashed = {Path(p).as_posix() for p in (already_stashed or set())}
    seen: set[str] = set()
    found: list[Path] = []
    # Maven/javac diagnostics commonly render as:
    #   [ERROR] /abs/project/src/test/java/FooTest.java:[12,9] ...
    #   [ERROR] src/test/java/FooTest.java:[12,9] ...
    java_path_re = re.compile(r"(?P<path>(?:/[^\s:\[]+|(?:\./)?src/[^\s:\[]+)[^\s:\[]*\.java)(?=:\[|:|\s|$)")
    for match in java_path_re.finditer(text):
        raw = match.group("path").strip()
        if not raw:
            continue
        candidate = Path(raw)
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve()
                rel = resolved.relative_to(project)
            except (OSError, ValueError):
                continue
        else:
            rel = Path(raw[2:] if raw.startswith("./") else raw)
            resolved = (project / rel).resolve()
            try:
                rel = resolved.relative_to(project)
            except (OSError, ValueError):
                continue
        rel_posix = rel.as_posix()
        if not rel_posix.startswith("src/test/") or not rel_posix.endswith(".java"):
            continue
        if rel_posix in stashed:
            continue
        if rel_posix in seen:
            continue
        seen.add(rel_posix)
        found.append(rel)
    return found


def _preflight_blocking_category(blocking_sources: list[Path], project_root: Path) -> str:
    """Classify preflight blockers without changing project files."""

    if not blocking_sources:
        return ""
    for rel in blocking_sources:
        path = project_root / rel
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _looks_like_tatg_generated_test_pollution(text):
            return "residual_generated_test_compile_failure"
    return "project_native_preflight_failed"


def _preflight_isolation_retry_sources(
    *,
    json_out: Path,
    log_path: Path,
    project_root: Path,
    backed_up_paths: set[Path],
    managed_target_tests: set[Path],
    max_sources: int = 8,
) -> list[Path]:
    """Select unrelated test sources safe to quarantine for one preflight retry.

    Maven's ``-Dtest`` limits Surefire execution but does not limit
    ``testCompile``.  A broken project-native test can therefore prevent an
    otherwise isolated generated test from reaching generation.  This gate is
    deliberately evidence-only: every source must be named by the preflight
    artifact, exist in the byte-for-byte batch backup, and be unrelated to the
    target test managed by this run.
    """

    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    logged = _preflight_blocking_test_sources_from_log(log_path, project_root)
    if not logged:
        return []

    data = _safe_read_json(json_out)
    preflight: dict = {}
    declared_raw: list[object] = list(logged)
    if isinstance(data, dict):
        status = str(
            data.get("coverage_measurement_status")
            or data.get("coverage_missing_reason")
            or data.get("status")
            or ""
        ).strip()
        preflight_raw = data.get("preflight_failure")
        preflight = preflight_raw if isinstance(preflight_raw, dict) else {}
        if status and status not in {
            "pre_generation_process_failed",
            "preflight_failed",
            "project_native_preflight_failed",
        }:
            return []
        if bool(preflight.get("prompt_generated")):
            return []
        raw_sources = preflight.get("blocking_test_sources")
        if isinstance(raw_sources, list) and raw_sources:
            declared_raw = raw_sources
    elif "Preflight test-command check failed" not in log_text:
        # Early preflight aborts can leave only a partial artifact.  The
        # explicit runner marker plus project-local javac paths is sufficient
        # evidence; generation has not started at this point.
        return []

    declared: list[Path] = []
    for raw in declared_raw:
        rel = Path(str(raw or "").strip())
        if (
            not rel.as_posix().startswith("src/test/")
            or rel.suffix != ".java"
            or rel.is_absolute()
        ):
            return []
        if rel not in declared:
            declared.append(rel)
    if not declared or len(declared) > max(1, int(max_sources)):
        return []

    # Re-read the full log instead of trusting a truncated JSON tail.  Refuse
    # isolation when another project-local test source also appears in javac
    # diagnostics but was not captured in the artifact.
    if set(logged) != set(declared):
        return []

    managed = {Path(path) for path in managed_target_tests}
    backup = {Path(path) for path in backed_up_paths}
    selected: list[Path] = []
    for rel in declared:
        path = project_root / rel
        if rel in managed or rel not in backup or not path.is_file():
            return []
        selected.append(rel)
    return selected


def _archive_preflight_isolation_attempt(
    *,
    run_out_dir: Path,
    out_stem: str,
    paths: list[Path],
) -> Path:
    """Preserve the first preflight failure before a one-shot isolation retry."""

    archive_dir = run_out_dir / "preflight-isolation-attempt"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        if path.exists():
            shutil.copy2(path, archive_dir / path.name)
    return archive_dir


def _record_preflight_isolation_retry(
    artifact_path: Path,
    *,
    sources: list[Path],
    first_return_code: int,
    retry_return_code: int,
    first_attempt_archive: Path,
) -> None:
    data = _safe_read_json(artifact_path)
    if not isinstance(data, dict):
        return
    data["preflight_isolation_retry"] = {
        "attempted": True,
        "blocking_test_sources": [path.as_posix() for path in sources],
        "first_return_code": first_return_code,
        "retry_return_code": retry_return_code,
        "first_attempt_archive": str(first_attempt_archive),
        "restore_guard": "batch_finally_byte_for_byte",
    }
    artifact_path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


JAVA_KEYWORDS = frozenset({
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "var", "void", "volatile", "while",
})


def _java_type_to_descriptor(type_name: str) -> str:
    raw = re.sub(r"\s+", "", str(type_name or "").strip())
    if not raw:
        return "Ljava/lang/Object;"
    array_depth = raw.count("[]")
    raw = raw.replace("[]", "")
    raw = re.sub(r"<.*>", "", raw)
    primitive = {
        "void": "V",
        "boolean": "Z",
        "byte": "B",
        "char": "C",
        "short": "S",
        "int": "I",
        "long": "J",
        "float": "F",
        "double": "D",
    }.get(raw)
    if primitive:
        desc = primitive
    else:
        common = {
            "String": "java/lang/String",
            "CharSequence": "java/lang/CharSequence",
            "Appendable": "java/lang/Appendable",
            "Reader": "java/io/Reader",
            "Writer": "java/io/Writer",
            "Object": "java/lang/Object",
            "Integer": "java/lang/Integer",
            "Long": "java/lang/Long",
            "Boolean": "java/lang/Boolean",
            "Double": "java/lang/Double",
            "Float": "java/lang/Float",
            "List": "java/util/List",
            "Map": "java/util/Map",
            "Set": "java/util/Set",
        }.get(raw, raw.replace(".", "/"))
        desc = f"L{common};"
    return "[" * array_depth + desc


def _split_java_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    angle_depth = 0
    paren_depth = 0
    bracket_depth = 0
    for idx, ch in enumerate(str(text or "")):
        if ch == "<":
            angle_depth += 1
        elif ch == ">" and angle_depth > 0:
            angle_depth -= 1
        elif ch == "(":
            paren_depth += 1
        elif ch == ")" and paren_depth > 0:
            paren_depth -= 1
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]" and bracket_depth > 0:
            bracket_depth -= 1
        elif ch == "," and angle_depth == 0 and paren_depth == 0 and bracket_depth == 0:
            part = text[start:idx].strip()
            if part:
                parts.append(part)
            start = idx + 1
    tail = str(text or "")[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _java_type_variables_from_decl(text: str) -> set[str]:
    raw = str(text or "").strip()
    if raw.startswith("<") and raw.endswith(">"):
        raw = raw[1:-1]
    out: set[str] = set()
    for part in _split_java_top_level_commas(raw):
        name = re.split(r"\s+(?:extends|super)\s+", part.strip(), maxsplit=1)[0].strip()
        if re.fullmatch(r"[A-Z][A-Za-z0-9_]*", name):
            out.add(name)
    return out


def _target_class_type_variables(source: str, class_name: str) -> set[str]:
    pattern = re.compile(
        rf"\b(?:class|interface|enum|@interface)\s+{re.escape(class_name)}\s*(?P<vars><[^{{;]+>)?"
    )
    match = pattern.search(source or "")
    return _java_type_variables_from_decl(match.group("vars") if match else "")


def _nested_type_names_for_class(source: str, class_name: str) -> set[str]:
    body = _targeting_extract_type_body(source or "", class_name)
    if body is None:
        return set()
    return set(JAVA_TYPE_DECL_RE.findall(body))


def _direct_super_type_names(source: str, class_name: str) -> list[str]:
    """Return direct source-declared supertypes without guessing classpath types."""
    match = re.search(
        rf"\b(?:class|interface|enum)\s+{re.escape(class_name)}"
        r"(?:\s*<[^{};]+>)?\s*(?P<header>[^{{;]*)\{",
        source or "",
        flags=re.DOTALL,
    )
    if not match:
        return []
    header = str(match.group("header") or "")
    chunks: list[str] = []
    extends = re.search(
        r"\bextends\s+(?P<types>.*?)(?=\bimplements\b|$)",
        header,
        flags=re.DOTALL,
    )
    implements = re.search(
        r"\bimplements\s+(?P<types>.*)$",
        header,
        flags=re.DOTALL,
    )
    if extends:
        chunks.extend(_split_java_top_level_commas(extends.group("types")))
    if implements:
        chunks.extend(_split_java_top_level_commas(implements.group("types")))
    return [
        _strip_java_type_noise(item)
        for item in chunks
        if _strip_java_type_noise(item)
    ]


def _source_type_fqn(
    type_name: str,
    *,
    package_name: str,
    explicit_imports: dict[str, str],
) -> str:
    raw = _strip_java_type_noise(type_name)
    if not raw:
        return ""
    if raw in explicit_imports:
        return explicit_imports[raw]
    if "." in raw:
        first, rest = raw.split(".", 1)
        if first in explicit_imports:
            return f"{explicit_imports[first]}.{rest}"
        if first and first[0].islower():
            return raw
    return f"{package_name}.{raw}" if package_name else raw


def _java_source_root(source_path: Path) -> Path | None:
    parts = source_path.parts
    for idx in range(len(parts) - 2):
        if parts[idx : idx + 3] in (
            ("src", "main", "java"),
            ("src", "test", "java"),
        ):
            return Path(*parts[: idx + 3])
    return None


def _conflicting_tests_for_run(
    target: HardMethodTarget,
    *,
    retain_existing_tests: bool,
) -> list[str]:
    """Return tests to isolate without contradicting retained-test mode.

    The target's default test file is the intended structural baseline when
    ``--retain-existing-tests`` is explicit. Other configured conflicting
    tests remain isolated because they may share a Surefire selector or break
    project preflight independently of the target suite.
    """

    default_test = _default_test_file_for_target(target)
    conflicting = [
        path
        for path in target.conflicting_tests
        if not (retain_existing_tests and path == default_test)
    ]
    if (
        default_test
        and not retain_existing_tests
        and default_test not in conflicting
    ):
        conflicting.append(default_test)
    return conflicting


def _retained_test_seed(
    manifest_path: Path | None,
    *,
    target_key: str,
    expected_test_name: str,
    expected_target_descriptor: str | None = None,
    expected_source_file: Path | None = None,
    expected_class_name: str | None = None,
    expected_method_name: str | None = None,
) -> dict[str, Any] | None:
    """Resolve one explicitly mapped, immutable retained-test seed.

    The manifest is deliberately exact-keyed.  We do not scan archives or
    guess by class/method name because overloads and repeated historical runs
    can otherwise seed a different target's suite.
    """

    if manifest_path is None:
        return None
    payload = _safe_read_json(manifest_path)
    entries = payload.get("tests") if isinstance(payload.get("tests"), dict) else payload
    raw = entries.get(target_key) if isinstance(entries, dict) else None
    if raw is None:
        return None
    item = {"path": raw} if isinstance(raw, str) else dict(raw or {})
    raw_path = str(item.get("path") or "").strip()
    if not raw_path:
        raise ValueError(f"retained-test seed {target_key!r} has no path")
    source = Path(raw_path).expanduser()
    if not source.is_absolute():
        source = manifest_path.parent / source
    source = source.resolve()
    if not source.is_file() or source.suffix != ".java":
        raise ValueError(f"retained-test seed is not a Java file: {source}")
    if source.name != expected_test_name:
        raise ValueError(
            f"retained-test seed filename mismatch for {target_key}: "
            f"expected {expected_test_name}, got {source.name}"
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    expected_digest = str(item.get("sha256") or "").strip().lower()
    if expected_digest and expected_digest != digest:
        raise ValueError(
            f"retained-test seed checksum mismatch for {target_key}: "
            f"expected {expected_digest}, got {digest}"
        )
    seed: dict[str, Any] = {
        "path": str(source),
        "sha256": digest,
        "target_key": target_key,
    }
    proof_raw = item.get("coverage_proof")
    if isinstance(proof_raw, dict) and proof_raw:
        proof = dict(proof_raw)
        proof_path_raw = str(proof.get("artifact_path") or "").strip()
        proof_digest_expected = str(proof.get("artifact_sha256") or "").strip().lower()
        if not proof_path_raw or not proof_digest_expected:
            raise ValueError(
                f"retained-test seed coverage proof for {target_key} requires "
                "artifact_path and artifact_sha256"
            )
        proof_path = Path(proof_path_raw).expanduser()
        if not proof_path.is_absolute():
            proof_path = manifest_path.parent / proof_path
        proof_path = proof_path.resolve()
        if not proof_path.is_file() or proof_path.suffix != ".json":
            raise ValueError(
                f"retained-test seed coverage proof is not a JSON artifact: {proof_path}"
            )
        proof_digest = hashlib.sha256(proof_path.read_bytes()).hexdigest()
        if proof_digest != proof_digest_expected:
            raise ValueError(
                f"retained-test seed coverage proof checksum mismatch for {target_key}: "
                f"expected {proof_digest_expected}, got {proof_digest}"
            )
        artifact = _safe_read_json(proof_path)
        current_source = (
            expected_source_file.resolve()
            if expected_source_file is not None
            else None
        )
        expected_source_digest = str(proof.get("source_sha256") or "").strip().lower()
        if current_source is not None:
            if not expected_source_digest:
                raise ValueError(
                    f"retained-test seed coverage proof for {target_key} requires "
                    "source_sha256 so JaCoCo line evidence cannot drift"
                )
            current_source_digest = hashlib.sha256(current_source.read_bytes()).hexdigest()
            if current_source_digest != expected_source_digest:
                raise ValueError(
                    f"retained-test seed coverage proof source checksum mismatch for "
                    f"{target_key}: expected {expected_source_digest}, got "
                    f"{current_source_digest}"
                )
        descriptor = str(
            artifact.get("target_descriptor")
            or artifact.get("target_method_descriptor")
            or proof.get("target_descriptor")
            or ""
        ).strip()
        expected_descriptor = str(expected_target_descriptor or "").strip()
        if expected_descriptor and descriptor != expected_descriptor:
            raise ValueError(
                f"retained-test seed coverage proof descriptor mismatch for {target_key}: "
                f"expected {expected_descriptor}, got {descriptor or '<missing>'}"
            )
        line_coverage = artifact.get("target_line_coverage_display")
        if line_coverage is None:
            line_coverage = artifact.get("target_line_coverage")
        branch_coverage = artifact.get("target_branch_coverage_display")
        if branch_coverage is None:
            branch_coverage = artifact.get("target_branch_coverage")
        try:
            line_value = float(line_coverage)
        except (TypeError, ValueError):
            line_value = 0.0
        try:
            branch_value = float(branch_coverage)
        except (TypeError, ValueError):
            branch_value = 0.0
        target_positive_count = max(
            int(artifact.get("retained_target_positive_count") or 0),
            int(artifact.get("target_positive_passed_count") or 0),
            int(artifact.get("candidate_passed_target_positive_count") or 0),
        )
        final_validation = artifact.get("final_validation")
        final_green = not isinstance(final_validation, dict) or int(
            final_validation.get("return_code") or 0
        ) == 0
        if not final_green or target_positive_count <= 0 or line_value <= 0.0:
            raise ValueError(
                f"retained-test seed coverage proof for {target_key} is not a "
                "green target-positive measured artifact"
            )
        branch_frontier: list[dict[str, int]] = []
        provenance = artifact.get("artifact_provenance")
        evaluation_artifacts = (
            list(provenance.get("evaluation_artifacts") or [])
            if isinstance(provenance, dict)
            else []
        )
        xml_rows = [
            row
            for row in evaluation_artifacts
            if isinstance(row, dict)
            and str(row.get("snapshot_path") or "").endswith(".xml")
            and str(row.get("sha256") or "").strip()
        ]
        if xml_rows and expected_class_name and expected_method_name and descriptor:
            xml_row = xml_rows[0]
            xml_path = Path(str(xml_row["snapshot_path"])).expanduser().resolve()
            xml_expected_digest = str(xml_row["sha256"]).strip().lower()
            if xml_path.is_file():
                xml_digest = hashlib.sha256(xml_path.read_bytes()).hexdigest()
                if xml_digest != xml_expected_digest:
                    raise ValueError(
                        f"retained-test seed JaCoCo checksum mismatch for {target_key}: "
                        f"expected {xml_expected_digest}, got {xml_digest}"
                    )
                root = ET.parse(xml_path).getroot()
                expected_simple = str(expected_class_name).rsplit(".", 1)[-1]
                expected_source_name = (
                    current_source.name
                    if current_source is not None
                    else f"{expected_simple.split('$', 1)[0]}.java"
                )
                for package in root.findall("package"):
                    source_node = package.find(
                        f"sourcefile[@name='{expected_source_name}']"
                    )
                    if source_node is None:
                        continue
                    matching_method = False
                    method_start = 0
                    method_end = 2**31 - 1
                    for class_node in package.findall("class"):
                        class_binary_name = str(class_node.get("name") or "")
                        if not (
                            class_binary_name.endswith("/" + expected_simple)
                            or class_binary_name == expected_simple
                        ):
                            continue
                        class_methods = list(class_node.findall("method"))
                        target_method_node = next(
                            (
                                method
                                for method in class_methods
                                if str(method.get("name") or "") == expected_method_name
                                and str(method.get("desc") or "") == descriptor
                            ),
                            None,
                        )
                        matching_method = target_method_node is not None
                        if matching_method:
                            method_start = int(target_method_node.get("line") or 0)
                            later_starts = sorted(
                                int(method.get("line") or 0)
                                for method in class_methods
                                if int(method.get("line") or 0) > method_start
                            )
                            if later_starts:
                                method_end = later_starts[0] - 1
                            break
                    if not matching_method:
                        continue
                    for line in source_node.findall("line"):
                        line_number = int(line.get("nr") or 0)
                        if line_number < method_start or line_number > method_end:
                            continue
                        missed = int(line.get("mb") or 0)
                        covered = int(line.get("cb") or 0)
                        if missed <= 0:
                            continue
                        branch_frontier.append(
                            {
                                "line": line_number,
                                "branch_covered": covered,
                                "branch_missed": missed,
                            }
                        )
                    break
        seed["coverage_proof"] = {
            "status": "archive_target_positive_measured",
            "target_key": target_key,
            "target_descriptor": descriptor,
            "target_line_coverage": line_value,
            "target_branch_coverage": branch_value,
            "artifact_path": str(proof_path),
            "artifact_sha256": proof_digest,
            "source_sha256": expected_source_digest,
            "branch_frontier": branch_frontier,
        }
    return seed


def _record_retained_test_seed(json_path: Path, seed: dict[str, Any] | None) -> None:
    if not seed or not json_path.exists():
        return
    payload = _safe_read_json(json_path)
    payload["retained_test_seed"] = {
        "target_key": seed["target_key"],
        "source_path": seed["path"],
        "sha256": seed["sha256"],
    }
    if isinstance(seed.get("coverage_proof"), dict):
        payload["retained_test_seed"]["coverage_proof"] = dict(
            seed["coverage_proof"]
        )
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _source_path_for_type(
    *,
    configured_source_path: Path,
    type_fqn: str,
) -> Path | None:
    simple_name = type_fqn.rsplit(".", 1)[-1].split("$", 1)[0]
    same_package = configured_source_path.with_name(f"{simple_name}.java")
    if same_package.exists():
        return same_package
    source_root = _java_source_root(configured_source_path)
    if source_root is None:
        return None
    imported_path = source_root / Path(*type_fqn.split("."))
    imported_path = imported_path.with_suffix(".java")
    return imported_path if imported_path.exists() else None


def _nested_type_owners_for_target(
    target: HardMethodTarget,
    source: str,
) -> dict[str, str]:
    """Map inherited simple nested names to the class that declares them.

    Java permits a subclass to use an inherited member type by its simple name.
    The JVM descriptor, however, must use the declaring owner's binary name.
    """
    configured_source_path = (
        target.subjects_root.resolve() / target.project / target.target_file
    )
    owners: dict[str, str] = {}
    visited: set[str] = set()

    def visit(
        current_source: str,
        current_class: str,
        current_path: Path,
    ) -> None:
        package_name, explicit_imports, _ = _source_package_and_imports(current_source)
        current_fqn = (
            f"{package_name}.{current_class}" if package_name else current_class
        )
        if current_fqn in visited:
            return
        visited.add(current_fqn)
        for nested_name in _nested_type_names_for_class(
            current_source,
            current_class,
        ):
            owners.setdefault(nested_name, current_fqn)
        for super_name in _direct_super_type_names(current_source, current_class):
            super_fqn = _source_type_fqn(
                super_name,
                package_name=package_name,
                explicit_imports=explicit_imports,
            )
            if not super_fqn or super_fqn in visited:
                continue
            super_path = _source_path_for_type(
                configured_source_path=current_path,
                type_fqn=super_fqn,
            )
            if super_path is None:
                continue
            try:
                super_source = super_path.read_text(
                    encoding="utf-8",
                    errors="ignore",
                )
            except OSError:
                continue
            visit(
                super_source,
                super_fqn.rsplit(".", 1)[-1],
                super_path,
            )

    visit(source, target.java_class_name, configured_source_path)
    return owners


def _java_method_decl_pattern(method_name: str) -> re.Pattern[str]:
    return re.compile(
        r"(?m)(?:^|[;{}\n])\s*"
        r"(?:@\w+(?:\([^)]*\))?\s*)*"
        r"(?:(?:public|protected|private|static|final|synchronized|native|abstract|strictfp|default)\s+)*"
        r"(?P<typeparams><[^>{};]+>\s*)?"
        # A Java wildcard may occur inside generic arguments, but it cannot
        # begin a method return type. Excluding it here prevents a ternary
        # call such as "? handle(value)" from being expanded as an overload.
        r"(?P<return>[\w.$\[\]]+(?:\s*<[^;{}()]+>)?(?:\s*\[\])?)\s+"
        rf"(?P<name>{re.escape(method_name)})\s*\((?P<params>[^;{{}}]*)\)\s*(?:throws\s+[^;{{]+)?\s*(?:\{{|;)"
    )


def _detected_method_overload_rows(source: str, method_name: str) -> list[dict]:
    rows: list[dict] = []
    pattern = _java_method_decl_pattern(method_name)
    for match in pattern.finditer(source or ""):
        params_raw = str(match.group("params") or "").strip()
        param_types: list[str] = []
        if params_raw:
            for part in _split_java_top_level_commas(params_raw):
                type_token = _java_param_decl_type(part)
                if not type_token:
                    continue
                param_types.append(type_token)
        return_type = str(match.group("return") or "void").strip()
        if _strip_java_type_noise(return_type) in JAVA_KEYWORDS - {"void", "boolean", "byte", "char", "short", "int", "long", "float", "double"}:
            continue
        descriptor = "(" + "".join(_java_type_to_descriptor(p) for p in param_types) + ")" + _java_type_to_descriptor(return_type)
        rows.append({
            "method": method_name,
            "param_types": param_types,
            "return_type": return_type,
            "descriptor": descriptor,
        })
    return rows


def _source_text_for_target(target: HardMethodTarget) -> str:
    source_path = target.subjects_root.resolve() / target.project / target.target_file
    try:
        return source_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def _source_package_and_imports(source: str) -> tuple[str, dict[str, str], list[str]]:
    package_match = re.search(r"(?m)^\s*package\s+([\w.]+)\s*;", source or "")
    package = package_match.group(1) if package_match else ""
    explicit_imports: dict[str, str] = {}
    wildcard_imports: list[str] = []
    for match in re.finditer(
        r"(?m)^\s*import\s+(?P<static>static\s+)?"
        r"(?P<import>[\w.*]+)\s*;",
        source or "",
    ):
        # Static imports expose fields, methods, and (rarely) nested member
        # types; they do not name a package wildcard. Treating
        # ``import static Owner.*`` as a package import makes an unrelated
        # regular wildcard ambiguous and silently emits a wrong JVM
        # descriptor. Nested member types require owner-aware proof and are
        # deliberately left to the existing nested-type resolver.
        if match.group("static"):
            continue
        imp = match.group("import")
        if imp.endswith(".*"):
            wildcard_imports.append(imp[:-2])
        else:
            explicit_imports[imp.rsplit(".", 1)[-1]] = imp
    return package, explicit_imports, wildcard_imports


def _strip_java_type_noise(type_name: str) -> str:
    raw = _strip_java_annotations(str(type_name or ""))
    raw = re.sub(r"\b(?:final|volatile|transient)\b\s*", "", raw)
    raw = re.sub(r"<.*>", "", raw)
    raw = raw.replace("...", "[]")
    return re.sub(r"\s+", "", raw).strip()


def _java_param_decl_type(param_decl: str) -> str:
    text = _strip_java_annotations(str(param_decl or ""))
    text = re.sub(r"\b(?:final|volatile|transient)\b\s*", "", text)
    text = text.replace("...", "[]").strip()
    parts = text.split()
    if len(parts) >= 2:
        return " ".join(parts[:-1]).strip()
    return parts[0] if parts else ""


def _strip_java_annotations(text: str) -> str:
    """Remove Java annotations without leaking nested argument text.

    Parameter annotations may contain quoted punctuation and nested calls.
    A ``[^)]*`` regex stops at the first inner close parenthesis and can turn
    annotation prose into a fake Java parameter type.  This scanner removes
    the complete balanced annotation while preserving the declaration around
    it.  Malformed annotations are left in place so target selection fails
    visibly instead of silently inventing a descriptor.
    """

    source = str(text or "")
    out: list[str] = []
    index = 0
    while index < len(source):
        if source[index] != "@":
            out.append(source[index])
            index += 1
            continue
        name = re.match(r"@[A-Za-z_$][A-Za-z0-9_$.]*", source[index:])
        if name is None:
            out.append(source[index])
            index += 1
            continue
        end = index + name.end()
        cursor = end
        while cursor < len(source) and source[cursor].isspace():
            cursor += 1
        if cursor < len(source) and source[cursor] == "(":
            depth = 0
            quote = ""
            escaped = False
            scan = cursor
            while scan < len(source):
                char = source[scan]
                if quote:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == quote:
                        quote = ""
                elif char in {'"', "'"}:
                    quote = char
                elif char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0:
                        scan += 1
                        break
                scan += 1
            if depth != 0:
                out.append(source[index])
                index += 1
                continue
            end = scan
        while end < len(source) and source[end].isspace():
            end += 1
        out.append(" ")
        index = end
    return "".join(out)


@functools.lru_cache(maxsize=256)
def _implicit_java_lang_fqn(simple_name: str) -> str | None:
    """Resolve an implicitly imported ``java.lang`` type generically.

    A hand-maintained list misses legal types such as StringBuffer and causes
    descriptor-qualified coverage to reject tests that really hit the target.
    ``javap`` is deterministic and already part of the required JDK toolchain.
    """

    simple = str(simple_name or "").strip()
    if not re.fullmatch(r"[A-Z][A-Za-z0-9_$]*", simple):
        return None
    fqn = f"java.lang.{simple}"
    javap_candidates: list[str] = []
    java_home = _resolve_compatible_target_java_home(
        None,
        "8",
        env=dict(os.environ),
    )
    if java_home:
        javap_candidates.append(str(Path(java_home) / "bin" / "javap"))
    path_javap = shutil.which("javap")
    if path_javap and path_javap not in javap_candidates:
        javap_candidates.append(path_javap)

    for javap in javap_candidates:
        try:
            completed = subprocess.run(
                [javap, "-public", fqn],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if completed.returncode != 0:
            continue
        declaration = completed.stdout or ""
        if re.search(
            rf"\b(?:class|interface|enum|record)\s+"
            rf"{re.escape(fqn)}(?:\b|<)",
            declaration,
        ):
            return fqn
    return None


def _java_type_to_descriptor_in_source(
    type_name: str,
    *,
    package_name: str,
    explicit_imports: dict[str, str],
    wildcard_imports: list[str],
    enclosing_class: str = "",
    nested_type_names: set[str] | frozenset[str] = frozenset(),
    nested_type_owners: dict[str, str] | None = None,
    same_package_type_names: set[str] | frozenset[str] = frozenset(),
    type_variables: set[str] | frozenset[str] = frozenset(),
) -> str:
    raw = _strip_java_type_noise(type_name)
    if not raw:
        return "Ljava/lang/Object;"
    array_depth = raw.count("[]")
    raw = raw.replace("[]", "")
    if raw in type_variables:
        return "[" * array_depth + "Ljava/lang/Object;"
    primitive = {
        "void": "V",
        "boolean": "Z",
        "byte": "B",
        "char": "C",
        "short": "S",
        "int": "I",
        "long": "J",
        "float": "F",
        "double": "D",
    }.get(raw)
    if primitive:
        return "[" * array_depth + primitive
    common = {
        "String": "java.lang.String",
        "CharSequence": "java.lang.CharSequence",
        "Appendable": "java.lang.Appendable",
        "Object": "java.lang.Object",
        "Number": "java.lang.Number",
        "Byte": "java.lang.Byte",
        "Short": "java.lang.Short",
        "Integer": "java.lang.Integer",
        "Long": "java.lang.Long",
        "Boolean": "java.lang.Boolean",
        "Double": "java.lang.Double",
        "Float": "java.lang.Float",
        "Character": "java.lang.Character",
        "Void": "java.lang.Void",
        "Reader": "java.io.Reader",
        "Writer": "java.io.Writer",
        "Class": "java.lang.Class",
        "Iterable": "java.lang.Iterable",
        "Comparable": "java.lang.Comparable",
        "Enum": "java.lang.Enum",
        "Thread": "java.lang.Thread",
        "Throwable": "java.lang.Throwable",
        "Exception": "java.lang.Exception",
        "RuntimeException": "java.lang.RuntimeException",
        "Type": "java.lang.reflect.Type",
        "Method": "java.lang.reflect.Method",
        "List": "java.util.List",
        "Map": "java.util.Map",
        "Set": "java.util.Set",
        "Collection": "java.util.Collection",
        "Iterator": "java.util.Iterator",
    }
    # Java resolves types declared/imported by the compilation unit before
    # the implicit java.lang import. Keeping that order prevents a project
    # type such as com.acme.Number from being rewritten to java.lang.Number.
    if raw in explicit_imports:
        fqn = explicit_imports[raw]
    elif nested_type_owners and raw in nested_type_owners:
        fqn = f"{nested_type_owners[raw]}${raw}"
    elif raw in nested_type_names and enclosing_class:
        fqn = f"{package_name}.{enclosing_class}${raw}" if package_name else f"{enclosing_class}${raw}"
    elif raw in same_package_type_names:
        fqn = f"{package_name}.{raw}" if package_name else raw
    elif raw in common:
        fqn = common[raw]
    else:
        implicit_java_lang_fqn = _implicit_java_lang_fqn(raw)
        if implicit_java_lang_fqn:
            fqn = implicit_java_lang_fqn
        elif "." in raw:
            first = raw.split(".", 1)[0]
            rest = raw.split(".", 1)[1]
            if first in explicit_imports:
                fqn = f"{explicit_imports[first]}${rest.replace('.', '$')}"
            elif first and first[0].isupper():
                fqn = f"{package_name}.{raw.replace('.', '$')}" if package_name else raw.replace(".", "$")
            else:
                fqn = raw
        elif re.fullmatch(r"[A-Z]", raw):
            fqn = "java.lang.Object"
        else:
            non_jdk_wildcards = [item for item in wildcard_imports if not item.startswith("java.")]
            if len(non_jdk_wildcards) == 1:
                fqn = f"{non_jdk_wildcards[0]}.{raw}"
            elif package_name:
                fqn = f"{package_name}.{raw}"
            else:
                fqn = raw
    return "[" * array_depth + f"L{fqn.replace('.', '/')};"


def _source_overloads_for_target(target: HardMethodTarget, method_name: str) -> list[dict]:
    source = _source_text_for_target(target)
    package_name, explicit_imports, wildcard_imports = _source_package_and_imports(source)
    source_path = (
        target.subjects_root.resolve()
        / target.project
        / target.target_file
    )
    try:
        same_package_type_names = {
            path.stem
            for path in source_path.parent.glob("*.java")
            if path.is_file()
        }
    except OSError:
        same_package_type_names = set()
    class_body = _targeting_extract_type_body(source, target.java_class_name)
    method_source = (
        _targeting_strip_nested_type_bodies(class_body)
        if class_body is not None
        else source
    )
    class_type_variables = _target_class_type_variables(source, target.java_class_name)
    nested_type_names = _nested_type_names_for_class(source, target.java_class_name)
    nested_type_owners = _nested_type_owners_for_target(target, source)
    rows: list[dict] = []
    for match in _java_method_decl_pattern(method_name).finditer(method_source or ""):
        params_raw = str(match.group("params") or "").strip()
        param_types: list[str] = []
        if params_raw:
            for part in _split_java_top_level_commas(params_raw):
                type_token = _java_param_decl_type(part)
                if type_token:
                    param_types.append(type_token)
        return_type = str(match.group("return") or "void").strip()
        if _strip_java_type_noise(return_type) in JAVA_KEYWORDS - {"void", "boolean", "byte", "char", "short", "int", "long", "float", "double"}:
            continue
        method_type_variables = _java_type_variables_from_decl(str(match.group("typeparams") or ""))
        type_variables = class_type_variables | method_type_variables
        descriptor = (
            "("
            + "".join(
                _java_type_to_descriptor_in_source(
                    p,
                    package_name=package_name,
                    explicit_imports=explicit_imports,
                    wildcard_imports=wildcard_imports,
                    enclosing_class=target.java_class_name,
                    nested_type_names=nested_type_names,
                    nested_type_owners=nested_type_owners,
                    same_package_type_names=same_package_type_names,
                    type_variables=type_variables,
                )
                for p in param_types
            )
            + ")"
            + _java_type_to_descriptor_in_source(
                return_type,
                package_name=package_name,
                explicit_imports=explicit_imports,
                wildcard_imports=wildcard_imports,
                enclosing_class=target.java_class_name,
                nested_type_names=nested_type_names,
                nested_type_owners=nested_type_owners,
                same_package_type_names=same_package_type_names,
                type_variables=type_variables,
            )
        )
        rows.append({
            "method": method_name,
            "param_types": param_types,
            "return_type": return_type,
            "descriptor": descriptor,
            "overload_short": _jvm_desc_to_short(descriptor),
        })
    if not rows:
        rows = [
            {
                **row,
                "overload_short": _jvm_desc_to_short(str(row.get("descriptor") or "")),
            }
            for row in _detected_method_overload_rows(source, method_name)
        ]
    return rows


def _candidate_source_files_for_selector(
    *,
    project_root: Path,
    class_name: str,
    method_name: str,
    expected_source: Path,
    limit: int = 12,
) -> list[dict]:
    candidates: list[dict] = []
    search_roots = [project_root / "src"] if (project_root / "src").exists() else [project_root]
    for base in search_roots:
        for path in sorted(base.rglob("*.java")):
            path_str = str(path)
            if "/target/" in path_str or "/.git/" in path_str:
                continue
            try:
                source = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            try:
                detected_classes = _extract_classes(path, source)
                detected_methods = _targeting_extract_methods(path, source, class_name)
            except Exception:
                continue
            class_match = class_name in detected_classes
            method_match = method_name in detected_methods
            if not class_match and not method_match:
                continue
            candidates.append({
                "source_file": str(path),
                "relative_source_file": str(path.relative_to(project_root)) if path.is_relative_to(project_root) else str(path),
                "is_configured_target_file": path.resolve() == expected_source.resolve(),
                "class_match": class_match,
                "method_match": method_match,
                "detected_classes": detected_classes[:12],
                "near_miss_methods": difflib.get_close_matches(method_name, detected_methods, n=8, cutoff=0.55),
            })
            if len(candidates) >= limit:
                return candidates
    return candidates


def _match_explicit_overload(overloads: list[dict], selector: str) -> dict | None:
    wanted = str(selector or "").strip().lstrip("#")
    if not wanted:
        return None
    normalized = re.sub(r"[^A-Za-z0-9$]+", "_", wanted.strip("()")).strip("_")
    for row in overloads:
        descriptor = str(row.get("descriptor") or "")
        short = str(row.get("overload_short") or _jvm_desc_to_short(descriptor))
        leaf_short = ",".join(part.rsplit("$", 1)[-1] for part in short.split(","))
        candidates = {
            descriptor,
            short,
            leaf_short,
            re.sub(r"[^A-Za-z0-9$]+", "_", short).strip("_"),
            re.sub(r"[^A-Za-z0-9$]+", "_", short.replace(",", "_")).strip("_"),
            re.sub(r"[^A-Za-z0-9$]+", "_", leaf_short.replace(",", "_")).strip("_"),
        }
        if wanted in candidates or normalized in candidates:
            return row
    return None


def _descriptor_tasks_for_method(
    class_name: str,
    target: HardMethodTarget,
    method_name: str,
) -> list[HardMethodRunTask]:
    base_method, sep, explicit_selector = method_name.partition("#")
    if sep:
        overloads = _source_overloads_for_target(target, base_method)
        matched = _match_explicit_overload(overloads, explicit_selector)
        if matched:
            return [HardMethodRunTask(class_name, target, base_method, str(matched.get("descriptor") or ""))]
        return [HardMethodRunTask(class_name, target, base_method, None)]
    method_name = base_method
    explicit_descriptor = str(target.target_descriptors.get(method_name) or "").strip()
    if explicit_descriptor:
        return [HardMethodRunTask(class_name, target, method_name, explicit_descriptor)]

    rows = _source_overloads_for_target(target, method_name)
    descriptors: list[str] = []
    for row in rows:
        descriptor = str(row.get("descriptor") or "").strip()
        if descriptor and descriptor not in descriptors:
            descriptors.append(descriptor)
    if len(descriptors) == 1:
        # Preserve exact method identity before the first coverage run so
        # TATG can resolve descriptor-referenced support types immediately.
        return [
            HardMethodRunTask(
                class_name,
                target,
                method_name,
                descriptors[0],
            )
        ]
    if not descriptors:
        return [HardMethodRunTask(class_name, target, method_name, None)]
    return [
        HardMethodRunTask(class_name, target, method_name, descriptor)
        for descriptor in descriptors
    ]


def _expand_method_tasks(selected: dict[str, tuple[str, ...]]) -> list[HardMethodRunTask]:
    tasks: list[HardMethodRunTask] = []
    for class_name, methods in selected.items():
        target = _TARGETS_BY_NAME.get(class_name)
        if target is None:
            continue
        for method_name in methods:
            tasks.extend(_descriptor_tasks_for_method(class_name, target, method_name))
    return tasks


def _write_minimal_failure_json(
    *,
    json_out: Path,
    target: HardMethodTarget,
    method_name: str,
    project_root: Path,
    command: list[str],
    process_returncode: int | None,
    failure_reason: str,
    log_path: Path,
    target_descriptor: str | None,
) -> None:
    if json_out.exists():
        return
    partial_path = _partial_json_path(json_out)
    partial_snapshot: dict = {}
    if partial_path.exists():
        try:
            loaded = json.loads(partial_path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(loaded, dict):
                partial_snapshot = loaded
        except Exception:  # noqa: BLE001
            partial_snapshot = {}
    log_tail = ""
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            log_tail = "\n".join(lines[-80:])
        except Exception:  # noqa: BLE001
            log_tail = ""
    normalized_failure_reason = (
        _class_version_failure_reason(log_tail)
        or _api_preflight_failure_reason(log_tail)
        or failure_reason
    )
    class_version_detail = _class_version_failure_detail(log_tail)
    preflight_blocking_sources = _preflight_blocking_test_sources_from_log(log_path, project_root)
    preflight_blocking_category = ""
    if normalized_failure_reason == "pre_generation_process_failed" and preflight_blocking_sources:
        preflight_blocking_category = _preflight_blocking_category(preflight_blocking_sources, project_root)
        if preflight_blocking_category:
            normalized_failure_reason = preflight_blocking_category
    detected_classes: list[str] = []
    detected_methods: list[str] = []
    detected_method_overloads: list[dict] = []
    candidate_source_files: list[dict] = []
    near_miss_methods: list[str] = []
    source_path = project_root / target.target_file
    skip_expensive_diagnostics = normalized_failure_reason in {
        "startup_artifact_timeout",
        "method_watchdog_timeout",
        "generation_worker_stalled",
    }
    try:
        from tatg.targeting import _extract_classes, _extract_methods  # type: ignore[attr-defined]

        source = source_path.read_text(encoding="utf-8", errors="ignore")
        if skip_expensive_diagnostics:
            detected_classes = []
            detected_methods = []
        else:
            detected_classes = _extract_classes(source_path, source)
            detected_methods = _extract_methods(source_path, source, target.java_class_name)
            detected_method_overloads = _source_overloads_for_target(target, method_name) or _detected_method_overload_rows(source, method_name)
            if method_name not in detected_methods:
                near_miss_methods = difflib.get_close_matches(method_name, detected_methods, n=8, cutoff=0.55)
            candidate_source_files = _candidate_source_files_for_selector(
                project_root=project_root,
                class_name=target.java_class_name,
                method_name=method_name,
                expected_source=source_path,
            )
    except Exception:  # noqa: BLE001
        detected_classes = []
        detected_methods = []
        detected_method_overloads = []
        candidate_source_files = []
        near_miss_methods = []
    descriptor_values = {
        str(item.get("descriptor") or "")
        for item in detected_method_overloads
        if str(item.get("descriptor") or "")
    }
    if normalized_failure_reason == "target_resolution_failed":
        if method_name in detected_methods and target_descriptor and descriptor_values and target_descriptor not in descriptor_values:
            normalized_failure_reason = "target_descriptor_mismatch"
        elif source_path.exists() and detected_classes and target.java_class_name not in detected_classes:
            normalized_failure_reason = "source_path_mismatch"
        elif (not source_path.exists()) or (
            method_name not in detected_methods
            and any(item.get("method_match") for item in candidate_source_files if not item.get("is_configured_target_file"))
        ):
            normalized_failure_reason = "source_path_mismatch"
    coverage_status = {
        "coverage_missing_reason": normalized_failure_reason,
        "target_line_coverage_display": 0.0,
        "target_branch_coverage_display": 0.0,
        "coverage_measurement_status": normalized_failure_reason,
        "raw_target_line_coverage": None,
        "raw_target_branch_coverage": None,
    }
    payload = {
        "status": normalized_failure_reason,
        "stage": normalized_failure_reason,
        "partial_artifact_used": bool(partial_snapshot),
        "partial_stage": partial_snapshot.get("stage") if partial_snapshot else None,
        "partial_artifact_prior_stage": partial_snapshot.get("stage") if partial_snapshot else None,
        "project_root": str(project_root),
        "source_file": str(source_path),
        "class_name": target.java_class_name,
        "method_name": method_name,
        "target_descriptor": target_descriptor,
        "generated_test_count": 0,
        "return_codes": [process_returncode] if process_returncode is not None else [],
        "final_validation": {"return_code": process_returncode, "failed_test_count": None},
        "report_summary": {
            "coverage": coverage_status,
            "counts": {"generated_artifact_file_count": 0, "retained_test_file_count": 0},
        },
        "coverage_measurement_status": normalized_failure_reason,
        "target_line_coverage_display": 0.0,
        "target_branch_coverage_display": 0.0,
        "target_resolution": {
            "selector": f"{target.java_class_name}.{method_name}",
            "source_file": str(source_path),
            "source_file_exists": source_path.exists(),
            "class_filter": target.java_class_name,
            "method_filter": method_name,
            "detected_classes": detected_classes,
            "detected_methods": detected_methods,
            "method_filter_matched": method_name in detected_methods,
            "detected_method_overloads": detected_method_overloads,
            "detected_method_descriptors": detected_method_overloads,
            "near_miss_methods": near_miss_methods,
            "candidate_source_files": candidate_source_files,
            "failure_reason": normalized_failure_reason,
        },
        "preflight_failure": {
            "category": normalized_failure_reason,
            "command": command,
            "return_code": process_returncode,
            "log_tail": log_tail,
            "prompt_generated": False,
            "blocking_test_sources": [path.as_posix() for path in preflight_blocking_sources],
            "blocking_test_source_category": preflight_blocking_category,
        },
    }
    if skip_expensive_diagnostics:
        payload["target_resolution"]["diagnostics_skipped_reason"] = normalized_failure_reason
    if class_version_detail:
        payload["preflight_failure"]["class_version_mismatch"] = class_version_detail
    if partial_snapshot:
        payload["partial_artifact_snapshot"] = {
            key: partial_snapshot.get(key)
            for key in (
                "status",
                "stage",
                "updated_at",
                "generation_checkpoint",
                "generation_checkpoints",
                "validation_policy",
            )
            if key in partial_snapshot
        }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _target_method_coverage(json_path: Path, method_name: str, target_descriptor: str | None = None) -> str:
    data, source = _read_json_with_source(json_path)
    if not data:
        return "target_method_cov=json_missing"
    coverage = ((data.get("report_summary") or {}).get("coverage") or {})
    artifact = " artifact=partial" if source == "partial" else ""
    target_row = coverage.get("target_method") if isinstance(coverage.get("target_method"), dict) else None
    if target_row and str(target_row.get("method_name") or "") == method_name:
        if not target_descriptor or str(target_row.get("descriptor") or "") == target_descriptor:
            return (
                f"target_method_cov=line={target_row.get('line_coverage')} "
                f"branch={target_row.get('branch_coverage')} "
                f"line_missed={target_row.get('line_missed')} "
                f"branch_missed={target_row.get('branch_missed')} "
                f"descriptor={target_row.get('descriptor')}{artifact}"
            )
    runs = data.get("test_runs") or []
    methods = []
    for run in runs:
        methods.extend(run.get("methods_coverage") or [])
    matches = [m for m in methods if m.get("method_name") == method_name]
    if target_descriptor:
        matches = [m for m in matches if m.get("descriptor") == target_descriptor]
    if not matches:
        candidate_best = _candidate_best_target_coverage(data, method_name, target_descriptor)
        if candidate_best:
            return (
                f"target_method_cov=candidate_evidence "
                f"line={candidate_best.get('line_coverage')} "
                f"branch={candidate_best.get('branch_coverage')} "
                f"line_missed={candidate_best.get('line_missed')} "
                f"branch_missed={candidate_best.get('branch_missed')} "
                f"descriptor={candidate_best.get('descriptor')}{artifact}"
            )
        return f"target_method_cov=missing{artifact}"
    best = max(
        matches,
        key=lambda item: float(item.get("line_coverage") or 0.0) + float(item.get("branch_coverage") or 0.0),
    )
    return (
        f"target_method_cov=line={best.get('line_coverage')} "
        f"branch={best.get('branch_coverage')} "
        f"line_missed={best.get('line_missed')} "
        f"branch_missed={best.get('branch_missed')} "
        f"descriptor={best.get('descriptor')}{artifact}"
    )


def _rewrite_post_eval_coverage_for_display_scope(json_path: Path, message: str) -> str:
    """Use method-scope display coverage in terminal summaries when available.

    The JSON keeps report-level ``line_coverage`` / ``branch_coverage`` for
    compatibility.  Batch stdout is user-facing, so for method runs it should
    prefer the explicit display coverage chosen by tatg.cli.
    """
    try:
        data = _read_json(json_path)
    except Exception:
        return message
    coverage = ((data.get("report_summary") or {}).get("coverage") or {})
    if coverage.get("display_scope") != "target_method":
        return message
    display_line = coverage.get("display_line_coverage")
    display_branch = coverage.get("display_branch_coverage")
    report_line = coverage.get("line_coverage")
    report_branch = coverage.get("branch_coverage")
    if display_line is None and display_branch is None:
        return message
    rewritten = message
    if display_line is not None:
        if re.search(r"(?<!report_)line=[^\s]+", rewritten):
            rewritten = re.sub(r"(?<!report_)line=[^\s]+", f"line={display_line}", rewritten, count=1)
        else:
            rewritten = f"{rewritten} line={display_line}"
    if display_branch is not None:
        if re.search(r"(?<!report_)branch=[^\s]+", rewritten):
            rewritten = re.sub(r"(?<!report_)branch=[^\s]+", f"branch={display_branch}", rewritten, count=1)
        else:
            rewritten = f"{rewritten} branch={display_branch}"
    if "coverage_scope=" not in rewritten:
        rewritten = f"{rewritten} coverage_scope=target_method"
    if report_line is not None and "report_line=" not in rewritten:
        rewritten = f"{rewritten} report_line={report_line}"
    if report_branch is not None and "report_branch=" not in rewritten:
        rewritten = f"{rewritten} report_branch={report_branch}"
    return rewritten


def _coverage_status(row: dict) -> str:
    """Classify a method result for report triage."""
    if not _artifact_exists_for_summary(Path(str(row.get("json_path") or ""))):
        return "no_json"
    if row.get("partial_artifact_used") and row.get("final_artifact_missing"):
        if not isinstance(row.get("target_line_coverage"), (int, float)) and not isinstance(row.get("target_branch_coverage"), (int, float)):
            stage = str(row.get("partial_stage") or "").strip()
            return f"partial_artifact_incomplete:{stage}" if stage else "partial_artifact_incomplete"
    reported_measurement_status = str(
        row.get("reported_coverage_measurement_status")
        or row.get("json_coverage_measurement_status")
        or ""
    ).strip()
    if (
        reported_measurement_status
        and not isinstance(row.get("target_line_coverage"), (int, float))
        and not isinstance(row.get("target_branch_coverage"), (int, float))
        and not isinstance(row.get("candidate_best_target_line_coverage"), (int, float))
        and not isinstance(row.get("candidate_best_target_branch_coverage"), (int, float))
        and not isinstance(row.get("candidate_failed_best_target_line_coverage"), (int, float))
        and not isinstance(row.get("candidate_failed_best_target_branch_coverage"), (int, float))
        and reported_measurement_status not in {"measured", "measured_partial_artifact"}
    ):
        return reported_measurement_status
    missing_reason = str(row.get("coverage_missing_reason") or "").strip()
    if missing_reason in {
        "candidate_extraction_empty",
        "final_validation_no_tests_run",
        "target_resolution_failed",
        "pre_generation_process_failed",
        "method_watchdog_timeout",
        "generation_worker_stalled",
        "preflight_failed",
        "coverage_report_not_generated",
        "no_target_positive_candidate",
        "no_retained_tests",
    } and not (
        isinstance(row.get("candidate_best_target_line_coverage"), (int, float))
        or isinstance(row.get("candidate_best_target_branch_coverage"), (int, float))
    ):
        return missing_reason
    line = row.get("target_line_coverage")
    branch = row.get("target_branch_coverage")
    has_line = isinstance(line, (int, float))
    has_branch = isinstance(branch, (int, float))
    if not has_line and not has_branch:
        candidate_line = row.get("candidate_best_target_line_coverage")
        candidate_branch = row.get("candidate_best_target_branch_coverage")
        has_candidate_measurement = isinstance(candidate_line, (int, float)) or isinstance(candidate_branch, (int, float))
        if has_candidate_measurement:
            retained_target_positive = int(row.get("retained_target_positive_count") or 0)
            retained_files = int(row.get("retained_test_file_count") or 0)
            if retained_target_positive > 0 or retained_files > 0:
                return "covered_with_candidate_evidence"
            if int(row.get("combination_rejected_count") or 0) > 0 or row.get("combination_conflict_failed_candidates"):
                return "combination_rejected_best_snapshot_only"
            return "best_snapshot_only"
        if isinstance(row.get("candidate_failed_best_target_line_coverage"), (int, float)) or isinstance(row.get("candidate_failed_best_target_branch_coverage"), (int, float)):
            return "coverage_positive_failed_best_candidate"
        if int(row.get("streaming_validation_submitted") or 0) > 0:
            # Method-level streaming can submit and reject many candidates
            # while the legacy generated/retained counters stay at zero.  Keep
            # that as a lifecycle/validation blocker instead of collapsing it
            # to a generic "no retained tests" status.
            if reported_measurement_status and reported_measurement_status not in {"measured", "measured_partial_artifact"}:
                return reported_measurement_status
            return "streaming_candidates_final_coverage_missing"
        retained_target_positive = row.get("retained_target_positive_count")
        retained_files = row.get("retained_test_file_count")
        has_retained_counters = retained_target_positive is not None or retained_files is not None
        if has_retained_counters and int(retained_target_positive or 0) <= 0 and int(retained_files or 0) <= 0:
            if int(row.get("passing_no_target_count") or 0) > 0:
                return "no_target_positive_candidate"
            return "no_retained_tests"
        if int(row.get("generated_test_count") or 0) == 0:
            # Streaming/deferred validation can submit method-level candidates
            # even when the legacy generated artifact counter remains zero.
            # Keep that distinct from a truly empty generation.
            if int(row.get("streaming_validation_submitted") or 0) > 0:
                return "streaming_candidates_final_coverage_missing"
            return "no_retained_tests"
        # If mutation score > 0 the method is exercised but JaCoCo can't
        # attribute line coverage (e.g. package-private, inlined, or indirect
        # call path).  Report as coverage_proxy_only so the metric is not
        # silently discarded.
        mutation_score = row.get("target_mutation_score")
        if isinstance(mutation_score, (int, float)) and float(mutation_score) > 0.0:
            return "coverage_proxy_only"
        return "coverage_missing"
    line_value = float(line) if has_line else 0.0
    branch_value = float(branch) if has_branch else 0.0
    if line_value == 0.0 and branch_value == 0.0:
        if (
            reported_measurement_status
            and reported_measurement_status not in {"measured", "measured_partial_artifact", "no_target_hit"}
            and int(row.get("retained_target_positive_count") or 0) <= 0
            and (
                int(row.get("candidate_target_positive_count") or 0) > 0
                or int(row.get("coverage_positive_failed_count") or 0) > 0
                or int(row.get("streaming_validation_submitted") or 0) > 0
            )
        ):
            return reported_measurement_status
        return "no_target_hit"
    if (has_line and line_value < 0.75) or (has_branch and branch_value < 0.75):
        return "low_target_coverage"
    if (row.get("pipeline_rc") or 0) != 0 or (row.get("process_rc") or 0) != 0:
        return "covered_with_intermediate_failures"
    return "clean_covered"


def _coverage_measurement_status(row: dict) -> str:
    status = str(row.get("coverage_status") or "").strip()
    reported_measurement_status = str(
        row.get("reported_coverage_measurement_status")
        or row.get("json_coverage_measurement_status")
        or ""
    ).strip()
    has_numeric_target = isinstance(row.get("target_line_coverage"), (int, float)) or isinstance(row.get("target_branch_coverage"), (int, float))
    target_line_value = float(row.get("target_line_coverage") or 0.0)
    target_branch_value = float(row.get("target_branch_coverage") or 0.0)
    if (
        has_numeric_target
        and target_line_value == 0.0
        and target_branch_value == 0.0
        and reported_measurement_status
        and reported_measurement_status not in {"measured", "measured_partial_artifact", "no_target_hit"}
    ):
        return reported_measurement_status
    if has_numeric_target:
        return "measured_partial_artifact" if row.get("partial_artifact_used") else "measured"
    if isinstance(row.get("candidate_best_target_line_coverage"), (int, float)) or isinstance(row.get("candidate_best_target_branch_coverage"), (int, float)):
        if status in {"combination_rejected_best_snapshot_only", "combination_rejected_best_snapshot_measured"}:
            return "combination_rejected_best_snapshot_measured"
        return "best_snapshot_only_measured"
    if isinstance(row.get("candidate_failed_best_target_line_coverage"), (int, float)) or isinstance(row.get("candidate_failed_best_target_branch_coverage"), (int, float)):
        return "coverage_positive_failed_best_candidate"
    if (
        reported_measurement_status
        and reported_measurement_status not in {"measured", "measured_partial_artifact"}
        and float(row.get("target_line_coverage") or 0.0) == 0.0
        and float(row.get("target_branch_coverage") or 0.0) == 0.0
        and int(row.get("retained_target_positive_count") or 0) <= 0
        and (
            int(row.get("candidate_target_positive_count") or 0) > 0
            or int(row.get("coverage_positive_failed_count") or 0) > 0
            or int(row.get("streaming_validation_submitted") or 0) > 0
        )
    ):
        return reported_measurement_status
    if reported_measurement_status and reported_measurement_status not in {"measured", "measured_partial_artifact"}:
        return reported_measurement_status
    if status.startswith("partial_artifact_incomplete"):
        return status
    if status == "no_json":
        return "artifact_missing"
    if status in {"clean_covered", "covered_with_intermediate_failures", "low_target_coverage", "no_target_hit", "coverage_proxy_only"}:
        return status
    return status or str(row.get("coverage_missing_reason") or "coverage_missing")


def _coverage_display_values(row: dict) -> tuple[float, float]:
    line = row.get("target_line_coverage")
    branch = row.get("target_branch_coverage")
    candidate_line = row.get("candidate_best_target_line_coverage")
    candidate_branch = row.get("candidate_best_target_branch_coverage")
    best_snapshot_status = str(row.get("coverage_measurement_status") or "") in {
        "best_snapshot_only_measured",
        "combination_rejected_best_snapshot_measured",
    }
    line_display = (
        float(candidate_line) if best_snapshot_status and isinstance(candidate_line, (int, float))
        else float(line) if isinstance(line, (int, float))
        else float(candidate_line) if isinstance(candidate_line, (int, float))
        else 0.0
    )
    branch_display = (
        float(candidate_branch) if best_snapshot_status and isinstance(candidate_branch, (int, float))
        else float(branch) if isinstance(branch, (int, float))
        else float(candidate_branch) if isinstance(candidate_branch, (int, float))
        else 0.0
    )
    return line_display, branch_display


def _reproducing_target_branch_counts(row: dict) -> tuple[int, int] | None:
    """Recover the target's branch counts from a source that reproduces the ratio.

    ``target_branch_covered``/``target_branch_missed`` are lifted from
    ``_best_method_row``, which matches only against
    ``test_runs[*].methods_coverage``.  When that lookup misses -- most often
    because the target descriptor was never resolved -- the counts are dropped
    even though ``target_branch_coverage`` itself was measured and the counts
    are present in the report's own target-method block.  On the 141-method
    benchmark that lost the counts on 76 targets, which reported them as
    ``coverage_unmeasured`` and made paired A/B decisions unreliable.

    The guard is the ratio: a candidate source is accepted only when
    ``covered / (covered + missed)`` reproduces the already-trusted
    ``target_branch_coverage``.  That proves the counts are the ones that
    produced the trusted number without relying on descriptor formatting,
    which is exactly what is unreliable here.  A source carrying an explicit
    descriptor that contradicts a known target descriptor is rejected outright
    so a sibling overload can never be attributed to the target.
    """

    trusted = row.get("target_branch_coverage")
    if not isinstance(trusted, (int, float)) or isinstance(trusted, bool):
        return None
    target_descriptor = str(row.get("target_descriptor") or "").strip()
    summary = row.get("iteration_target_method_coverage_summary")
    candidates = [
        row.get("final_method_coverage"),
        (summary or {}).get("best") if isinstance(summary, dict) else None,
    ]
    for source in candidates:
        if not isinstance(source, dict):
            continue
        covered = source.get("branch_covered")
        missed = source.get("branch_missed")
        if not isinstance(covered, int) or not isinstance(missed, int):
            continue
        total = covered + missed
        if total <= 0:
            continue
        source_descriptor = str(source.get("descriptor") or "").strip()
        if target_descriptor and source_descriptor and source_descriptor != target_descriptor:
            continue
        if abs((covered / total) - float(trusted)) > 1e-6:
            continue
        return covered, missed
    return None


def _branch_measurement_status(row: dict) -> str:
    """Distinguish measured branchless methods from unmeasured coverage.

    JaCoCo reports ``0/0`` for a method with no branch instructions.  The
    numeric compatibility display remains ``0.0``, but consumers must not
    interpret that value as measured 0% branch coverage.
    """

    measurement_status = str(row.get("coverage_measurement_status") or "")
    measured = measurement_status in {
        "measured",
        "measured_partial_artifact",
        "best_snapshot_only_measured",
        "combination_rejected_best_snapshot_measured",
    }
    if not measured:
        return "coverage_unmeasured"
    branch_covered = row.get("target_branch_covered")
    branch_missed = row.get("target_branch_missed")
    if not (
        isinstance(branch_covered, (int, float))
        and isinstance(branch_missed, (int, float))
    ):
        recovered = _reproducing_target_branch_counts(row)
        if recovered is None:
            return "coverage_unmeasured"
        branch_covered, branch_missed = recovered
        row["target_branch_covered"] = branch_covered
        row["target_branch_missed"] = branch_missed
        row["target_branch_counts_source"] = "ratio_reproducing_report_target_method"
    if int(branch_covered) + int(branch_missed) == 0:
        return "not_applicable_no_branches"
    return "measured"


def _classify_failure_text(text: str) -> str:
    lower = str(text or "").lower()
    if "illegal character" in lower and ("```" in lower or "illegal character: '`'" in lower):
        return "residual_generated_test_compile_failure"
    if "unterminated string" in lower or "invalid json" in lower or "json_patch_rejected" in lower:
        return "llm_output_truncated_or_invalid_json"
    if "cannot find symbol" in lower:
        if " class " in lower:
            return "unavailable_type"
        if " method " in lower:
            return "missing_known_method"
        return "compile_blocker"
    if "actual and formal argument lists differ" in lower or "cannot be applied to given types" in lower:
        return "wrong_overload_or_arity"
    if "incompatible types" in lower:
        return "compile-type-mismatch"
    if "has private access" in lower or "is not public" in lower:
        return "inaccessible_api"
    if "abstract; cannot be instantiated" in lower or "is abstract; cannot be instantiated" in lower:
        return "abstract_instantiation"
    return ""


def _obvious_malformed_java_test_reason(path: Path) -> str:
    """Return a precise reason for Java test files that Maven cannot compile.

    This intentionally detects only deterministic, syntax-level artifacts
    produced by LLM/schema failures.  It does not try to lint arbitrary user
    tests and therefore should not classify ordinary failing tests.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    stripped = text.lstrip()
    if stripped.startswith("```"):
        return "markdown_fenced_java_fragment"
    if stripped.startswith("{") or stripped.startswith("["):
        return "json_instead_of_java_test"
    # A common schema-salvage failure is imports/statements without a test class.
    if "@Test" in text and "class " not in text[: min(len(text), 4000)]:
        return "test_method_fragment_without_class"
    return ""


def _find_residual_generated_test_blockers(project_root: Path) -> list[dict]:
    """Find obvious malformed test files that would poison project testCompile."""
    test_root = project_root / "src/test/java"
    if not test_root.exists():
        return []
    blockers: list[dict] = []
    for path in sorted(test_root.rglob("*Test.java")):
        reason = _obvious_malformed_java_test_reason(path)
        if not reason:
            continue
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            rel = path
        blockers.append({
            "path": rel.as_posix(),
            "reason": reason,
        })
    return blockers


def _write_residual_generated_test_failure_json(
    *,
    json_out: Path,
    target: HardMethodTarget,
    method_name: str,
    project_root: Path,
    command: list[str],
    log_path: Path,
    target_descriptor: str | None,
    blockers: list[dict],
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "residual_generated_test_compile_failure: malformed generated test files detected before run\n"
        + "\n".join(f"- {item.get('path')}: {item.get('reason')}" for item in blockers)
        + "\n",
        encoding="utf-8",
    )
    _write_minimal_failure_json(
        json_out=json_out,
        target=target,
        method_name=method_name,
        project_root=project_root,
        command=command,
        process_returncode=1,
        failure_reason="residual_generated_test_compile_failure",
        log_path=log_path,
        target_descriptor=target_descriptor,
    )
    try:
        payload = json.loads(json_out.read_text(encoding="utf-8", errors="ignore"))
        payload["residual_generated_test_blockers"] = blockers
        preflight = payload.setdefault("preflight_failure", {})
        if isinstance(preflight, dict):
            preflight["blocking_test_sources"] = [str(item.get("path")) for item in blockers]
            preflight["blocking_test_source_category"] = "residual_generated_test_compile_failure"
        json_out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _dominant_missing_reason(data: dict, row: dict) -> str:
    """Roll up the most actionable reason for 0/missing coverage from candidate evidence."""
    # This field is for missing/0-coverage diagnosis.  If we already have
    # measured non-zero target coverage, candidate audit rows may still contain
    # target-positive passed/repaired candidates that were not in the final
    # retained suite, but that is not a missing-coverage reason.  Keeping it here
    # made low-but-measured targets look like lifecycle failures.
    line = row.get("target_line_coverage")
    branch = row.get("target_branch_coverage")
    if (
        (isinstance(line, (int, float)) and float(line) > 0.0)
        or (isinstance(branch, (int, float)) and float(branch) > 0.0)
    ):
        return ""
    roots: list[dict] = []
    if isinstance(data, dict):
        roots.append(data)
        validation_policy = data.get("validation_policy")
        if isinstance(validation_policy, dict):
            roots.append(validation_policy)
        report_summary = data.get("report_summary")
        if isinstance(report_summary, dict) and isinstance(report_summary.get("validation_policy"), dict):
            roots.append(report_summary.get("validation_policy") or {})

    reason_counts: dict[str, int] = {}

    def add(reason: object) -> None:
        text = str(reason or "").strip()
        if not text or text in {"unknown", "none", "null"}:
            return
        normalized = {
            "compile": "compile_blocker",
            "runtime-validation-failure": "runtime_validation_failure",
        }.get(text, text)
        reason_counts[normalized] = reason_counts.get(normalized, 0) + 1

    for root in roots:
        for item in root.get("candidate_extraction_empty_details") or []:
            if not isinstance(item, dict):
                continue
            add(_classify_failure_text(str(item.get("stdout_tail") or "")) or item.get("failure_type") or item.get("reason"))
        for key in (
            "streaming_candidate_coverage_audit_details",
            "candidate_coverage_audit_details",
            "candidate_audit",
            "streaming_validation_failed_details",
        ):
            for item in root.get(key) or []:
                if not isinstance(item, dict):
                    continue
                status = str(item.get("status") or "").lower()
                gate = item.get("coverage_gate") if isinstance(item.get("coverage_gate"), dict) else {}
                cov = item.get("candidate_target_coverage") if isinstance(item.get("candidate_target_coverage"), dict) else {}
                target_positive = bool(
                    item.get("target_positive")
                    or item.get("has_target_coverage")
                    or gate.get("has_target_coverage")
                    or cov.get("has_target_coverage")
                )
                if status == "passed" and target_positive:
                    if int(row.get("combination_rejected_count") or 0) > 0 or row.get("combination_conflict_failed_candidates"):
                        add("combination_rejected_best_snapshot_only")
                    else:
                        add("best_snapshot_only")
                    continue
                failure_text = "\n".join(
                    str(item.get(key) or "")
                    for key in ("failure_report_tail", "stdout_tail", "stderr_tail", "message")
                )
                failure_reason = (
                    item.get("failure_category")
                    or _classify_failure_text(failure_text)
                    or item.get("promotion_reject_reason")
                    or item.get("failure_type")
                    or (gate.get("reason") if isinstance(gate, dict) else "")
                )
                if target_positive and failure_reason:
                    normalized_failure = str(failure_reason).strip().replace("_", "-")
                    if normalized_failure in {
                        "oracle-mismatch",
                        "assertion-oracle-mismatch",
                        "state-sequence-or-oracle-mismatch",
                        "runtime-validation-failure",
                    }:
                        failure_reason = (
                            "coverage_positive_failed_"
                            + normalized_failure.replace("-", "_")
                        )
                add(failure_reason)

    if not reason_counts:
        for key in ("coverage_measurement_status", "coverage_status", "coverage_missing_reason"):
            add(row.get(key))
    if not reason_counts:
        return ""
    priority = {
        "combination_rejected_best_snapshot_only": 0,
        "best_snapshot_only": 0,
        "target_positive_candidate_not_promoted": 0,
        "coverage_positive_failed_oracle_mismatch": 1,
        "coverage_positive_failed_assertion_oracle_mismatch": 1,
        "coverage_positive_failed_state_sequence_or_oracle_mismatch": 1,
        "coverage_positive_failed_runtime_validation_failure": 1,
        "wrong_overload_or_arity": 2,
        "compile-type-mismatch": 3,
        "unavailable_type": 4,
        "missing_known_method": 5,
        "inaccessible_api": 6,
        "abstract_instantiation": 7,
        "compilation-failure": 8,
        "compile_blocker": 8,
        "candidate_extraction_empty": 9,
        "llm_empty_response": 10,
        "llm_output_truncated_or_invalid_json": 10,
    }
    return min(reason_counts, key=lambda reason: (priority.get(reason, 100), -reason_counts[reason], reason))


def _refine_generic_measurement_status(row: dict) -> None:
    status = str(row.get("coverage_measurement_status") or "")
    reason = str(row.get("dominant_missing_reason") or "")
    if status in {"no_target_positive_candidate", "no_retained_tests", "coverage_missing", "llm_generation_failed"} and reason:
        row["coverage_measurement_status"] = reason


def _row_has_usable_final_result(row: dict) -> bool:
    if int(row.get("final_validation_rc") or 0) != 0:
        return False
    if int(row.get("final_failed_tests") or 0) != 0:
        return False
    if (
        int(row.get("generated_test_count") or 0) <= 0
        and int(row.get("streaming_validation_submitted") or 0) <= 0
        and int(row.get("retained_target_positive_count") or 0) <= 0
    ):
        return False
    status = str(row.get("coverage_status") or "")
    return status in {
        "clean_covered",
        "covered_with_intermediate_failures",
        "low_target_coverage",
        "coverage_proxy_only",
        "covered_with_candidate_evidence",
        "combination_rejected_best_snapshot_only",
        "best_snapshot_only",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Method selector
# ──────────────────────────────────────────────────────────────────────────────

def _select_hard_methods(selector: str) -> dict[str, tuple[str, ...]]:
    """Filter HARD_METHODS by a comma-separated selector.

    Each token may be:
    - a method name, e.g. ``writeString``
    - a class name, e.g. ``ToXmlGenerator``
    - class.method, e.g. ``ToXmlGenerator.writeString``
    """
    tokens = _comma_values(selector)
    if not tokens:
        return HARD_METHODS

    selected: dict[str, list[str]] = {}

    def _add(class_name: str, method_name: str) -> None:
        selected.setdefault(class_name, [])
        if method_name not in selected[class_name]:
            selected[class_name].append(method_name)

    class_to_methods = {name: set(methods) for name, methods in HARD_METHODS.items()}
    method_to_classes: dict[str, list[str]] = {}
    for class_name, methods in HARD_METHODS.items():
        for method_name in methods:
            method_to_classes.setdefault(method_name, []).append(class_name)

    for token in tokens:
        if "." in token:
            class_name, method_name = token.split(".", 1)
            class_name, method_name = class_name.strip(), method_name.strip()
            base_method = method_name.split("#", 1)[0]
            if class_name not in class_to_methods:
                raise SystemExit(f"Unknown hard-method class selector: {class_name}")
            if base_method not in class_to_methods[class_name]:
                raise SystemExit(f"Unknown hard-method selector: {class_name}.{method_name}")
            _add(class_name, method_name)
            continue

        if token in class_to_methods:
            for method_name in HARD_METHODS[token]:
                _add(token, method_name)
            continue

        classes = method_to_classes.get(token)
        if not classes and "#" in token:
            base_method = token.split("#", 1)[0]
            classes = method_to_classes.get(base_method)
        if not classes:
            raise SystemExit(
                f"Unknown hard-method selector: {token}. "
                "Use method, class, or class.method from HARD_METHODS."
            )
        if len(classes) > 1:
            suggestions = ", ".join(f"{c}.{token}" for c in classes)
            raise SystemExit(
                f"Ambiguous method selector '{token}': matches {len(classes)} classes "
                f"({', '.join(classes)}). Use class.method form: {suggestions}"
            )
        for class_name in classes:
            _add(class_name, token)

    # Preserve original HARD_METHODS ordering.
    ordered: dict[str, tuple[str, ...]] = {}
    for class_name, methods in HARD_METHODS.items():
        chosen = selected.get(class_name)
        if chosen:
            ordered_methods: list[str] = []
            for method in methods:
                for chosen_method in chosen:
                    if chosen_method == method or chosen_method.split("#", 1)[0] == method:
                        ordered_methods.append(chosen_method)
            ordered[class_name] = tuple(ordered_methods)
    return ordered


# ──────────────────────────────────────────────────────────────────────────────
# Run one method target
# ──────────────────────────────────────────────────────────────────────────────

def _recover_crashed_conflicting_tests(
    *,
    project_root: Path,
    backup_root: Path,
    conflicting_tests: list[str],
    crash_residue_root: Path,
) -> list[dict[str, str]]:
    recovered: list[dict[str, str]] = []
    if not backup_root.exists():
        return recovered
    for rel_text in conflicting_tests:
        rel = Path(rel_text)
        current = project_root / rel
        original = backup_root / rel
        residue_saved = False
        if current.exists() and (
            not original.exists() or current.read_bytes() != original.read_bytes()
        ):
            residue = crash_residue_root / rel
            residue.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, residue)
            residue_saved = True
        if original.exists():
            current.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, current)
            action = "restored"
        elif current.exists():
            current.unlink()
            action = "removed_generated_residue"
        else:
            action = "unchanged_missing"
        recovered.append({
            "path": str(rel),
            "action": action,
            "residue_saved": str(residue_saved).lower(),
        })
    return recovered


def _build_command(
    args: Namespace,
    target: HardMethodTarget,
    project_root: Path,
    method_name: str,
    json_out: Path,
    *,
    scope: str = "method",
    target_descriptor: str | None = None,
) -> list[str]:
    inferred_java_version = _infer_target_java_version(project_root, target)
    effective_java_home = _resolve_compatible_target_java_home(
        target.java_home,
        inferred_java_version,
    )
    command = [
        args.python,
        "-m", "tatg.cli", "generate",
        "--project-root", str(project_root),
        "--target-file", target.target_file,
        "--scope", scope,
        "--class-name", target.java_class_name,
        "--backend", args.backend,
        "--api-base", args.api_base,
        "--model", args.model,
        "--api-key-env", args.api_key_env,
        "--max-iterations", str(args.max_iterations),
        "--repair-iterations", str(args.repair_iterations),
        "--mutation-extra-iterations", str(max(0, int(getattr(args, "mutation_extra_iterations", 1) or 0))),
        "--streaming-repair-mode", args.streaming_repair_mode,
        "--llm-timeout-seconds", str(args.llm_timeout_seconds),
        "--tatg-preanalysis-timeout-seconds", str(args.tatg_preanalysis_timeout_seconds),
        "--tatg-branch-analysis-timeout-seconds",
        str(
            getattr(
                args,
                "tatg_branch_analysis_timeout_seconds",
                TATG_BRANCH_ANALYSIS_TIMEOUT_SECONDS,
            )
        ),
        "--tatg-generation-timeout-seconds", str(args.tatg_generation_timeout_seconds),
        "--log-prompts",
        "--log-unrepaired-tests",
        "--test-command", _make_test_command(
            target.java_class_name,
            target.extra_mvn_flags,
            target.test_goals,
            project_root,
            target.jacoco_mode,
            effective_java_home,
        ),
        "--json-out", str(json_out),
        "--e2e-timeout-seconds", str(args.e2e_timeout_seconds),
        "--generation-parallelism", str(args.generation_parallelism),
        "--repair-parallelism", str(args.repair_parallelism),
        "--validation-parallelism", str(args.validation_parallelism),
        "--test-timeout-seconds", str(args.test_timeout_seconds),
    ]
    if scope == "method":
        command.extend(["--method-name", method_name])
    if target_descriptor:
        command.extend(["--target-descriptor", target_descriptor])
    if bool(getattr(args, "retain_existing_tests", False)):
        command.append("--retain-existing-tests")
    if inferred_java_version:
        command.extend(["--java-version", inferred_java_version])
    if args.with_mutation:
        command.append("--with-mutation")
    return command


def _safe_path_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._$-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def _descriptor_artifact_suffix(descriptor: str | None) -> str:
    short = _jvm_desc_to_short(str(descriptor or ""))
    if short == "()":
        return "noargs"
    return _safe_path_component(short.replace(",", "_"))


def _method_output_dir(
    args: Namespace,
    target: HardMethodTarget,
    method_name: str,
    target_descriptor: str | None = None,
) -> Path:
    return (
        args.out_dir
        / _safe_path_component(args.backend)
        / _safe_path_component(target.name)
        / _safe_path_component(_artifact_method_name(method_name, target_descriptor))
    )


def _artifact_method_name(method_name: str, target_descriptor: str | None = None) -> str:
    descriptor = str(target_descriptor or "").strip()
    if not descriptor:
        return method_name
    short = _descriptor_artifact_suffix(descriptor)
    return f"{method_name}__{short or 'descriptor'}"


def _method_output_stem(
    target: HardMethodTarget,
    method_name: str,
    target_descriptor: str | None = None,
) -> str:
    return f"{_safe_path_component(target.name)}.{_safe_path_component(_artifact_method_name(method_name, target_descriptor))}"


def _method_json_path(
    args: Namespace,
    target: HardMethodTarget,
    method_name: str,
    target_descriptor: str | None = None,
) -> Path:
    return _method_output_dir(args, target, method_name, target_descriptor) / f"{_method_output_stem(target, method_name, target_descriptor)}.json"


def _iter_payload_artifact_paths(value: object) -> list[Path]:
    paths: list[Path] = []

    def visit(item: object, key_hint: str = "") -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, str(key))
            return
        if isinstance(item, list):
            for child in item:
                visit(child, key_hint)
            return
        if not isinstance(item, str) or not item.strip():
            return
        key_l = key_hint.lower()
        if not (
            key_l.endswith("path")
            or key_l.endswith("xml")
            or key_l.endswith("csv")
            or "report_path" in key_l
            or "pitest_xml_path" in key_l
        ):
            return
        candidate = Path(item)
        if candidate.name in {"jacoco.xml", "jacoco.csv", "mutations.xml", "mutations.csv", "mutations.json"}:
            paths.append(candidate)

    visit(value)
    return paths


def _evaluation_artifact_candidates(project_root: Path, payload: dict | None = None) -> list[Path]:
    candidates: list[Path] = []
    if isinstance(payload, dict):
        for path in _iter_payload_artifact_paths(payload):
            candidates.append(path if path.is_absolute() else project_root / path)
    conventional = [
        project_root / "target" / "site" / "jacoco" / "jacoco.xml",
        project_root / "target" / "site" / "jacoco" / "jacoco.csv",
        project_root / "target" / "jacoco" / "jacoco.xml",
        project_root / "target" / "jacoco" / "jacoco.csv",
        project_root / "target" / "pit-reports" / "mutations.xml",
        project_root / "target" / "pit-reports" / "mutations.csv",
        project_root / "target" / "pit-reports" / "mutations.json",
    ]
    candidates.extend(conventional)
    pit_root = project_root / "target" / "pit-reports"
    if pit_root.exists():
        candidates.extend(pit_root.rglob("mutations.xml"))
        candidates.extend(pit_root.rglob("mutations.csv"))
        candidates.extend(pit_root.rglob("mutations.json"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if resolved in seen or not resolved.exists() or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _artifact_kind(path: Path) -> str:
    text = "/".join(path.parts).lower()
    if "pit-reports" in text or path.name.startswith("mutations"):
        return "pitest"
    if "jacoco" in text:
        return "jacoco"
    return "evaluation"


def _archive_evaluation_artifacts(
    *,
    project_root: Path,
    json_out: Path,
    run_out_dir: Path,
    command: list[str],
    env: dict[str, str],
) -> list[dict[str, object]]:
    if not json_out.exists():
        return []
    try:
        payload = json.loads(json_out.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    artifacts_dir = run_out_dir / "artifacts"
    archived: list[dict[str, object]] = []
    for source in _evaluation_artifact_candidates(project_root, payload):
        kind = _artifact_kind(source)
        try:
            rel = source.relative_to(project_root)
            rel_name = "__".join(rel.parts)
        except ValueError:
            rel_name = source.name
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        snapshot = artifacts_dir / kind / f"{digest[:12]}-{rel_name}"
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copy2(source, snapshot)
        archived.append(
            {
                "kind": kind,
                "source_path": str(source),
                "snapshot_path": str(snapshot),
                "sha256": digest,
                "bytes": snapshot.stat().st_size,
            }
        )
    payload["artifact_provenance"] = {
        "schema_version": 1,
        "command": list(command),
        "cwd": str(project_root),
        "java_home": env.get("JAVA_HOME"),
        "evaluation_artifacts": archived,
    }
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return archived


def _copy_declared_generated_tests(project_root: Path, json_out: Path, generated_out: Path) -> list[Path]:
    if not json_out.exists():
        return []
    data = _safe_read_json(json_out)
    path_keys = {"test_file", "generated_tests", "output_file"}
    candidates: set[Path] = set()

    def eligible(container: dict[str, object]) -> bool:
        status = str(container.get("status") or "").strip().lower()
        if status in {"passed", "success", "partial_success"}:
            return True
        if bool(container.get("converted")):
            return True
        gate = container.get("coverage_gate")
        if isinstance(gate, dict) and bool(gate.get("accepted", False)):
            return True
        for marker in ("generated_code", "final_code", "candidate_code"):
            if marker in container:
                return True
        if "generated_tests" in container:
            return True
        return False

    def maybe_collect_path(value: object) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text.endswith(".java"):
            return
        path = Path(text)
        if not path.is_absolute():
            path = project_root / path
        try:
            rel = path.resolve().relative_to(project_root.resolve())
        except Exception:
            return
        if "src/test" not in "/".join(rel.parts):
            return
        if path.exists() and path.is_file():
            candidates.add(rel)

    def collect(value: object, key: str | None = None) -> None:
        if isinstance(value, dict):
            if eligible(value):
                for path_key in path_keys:
                    maybe_collect_path(value.get(path_key))
            for child_key, child_value in value.items():
                collect(child_value, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                collect(child, key)
            return
        if key == "generated_tests":
            maybe_collect_path(value)

    collect(data)
    copied: list[Path] = []
    generated_out.mkdir(parents=True, exist_ok=True)
    for rel_path in sorted(candidates):
        source = project_root / rel_path
        destination = generated_out / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)
    return copied


TATG_STRATEGY_REGRESSION_METHODS = (
    "FilteringParserDelegate.nextToken",
    "ApplicationSubmissionHelper.getDriverSpec",
    "ApplicationSubmissionHelper.getExecutorSpec",
    "FromXmlParser.nextTextValue",
    "ReActAgent.doCall",
    "ConstructorConstructor.newDefaultImplementationConstructor",
    "DefaultRequestHandler.onMessageSend",
)


# S1: stratified tuning subset used to accept or reject a TATG strategy before
# it is promoted to the full 141.  28 targets across 19 projects, balanced over
# the current branch-coverage buckets so a change cannot look good by moving one
# family only.  Ten targets below 0.50 (the coverage we must win), ten between
# 0.50 and 0.70 (where the remaining gap is most reachable), six at or above
# 0.70 that act as the regression guard, and two that currently fail to produce
# a measurement at all.
TATG_S1_TUNING_METHODS = (
    # tail, < 0.50
    "Rotation.getAngles",
    "NumberUtils.createNumber",
    "FilteringParserDelegate.nextToken",
    "Metaphone.metaphone",
    "AbstractPatriciaTrie.put",
    "RuleCompiler.processMatchExpression",
    "AbstractXMLOutputProcessor.printDocument",
    "$Gson$Types.resolve",
    "SlidingWindowEventCompactor.getCompactionEvents",
    "JCurlCommandProcessor.resolveCurlCommand",
    # middle, 0.50 - 0.70
    "Flat3MapNew.get",
    "JsonReader.readEscapeCharacter",
    "DoubleMetaphone.handleG",
    "ToXmlGenerator.writeString",
    "DefaultRequestHandler.onMessageSend",
    "SubmissionSummary.copyFrom",
    "CSVParser.createHeaders",
    "SAXHandler.startElement",
    "ByteMachine.addMatchReferences",
    "ServerSentEventParser.processLine",
    # regression guard, >= 0.70
    "GaussNewtonOptimizer.doOptimize",
    "Rule.pattern",
    "Entities.escape",
    "JsonReader.doPeek",
    "Element.getNamespacesInScope",
    "AntPathMatcher.isMatch",
    # currently unmeasured / infrastructure
    "PatternOptionBuilder.getValueType",
    "Contents.rearrangeEventsForLatestFunctionResponse",
)

# H1: held-out generality set.  Never used while tuning a strategy; run once
# after S1 acceptance.  A strategy that wins on S1 and does nothing here was
# fitted to S1 rather than to the underlying problem.
TATG_H1_HOLDOUT_METHODS = (
    "DoubleMetaphone.doubleMetaphone",
    "AbstractPatriciaTrie.higherEntry",
    "Ruler.matches",
    "XPathHelper.getSingleStep",
    "JsonReader.peekNumber",
    "PlanNotebook.reviseCurrentPlan",
    "ApplicationSubmissionHelper.getExecutorSpec",
    "FakeValuesService.bothify",
)

VALIDATION_SUITES: dict[str, tuple[str, ...]] = {
    "tatg-strategy-regression": TATG_STRATEGY_REGRESSION_METHODS,
    "tatg-s1-tuning": TATG_S1_TUNING_METHODS,
    "tatg-h1-holdout": TATG_H1_HOLDOUT_METHODS,
}


def _effective_method_selector(*, methods: str, validation_suite: str | None = None) -> str:
    suite = str(validation_suite or "").strip()
    selected = VALIDATION_SUITES.get(suite)
    if selected:
        return ",".join(selected)
    if suite:
        raise SystemExit(
            f"unknown --validation-suite {suite!r}; "
            f"available: {', '.join(sorted(VALIDATION_SUITES))}"
        )
    return str(methods or "")


def _cleanup_forced_method_outputs(run_out_dir: Path, stem: str, backend: str) -> None:
    for suffix in (
        ".json",
        ".json.partial",
        ".json.prompts.jsonl",
        ".json.partial.prompts.jsonl",
        f".{backend}.log",
        f".{backend}.progress.log",
        ".tatg.log",
        ".tatg.progress.log",
    ):
        (run_out_dir / f"{stem}{suffix}").unlink(missing_ok=True)


def _has_duplicate_java_method_signatures(text: str) -> bool:
    seen: set[tuple[str, int]] = set()
    method_re = re.compile(
        r"\b(?:public|protected|private|static|final|synchronized|\s)+"
        r"[\w.<>,? \[\]]+\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{"
    )
    for match in method_re.finditer(text or ""):
        name = match.group(1)
        params = [p for p in (part.strip() for part in match.group(2).split(",")) if p]
        key = (name, len(params))
        if key in seen:
            return True
        seen.add(key)
    return False


def _looks_like_tatg_generated_test_pollution(text: str) -> bool:
    content = str(text or "")
    if "```" in content:
        return True
    if re.search(r"\b\w+__regen\d+\b", content):
        return True
    if _has_duplicate_java_method_signatures(content):
        return True
    return False


def _discover_generated_test_pollution(project_root: Path, active_tests: list[Path]) -> list[Path]:
    active = {Path(p).as_posix() for p in active_tests}
    discovered: list[Path] = []
    for java_file in sorted(project_root.rglob("src/test/java/**/*.java")):
        rel = java_file.relative_to(project_root)
        if rel.as_posix() in active:
            continue
        try:
            text = java_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _looks_like_tatg_generated_test_pollution(text):
            discovered.append(rel)
    return discovered


def _copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def _method_best_score(
    data: dict,
    target: HardMethodTarget,
    method_name: str,
    target_descriptor: str | None = None,
) -> float:
    row = _best_method_row(
        data,
        target.java_class_name,
        method_name,
        target_descriptor or target.target_descriptors.get(method_name),
    )
    if isinstance(row, dict):
        try:
            return float(row.get("line_coverage") or 0.0)
        except (TypeError, ValueError):
            pass
    cov = data.get("report_summary", {}).get("coverage", {}) if isinstance(data.get("report_summary"), dict) else {}
    target_method = cov.get("target_method") if isinstance(cov, dict) else {}
    if isinstance(target_method, dict):
        try:
            line = float(target_method.get("line_coverage") or cov.get("display_line_coverage") or 0.0)
            branch = target_method.get("branch_coverage", cov.get("display_branch_coverage"))
            return line + (float(branch) if isinstance(branch, (int, float)) else 0.0)
        except (TypeError, ValueError):
            return 0.0
    try:
        return float(cov.get("display_line_coverage") or cov.get("line_coverage") or 0.0) if isinstance(cov, dict) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _snapshot_previous_method_best(
    *,
    json_out: Path,
    generated_out: Path,
    backup_dir: Path,
    target: HardMethodTarget,
    method_name: str,
) -> dict[str, object]:
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    snapshot: dict[str, object] = {"exists": False, "score": 0.0, "generated_tests_snapshot": False}
    if json_out.exists():
        data = _safe_read_json(json_out)
        snapshot.update({"exists": True, "score": _method_best_score(data, target, method_name)})
        shutil.copy2(json_out, backup_dir / json_out.name)
        prompt_logs = data.get("prompt_logs") if isinstance(data, dict) else None
        if isinstance(prompt_logs, list) and prompt_logs:
            prompt_sidecar = backup_dir / (json_out.name + ".prompts.jsonl")
            prompt_sidecar.write_text(
                "\n".join(json.dumps(item, ensure_ascii=True) for item in prompt_logs if isinstance(item, dict)) + "\n",
                encoding="utf-8",
            )
        prompt_log = Path(str(json_out) + ".prompts.jsonl")
        if prompt_log.exists() and not (backup_dir / prompt_log.name).exists():
            shutil.copy2(prompt_log, backup_dir / prompt_log.name)
    if generated_out.exists():
        _copy_tree_contents(generated_out, backup_dir / "generated-tests")
        snapshot["exists"] = True
        snapshot["generated_tests_snapshot"] = True
    return snapshot


def _strip_mutation_for_disabled_run(data: dict) -> None:
    data.pop("mutation_evaluation", None)
    for run in data.get("test_runs") or []:
        if not isinstance(run, dict):
            continue
        for row in run.get("methods_coverage") or []:
            if isinstance(row, dict):
                for key in list(row):
                    if "mutation" in key:
                        row.pop(key, None)
    report = data.setdefault("report_summary", {})
    if isinstance(report, dict):
        report["mutation"] = {"skipped_reason": "mutation_disabled_for_current_run"}


def _restore_previous_method_best_if_regressed(
    *,
    json_out: Path,
    generated_out: Path,
    backup_dir: Path,
    snapshot: dict[str, object],
    target: HardMethodTarget,
    method_name: str,
    preserve_mutation: bool = True,
    historical_fallback_only: bool = True,
) -> bool:
    if not snapshot or not bool(snapshot.get("exists")):
        return False
    previous_score = float(snapshot.get("score") or 0.0)
    current_data = _safe_read_json(json_out) if json_out.exists() else {}
    current_score = _method_best_score(current_data, target, method_name) if current_data else 0.0
    current_generated = int(current_data.get("generated_test_count") or 0) if isinstance(current_data, dict) else 0
    tests_only = bool(snapshot.get("generated_tests_snapshot")) and not (backup_dir / json_out.name).exists()
    regressed = tests_only or current_generated <= 0 or current_score < previous_score
    if historical_fallback_only and current_generated > 0 and current_score > 0.0:
        regressed = False
    if not regressed:
        return False
    backup_json = backup_dir / json_out.name
    if backup_json.exists():
        data = _safe_read_json(backup_json)
    else:
        data = dict(current_data) if isinstance(current_data, dict) else {}
        data["generated_test_count"] = 1 if tests_only else int(data.get("generated_test_count") or 0)
    if not preserve_mutation:
        _strip_mutation_for_disabled_run(data)
    data["batch_regression_guard"] = {
        "restored_previous_best_snapshot": True,
        "restored_previous_generated_tests_snapshot": bool(snapshot.get("generated_tests_snapshot")),
        "previous_score": previous_score,
        "current_score": current_score,
    }
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    backup_prompt = backup_dir / (json_out.name + ".prompts.jsonl")
    if backup_prompt.exists():
        shutil.copy2(backup_prompt, Path(str(json_out) + ".prompts.jsonl"))
    backup_tests = backup_dir / "generated-tests"
    if backup_tests.exists():
        if generated_out.exists():
            shutil.rmtree(generated_out)
        _copy_tree_contents(backup_tests, generated_out)
    return True


def _archive_current_method_best_if_better(
    *,
    json_out: Path,
    generated_out: Path,
    backup_dir: Path,
    target: HardMethodTarget,
    method_name: str,
    target_descriptor: str | None = None,
) -> dict[str, object]:
    current = _safe_read_json(json_out) if json_out.exists() else {}
    current_score = _method_best_score(current, target, method_name, target_descriptor) if isinstance(current, dict) else 0.0
    meta_path = backup_dir / "snapshot.json"
    old_meta = _safe_read_json(meta_path) if meta_path.exists() else {}
    old_score = float(old_meta.get("score") or -1.0) if isinstance(old_meta, dict) else -1.0
    if current_score >= old_score:
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        if json_out.exists():
            shutil.copy2(json_out, backup_dir / json_out.name)
        if generated_out.exists():
            _copy_tree_contents(generated_out, backup_dir / "generated-tests")
        old_meta = {"score": current_score}
        meta_path.write_text(json.dumps(old_meta), encoding="utf-8")
    return old_meta


def _archive_regressed_live_snapshot(*, json_out: Path, generated_out: Path, backup_dir: Path) -> bool:
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    backup_dir.mkdir(parents=True, exist_ok=True)
    copied = False
    if json_out.exists():
        shutil.copy2(json_out, backup_dir / json_out.name)
        copied = True
    if generated_out.exists():
        _copy_tree_contents(generated_out, backup_dir / "generated-tests")
        copied = True
    return copied


def _promote_completed_partial_json(json_out: Path) -> bool:
    """Recover a completed `.json.partial` when the final JSON was not materialized."""

    if json_out.exists():
        return False
    partial = Path(str(json_out) + ".partial")
    if not partial.exists():
        return False
    data = _safe_read_json(partial)
    if not isinstance(data, dict):
        return False
    if str(data.get("stage") or "") != "completed" and str(data.get("status") or "") != "completed":
        return False
    recovered = dict(data)
    recovered["partial_artifact_used"] = True
    recovered["partial_artifact_promoted_to_final"] = True
    recovered["partial_artifact_source"] = str(partial)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(recovered, indent=2, sort_keys=True), encoding="utf-8")
    return True


def _log_has_target_resolution_error(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    needles = (
        "Target resolution error",
        "No targets found for the given scope/filters",
    )
    return any(needle in text for needle in needles)


def _log_has_generation_worker_stall(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return "generation_worker_stalled" in text or "post_coverage_generation_finalization_timeout" in text


def _default_stale_partial_stage_timeout_seconds(args: Namespace) -> int:
    """Keep the stale-stage watchdog outside configured child-operation budgets.

    A partial stage can legitimately remain unchanged while a slow local model
    is serving an LLM request.  Killing it sooner than the configured LLM or
    generation timeout turns slow inference into a false infrastructure
    failure.  The overall per-method watchdog remains the final upper bound.
    """

    component_timeouts: list[int] = []
    for name in (
        "llm_timeout_seconds",
        "tatg_preanalysis_timeout_seconds",
        "tatg_branch_analysis_timeout_seconds",
        "tatg_generation_timeout_seconds",
        "test_timeout_seconds",
    ):
        try:
            value = max(0, int(getattr(args, name, 0) or 0))
        except (TypeError, ValueError):
            value = 0
        if value:
            component_timeouts.append(value)
    longest_component = max(component_timeouts, default=0)
    return max(420, longest_component + 120)


def _run_one(
    args: Namespace,
    target: HardMethodTarget,
    env: dict[str, str],
    method_name: str,
    target_descriptor: str | None = None,
) -> int:
    project_root = target.subjects_root.resolve() / target.project
    target_descriptor = str(target_descriptor or target.target_descriptors.get(method_name) or "").strip() or None
    out_stem = _method_output_stem(target, method_name, target_descriptor)
    run_out_dir = _method_output_dir(args, target, method_name, target_descriptor)
    run_out_dir.mkdir(parents=True, exist_ok=True)
    json_out = run_out_dir / f"{out_stem}.json"
    log_out = run_out_dir / f"{out_stem}.{args.backend}.log"
    progress_log_out = run_out_dir / f"{out_stem}.{args.backend}.progress.log"
    generated_out = (
        args.out_dir
        / "generated-tests"
        / args.backend
        / _safe_path_component(target.name)
        / _safe_path_component(_artifact_method_name(method_name, target_descriptor))
    )

    def _interrupted_returncode(returncode: int | None) -> bool:
        if returncode is None:
            return False
        return returncode < 0 or returncode in {130, 143}

    def _post_eval_skipped_reason(returncode: int | None) -> str:
        if returncode is not None and returncode < 0:
            return f"interrupted_by_signal={abs(returncode)} post_run_skipped"
        return f"interrupted_rc={returncode} post_run_skipped"

    command = _build_command(
        args,
        target,
        project_root,
        method_name,
        json_out,
        scope="method",
        target_descriptor=target_descriptor,
    )
    label = out_stem
    print(f"[{label}] command: {' '.join(command)}")

    if args.dry_run:
        return 0

    if json_out.exists() and not args.force:
        print(f"[{label}] skip: {json_out} already exists (use --force to rerun)")
        return 0
    if args.force:
        json_out.unlink(missing_ok=True)
        (run_out_dir / f"{out_stem}.json.partial").unlink(missing_ok=True)
        log_out.unlink(missing_ok=True)
        progress_log_out.unlink(missing_ok=True)

    seed: dict[str, Any] | None = None
    seed_manifest_raw = str(
        getattr(args, "retained_test_seed_manifest", "") or ""
    ).strip()
    default_test = _default_test_file_for_target(target)
    if seed_manifest_raw:
        if not bool(getattr(args, "retain_existing_tests", False)):
            raise ValueError(
                "--retained-test-seed-manifest requires --retain-existing-tests"
            )
        if not default_test:
            raise ValueError(f"[{label}] target has no deterministic test path")
        seed = _retained_test_seed(
            Path(seed_manifest_raw).expanduser().resolve(),
            target_key=label,
            expected_test_name=Path(default_test).name,
            expected_target_descriptor=target_descriptor,
            expected_source_file=project_root / target.target_file,
            expected_class_name=target.name,
            expected_method_name=method_name,
        )

    backup_root = args.out_dir / ".tree-backups" / args.backend / out_stem
    conflicting_tests = _conflicting_tests_for_run(
        target,
        retain_existing_tests=bool(
            getattr(args, "retain_existing_tests", False)
        ),
    )
    # Crash recovery must restore the batch-managed test paths even when a
    # generated replacement still exists. Restoring only missing files lets a
    # killed run's generated suite become the next run's "original" baseline.
    if backup_root.exists():
        crash_residue_root = (
            args.out_dir / ".crash-residue" / args.backend / out_stem
        )
        for recovery in _recover_crashed_conflicting_tests(
            project_root=project_root,
            backup_root=backup_root,
            conflicting_tests=conflicting_tests,
            crash_residue_root=crash_residue_root,
        ):
            rel = recovery["path"]
            if recovery["action"] == "restored":
                print(f"[{label}] crash-recovery: restored {rel} from prior backup")
            elif recovery["action"] == "removed_generated_residue":
                print(
                    f"[{label}] crash-recovery: removed generated residue {rel}; "
                    "the path did not exist before the interrupted run"
                )
        shutil.rmtree(backup_root)
    before = _backup_java_files(project_root, backup_root)
    if seed:
        destination = project_root / default_test
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed["path"], destination)
            print(
                f"[{label}] installed retained-test seed: {seed['path']} "
                f"sha256={seed['sha256']}"
            )
        except Exception:
            _restore_java_files(project_root, before, backup_root)
            raise
    residual_blockers = _find_residual_generated_test_blockers(project_root)
    residual_stashed: list[dict] = []
    if residual_blockers:
        for item in residual_blockers:
            rel_text = str(item.get("path") or "").strip()
            if not rel_text:
                continue
            rel_path = Path(rel_text)
            path = project_root / rel_path
            # Only quarantine files that were included in the backup and still
            # match the deterministic malformed-generated-test detector.  That
            # keeps ordinary user tests and project-native failures untouched.
            if rel_path in before and path.exists() and _obvious_malformed_java_test_reason(path):
                path.unlink()
                residual_stashed.append(dict(item))
                print(f"[{label}] stashed residual malformed generated test: {rel_text}")
        residual_blockers = _find_residual_generated_test_blockers(project_root)
    if residual_blockers:
        _write_residual_generated_test_failure_json(
            json_out=json_out,
            target=target,
            method_name=method_name,
            project_root=project_root,
            command=command,
            log_path=log_out,
            target_descriptor=target_descriptor,
            blockers=residual_blockers,
        )
        print(
            f"[{label}] residual generated-test compile blocker before run: "
            f"{len(residual_blockers)} malformed test file(s); see {log_out}"
        )
        _restore_java_files(project_root, before, backup_root)
        restore_mismatches = _java_restore_mismatches(project_root, before, backup_root)
        if restore_mismatches:
            print(
                f"[{label}] ERROR: Java tree restore failed after preflight abort: "
                f"{restore_mismatches[:5]}"
            )
        return 1
    start = time.monotonic()
    print(f"[{label}] starting; log={log_out} progress_log={progress_log_out}")
    process_returncode = 130
    copied: list[Path] = []
    post_eval_result = "post_run_skipped"
    # Per-target env: override JAVA_HOME when the project requires an older JDK.
    target_env = dict(env)
    inferred_java_version = _infer_target_java_version(project_root, target)
    effective_java_home = _configure_target_java_env(
        target_env,
        preferred_java_home=target.java_home,
        required_release=inferred_java_version,
    )
    if effective_java_home:
        print(f"[{label}] JAVA_HOME override: {effective_java_home}")
    elif target.java_home or inferred_java_version:
        print(
            f"[{label}] no compatible configured JAVA_HOME found; "
            f"preferred={target.java_home or 'none'} "
            f"target release={inferred_java_version or 'unknown'}"
        )
    if target_descriptor:
        target_env["TATG_TARGET_DESCRIPTOR"] = target_descriptor
        print(f"[{label}] target descriptor: {target_descriptor}")
    if seed and isinstance(seed.get("coverage_proof"), dict):
        target_env["TATG_RETAINED_TEST_SEED_EVIDENCE"] = json.dumps(
            seed["coverage_proof"],
            separators=(",", ":"),
            sort_keys=True,
        )
    partial_json_out = run_out_dir / f"{out_stem}.json.partial"
    existing_startup_paths = [
        part for part in str(target_env.get("TATG_STARTUP_ARTIFACT_PATHS") or "").split(os.pathsep)
        if part.strip()
    ]
    startup_paths = [*existing_startup_paths, str(partial_json_out), str(json_out)]
    target_env["TATG_STARTUP_ARTIFACT_PATHS"] = os.pathsep.join(dict.fromkeys(startup_paths))
    target_env.setdefault("TATG_STARTUP_ARTIFACT_TIMEOUT_SECONDS", "180")
    target_env.setdefault("TATG_STALE_PARTIAL_PATHS", str(partial_json_out))
    target_env.setdefault(
        "TATG_STALE_PARTIAL_STAGE_TIMEOUT_SECONDS",
        str(_default_stale_partial_stage_timeout_seconds(args)),
    )
    target_env.setdefault(
        "TATG_STALE_PARTIAL_STAGES",
        "post_coverage_generation,deferred_streaming_repair_done,post_mutation_extra,"
        "coverage_feedback_loop_done,post_generation_count_done,post_repair_pipeline_done,"
        "repair_summary_done,progress_finish_done,final_report_assembly_start",
    )
    try:
        # Stash conflicting tests inside the try block so the finally always restores them.
        # The backup already captured these files above, so _restore_java_files will put
        # them back regardless of how the process exits (Ctrl+C, exception, rc!=0).
        for rel in conflicting_tests:
            p = project_root / rel
            rel_path = Path(rel)
            if p.exists() and rel_path in before:
                p.unlink()
                print(f"[{label}] stashed conflicting test: {rel}")
        method_watchdog_timeout_seconds = max(
            0,
            int(
                getattr(args, "e2e_timeout_seconds", 0)
                or getattr(args, "tatg_generation_timeout_seconds", 0)
                or 0
            ),
        )
        process_returncode = _run_command_with_timestamped_log(
            command,
            project_root,
            target_env,
            log_out,
            progress_log_out,
            show_progress=bool(getattr(args, "show_progress", False)),
            timeout_seconds=method_watchdog_timeout_seconds,
        )
        first_preflight_return_code = process_returncode
        managed_target_tests = {
            Path(rel)
            for rel in conflicting_tests
            if str(rel or "").strip()
        }
        isolation_sources = _preflight_isolation_retry_sources(
            json_out=json_out,
            log_path=log_out,
            project_root=project_root,
            backed_up_paths=set(before),
            managed_target_tests=managed_target_tests,
        )
        if isolation_sources:
            first_attempt_archive = _archive_preflight_isolation_attempt(
                run_out_dir=run_out_dir,
                out_stem=out_stem,
                paths=[json_out, partial_json_out, log_out, progress_log_out],
            )
            for rel in isolation_sources:
                (project_root / rel).unlink()
                print(
                    f"[{label}] preflight isolation retry: temporarily stashed "
                    f"unrelated blocking test {rel}"
                )
            json_out.unlink(missing_ok=True)
            partial_json_out.unlink(missing_ok=True)
            progress_log_out.unlink(missing_ok=True)
            process_returncode = _run_command_with_timestamped_log(
                command,
                project_root,
                target_env,
                log_out,
                progress_log_out,
                show_progress=bool(getattr(args, "show_progress", False)),
                timeout_seconds=method_watchdog_timeout_seconds,
            )
            for artifact_path in (json_out, partial_json_out):
                _record_preflight_isolation_retry(
                    artifact_path,
                    sources=isolation_sources,
                    first_return_code=first_preflight_return_code,
                    retry_return_code=process_returncode,
                    first_attempt_archive=first_attempt_archive,
                )
        interrupted = _interrupted_returncode(process_returncode)
        if _promote_completed_partial_json(json_out):
            print(f"[{label}] recovered completed partial artifact: {json_out}")
        if bool(getattr(args, "post_run_evaluate", False)) and json_out.exists() and not interrupted:
            json_data = _safe_read_json(json_out)
            _, msg = _post_run_evaluate_and_merge(
                project_root=project_root,
                json_out=json_out,
                generated_test_count=int(json_data.get("generated_test_count") or 0),
                run_mutation_if_missing=bool(getattr(args, "with_mutation", False)),
            )
            post_eval_result = msg
        elif interrupted:
            post_eval_result = _post_eval_skipped_reason(process_returncode)
        if json_out.exists() and not interrupted:
            _record_retained_test_seed(json_out, seed)
            archived = _archive_evaluation_artifacts(
                project_root=project_root,
                json_out=json_out,
                run_out_dir=run_out_dir,
                command=command,
                env=target_env,
            )
            if archived:
                post_eval_result = f"{post_eval_result} artifacts_archived={len(archived)}"
            if residual_stashed:
                try:
                    data = json.loads(json_out.read_text(encoding="utf-8", errors="ignore"))
                    data["residual_generated_test_stashed_before_run"] = residual_stashed
                    json_out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
                except Exception:
                    pass
        if not interrupted:
            copied = _copy_changed_java_files(project_root, before, backup_root, generated_out)
            declared_copied = _copy_declared_generated_tests(project_root, json_out, generated_out)
            if declared_copied:
                by_path = {path.resolve(): path for path in copied}
                for path in declared_copied:
                    by_path.setdefault(path.resolve(), path)
                copied = sorted(by_path.values())
        if (
            process_returncode != 0
            and not json_out.exists()
            and _log_has_target_resolution_error(log_out)
        ):
            _write_minimal_failure_json(
                json_out=json_out,
                target=target,
                method_name=method_name,
                project_root=project_root,
                command=command,
                process_returncode=process_returncode,
                failure_reason="target_resolution_failed",
                log_path=log_out,
                target_descriptor=target_descriptor,
            )
            print(
                f"[{label}] FATAL: target resolution failed for method scope — "
                "the method was not found in the source file. "
                "Check that the method name is correct and that targeting.py extracts it. "
                "Not falling back to class scope (would generate tests for the entire class)."
            )
    finally:
        _restore_java_files(project_root, before, backup_root)
        restore_mismatches = _java_restore_mismatches(project_root, before, backup_root)
        if restore_mismatches:
            # A second deterministic restore handles transient file replacement
            # races. Persistent mismatches are infrastructure failures and must
            # not be allowed to contaminate the next target.
            _restore_java_files(project_root, before, backup_root)
            restore_mismatches = _java_restore_mismatches(
                project_root,
                before,
                backup_root,
            )
        if restore_mismatches:
            process_returncode = process_returncode or 1
            print(
                f"[{label}] ERROR: Java tree restore verification failed: "
                f"{restore_mismatches[:5]}"
            )
    elapsed = time.monotonic() - start
    if (
        process_returncode != 0
        and not json_out.exists()
        and not _interrupted_returncode(process_returncode)
    ):
        if process_returncode == 124 and _log_has_generation_worker_stall(log_out):
            reason = "generation_worker_stalled"
        else:
            reason = "method_watchdog_timeout" if process_returncode == 124 else "pre_generation_process_failed"
        _write_minimal_failure_json(
            json_out=json_out,
            target=target,
            method_name=method_name,
            project_root=project_root,
            command=command,
            process_returncode=process_returncode,
            failure_reason=reason,
            log_path=log_out,
            target_descriptor=target_descriptor,
        )
    json_outcome = _json_outcome(json_out)
    if _interrupted_returncode(process_returncode):
        json_outcome = f"json=ignored_interrupted_process ({json_outcome})"
    elif json_out.exists():
        post_eval_result = _rewrite_post_eval_coverage_for_display_scope(json_out, post_eval_result)
    print(
        f"[{label}] rc={process_returncode} elapsed={elapsed:.1f}s "
        f"changed_files_copied={len(copied)} {post_eval_result} {json_outcome}"
    )
    return process_returncode


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TATG hard-method batches for all curated targets (defects4j + new projects)."
    )
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=BACKEND if BACKEND in BACKEND_CHOICES else "tatg",
        help="TATG backend to run. Default: %(default)s. Can also be set with TATG_BACKEND.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help=(
            "Comma-separated selector list. Each token can be a method name, class name, "
            "or class.method (e.g. 'writeString,FunctionCallingUtils,ToXmlGenerator.writeString'). "
            "Default: run all HARD_METHODS."
        ),
    )
    parser.add_argument(
        "--list-hard-methods",
        action="store_true",
        help="Print available hard-method selectors (class.method) and exit.",
    )
    parser.add_argument(
        "--validation-suite",
        default="",
        help=("Named validation subset to run: tatg-s1-tuning (28-target stratified "
             "strategy subset), tatg-h1-holdout (8-target generality set), or "
             "tatg-strategy-regression."),
    )
    parser.add_argument(
        "--allow-historical-snapshot-restore",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow restoring a previous on-disk best snapshot when a force rerun regresses.",
    )
    parser.add_argument(
        "--show-progress",
        action=argparse.BooleanOptionalAction,
        default=SHOW_PROGRESS,
        help="Display live TTY progress panel. Use --no-show-progress for log-only mode.",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=FORCE,
        help="Delete existing JSON/log outputs for each selected target before running.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=DRY_RUN,
        help="Print commands only without invoking TATG. Can also be set with TATG_DRY_RUN=1.",
    )
    parser.add_argument(
        "--with-mutation",
        action=argparse.BooleanOptionalAction,
        default=WITH_MUTATION,
        help="Run PITest mutation evaluation and mutation feedback. Use --no-with-mutation to disable.",
    )
    parser.add_argument(
        "--retain-existing-tests",
        action=argparse.BooleanOptionalAction,
        default=str(
            os.environ.get("TATG_RETAIN_EXISTING_TESTS", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"},
        help=(
            "Preserve the target project's existing green test file as the "
            "structural-generation baseline. Disabled by default."
        ),
    )
    parser.add_argument(
        "--retained-test-seed-manifest",
        default=str(os.environ.get("TATG_RETAINED_TEST_SEED_MANIFEST", "") or ""),
        help=(
            "Optional JSON manifest mapping exact descriptor artifact keys to "
            "read-only Java test snapshots. Requires --retain-existing-tests; "
            "project files are restored after every target."
        ),
    )
    parser.add_argument(
        "--mutation-extra-iterations",
        type=int,
        default=int(os.environ.get("TATG_MUTATION_EXTRA_ITERATIONS", "1") or "1"),
        help="Final mutation-focused generation rounds passed to TATG (default: 1).",
    )
    parser.add_argument(
        "--post-run-evaluate",
        action=argparse.BooleanOptionalAction,
        default=POST_RUN_EVALUATE,
        help="Run post-generation coverage/mutation evaluation and merge it into the JSON report.",
    )
    parser.add_argument(
        "--streaming-repair-mode",
        default=STREAMING_REPAIR_MODE,
        help="Streaming validation repair mode passed to TATG. Can also be set with TATG_STREAMING_REPAIR_MODE.",
    )
    parser.add_argument(
        "--generation-parallelism",
        type=int,
        default=GENERATION_PARALLELISM,
        help="Generation worker parallelism passed to TATG.",
    )
    parser.add_argument(
        "--repair-parallelism",
        type=int,
        default=REPAIR_PARALLELISM,
        help="Repair worker parallelism passed to TATG.",
    )
    parser.add_argument(
        "--validation-parallelism",
        type=int,
        default=VALIDATION_PARALLELISM,
        help="Streaming validation worker parallelism passed to TATG.",
    )
    parser.add_argument(
        "--tatg-preanalysis-timeout-seconds",
        type=int,
        default=TATG_PREANALYSIS_TIMEOUT_SECONDS,
        help="Per-target timeout (seconds) for TATG pre-analysis.",
    )
    parser.add_argument(
        "--tatg-branch-analysis-timeout-seconds",
        type=int,
        default=TATG_BRANCH_ANALYSIS_TIMEOUT_SECONDS,
        help="Per-target timeout (seconds) for semantic LLM branch/state analysis.",
    )
    parser.add_argument(
        "--tatg-generation-timeout-seconds",
        type=int,
        default=TATG_GENERATION_TIMEOUT_SECONDS,
        help="Per-target hard timeout (seconds) for the TATG generation worker.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=MAX_ITERATIONS,
        help="Maximum generation/coverage-feedback iterations. Can also be set with TATG_MAX_ITERATIONS.",
    )
    parser.add_argument(
        "--repair-iterations",
        type=int,
        default=REPAIR_ITERATIONS,
        help="Maximum repair iterations. Can also be set with TATG_REPAIR_ITERATIONS.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        default=LLM_TIMEOUT_SECONDS,
        help="Timeout for each LLM generation/repair call. Can also be set with TATG_LLM_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--e2e-timeout-seconds",
        type=int,
        default=E2E_TIMEOUT_SECONDS,
        help="Overall timeout for each method run; 0 disables the limit. Can also be set with TATG_E2E_TIMEOUT_SECONDS.",
    )
    parser.add_argument(
        "--test-timeout-seconds",
        type=int,
        default=TEST_TIMEOUT_SECONDS,
        help="Timeout for each Maven/test command. Can also be set with TATG_TEST_TIMEOUT_SECONDS.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────────

def _selector_class_name_from_artifact(json_path: Path, data: dict) -> str:
    """Return the batch selector identity encoded by the artifact path.

    ``class_name`` in the JSON is the Java class and can intentionally differ
    from the selector name (for example two revisions of the same Java class).
    The directory immediately below the backend preserves that selector
    identity and must therefore win during archive replay.
    """

    parts = json_path.parts
    for backend in BACKEND_CHOICES:
        try:
            index = parts.index(backend)
        except ValueError:
            continue
        if index + 1 < len(parts):
            return str(parts[index + 1])
    return str(data.get("selector_class_name") or data.get("class_name") or "").strip()


def _descriptor_leaf_signature(descriptor: str) -> tuple[str, ...]:
    short = _jvm_desc_to_short(str(descriptor or ""))
    if short in {"", "()"}:
        return ()
    return tuple(
        token.strip().replace(".", "$").rsplit("$", 1)[-1]
        for token in short.split(",")
    )


def _canonical_artifact_descriptor(
    *,
    target: HardMethodTarget | None,
    method_name: str,
    artifact_descriptor: str | None,
    json_path: Path,
) -> tuple[str | None, str]:
    """Map a legacy/simple nested-type descriptor to a source descriptor.

    Canonicalisation is intentionally conservative: it requires an exact
    descriptor, a unique source overload, an exact artifact suffix, or a
    unique parameter leaf signature.  Ambiguous overloads remain unresolved.
    """

    descriptor = str(artifact_descriptor or "").strip()
    if target is None:
        return (descriptor or None, "artifact" if descriptor else "")
    overloads = _source_overloads_for_target(target, method_name)
    source_descriptors = [
        str(item.get("descriptor") or "").strip()
        for item in overloads
        if str(item.get("descriptor") or "").strip()
    ]
    if descriptor and descriptor in source_descriptors:
        return descriptor, "artifact"
    if len(source_descriptors) == 1:
        canonical = source_descriptors[0]
        source = "source_unique"
        if descriptor and descriptor != canonical:
            source = "legacy_artifact_suffix_canonicalized"
        return canonical, source

    artifact_component = json_path.parent.name
    suffix = artifact_component.split("__", 1)[1] if "__" in artifact_component else ""
    suffix_normalized = _safe_path_component(suffix)
    suffix_matches = [
        candidate
        for candidate in source_descriptors
        if suffix_normalized
        and suffix_normalized
        in {
            _safe_path_component(_jvm_desc_to_short(candidate).replace(",", "_")),
            _safe_path_component(
                "_".join(_descriptor_leaf_signature(candidate))
            ),
        }
    ]
    if len(suffix_matches) == 1:
        canonical = suffix_matches[0]
        source = "artifact_suffix"
        if descriptor and descriptor != canonical:
            source = "legacy_artifact_suffix_canonicalized"
        return canonical, source

    if descriptor:
        leaf_signature = _descriptor_leaf_signature(descriptor)
        leaf_matches = [
            candidate
            for candidate in source_descriptors
            if _descriptor_leaf_signature(candidate) == leaf_signature
        ]
        if len(leaf_matches) == 1:
            return leaf_matches[0], "legacy_artifact_suffix_canonicalized"
        return descriptor, "artifact_unverified"
    return None, ""


def _source_class_fqn(source_path: Path | None, class_name: str) -> str:
    if source_path is None or not source_path.exists():
        return class_name
    try:
        source = source_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return class_name
    package_name, _imports, _wildcards = _source_package_and_imports(source)
    return f"{package_name}.{class_name}" if package_name else class_name


def _backfill_artifact_method_coverage(
    *,
    data: dict,
    project_root: Path,
    source_path: Path | None,
    class_name: str,
    method_name: str,
    target_descriptor: str | None,
) -> None:
    """Backfill a missing target row from the artifact or JaCoCo report."""

    report_coverage = (
        (data.get("report_summary") or {}).get("coverage")
        if isinstance(data.get("report_summary"), dict)
        else {}
    )
    report_target = (
        report_coverage.get("target_method")
        if isinstance(report_coverage, dict)
        else None
    )
    if isinstance(report_target, dict):
        report_name = str(report_target.get("method_name") or method_name)
        report_descriptor = str(report_target.get("descriptor") or "").strip()
        if (
            report_name == method_name
            and (not target_descriptor or not report_descriptor or report_descriptor == target_descriptor)
            and isinstance(report_target.get("line_coverage"), (int, float))
        ):
            row = dict(report_target)
            row["class_name"] = class_name
            row["method_name"] = method_name
            if target_descriptor:
                row["descriptor"] = target_descriptor
            data.setdefault("test_runs", []).append({"methods_coverage": [row]})
            return

    class_fqn = _source_class_fqn(source_path, class_name)
    try:
        from tatg.evaluation.coverage import parse_jacoco_xml_per_method
    except Exception:
        return
    for candidate in _evaluation_artifact_candidates(project_root, data):
        if candidate.name != "jacoco.xml":
            continue
        try:
            rows = parse_jacoco_xml_per_method(candidate, class_fqn=class_fqn)
        except Exception:
            continue
        matches = [
            row
            for row in rows
            if str(row.get("method_name") or "") == method_name
            and (
                not target_descriptor
                or str(row.get("descriptor") or "") == target_descriptor
            )
        ]
        if len(matches) != 1:
            continue
        row = dict(matches[0])
        row["class_name"] = class_name
        data.setdefault("test_runs", []).append({"methods_coverage": [row]})
        coverage = data.setdefault("report_summary", {}).setdefault("coverage", {})
        if isinstance(coverage, dict):
            coverage["coverage_measurement_status"] = "measured"
            coverage["target_method"] = dict(row)
        return


def _hard_method_report_row_from_existing_json(json_path: Path) -> dict:
    """Build one descriptor-level report row from a final or partial artifact."""

    data, artifact_source = _read_json_with_source(json_path)
    selector_class_name = _selector_class_name_from_artifact(json_path, data)
    target = _TARGETS_BY_NAME.get(selector_class_name)
    class_name = str(
        data.get("class_name")
        or (target.java_class_name if target is not None else selector_class_name)
        or selector_class_name
    ).strip()
    method_name = str(data.get("method_name") or "").strip()
    if not method_name:
        stem = json_path.stem
        method_part = stem.split(".", 1)[1] if "." in stem else stem
        method_name = method_part.split("__", 1)[0]

    artifact_descriptor = _resolved_target_descriptor(data, method_name)
    target_descriptor, descriptor_source = _canonical_artifact_descriptor(
        target=target,
        method_name=method_name,
        artifact_descriptor=artifact_descriptor,
        json_path=json_path,
    )

    # Work on a private copy because archive replay must remain read-only.
    normalized = json.loads(json.dumps(data)) if data else {}
    if target_descriptor and artifact_descriptor and target_descriptor != artifact_descriptor:
        for run in normalized.get("test_runs") or []:
            for method_row in run.get("methods_coverage") or []:
                if (
                    str(method_row.get("method_name") or "") == method_name
                    and str(method_row.get("descriptor") or "") == artifact_descriptor
                ):
                    method_row["descriptor"] = target_descriptor
        coverage = (
            (normalized.get("report_summary") or {}).get("coverage")
            if isinstance(normalized.get("report_summary"), dict)
            else {}
        )
        report_target = coverage.get("target_method") if isinstance(coverage, dict) else None
        if (
            isinstance(report_target, dict)
            and str(report_target.get("method_name") or method_name) == method_name
            and str(report_target.get("descriptor") or "") == artifact_descriptor
        ):
            report_target["descriptor"] = target_descriptor
    if target_descriptor:
        normalized["target_descriptor"] = target_descriptor

    project_root = Path(str(normalized.get("project_root") or ""))
    if not project_root.exists() and target is not None:
        project_root = target.subjects_root.resolve() / target.project
    source_path: Path | None = None
    source_value = str(normalized.get("source_file") or "").strip()
    if source_value:
        source_path = Path(source_value)
        if not source_path.is_absolute() and project_root:
            source_path = project_root / source_path
    elif target is not None:
        source_path = project_root / target.target_file

    _backfill_artifact_method_coverage(
        data=normalized,
        project_root=project_root,
        source_path=source_path,
        class_name=class_name,
        method_name=method_name,
        target_descriptor=target_descriptor,
    )
    if not target_descriptor:
        inferred_descriptor = _resolved_target_descriptor(normalized, method_name)
        if inferred_descriptor:
            target_descriptor = inferred_descriptor
            descriptor_source = "coverage_artifact"
            normalized["target_descriptor"] = inferred_descriptor

    final = normalized.get("final_validation") or {}
    return_codes = normalized.get("return_codes")
    latest = (normalized.get("test_runs") or [{}])[-1]
    best_method = _best_method_row(
        normalized,
        class_name,
        method_name,
        target_descriptor,
    )
    line_cov, branch_cov = _best_method_cov_pair(
        normalized,
        class_name,
        method_name,
        target_descriptor,
    )
    candidate_best = _candidate_best_target_coverage(
        normalized,
        method_name,
        target_descriptor,
    )
    candidate_failed_best = _candidate_best_failed_target_coverage(normalized)
    report_coverage = (
        (normalized.get("report_summary") or {}).get("coverage")
        if isinstance(normalized.get("report_summary"), dict)
        else {}
    )
    process_rc = normalized.get("process_rc")
    if not isinstance(process_rc, int):
        process_rc = final.get("return_code")
    if not isinstance(process_rc, int):
        process_rc = _pipeline_rc(return_codes)

    row = {
        "project": (
            normalized.get("project")
            or (target.project if target is not None else project_root.name)
        ),
        "selector_class_name": selector_class_name,
        "class_name": class_name,
        "method_name": method_name,
        "json_path": str(json_path),
        "process_rc": process_rc,
        "pipeline_rc": _pipeline_rc(return_codes),
        "return_codes": return_codes,
        "has_nonzero_intermediate_rc": _has_nonzero_intermediate_rc(return_codes),
        "final_validation_rc": final.get("return_code"),
        "final_failed_tests": final.get("failed_test_count"),
        "generated_test_count": normalized.get("generated_test_count"),
        "generation_timeout_count": int(
            (normalized.get("validation_policy") or {}).get("generation_timeout_count")
            or 0
        ),
        "preanalysis_elapsed_seconds": (
            normalized.get("tatg_preanalysis") or {}
        ).get("elapsed_seconds"),
        "pre_e2e_seconds": normalized.get("pre_e2e_seconds"),
        "e2e_seconds": normalized.get("e2e_seconds"),
        "coverage_missing_reason": (
            report_coverage.get("coverage_missing_reason")
            if isinstance(report_coverage, dict)
            else ""
        ),
        "reported_coverage_measurement_status": (
            normalized.get("coverage_measurement_status")
            or (
                report_coverage.get("coverage_measurement_status")
                if isinstance(report_coverage, dict)
                else None
            )
        ),
        "target_line_coverage": line_cov,
        "target_branch_coverage": branch_cov,
        "target_line_covered": (
            best_method.get("line_covered") if isinstance(best_method, dict) else None
        ),
        "target_line_missed": (
            best_method.get("line_missed") if isinstance(best_method, dict) else None
        ),
        "target_branch_covered": (
            best_method.get("branch_covered") if isinstance(best_method, dict) else None
        ),
        "target_branch_missed": (
            best_method.get("branch_missed") if isinstance(best_method, dict) else None
        ),
        "candidate_best_target_line_coverage": (
            candidate_best.get("line_coverage") if isinstance(candidate_best, dict) else None
        ),
        "candidate_best_target_branch_coverage": (
            candidate_best.get("branch_coverage") if isinstance(candidate_best, dict) else None
        ),
        "candidate_best_target_coverage_source": (
            candidate_best.get("source") if isinstance(candidate_best, dict) else None
        ),
        "candidate_best_target_test_method": (
            candidate_best.get("test_method") if isinstance(candidate_best, dict) else None
        ),
        "candidate_best_target_descriptor": (
            candidate_best.get("descriptor") if isinstance(candidate_best, dict) else None
        ),
        "candidate_failed_best_target_line_coverage": (
            candidate_failed_best.get("line_coverage")
            if isinstance(candidate_failed_best, dict)
            else None
        ),
        "candidate_failed_best_target_branch_coverage": (
            candidate_failed_best.get("branch_coverage")
            if isinstance(candidate_failed_best, dict)
            else None
        ),
        "candidate_failed_best_target_failure_category": (
            candidate_failed_best.get("failure_category")
            if isinstance(candidate_failed_best, dict)
            else None
        ),
        "partial_artifact_used": bool(
            artifact_source == "partial" or normalized.get("partial_artifact_used")
        ),
        "final_artifact_missing": bool(normalized.get("final_artifact_missing")),
        "partial_stage": (
            normalized.get("stage")
            if artifact_source == "partial" or normalized.get("partial_artifact_used")
            else None
        ),
        "target_descriptor": target_descriptor,
        "target_descriptor_source": descriptor_source,
        "overload_short": (
            _jvm_desc_to_short(str(target_descriptor)) if target_descriptor else ""
        ),
        "overall_line_coverage": latest.get("line_coverage"),
        "overall_branch_coverage": latest.get("branch_coverage"),
        "overloads": _per_overload_coverage(
            normalized,
            method_name,
            target_descriptor,
        ),
        "overload_tests": _per_overload_tests(normalized),
    }
    row.update(_mutation_status_summary(normalized, best_method))
    row.update(_streaming_validation_summary(normalized))
    row.update(_latest_obligation_summary(normalized))
    row["coverage_status"] = _coverage_status(row)
    row["coverage_measurement_status"] = _coverage_measurement_status(row)
    row["mutation_run_status"] = _mutation_run_status(row)
    row["dominant_missing_reason"] = _dominant_missing_reason(normalized, row)
    _refine_generic_measurement_status(row)
    line_display, branch_display = _coverage_display_values(row)
    row["target_line_coverage_display"] = line_display
    row["target_branch_coverage_display"] = branch_display
    row["branch_measurement_status"] = _branch_measurement_status(row)
    return row


def _report_row_quality(row: dict) -> tuple[int, int, float, int]:
    measured = str(row.get("coverage_measurement_status") or "") in {
        "measured",
        "measured_partial_artifact",
        "best_snapshot_only_measured",
        "combination_rejected_best_snapshot_measured",
    }
    explicit_descriptor = str(row.get("target_descriptor_source") or "") in {
        "artifact",
        "artifact_suffix",
    }
    score = float(row.get("target_line_coverage_display") or 0.0) + float(
        row.get("target_branch_coverage_display") or 0.0
    )
    final_artifact = not bool(row.get("partial_artifact_used"))
    return int(measured), int(explicit_descriptor), score, int(final_artifact)


def _rebuild_hard_method_report_from_artifacts(
    *,
    out_dir: Path,
    backend: str,
    wanted_labels: set[str] | None = None,
) -> int:
    """Rebuild an incremental-safe report without running any target."""

    artifact_root = out_dir / str(backend)
    candidates: list[dict] = []
    if artifact_root.exists():
        for json_path in sorted(artifact_root.rglob("*.json")):
            if json_path.name.endswith(".prompts.json") or "artifacts" in json_path.parts:
                continue
            row = _hard_method_report_row_from_existing_json(json_path)
            selector = str(row.get("selector_class_name") or row.get("class_name") or "")
            method_name = str(row.get("method_name") or "")
            short = str(row.get("overload_short") or "")
            labels = {
                f"{selector}.{method_name}",
                f"{selector}.{method_name}#{short}" if short else "",
            }
            if wanted_labels and not (labels & wanted_labels):
                continue
            candidates.append(row)

    best_by_identity: dict[tuple[str, str, str], dict] = {}
    for row in candidates:
        identity = (
            str(row.get("selector_class_name") or row.get("class_name") or ""),
            str(row.get("method_name") or ""),
            str(row.get("target_descriptor") or row.get("overload_short") or ""),
        )
        existing = best_by_identity.get(identity)
        if existing is None or _report_row_quality(row) > _report_row_quality(existing):
            best_by_identity[identity] = row
    rows = sorted(
        best_by_identity.values(),
        key=lambda row: (
            str(row.get("selector_class_name") or row.get("class_name") or ""),
            str(row.get("method_name") or ""),
            str(row.get("target_descriptor") or ""),
        ),
    )
    if not rows:
        return 1
    _write_hard_method_report(out_dir, rows)
    return 0


def _append_incremental_hard_method_report(out_dir: Path, row: dict) -> None:
    """Durably append one completed descriptor row during a long batch."""

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "hard_method_report.incremental.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=True))
        fh.write("\n")

    csv_path = out_dir / "hard_method_report.incremental.csv"
    fieldnames: list[str]
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if write_header:
        fieldnames = list(row.keys())
    else:
        try:
            with csv_path.open(encoding="utf-8", newline="") as fh:
                fieldnames = next(csv.reader(fh))
        except (OSError, StopIteration):
            fieldnames = list(row.keys())
            write_header = True
    with csv_path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _write_hard_method_report(out_dir: Path, hard_rows: list[dict]) -> None:
    csv_path = out_dir / "hard_method_report.csv"
    json_path = out_dir / "hard_method_report.json"
    md_path = out_dir / "hard_method_report.md"

    fieldnames = [
        "project", "class_name", "method_name", "overload_short", "target_descriptor", "json_path",
        "process_rc", "pipeline_rc", "return_codes",
        "has_nonzero_intermediate_rc", "final_validation_rc", "final_failed_tests",
        "generated_test_count", "coverage_status",
        "generated_artifact_file_count", "retained_test_file_count",
        "streaming_retained_target_positive_count",
        "retained_target_positive_count", "final_green_test_method_count",
        "final_suite_target_positive", "final_suite_green_test_count",
        "final_suite_reconciled_after_streaming", "passing_no_target_count",
        "coverage_positive_failed_count", "coverage_positive_repair_passed",
        "coverage_positive_repair_submitted", "coverage_positive_repair_rescued",
        "size_gated_round_skipped",
        "ladder_failed_items_seen",
        "streaming_validation_written_files", "streaming_validation_applied_candidates",
        "streaming_validation_passed", "streaming_validation_failed",
        "streaming_validation_target_coverage_gate_rejected", "streaming_validation_failed_detail_count",
        "streaming_validation_failure_types", "streaming_validation_gate_reasons",
        "generation_timeout_count", "preanalysis_elapsed_seconds",
        "pre_e2e_seconds", "e2e_seconds",
        "coverage_missing_reason", "dominant_missing_reason", "candidate_extraction_empty_count",
        "partial_artifact_used", "coverage_measurement_status",
        "partial_stage",
        "target_line_coverage", "target_branch_coverage",
        "target_line_coverage_display", "target_branch_coverage_display",
        "target_line_covered", "target_line_missed",
        "target_branch_covered", "target_branch_missed",
        "branch_measurement_status",
        "candidate_best_target_line_coverage", "candidate_best_target_branch_coverage",
        "candidate_best_target_coverage_source", "candidate_best_target_test_method",
        "candidate_failed_best_target_line_coverage", "candidate_failed_best_target_branch_coverage",
        "candidate_failed_best_target_failure_category",
        "target_mutation_score", "target_mutation_total", "target_mutation_raw_total",
        "target_mutation_non_viable", "target_mutation_killed",
        "target_mutation_survived", "target_mutation_no_coverage",
        "target_mutation_applicability_status",
        "mutation_status", "mutation_return_code", "mutation_error_hint", "mutation_junit5_plugin_injected",
        "pitest_tests_run", "pitest_retry_command_count", "mutation_feedback_useful",
        "mutation_run_status", "mutation_extra_iterations_run", "mutation_extra_round_skipped_reason",
        "mutation_extra_delta", "mutation_no_gain_reason",
        "coverage_obligation_satisfied", "coverage_obligation_total", "coverage_obligation_rate", "coverage_obligation_declared_ids", "coverage_obligation_streaming_ids",
        "overall_line_coverage", "overall_branch_coverage",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(hard_rows)
    json_path.write_text(json.dumps(hard_rows, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# TATG Hard-Method Report (All Projects)",
        "",
        "- `process_rc`: wrapper process return code for this method invocation.",
        "- `pipeline_rc`: max value in `return_codes` (intermediate/pipeline status).",
        "- `final_validation_rc`: final validation return code (`0` means final tests passed).",
        "- target line/branch: display coverage for the target method; raw values remain `null` in JSON when measurement was unavailable.",
        "- `coverage_measurement_status` explains whether display `0.0` means no retained/no target/infra failure rather than a measured JaCoCo row.",
        "- Branch coverage is shown as `N/A` when JaCoCo reports no branch denominator (`0/0`).",
        "- target mutation: exact descriptor mutation score when PITest generated a parseable `mutations.xml`; mutation status explains missing scores.",
        "- `written/applied/failed`: streaming validation promotion counters; use these with `pitest_tests_run` to distinguish generated candidates from retained, PITest-discoverable tests.",
        "- `coverage_status`: triage label: clean coverage, low/zero target hit, no retained tests, missing JSON/coverage, or coverage_proxy_only (JaCoCo missing but mutation score > 0 confirms method is exercised).",
        "",
        "| Method | project | status | measurement | generated | written | applied | failed | PITest tests | process_rc | pipeline_rc | final_rc | gen_timeout | preanalysis(s) | target line | target branch | target mutation | mutation status | mutation run | CGPO |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for row in hard_rows:
        method_display = f"{row['class_name']}.{row['method_name']}"
        if row.get("overload_short"):
            method_display += f"#{row.get('overload_short')}"
        lines.append(
            f"| {method_display}"
            f" | {row['project']}"
            f" | {row.get('coverage_status', '')}"
            f" | {row.get('coverage_measurement_status', '')}"
            f" | {row.get('generated_test_count', '')}"
            f" | {row.get('streaming_validation_written_files', '')}"
            f" | {row.get('streaming_validation_applied_candidates', '')}"
            f" | {row.get('streaming_validation_failed', '')}"
            f" | {row.get('pitest_tests_run', '')}"
            f" | {row['process_rc']}"
            f" | {row['pipeline_rc']}"
            f" | {row['final_validation_rc']}"
            f" | {row.get('generation_timeout_count', '')}"
            f" | {row.get('preanalysis_elapsed_seconds', '')}"
            f" | {_pct(row.get('target_line_coverage_display', row.get('target_line_coverage')))}"
            f" | {('N/A' if row.get('branch_measurement_status') == 'not_applicable_no_branches' else _pct(row.get('target_branch_coverage_display', row.get('target_branch_coverage'))))}"
            f" | {_pct(row.get('target_mutation_score'))}"
            f" | {row.get('mutation_status') or ''}:{row.get('mutation_error_hint') or ''}"
            f" | {row.get('mutation_run_status') or ''}"
            f" | {_format_cgpo_cell(row)} |"
        )
    # Per-overload breakdown for methods that have multiple overloads.
    overloaded_rows = [r for r in hard_rows if len(r.get("overloads") or []) > 1]
    if overloaded_rows:
        lines += [
            "",
            "## Per-Overload Coverage Breakdown",
            "",
            "| Method | target | overload params | line% | branch% | lines (cov/tot) | branches (cov/tot) | tests pass/fail |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        for row in overloaded_rows:
            ot = row.get("overload_tests") or {}
            for ov in (row.get("overloads") or []):
                short = ov["short"] or ov["descriptor"]
                # Match overload_tests by comparing the overload_id's type tag (e.g.
                # "writeString#String") against the first param type of this overload.
                first_param = short.split(",")[0].strip() if short else ""
                matched_tests = None
                for oid, tc in ot.items():
                    oid_type = oid.split("#", 1)[-1] if "#" in oid else oid
                    if oid_type and first_param == oid_type:
                        matched_tests = tc
                        break
                test_str = (
                    f"{matched_tests['passed']}/{matched_tests['failed']}"
                    if matched_tests else "—"
                )
                lc = ov["line_coverage"]
                bc = ov["branch_coverage"]
                lcov = ov["line_covered"]
                ltot = lcov + ov["line_missed"]
                bcov = ov["branch_covered"]
                btot = bcov + ov["branch_missed"]
                target_mark = "yes" if ov.get("is_target_descriptor") else ""
                lines.append(
                    f"| {row['class_name']}.{row['method_name']}"
                    f" | {target_mark}"
                    f" | `({short})`"
                    f" | {_pct(lc)}"
                    f" | {_pct_with_total(bc, btot)}"
                    f" | {lcov}/{ltot}"
                    f" | {bcov}/{btot}"
                    f" | {test_str} |"
                )

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"hard_method_report_csv={csv_path}")
    print(f"hard_method_report_json={json_path}")
    print(f"hard_method_report_md={md_path}")


def _format_cgpo_cell(row: dict) -> str:
    satisfied = row.get("coverage_obligation_satisfied")
    total = row.get("coverage_obligation_total")
    prefix = f"{satisfied if satisfied is not None else ''}/{total if total is not None else ''}"
    declared = str(row.get("coverage_obligation_declared_ids") or "")
    streaming = str(row.get("coverage_obligation_streaming_ids") or "")
    ids = declared or streaming
    if ids:
        count = len([item for item in ids.split(",") if item])
        return f"{prefix} ids={count}"
    return prefix


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    cli_args = _parse_cli_args()

    if bool(getattr(cli_args, "list_hard_methods", False)):
        print("Available HARD_METHODS selectors:")
        for class_name, methods in HARD_METHODS.items():
            target = _TARGETS_BY_NAME.get(class_name)
            group = f"  [{target.project}]" if target else ""
            for method_name in methods:
                print(f"  {class_name}.{method_name}{group}")
        return 0

    repo_root = _repo_root()
    out_dir = OUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Warn early if the Java 11 home is missing (needed for JacksonXml-5f, Jsoup-93f).
    if not JAVA11_HOME:
        print(
            "[WARN] TATG_JAVA11_HOME is not configured; targets requiring "
            "Java 11 may fail compilation.",
            file=sys.stderr,
        )
    elif not Path(JAVA11_HOME).exists():
        print(
            f"[WARN] TATG_JAVA11_HOME={JAVA11_HOME} not found — "
            "FromXmlParser, ToXmlGenerator, and Entities targets will fail compilation. "
            "Install Java 11 or update TATG_JAVA11_HOME.",
            file=sys.stderr,
        )

    env = os.environ.copy()
    try:
        api_config = resolve_api_config(
            api_base=API_BASE,
            model=MODEL,
            api_key_env=API_KEY_ENV,
            env=env,
        )
        if not bool(getattr(cli_args, "dry_run", DRY_RUN)):
            preflight_timeout = int(getattr(cli_args, "llm_timeout_seconds", LLM_TIMEOUT_SECONDS) or LLM_TIMEOUT_SECONDS)
            preflight_api_config(api_config, timeout_seconds=min(15, max(5, preflight_timeout)))
        apply_api_config_to_env(api_config, env)
    except Exception as exc:
        print(f"API configuration error: {exc}", file=sys.stderr)
        return 1

    src_path = str(repo_root / "src")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src_path if not existing_pythonpath else f"{src_path}{os.pathsep}{existing_pythonpath}"
    _augment_tool_path(env)

    tatg_preanalysis_timeout = max(
        1,
        int(getattr(cli_args, "tatg_preanalysis_timeout_seconds", TATG_PREANALYSIS_TIMEOUT_SECONDS) or TATG_PREANALYSIS_TIMEOUT_SECONDS),
    )
    tatg_branch_analysis_timeout = max(
        30,
        int(
            getattr(
                cli_args,
                "tatg_branch_analysis_timeout_seconds",
                TATG_BRANCH_ANALYSIS_TIMEOUT_SECONDS,
            )
            or TATG_BRANCH_ANALYSIS_TIMEOUT_SECONDS
        ),
    )
    tatg_generation_timeout = max(
        1,
        int(getattr(cli_args, "tatg_generation_timeout_seconds", TATG_GENERATION_TIMEOUT_SECONDS) or TATG_GENERATION_TIMEOUT_SECONDS),
    )
    max_iterations = max(1, int(getattr(cli_args, "max_iterations", MAX_ITERATIONS) or MAX_ITERATIONS))
    repair_iterations_raw = getattr(cli_args, "repair_iterations", REPAIR_ITERATIONS)
    repair_iterations = max(0, int(REPAIR_ITERATIONS if repair_iterations_raw is None else repair_iterations_raw))
    llm_timeout_seconds = max(1, int(getattr(cli_args, "llm_timeout_seconds", LLM_TIMEOUT_SECONDS) or LLM_TIMEOUT_SECONDS))
    e2e_timeout_seconds = max(0, int(getattr(cli_args, "e2e_timeout_seconds", E2E_TIMEOUT_SECONDS) or E2E_TIMEOUT_SECONDS))
    test_timeout_seconds = max(1, int(getattr(cli_args, "test_timeout_seconds", TEST_TIMEOUT_SECONDS) or TEST_TIMEOUT_SECONDS))
    llm_timeout_seconds = local_lmstudio_timeout_seconds(
        api_config.api_base,
        llm_timeout_seconds,
        operation="llm",
    )
    tatg_branch_analysis_timeout = local_lmstudio_timeout_seconds(
        api_config.api_base,
        tatg_branch_analysis_timeout,
        operation="branch_analysis",
    )
    tatg_generation_timeout = local_lmstudio_timeout_seconds(
        api_config.api_base,
        tatg_generation_timeout,
        operation="generation",
    )
    e2e_timeout_seconds = local_lmstudio_timeout_seconds(
        api_config.api_base,
        e2e_timeout_seconds,
        operation="e2e",
    )

    args = Namespace(
        out_dir=out_dir,
        python=PYTHON,
        api_base=api_config.api_base,
        model=api_config.model,
        api_key_env=api_config.api_key_env,
        max_iterations=max_iterations,
        repair_iterations=repair_iterations,
        mutation_extra_iterations=max(0, int(getattr(cli_args, "mutation_extra_iterations", 1) or 0)),
        streaming_repair_mode=str(getattr(cli_args, "streaming_repair_mode", STREAMING_REPAIR_MODE) or STREAMING_REPAIR_MODE),
        llm_timeout_seconds=llm_timeout_seconds,
        tatg_preanalysis_timeout_seconds=tatg_preanalysis_timeout,
        tatg_branch_analysis_timeout_seconds=tatg_branch_analysis_timeout,
        tatg_generation_timeout_seconds=tatg_generation_timeout,
        e2e_timeout_seconds=e2e_timeout_seconds,
        generation_parallelism=max(1, int(getattr(cli_args, "generation_parallelism", GENERATION_PARALLELISM) or GENERATION_PARALLELISM)),
        repair_parallelism=max(1, int(getattr(cli_args, "repair_parallelism", REPAIR_PARALLELISM) or REPAIR_PARALLELISM)),
        validation_parallelism=max(1, int(getattr(cli_args, "validation_parallelism", VALIDATION_PARALLELISM) or VALIDATION_PARALLELISM)),
        test_timeout_seconds=test_timeout_seconds,
        with_mutation=bool(getattr(cli_args, "with_mutation", WITH_MUTATION)),
        retain_existing_tests=bool(
            getattr(cli_args, "retain_existing_tests", False)
        ),
        retained_test_seed_manifest=str(
            getattr(cli_args, "retained_test_seed_manifest", "") or ""
        ),
        post_run_evaluate=bool(getattr(cli_args, "post_run_evaluate", POST_RUN_EVALUATE)),
        show_progress=bool(cli_args.show_progress),
        force=bool(cli_args.force),
        dry_run=bool(getattr(cli_args, "dry_run", DRY_RUN)),
        repo_root=repo_root,
        backend=str(cli_args.backend).lower(),
    )

    env["TATG_GENERATION_TIMEOUT_SECONDS"] = str(args.tatg_generation_timeout_seconds)
    for key, value in STATIC_ANALYSIS_DEFAULT_ENV.items():
        env.setdefault(key, value)
    if args.show_progress:
        env["TATG_FORCE_TTY_PROGRESS"] = "1"

    print(f"repo_root={repo_root}")
    print(f"out_dir={out_dir}")
    print(
        f"backend={args.backend} "
        f"max_iterations={args.max_iterations} repair_iterations={args.repair_iterations} "
        f"streaming_repair_mode={args.streaming_repair_mode} "
        f"llm_timeout_seconds={args.llm_timeout_seconds} "
        f"e2e_timeout_seconds={args.e2e_timeout_seconds} "
        f"test_timeout_seconds={args.test_timeout_seconds} "
        f"tatg_preanalysis_timeout_seconds={args.tatg_preanalysis_timeout_seconds} "
        f"tatg_branch_analysis_timeout_seconds={args.tatg_branch_analysis_timeout_seconds} "
        f"tatg_generation_timeout_seconds={args.tatg_generation_timeout_seconds} "
        f"parallelism={args.generation_parallelism}/{args.repair_parallelism}/{args.validation_parallelism} "
        f"with_mutation={args.with_mutation} post_run_evaluate={args.post_run_evaluate} "
        f"retain_existing_tests={args.retain_existing_tests} "
        f"retained_test_seed_manifest={args.retained_test_seed_manifest or 'none'} "
        f"show_progress={args.show_progress} force={args.force} dry_run={args.dry_run}"
    )
    print(
        "static_analysis_defaults="
        + ",".join(f"{key}={env.get(key)}" for key in sorted(STATIC_ANALYSIS_DEFAULT_ENV))
    )

    selector = _effective_method_selector(
        methods=str(getattr(cli_args, "methods", "") or ""),
        validation_suite=str(getattr(cli_args, "validation_suite", "") or ""),
    )
    selected = _select_hard_methods(selector)
    hard_rows: list[dict] = []
    rc = 0

    # Flatten into an ordered task list so we know the total count upfront.
    for class_name in selected:
        if class_name not in _TARGETS_BY_NAME:
            print(f"[WARN] {class_name} not in ALL_TARGETS — skipping.", file=sys.stderr)
    all_tasks = _expand_method_tasks(selected)
    for class_name, methods in selected.items():
        for method_name in methods:
            expanded = [
                task for task in all_tasks
                if task.class_name == class_name and (
                    task.method_name == method_name.split("#", 1)[0]
                )
            ]
            if len(expanded) > 1:
                print(
                    f"[{class_name}.{method_name}] expanding {len(expanded)} overload descriptors: "
                    + ", ".join(_jvm_desc_to_short(str(task.target_descriptor or "")) for task in expanded)
                )

    # Outer progress bar.  Written to stderr so it doesn't interfere with
    # stdout log lines.  When the inner rich panel is active (--show-progress),
    # the subprocess holds the TTY while our process is blocked in wait(), so
    # the bar is simply frozen during each run and updates when control returns.
    with _tqdm(
        all_tasks,
        desc="hard-methods",
        unit="method",
        file=sys.stderr,
        leave=True,
        dynamic_ncols=True,
        bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}",
    ) as pbar:
        for task in pbar:
            class_name, target, method_name = task.class_name, task.target, task.method_name
            target_descriptor = task.target_descriptor
            label = task.label
            pbar.set_postfix_str(label, refresh=True)

            process_rc = _run_one(
                args,
                target,
                env,
                method_name=method_name,
                target_descriptor=target_descriptor,
            )
            if args.dry_run:
                continue

            json_path = _method_json_path(args, target, method_name, target_descriptor)
            cov_line = _target_method_coverage(json_path, method_name, target_descriptor)
            pbar.write(f"[{label}] {cov_line}")

            data, artifact_source = _read_json_with_source(json_path)
            resolved_target_descriptor = _resolved_target_descriptor(
                data,
                method_name,
                target_descriptor,
            )
            final = data.get("final_validation") or {}
            return_codes = data.get("return_codes")
            latest = (data.get("test_runs") or [{}])[-1]
            best_method = _best_method_row(
                data,
                target.java_class_name,
                method_name,
                resolved_target_descriptor,
            )
            line_cov, branch_cov = _best_method_cov_pair(
                data,
                target.java_class_name,
                method_name,
                resolved_target_descriptor,
            )
            candidate_best = _candidate_best_target_coverage(
                data,
                method_name,
                resolved_target_descriptor,
            )
            candidate_failed_best = _candidate_best_failed_target_coverage(data)
            overloads = _per_overload_coverage(
                data,
                method_name,
                resolved_target_descriptor,
            )
            overload_tests = _per_overload_tests(data)
            if len(overloads) > 1:
                for ov in overloads:
                    pbar.write(
                        f"  [{label}({ov['short']}){' target' if ov.get('is_target_descriptor') else ''}] "
                        f"line={_pct(ov['line_coverage'])} branch={_pct_with_total(ov['branch_coverage'], int(ov['branch_covered'] or 0) + int(ov['branch_missed'] or 0))} "
                        f"lines={ov['line_covered']}/{ov['line_covered']+ov['line_missed']}"
                    )

            row = {
                "project": target.project,
                "class_name": class_name,
                "method_name": method_name,
                "json_path": str(json_path),
                "process_rc": process_rc,
                "pipeline_rc": _pipeline_rc(return_codes),
                "return_codes": return_codes,
                "has_nonzero_intermediate_rc": _has_nonzero_intermediate_rc(return_codes),
                "final_validation_rc": final.get("return_code"),
                "final_failed_tests": final.get("failed_test_count"),
                "generated_test_count": data.get("generated_test_count"),
                "generation_timeout_count": int(
                    (data.get("validation_policy") or {}).get("generation_timeout_count") or 0
                ),
                "preanalysis_elapsed_seconds": (data.get("tatg_preanalysis") or {}).get("elapsed_seconds"),
                "pre_e2e_seconds": data.get("pre_e2e_seconds"),
                "e2e_seconds": data.get("e2e_seconds"),
                "coverage_missing_reason": ((data.get("report_summary") or {}).get("coverage") or {}).get("coverage_missing_reason")
                if isinstance((data.get("report_summary") or {}).get("coverage"), dict)
                else "",
                "reported_coverage_measurement_status": (
                    data.get("coverage_measurement_status")
                    or (((data.get("report_summary") or {}).get("coverage") or {}).get("coverage_measurement_status")
                        if isinstance((data.get("report_summary") or {}).get("coverage"), dict)
                        else None)
                ),
                "target_line_coverage": line_cov,
                "target_branch_coverage": branch_cov,
                "target_line_covered": (
                    best_method.get("line_covered") if isinstance(best_method, dict) else None
                ),
                "target_line_missed": (
                    best_method.get("line_missed") if isinstance(best_method, dict) else None
                ),
                "target_branch_covered": (
                    best_method.get("branch_covered") if isinstance(best_method, dict) else None
                ),
                "target_branch_missed": (
                    best_method.get("branch_missed") if isinstance(best_method, dict) else None
                ),
                "candidate_best_target_line_coverage": (
                    candidate_best.get("line_coverage") if isinstance(candidate_best, dict) else None
                ),
                "candidate_best_target_branch_coverage": (
                    candidate_best.get("branch_coverage") if isinstance(candidate_best, dict) else None
                ),
                "candidate_best_target_coverage_source": (
                    candidate_best.get("source") if isinstance(candidate_best, dict) else None
                ),
                "candidate_best_target_test_method": (
                    candidate_best.get("test_method") if isinstance(candidate_best, dict) else None
                ),
                "candidate_best_target_descriptor": (
                    candidate_best.get("descriptor") if isinstance(candidate_best, dict) else None
                ),
                "candidate_failed_best_target_line_coverage": (
                    candidate_failed_best.get("line_coverage") if isinstance(candidate_failed_best, dict) else None
                ),
                "candidate_failed_best_target_branch_coverage": (
                    candidate_failed_best.get("branch_coverage") if isinstance(candidate_failed_best, dict) else None
                ),
                "candidate_failed_best_target_failure_category": (
                    candidate_failed_best.get("failure_category") if isinstance(candidate_failed_best, dict) else None
                ),
                "partial_artifact_used": bool(artifact_source == "partial" or data.get("partial_artifact_used")),
                "final_artifact_missing": bool(data.get("final_artifact_missing")),
                "partial_stage": data.get("stage") if artifact_source == "partial" or data.get("partial_artifact_used") else None,
                "target_descriptor": resolved_target_descriptor,
                "overload_short": (
                    _jvm_desc_to_short(str(resolved_target_descriptor))
                    if resolved_target_descriptor
                    else ""
                ),
                "tatg_preanalysis_details": (data.get("tatg_preanalysis") or {}).get("details"),
                "overall_line_coverage": latest.get("line_coverage"),
                "overall_branch_coverage": latest.get("branch_coverage"),
                "overloads": overloads,
                "overload_tests": overload_tests,
            }
            row.update(_mutation_status_summary(data, best_method))
            row.update(_streaming_validation_summary(data))
            row.update(_latest_obligation_summary(data))
            row["coverage_status"] = _coverage_status(row)
            row["coverage_measurement_status"] = _coverage_measurement_status(row)
            row["mutation_run_status"] = _mutation_run_status(row)
            row["dominant_missing_reason"] = _dominant_missing_reason(data, row)
            _refine_generic_measurement_status(row)
            line_display, branch_display = _coverage_display_values(row)
            row["target_line_coverage_display"] = line_display
            row["target_branch_coverage_display"] = branch_display
            row["branch_measurement_status"] = _branch_measurement_status(row)
            hard_rows.append(row)
            _append_incremental_hard_method_report(out_dir, row)
            if not _row_has_usable_final_result(row):
                rc = max(rc, process_rc or 1)

    if hard_rows:
        _write_hard_method_report(out_dir, hard_rows)

    # Stability check: pipeline errors paired with a clean final validation
    # mean at least one intermediate candidate/build failed. This is expected
    # during repair, but those methods should be reviewed separately from clean
    # coverage results.
    suspicious = [
        row for row in hard_rows
        if (row.get("pipeline_rc") or 0) != 0 and (row.get("final_validation_rc") or 0) == 0
    ]
    if suspicious:
        print(
            f"\n[WARN] {len(suspicious)} method(s) had pipeline_rc!=0 but final_validation_rc==0"
            " — intermediate candidates failed; inspect coverage_status before treating coverage as clean:"
        )
        for row in suspicious:
            print(
                f"  {row['class_name']}.{row['method_name']}  "
                f"pipeline_rc={row['pipeline_rc']} status={row.get('coverage_status')}"
            )

    print("hard-method batch complete")
    return rc


if __name__ == "__main__":
    sys.exit(main())
