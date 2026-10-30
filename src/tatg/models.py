from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Scope(str, Enum):
    METHOD = "method"
    CLASS = "class"
    SINGLE_MODULE = "singlemodule"
    MULTI_MODULE = "multimodule"


class PromptInsertPosition(str, Enum):
    BEFORE_SYSTEM = "before_system"
    AFTER_SYSTEM = "after_system"
    BEFORE_USER = "before_user"
    AFTER_USER = "after_user"


@dataclass(slots=True)
class Target:
    scope: Scope
    source_file: Path
    class_name: str | None = None
    method_name: str | None = None

    @property
    def label(self) -> str:
        chunks = [str(self.source_file)]
        if self.class_name:
            chunks.append(self.class_name)
        if self.method_name:
            chunks.append(self.method_name)
        return "::".join(chunks)


@dataclass(slots=True)
class PromptBundle:
    system: str
    user: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationRequest:
    backend: str
    scope: Scope
    project_root: Path
    target_file: Path | None
    class_name: str | None
    method_name: str | None
    model: str
    api_base: str | None
    api_key_env: str
    test_command: str
    coverage_report_path: str | None
    coverage_type: str
    prompt_extra_text: str
    prompt_extra_file: Path | None
    prompt_insert: PromptInsertPosition
    static_analysis_json: Path | None
    pass_specs: list[str]
    dry_run: bool
    max_iterations: int
    cat_engine_root: Path
    method_engine_root: Path
    target_descriptor: str = ""
    module_path: Path | None = None
    test_command_map: dict[str, list[str]] = field(default_factory=dict)
    java_version: str = ""
    junit_version: str = ""
    junit_family: str = ""
    repair_iterations: int = 1
    llm_timeout_seconds: int = 120
    test_timeout_seconds: int = 900
    e2e_timeout_seconds: int = 0
    parallel_targets: int = 1
    generation_parallelism: int = 1
    repair_parallelism: int = 1
    validation_parallelism: int = 4
    test_commands: list[str] = field(default_factory=list)
    filter_classes: list[str] = field(default_factory=list)
    filter_methods: list[str] = field(default_factory=list)
    filter_modules: list[str] = field(default_factory=list)
    repair_mode: str = "class"  # "class" | "method"
    streaming_repair_mode: str = "deferred"  # "deferred" | "eager"
    defer_test_writes: bool = False
    with_mutation: bool = False  # Append PITest mutationCoverage goal to the test command (Maven/Gradle only).
    mutation_extra_iterations: int = 1
    retain_existing_tests: bool = False
    retained_test_seed_evidence: dict[str, Any] = field(default_factory=dict)
    pre_generate_teardown: bool = True
    log_unrepaired_tests: bool = False
    log_prompts: bool = False
    log_prompts_limit: int = 20
    prompt_sidecar_path: Path | None = None
    prompt_sidecar_flushed_count: int = 0
    prompt_iteration: int = 1
    live_run_id: str = ""
    prompt_logs: list[dict[str, Any]] = field(default_factory=list)
    generation_coverage_feedback_by_label: dict[str, dict[str, Any]] = field(default_factory=dict)
    streaming_baseline_methods_coverage: list[dict[str, Any]] | None = None
    streaming_coverage_report_path: str | Path | None = None
    last_pre_feedback_pitest_result: dict[str, Any] = field(default_factory=dict)
    tatg_preanalysis_timeout_seconds: int = 180
    tatg_generation_timeout_seconds: int = 0
    tatg_preanalysis_cache: dict[str, Any] = field(default_factory=dict)
    tatg_preanalysis_stats: dict[str, Any] = field(default_factory=dict)
    tatg_branch_analysis_timeout_seconds: int = 600
    tatg_branch_analysis_cache: dict[str, Any] = field(default_factory=dict)
    tatg_branch_analysis_stats: dict[str, Any] = field(default_factory=dict)
    tatg_initial_branch_analysis_by_label: dict[str, dict[str, Any]] = field(default_factory=dict)
    tatg_initial_branch_analysis_stats: dict[str, Any] = field(default_factory=dict)
    tatg_mutation_semantic_verification_by_label: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    tatg_target_positive_test_methods_by_label: dict[str, list[str]] = field(
        default_factory=dict
    )
    tatg_structural_duplicate_retry_by_label: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    memory_dir: Path | None = None
    mutation_candidate_test_files: list[str] = field(default_factory=list)
    """When True, parse the final test-runner output after all repair rounds and
    persist any still-failing test method names in the JSON report under
    ``unrepaired_tests``.  Off by default (experimental)."""
    """Run a one-shot ``@AfterEach``/``@After`` generation pass before the per-method
    generation loop.  A lightweight source-scan heuristic decides whether a given
    target class is likely to need teardown (closeables, executor services, JDBC
    connections, mockStatic, …); only then is an LLM call made.  The generated
    teardown is patched into the freshly-seeded skeleton and re-validated by the
    project test command.  If validation fails, the teardown edit is rolled back
    and per-method generation proceeds without teardown (best-effort fallback)."""
