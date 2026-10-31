# TATG

TATG is a target-aware, LLM-driven unit-test generator for Java. You give it a
focal method. It analyses the method's branches and reachability, asks an
OpenAI-compatible model for JUnit tests, validates each test in a sandbox,
repairs failing tests, and iterates on measured JaCoCo coverage and,
optionally, PITest mutation results.

This repository contains the TATG method and the harness that runs it: the
CLI, Maven/Gradle build integration, sandbox validation, repair, and coverage
and mutation evaluation. It was split out of the QuasarGen framework. The other
QuasarGen backends (`mutgen`, `cat`, `fusion`) and the mock-transformation
tools are not included.

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [How the pipeline works](#how-the-pipeline-works)
- [CLI reference](#cli-reference)
- [Outputs](#outputs)
- [Running the benchmark](#running-the-benchmark)
- [Smoke test without an LLM](#smoke-test-without-an-llm)
- [Configuration and environment variables](#configuration-and-environment-variables)
- [Development](#development)
- [Known limitations](#known-limitations)

## Repository layout

```text
src/tatg/
  cli.py                  CLI entry point (`tatg`) and the generate/validate/repair loop
  backends/tatg_backend.py  wraps the engine as schedulable work units
  engines/tatg_adapter.py   TATG engine: prompt construction, LLM patches, candidate extraction
  branch_analysis.py      LLM branch-dependency / reachability analysis
  coverage_obligations.py coverage obligations and frontier sketches
  objectives.py           coverage and mutation objectives fed back to the model
  repair.py               failure classification and repair prompts
  analysis/               static analysis: API surface, classpath, type info,
                          execution recipes, optional SootUp call graph
  evaluation/             JaCoCo coverage, PITest mutation, failing-test pruning
  build_system.py         Maven/Gradle detection, JDK/JUnit version inference
  runner.py, progress.py  parallel work-unit runner and terminal progress
  openai_compat.py, lmstudio.py, api_config.py
                          OpenAI-compatible client (optional LM Studio support)
scripts/                  benchmark batch runners and result analysis
benchmark/README.md       the 141 benchmark focal methods
examples/stub-smoke/      no-LLM end-to-end smoke test (stub server and canned tests)
tools/external/sootup-callgraph-pathfinder/
                          optional SootUp call-graph helper (Java)
docs/                     design notes, reviews, and measurement methodology
tests/                    pytest suite (about 3,750 tests)
```

## Requirements

- Python 3.10 or newer. The package uses only the standard library at
  runtime. `pytest` is needed for development.
- A JDK (`java` and `javac` on `PATH`) that can build the subject project.
  Older subjects that use `source/target 1.7` need JDK 11. Set
  `TATG_JAVA11_HOME` for the batch scripts.
- Maven (or Gradle) for the subject project. TATG injects JaCoCo 0.8.12 and
  PITest 1.17.0 through Maven plugin coordinates, so the first run needs access
  to Maven Central or a mirror.
- An OpenAI-compatible chat-completions endpoint. The model must support
  `response_format` with `json_schema`, because TATG relies on structured
  output.

## Installation

```bash
git clone <this repo> TATG && cd TATG
python -m pip install -e .            # installs the `tatg` command
python -m pip install -e '.[dev]'     # adds pytest
```

You can also run TATG without installing it: `PYTHONPATH=src python -m tatg.cli ...`.

## Quick start

Generate tests for one focal method:

```bash
export OPENAI_API_KEY=...               # the key is read from the env var named by --api-key-env

tatg generate \
  --scope method \
  --project-root /path/to/repo \
  --target-file  /path/to/repo/src/main/java/com/acme/Calculator.java \
  --class-name   Calculator \
  --method-name  add \
  --model        openai/gpt-4o-mini \
  --api-base     https://api.openai.com/v1 \
  --api-key-env  OPENAI_API_KEY \
  --max-iterations 2 \
  --repair-iterations 1 \
  --json-out ./artifacts/Calculator.add.json
```

Notes:

- `--backend` defaults to `tatg` and has no other value, so you can omit it.
- For overloaded methods, add `--target-descriptor` (for example
  `'(Ljava/lang/String;I)V'`) to pin one overload.
- If you don't pass `--test-command`, TATG infers a Maven or Gradle command.
  For the reliable per-target coverage that the benchmark uses, pass an
  explicit command that runs only the generated test class and writes a
  JaCoCo report. `scripts/run_all_hard_methods_batch.py` builds that command
  for you (see `_make_test_command`), and `examples/stub-smoke/run_smoke.sh`
  reuses it.
- Add `--with-mutation --mutation-extra-iterations N` to run PITest on the
  final suite and add mutation-directed rounds.
- Other scopes: `--scope class` (all methods of `--class-name`),
  `--scope singlemodule`, and `--scope multimodule` (with `--filter-modules`,
  `--filter-classes`, and `--filter-methods`).

## How the pipeline works

One method-scope run goes through these stages. Each stage below appears in
the `[tatg]` progress log and in the JSON report.

1. **Preflight.** TATG checks the API endpoint with a connectivity probe and
   checks the test command. It detects the build system and infers the
   project's Java and JUnit versions.
2. **Pre-analysis.** TATG extracts the focal method's source, its
   source-visible API surface, the classpath, type information, and
   construction recipes. If configured, it also runs the SootUp call graph to
   find source-visible entry paths. Private targets must be reached through a
   verified public entry point. Overloaded targets are bound to an explicit
   `overload_id` contract.
3. **Branch analysis.** The LLM is asked for a JSON branch-dependency analysis:
   the required state, the setup actions, and how each branch is reached. The
   result becomes branch rows, coverage obligations, and baseline skeletons.
4. **Generation.** The LLM returns a strict-JSON test patch
   (`new_tests[].test_code_lines`, `new_imports_lines`, `overload_id`,
   `candidate_role`, ...). TATG extracts and gates each candidate. Candidates
   are rejected if they violate the overload contract, call a private target
   directly, or duplicate retained behaviour.
5. **Streaming sandbox validation.** Each `@Test` compiles and runs in an
   isolated sandbox. Only stable, passing candidates are promoted into the
   project's test file.
6. **Repair.** Failing candidates go through classified repair prompts.
   Coverage-positive failures get priority.
7. **Coverage feedback.** TATG measures JaCoCo at method level. Uncovered
   lines and branches, plus unmet obligations, are fed back into further
   generation rounds, up to `--max-iterations`.
8. **Mutation (optional).** PITest runs on the target class with the generated
   test class. Surviving mutants drive extra rounds
   (`--mutation-extra-iterations`).
9. **Report.** TATG writes the final suite, coverage, mutation score,
   per-candidate audits, timings, and token usage to `--json-out`.

## CLI reference

```text
tatg generate         generate, validate, repair, and measure tests
tatg evaluate         read or produce JaCoCo / PITest metrics for existing tests
tatg salvage-replay   replay coverage-positive failed candidates from a report
tatg config init      write a starter JSON config (.tatg.generate.json)
```

Frequently used `generate` options (run `tatg generate --help` for the full
list):

| Option | Purpose |
|---|---|
| `--scope {method,class,singlemodule,multimodule}` | target scope |
| `--project-root`, `--module-path`, `--target-file`, `--class-name`, `--method-name`, `--target-descriptor` | target selection |
| `--model`, `--api-base`, `--api-key-env`, `--llm-timeout-seconds` | model endpoint |
| `--test-command`, `--test-command-map`, `--build-system`, `--java-version`, `--junit-version` | build and test execution |
| `--max-iterations`, `--repair-iterations`, `--streaming-repair-mode` | generation and repair budget |
| `--with-mutation`, `--mutation-extra-iterations`, `--no-mutation-extra-round` | mutation stage |
| `--tatg-preanalysis-timeout-seconds`, `--tatg-branch-analysis-timeout-seconds`, `--tatg-generation-timeout-seconds` | per-stage TATG time budgets |
| `--test-timeout-seconds`, `--e2e-timeout-seconds` | run time limits |
| `--generation-parallelism`, `--repair-parallelism`, `--validation-parallelism`, `--parallel-targets` | concurrency |
| `--retain-existing-tests` | keep and seed from existing tests instead of starting clean |
| `--sootup-jar`, `--skip-sootup-preflight`, `--static-analysis-json` | static-analysis inputs |
| `--json-out`, `--print-json`, `--log-prompts`, `--log-unrepaired-tests`, `--no-progress` | reporting |
| `--dry-run`, `--preflight-only` | plan or check without generating |

Evaluate an existing suite:

```bash
tatg evaluate --project-root /path/to/repo --coverage-type jacoco \
  --run-coverage-if-missing --run-mutation-if-missing \
  --json-out ./artifacts/eval.json
```

## Outputs

- **Generated tests** are written into the subject project (for example
  `src/test/java/.../CalculatorTest.java`). The project's `.java` files are
  backed up and restored around sandbox runs.
- **`<json-out>`** is the final report. Key fields:
  - `final_test_count`, `generated_test_count`, `compilation_rate`
  - `final_method_coverage` and `iteration_target_method_coverage`
    (line and branch coverage at target-method level)
  - `coverage_measurement_status`
  - `mutation_evaluation` (PITest command, status, and per-method score)
  - `candidate_audit`, `coverage_positive_failed_details`, `token_usage`,
    `e2e_seconds`
- **`<json-out>.partial`** is an incremental checkpoint for long or
  interrupted runs.
- **`<json-out>.prompts.jsonl`** is the prompt log, written with
  `--log-prompts`.

## Running the benchmark

[benchmark/README.md](benchmark/README.md) lists the 141 focal methods, drawn
from Defects4J subjects and recent GitHub projects.
`scripts/run_all_hard_methods_batch.py` runs them:

```bash
cp config/artifact.env.example config/artifact.env   # edit paths, model, and endpoint
set -a; source config/artifact.env; set +a

python scripts/run_all_hard_methods_batch.py --list-hard-methods
python scripts/run_all_hard_methods_batch.py --methods Lexer.nextToken,HelpFormatter.renderOptions \
  --api-base "$TATG_API_BASE" --model "$TATG_MODEL" --api-key-env "$TATG_API_KEY_ENV" \
  --max-iterations 2 --with-mutation --mutation-extra-iterations 1
python scripts/run_all_hard_methods_batch.py --dry-run          # print commands only
```

The subject checkouts are expected under `TATG_SUBJECTS_ROOT` (Defects4J
subjects with their original tests removed) and `TATG_NEW_SUBJECTS_ROOT`.

Other scripts:

| Script | Purpose |
|---|---|
| `run_hard_method_source_batch.py`, `run_new_projects_batch.py`, `run_four_class_source_batch.py` | other batch variants |
| `merge_tatg_eval_archive.py` | merge archived results using the current triage logic (offline, no LLM) |
| `run_archived_mutation_validation.py` | re-run mutation on archived final suites |
| `compare_subset_runs.py` | paired per-method A/B comparison of two runs |
| `analyze_batch_artifacts.py`, `monitor_tatg_logs.py` | batch diagnostics |

## Smoke test without an LLM

`examples/stub-smoke/` checks the whole harness with no model or API key.
`stub_llm.py` is a small OpenAI-compatible server. It answers each TATG prompt
type deterministically: probe, teardown, branch analysis (a minimal
schema-valid JSON), and generation (hand-written JUnit tests from `canned/`,
returned in TATG's `new_tests` patch format and bound to an allowed
`overload_id`). TATG then runs its real pipeline on those tests: extraction,
contract gates, sandbox compile and run, promotion, JaCoCo, feedback rounds,
and PITest.

```bash
git clone --depth 1 --branch rel/commons-csv-1.10.0 https://github.com/apache/commons-csv.git ~/subjects/commons-csv
rm -rf ~/subjects/commons-csv/src/test/java/org        # benchmark subjects run without original tests

PYTHON=$(which python) examples/stub-smoke/run_smoke.sh ~/subjects/commons-csv \
  src/main/java/org/apache/commons/csv/Lexer.java Lexer nextToken \
  examples/stub-smoke/canned/LexerTest.java --max-iterations 2
```

These are the results of the smoke run on benchmark focal methods. Coverage
is method-level JaCoCo. The numbers measure the harness, not generation
quality, because the stub always returns the same canned tests.

| Benchmark focal method | Subject | Result |
|---|---|---|
| `Lexer.nextToken` | commons-csv 1.10.0 | 4/4 candidates passed sandbox validation; line 66.7%, branch 57.9%; feedback round 2 ran and rejected duplicate candidates |
| `CSVFormat.printWithQuotes` (private, 2 overloads) | commons-csv 1.10.0 | Candidates without `overload_id` were rejected by the overload contract. With a bound `overload_id`, tests reached the target through `print(...)`: line 50.7%, branch 43.4% |
| `HelpFormatter.renderOptions` (`--with-mutation`) | commons-cli 1.6.0 | line 97.4%, branch 84.6%; PITest method mutation score 0.53 |

## Configuration and environment variables

`tatg generate` reads JSON defaults from `--config-file`, or from the first of
`.tatg.generate.json`, `tatg.generate.json`, `.tatg.json`, or `tatg.json`
that it finds. CLI flags take precedence over the config file. Create a
template with `tatg config init [--minimal]`. See
`examples/tatg.binance.multimodule.json` for a multi-module example.

Useful environment variables:

| Variable | Effect |
|---|---|
| `TATG_SUBJECTS_ROOT`, `TATG_NEW_SUBJECTS_ROOT`, `TATG_*_OUT_DIR`, `TATG_JAVA11_HOME`, `TATG_PYTHON` | paths used by the benchmark scripts (see `config/artifact.env.example`) |
| `TATG_MEMORY_DIR`, `TATG_IMPORT_LIBRARY_DIR` | persistent memory, such as the learned Java import library |
| `SOOTUP_CG_JAR`, `SOOTUP_JAVA` | enable the optional SootUp call-graph analysis |
| `TATG_OPENAI_FORCE_STREAMING`, `TATG_OPENAI_REASONING_EFFORT`, `TATG_SOCKET_RETRY_ATTEMPTS` | LLM client behaviour |
| `TATG_PITEST_TIMEOUT_FACTOR`, `TATG_PITEST_TIMEOUT_CONSTANT_MS`, `TATG_PITEST_JUNIT5_PLUGIN_VERSION` | PITest tuning |
| `TATG_RESPECT_JAVA_HOME` | use `JAVA_HOME` as-is instead of choosing a JDK for the target release |

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

The live end-to-end tests in `tests/test_live_generate_evaluate_integration.py`
are skipped by default. Enable them with `TATG_LIVE_E2E=1` plus
`TATG_LIVE_MODEL`, `TATG_LIVE_API_BASE`, and `TATG_LIVE_API_KEY_ENV`.

Two unit tests are environment-sensitive and also fail in the original
QuasarGen checkout:

- `test_run_all_hard_methods_batch.py::test_build_command_ignores_java_home_older_than_target_release`
  depends on the JDKs installed on the host.
- `test_streaming_validation.py::test_mockito_inline_attach_retry_detection_is_specific`
  fails when `JAVA_TOOL_OPTIONS` is set.

