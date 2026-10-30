# TATG Optimization Plan

**Scope:** TATG generation pipeline.
**Goal:** mean branch coverage over the 141-method benchmark ≥ 0.70 (currently **0.6444**), then mean mutation score ≥ 0.70.
**Protocol constraint:** every strategy is first evaluated on a 28-method stratified subset (~4 h/round) and only promoted to the full 141 after it clears the acceptance gate.

---

## Part 1 — Diagnosis

All numbers below come from `branch-recovery-20260804-10/merged_replay_rows.json` (141 rows) and from the raw run artifacts under `tatg-eval-archives/branch-*/`.

### 1.1 Where the benchmark actually stands

| Branch-coverage bucket | Count |
|---|---|
| ≥ 0.70 | 60 |
| 0.50 – 0.70 | 43 |
| 0.30 – 0.50 | 23 |
| < 0.30 | 9 |
| no value | 6 |

Mean 0.6444, median 0.6667, n = 135. To lift every sub-70 % method to exactly 0.70 requires **+404 covered branches out of the 1 939 branches** those 81 methods contain. To lift the *mean* to 0.70 requires roughly +7.6 pp averaged over the 141 — i.e. about **+11 pp concentrated on the 81 methods below the line**.

### 1.2 The central finding: the LLM's branch reasoning is discarded before it can act

Aggregated over **691 branch-analysis detail blocks** in the raw archives:

```
raw LLM branch rows      2988
validated rows           3332   (includes rows added by later stages)
hard_eligible rows          75   →  2.5 %
```

In the merged 141-method run specifically, **`hard_eligible_count` is 0 in all 19 detail blocks**. Not one LLM-derived branch row became a binding obligation.

Why rows are rejected (counts across the 691 blocks):

| advisory reason | count | gate |
|---|---:|---|
| `required_state_not_established` | 2637 | `branch_analysis.py:19905` — all-or-nothing over up to 8 facts |
| `unmodeled_ancestor_control_path` | 1335 | `branch_analysis.py:19929` — every ancestor predicate must be modelled *and* set up |
| `observable_oracle_not_concrete` | 1331 | `_observable_oracle_is_concrete` @17414 — an **English-phrase blacklist** |
| `external_dependency_not_established` | 790 | `branch_analysis.py:19871` — all-or-nothing |
| `confidence_below_threshold` | 604 | `20065` — and `confidence` was already **deterministically capped to 0.5** at `20032` whenever no required-state fact verified, so this double-fires |
| `unverified_setup_actions` | 544 | `18405/18765` — one bad line out of 16 kills the row |
| `entry_input_projection_requires_replay` | 526 | `19988` |
| `field_relation_not_established` | 514 | `19905` |
| `target_entry_not_verified` | 456 | `19980` |
| `anchor_diversity_recovery_requires_candidate_replay` | 368 | `33645` — **unconditional** demotion regardless of validation |

`hard_eligible` is a 21-clause conjunction (`branch_analysis.py:20042-20067`). Most clauses are themselves all-or-nothing quantifiers over lists. The probability that a rich, multi-fact branch row clears all 21 is, empirically, 2.5 %.

**Consequence.** The system has three parallel guidance tracks and the wrong one is driving:

| Track | Mechanism | Status |
|---|---|---|
| **A** — deterministic rule engine | `coverage_obligations.py:2792-3682`: 20 hard-coded categories (`StatefulCursor`, `TokenPathMatrix`, `ModeDispatch`, …), regex-triggered, hand-written `scenario`/`program_fact`/`required_setup`/`oracle` prose | **This is what actually produces the O-ids that generation is contractually required to satisfy.** |
| **B** — LLM semantic branch analysis | `branch_analysis.py:analyze_branch_dependencies`, 11 LLM call sites, deep reasoning | 97.5 % demoted to `advisory_only`; survivors become SK-sketches which carry **no O-id**, cannot be required, are capped at 4 (`20534`, `del sketches[4:]` @21608), and several candidate classes are capped at **1** (`20687`, `20696`, `20711`) |
| **C** — raw prompt context | ~40 sections, 60 k char budget | Carries the rest |

The paper's claim is that Track B replaced Track A. In the shipped pipeline, Track A is load-bearing and Track B is decoration. This also explains the shape of the last three weeks of work: 755 archive directories, dominated by per-target rescue runs (`branch-ruler-*`, `branch-numberutils-rescue-*`, `branch-doublemetaphone-seeded-*`). Those tune Track A on individual benchmark methods, which is exactly the plateau.

### 1.3 Obligations are declared but almost never satisfied

Of the rows that record obligation accounting: **mean `coverage_obligation_rate` = 0.008**, with totals like `(7 declared, 0 satisfied)`, `(14, 1)`, `(16, 0)`, `(13, 0)`. The unified O-id contract is being emitted and then not met. So Track A is not converting either — it is producing *volume*, not *reachability*.

### 1.4 The feedback loop saturates at iteration 3

Marginal branch-coverage gain per additional iteration, over the merged run:

| iteration | n | mean Δbranch | positive | zero | negative |
|---|---:|---:|---:|---:|---:|
| 2 | 92 | **+0.1676** | 74 | 18 | 0 |
| 3 | 67 | **+0.0750** | 37 | 30 | 0 |
| 4 | 18 | **+0.0085** | 4 | 14 | 0 |
| 5 | 1 | +0.0286 | 1 | 0 | 0 |

By phase: `regular_generation` +0.1645 (n=82), `regular_generation_feedback` +0.0848 (n=75, 37 of them exactly zero), `mutation_extra` +0.0405 (n=16).

The cause is mechanical, not stochastic. The branch analysis is **cached by content hash** (`_analysis_cache_key` @15702, `_BRANCH_ANALYSIS_SCHEMA_VERSION = 50`) and the gates are deterministic. Round *N+1* sees the same source, produces the same rows, hits the same 21 clauses, gets the same rejections, and renders a near-identical prompt. The `tatg_structural_duplicate_retry_by_label` path exists precisely because the model then emits behaviourally duplicate tests.

**Attribution warning:** a third generation iteration alone is worth roughly +7.5 pp on methods that need it. That is budget, not method. It must be run as a control arm so the paper does not credit a method contribution to extra compute.

### 1.5 Coverage-positive tests are thrown away

Across the 141: **815 candidate target-positive tests, 749 retained, 234 coverage-positive failures** on 53 methods. Of the rows that record a category for the best *failing* target-positive candidate:

| failure category | count |
|---|---:|
| `oracle-mismatch` | 29 |
| `runtime-validation-failure` | 4 |
| `state-sequence-or-oracle-mismatch` | 3 |
| `missing-required-initialization` | 2 |
| other | 3 |

Worst cases: `Rotation.getAngles` — 30 candidates reached the target, **13 retained, 17 discarded**, final branch 0.386. `CSVParser.createHeaders` — 20 candidates, 7 retained, 13 discarded. `JsonRuleCompiler.writeRules` — 35 candidates, 22 retained, 13 discarded.

These tests *executed the branch*. They were deleted because the assertion was wrong. Nothing in the pipeline tries to repair an oracle from the observed value.

### 1.6 Measurement integrity is too weak to accept or reject a strategy

`branch_measurement_status` over the 141: **`coverage_unmeasured` 112, `measured` 29, `not_applicable_no_branches` 3**. `coverage_status = no_json` on 48 rows. Reported coverage mostly derives from `streaming_candidate_combination:*` snapshots (157 + 146 occurrences) rather than a final measured JaCoCo report.

At that noise level a +3 pp effect cannot be distinguished from run-to-run variance, which is a plausible reason several past "optimizations" looked ambiguous. **This has to be fixed before anything else is evaluated.**

### 1.7 Structural gaps that cap the ceiling

- **`external_collaborator` is a dead-end kind.** Supported by `safe_advisory` and `deterministic_dependency_advisory` only; explicitly excluded from `scaffold_delta_candidate` (`20766`), `_semantic_feasibility_candidate_rows` (`22140`) and `_source_visible_semantic_sandbox_contract_is_complete` (`24600`). It can never reach a replay/sandbox proof. 790 rejections. This kind dominates `adk-java`, `a2a-java`, `batch-processing-gateway`, `agentscope-java`.
- **`unknown` drops out of everything** except the weakest advisory path (`20644`).
- **`local_or_unknown` derived roots** → hard `return False, "derived_state_root_kind_not_supported"` (`11277`).
- **Benchmark-specific literals still live on the default path**, which will not survive review: the commons-math `getRowDimension|getColumnDimension` regex in **4 copies** (`recipe_synthesizer.py:45,127`; `coverage_obligations.py:667,731`); the commons-lang `NumberUtils` radix/suffix detector (`coverage_obligations.py:1268,1252`); Jackson's private `_inputBuffer` field name emitted into prompts for arbitrary projects (`recipe_synthesizer.py:1968`); Anthropic-SDK accessors inside `_DATA_FLOW_METHODS` (`coverage_obligations.py:2109`); `text`/`content` builder-setter ranking (`tatg_adapter.py:10235`); `@curlcommand` (`tatg_adapter.py:6531`). A further ~340 lines of verbatim AWS event-ruler rule JSON sit at `coverage_obligations.py:1552-1644` behind a constant-`False` gate — disabled, but present.

### 1.8 One-line summary

> Coverage is not limited by what the LLM can infer. It is limited by a static verification wall that rejects 97.5 % of what the LLM infers, a deterministic rule engine that fills the gap with benchmark-shaped guesses, a feedback loop that deterministically repeats itself after round 2, and a validator that deletes tests which already reached the branch.

---

## Part 2 — Strategy

### 2.1 The design inversion

Today the pipeline tries to **prove statically that a test plan will work**, and vetoes the plan when it cannot complete the proof. That is the wrong direction: static proof of reachability in Java with collaborators, protocol state and nested guards is exactly the hard problem, and every incomplete proof becomes a rejection.

Instead: **cheaply execute the plan and let JaCoCo be the judge.** Validation elapsed time is already ~15 s mean per snapshot, so a single-test micro-probe is affordable. Execution is also the only *general* mechanism available — it needs no category, no keyword, no per-project rule.

Static analysis is retained, but demoted from **veto** to **prior**: it orders which plans get probed first.

### 2.2 The unified representation

Replace the twin taxonomies (20 obligation categories + 12 recipe categories + 8 dependency kinds + SK/O/I/M/C/K id families) with a single IR. Everything the LLM infers lands in this shape; one executor handles all of it.

```
BranchGoal
  anchor            : {file, line, governing_predicate}      # source-verified, exact text
  path_condition    : [Clause]                                # ordered, outermost guard first
  outcome           : "taken" | "not_taken"

Clause
  subject   : SymbolRef        # param#i | receiver.field | collaborator(c).m() | cursor_state | static.CONST
  relation  : ==, !=, <, <=, >, >=, in, instanceof, isNull, nonNull, length, matches
  operand   : Literal | Expr

Achievement                    # how to make one Clause true — the only extensible axis
  kind     : direct_argument | constructor_arg | public_writer_call | builder_step
           | protocol_prefix  | collaborator_stub | reflection_field_set | input_encoding
  program  : [JavaStmt]        # executable, ordered
  evidence : SourceCitation | null
  status   : proven | probe_required | refuted        # set by the probe, not by a predicate ladder

Oracle
  expression : JavaExpr        # must compile and evaluate deterministically
  provenance : source_contract | differential | exception | observed_value
  status     : proven | observed_pending_adjudication | refuted
```

Three properties matter:

1. **Partial credit.** A `BranchGoal` carries *k* clauses with independent statuses. A goal with 5 proven and 1 `probe_required` clause is a live, actionable goal — today it is a rejected row. This alone addresses the 2637 `required_state_not_established` rejections.
2. **`Achievement.kind` is the single extension point.** Adding support for a new construction pattern means adding an achievement kind and a renderer, not a new obligation category with hand-written prose. `external_collaborator` and `unknown` stop being special-cased because they are just clauses whose achievement is `collaborator_stub` / `probe_required`.
3. **Proof is empirical.** `status` comes from the probe. `_observable_oracle_is_concrete`'s English blacklist is replaced by "compiles, runs, and yields the same value twice."

### 2.3 Strategy ranking (expected effect vs. cost)

| # | Strategy | Targets | Expected mean Δbranch | Cost |
|---|---|---|---:|---|
| S0 | Measurement integrity | §1.6 | 0 (enabling) | low |
| S1 | Anchor probe executor | enabling | 0 (enabling) | medium |
| S2 | Execution-backed admission (replaces `hard_eligible`) | §1.2, §1.7 | **+4 … +8 pp** | high |
| S3 | Oracle recovery from observed values | §1.5 | **+2 … +4 pp** | medium |
| S4 | Non-repeating attempt ledger | §1.4 | **+2 … +4 pp** | medium |
| S5 | Path-condition decomposition | `unmodeled_ancestor_control_path` (1335) | **+1 … +3 pp** | medium |
| S6 | First-class `external_collaborator` | 790 rejections | **+1 … +3 pp** | medium |
| S7 | Retire Track A + benchmark literals | §1.7, paper claim | **0 (must not drop)** | high |
| S8 | Mutation strengthening | phase 2 goal | — | medium |

Estimates are from the rejection volumes and the discard counts, not from prior runs. They are hypotheses to be falsified by the subset protocol, not commitments.

---

## Part 3 — Phased plan

### Phase 0 — Measurement integrity and harness *(prerequisite, claims nothing)*

**Problem.** 112/141 targets end `coverage_unmeasured`; the reported number comes from streaming candidate snapshots.

**Changes**
1. Force a **final measured JaCoCo run** per target after the last iteration. If it cannot be produced, report `target_branch_coverage = null` with a reason — never substitute a snapshot value.
2. Split the reported metric into `branch_coverage_measured` and `branch_coverage_snapshot_estimate`; the merged summary averages **only measured**, and prints `measured_fraction`.
3. Surface per-target into the merged report (all already computed, just not aggregated): `raw_branch_count`, `validated_count`, `hard_eligible_count`, `advisory_reason_counts`, `coverage_obligation_total/satisfied`, `candidate_target_positive_count`, `retained_target_positive_count`, `coverage_positive_failed_details[].category`.
4. Add `scripts/compare_subset_runs.py`: paired per-method A/B against a frozen baseline, emitting mean Δ, per-method Δ table, regression list, Wilcoxon signed-rank p, and wall-clock delta.
5. Freeze a **baseline re-run** of the 28-method subset at the exact experimental configuration (2 generation + 1 mutation, `--force`, same seed, same model) so every later comparison is paired against a run made under the current code, not against the historical archive.

**Exit criteria:** `measured_fraction ≥ 0.95` on the subset; two consecutive baseline re-runs agree to within ±1.5 pp mean. If they do not, raise `--validation-parallelism` determinism or pin surefire ordering before proceeding — an unstable baseline invalidates every subsequent decision.

---

### Phase 1 — Anchor probe executor *(the general mechanism)*

**New module** `src/tatg/probe.py`.

```
probe(target, java_stmts, act_call, oracle_expr|None) ->
  { compiles: bool, runs: bool, passes: bool|None,
    anchor_lines_covered: [int], branch_ids_covered: [str],
    observed_values: {expr -> repr}, stderr_digest: str, elapsed: float }
```

Implementation notes:
- Emits one throwaway `@Test` into an isolated generated test class; runs `mvn test -Dtest=<Probe>` scoped to that single method with the JaCoCo agent already configured by `build_system.py`; parses the anchor's line/branch status out of `jacoco.xml`.
- **Observed-value capture:** when `oracle_expr` is null, the probe wraps the act call and records `String.valueOf(result)` plus each `Clause.subject` reading, so S3 has data.
- **Caching:** key on `(target descriptor, sha256(program), sha256(act), oracle)`. Deterministic replays are free.
- **Budget:** `TATG_PROBE_MAX_PER_TARGET` (default 12), `TATG_PROBE_TIMEOUT_SECONDS` (default 90), and a per-target wall-clock ceiling so probing can never consume more than ~25 % of `--tatg-generation-timeout-seconds`.
- Reuse `evaluation/coverage.py` for the JaCoCo parse and `build_system.py` for the command; do not fork a second build path.

**Exit criteria:** probe round-trip median ≤ 25 s on the subset; cache hit rate ≥ 40 % within a target; zero project-file mutation leaking past a probe (assert via the existing project-restore machinery).

---

### Phase 2 — Execution-backed admission *(S2, the main lever)*

This is where the coverage comes from.

**2a. Decompose the conjunction.**
`branch_analysis.py:20042-20067`: `hard_eligible` stops being a boolean. Each clause of the current conjunction becomes a per-`Clause` or per-`Achievement` status, and the row carries `clauses_proven / clauses_total`. Gate on *actionability*, not completeness: a row is actionable if the anchor is source-verified and at least one clause has a renderable achievement.

Specifically retire as vetoes (keep as priors):
- `required_state_setup_verified` (`19905`) → per-fact status
- `ancestor_control_path_setup_verified` (`19929`) → handled by Phase 5
- `external_dependencies_verified` (`19871`) → per-dependency status
- `unverified_actions` (`18405/18765`) → per-action status; unverified actions become `probe_required`
- `entry_input_projection_verified` (`19988`) → probe decides
- the `confidence ≥ 0.65` clause (`20065`) and, critically, **delete the `confidence = min(confidence, 0.5)` cap at `20032-20033`** — it exists only to force the same rejection twice.

**2b. Replace the oracle classifier.**
Delete `_observable_oracle_is_concrete` (`17414-17462`) — a blacklist of English phrases plus a fixed word set, and itself one of the hard-coded heuristics the paper claims to have removed. New rule: an oracle is concrete iff its `expression` compiles inside the probe and evaluates to the same value on two consecutive probe runs. 1331 rejections re-enter the pipeline.

**2c. Probe-first admission loop.**
Per target, per uncovered anchor: rank candidate `BranchGoal`s by (clauses proven ÷ clauses total, then static confidence), probe the top *k* (`k` ≤ 4), and promote any goal whose probe covers the anchor to a **hard obligation with an O-id**. Probe outcomes write back into the goal (`refuted` clauses get pruned, `probe_required` ones become `proven`).

This is the unification the paper needs: the LLM's semantic inference becomes binding *because it executed*, not because it satisfied 21 static predicates.

**2d. Lift the caps once admission is real.**
`20534` `[:4]`, `21608` `del sketches[4:]`, and the `1`-item budgets at `20687/20696/20711` were sized for advisory noise. With execution-backed admission they should scale with `TATG_PROBE_MAX_PER_TARGET`. Also remove the two unconditional demotions at `33645-33663` and `34528-34549` — a probe-proven row must not be demoted because of the *stage* that produced it.

**Also fix:** `derived_state_root_kind_not_supported` (`11277`) — `local_or_unknown` roots become `probe_required` rather than a hard `False`.

**Instrumented exit criteria:** `hard_eligible` (renamed `probe_admitted`) rate rises from 2.5 % to **≥ 35 %**; `coverage_obligation_rate` rises from 0.008 to **≥ 0.25**; subset mean Δbranch ≥ +4 pp.

---

### Phase 3 — Oracle recovery from observed values *(S3)*

Recovers the 234 discarded coverage-positive tests.

**Trigger.** A candidate is target-positive (its execution touched new anchor lines) but failed validation with a category in `{oracle-mismatch, state-sequence-or-oracle-mismatch}`. Hook in at `repair.py:1555-1613`, adding a branch to the repair-mode ladder *before* the existing `has_expected_actual` arm.

**Procedure**
1. Re-run the test through `probe()` with the assertion stripped, capturing `observed_values`.
2. Ask the LLM one bounded question, with the method source, the assertion, expected and observed: *is the observed value the specified behaviour of this code, or evidence that the test's setup is unsound?* Structured output: `{verdict: specified|unsound|unknown, justification, source_citation}`.
3. `specified` **and** a source citation that resolves → rewrite the assertion to the observed value, re-validate, retain.
4. `unsound` or `unknown` → discard, and record the reason in the attempt ledger (Phase 4) so the same approach is not re-proposed.

**Guard rails — this is the risky strategy.** An observed-value oracle is a tautology if it is not constrained, and would inflate branch coverage while destroying mutation score.
- Never accept `observed_value` provenance for a clause the LLM marked `refuted`.
- Every recovered oracle is tagged `provenance = observed_value` and is **excluded from the mutation-adequacy count** unless it later kills at least one mutant.
- Add a subset-level guard to the acceptance rule: if S3 raises branch coverage but lowers mutation score on the mutation-enabled subset arm, **reject it**. The two goals are 70 % / 70 %; an oracle strategy that trades one for the other is not a win.
- Cap recovered oracles at 30 % of retained tests per target.

**Exit criteria:** ≥ 40 % of coverage-positive failures recovered; subset mean Δbranch ≥ +2 pp; **mutation score not lower** than baseline on the same subset.

---

### Phase 4 — Non-repeating attempt ledger *(S4, kills the plateau)*

**Problem.** Deterministic gates + content-hashed analysis cache ⇒ round 3 restates round 2.

**Change.** A per-target, cross-iteration ledger persisted alongside the run artifacts:

```
LedgerEntry: { anchor_id, clause_signature, achievement_kind,
               outcome: covered|not_covered|compile_fail|runtime_fail|oracle_fail,
               probe_evidence, iteration }
```

Rules:
1. An `(anchor, clause_signature, achievement_kind)` triple that produced `not_covered` or `compile_fail` **may not be re-proposed**; it is rendered into the prompt as an explicit exclusion with its evidence.
2. On repeated failure for an anchor, **escalate the achievement kind** along a fixed ladder: `direct_argument → constructor_arg → public_writer_call → builder_step → protocol_prefix → collaborator_stub → reflection_field_set`. The ladder is over *representations*, not over method names — it is project-agnostic.
3. Feed the ledger into `_analysis_cache_key` (`15702`) so round *N+1* genuinely re-analyses instead of returning the round-*N* object.
4. Anchors with `TATG_LEDGER_MAX_ATTEMPTS` (default 3) exhausted attempts are marked `probe_infeasible` and de-prioritised, releasing budget to reachable anchors.

**Exit criteria:** mean Δbranch at iteration 3 rises from +0.075 to ≥ +0.12, **and** iteration-2 gain rises — the point is to make 2 iterations do what 4 do now, since the production protocol is 2+2. Duplicate-test retry rate (`tatg_structural_duplicate_retry_by_label`) falls by ≥ 50 %.

---

### Phase 5 — Path-condition decomposition *(S5)*

**Problem.** `unmodeled_ancestor_control_path` — 1335 rejections. Deeply nested guards are represented as a single predicate, so the row can never be "modelled and set up".

**Change.** `BranchGoal.path_condition` is an ordered conjunction from the outermost enclosing guard inward. The solver walks it: satisfy clause 1, probe, confirm the outer guard is taken, then clause 2, and so on. Each step is verified by execution, so a 5-deep nest becomes 5 cheap probes instead of one unprovable predicate. Partially-satisfied paths are still valuable — they are exactly the "partial_progress" lifecycle the renderer already understands (`_coverage_obligation_lifecycle_status` @12996).

Also replaces `_concrete_control_path_node_verdict` (`9777`) symbolic evaluation with observed control flow.

**Exit criteria:** on the deep-nesting subset members (`Rotation.getAngles`, `RuleCompiler.processMatchExpression`, `AbstractPatriciaTrie.put`, `FilteringParserDelegate.nextToken`), mean Δbranch ≥ +6 pp with no regression elsewhere.

---

### Phase 6 — First-class `external_collaborator` *(S6)*

**Change.** Add `external_collaborator` (and `unknown` with a resolvable subject) to the candidate sets at `20766`, `22140`, `24600`. A collaborator clause gets `Achievement.kind = collaborator_stub` whose `program` is a Mockito stub or an anonymous implementation synthesised from the *source-visible interface*, then verified by probe. `_extract_collaborator_stub_contracts` (`tatg_adapter.py:23985`) and the `K<n>` collaborator map (`19687`) already produce the raw material; they currently feed prose, not an executable achievement.

**Exit criteria:** on the collaborator-heavy subset members (`DefaultRequestHandler.onMessageSend`, `SubmissionSummary.copyFrom`, `SlidingWindowEventCompactor.getCompactionEvents`, `ServerSentEventParser.processLine`), mean Δbranch ≥ +5 pp.

---

### Phase 7 — Retire the hard-coded engine *(S7, the paper claim)*

Only after Phases 2–6 clear their gates. Run as **four separate ablations**, each accepted only if the subset mean does **not** drop by more than 1 pp:

| Ablation | Removes |
|---|---|
| 7a | The live benchmark literals: `getRowDimension|getColumnDimension` ×4 (`recipe_synthesizer.py:45,127`, `coverage_obligations.py:667,731`); `NumberUtils` radix/suffix (`coverage_obligations.py:1252,1268`); `_inputBuffer` fallback (`recipe_synthesizer.py:1968`); `_DATA_FLOW_METHODS` SDK accessors (`coverage_obligations.py:2109`); `text`/`content` setter ranking (`tatg_adapter.py:10235`); `@curlcommand` (`tatg_adapter.py:6531`) |
| 7b | The ~340 lines of verbatim AWS event-ruler JSON (`coverage_obligations.py:1346-1682`) and the gated Jackson/SAX/JDOM fact extractors (`1165-1208`, `1795-1820`, `1955-2031`) — currently dead behind constant-`False` gates, but present in the artifact |
| 7c | The 20-category obligation engine (`coverage_obligations.py:2792-3682`) and the 12 recipe categories (`recipe_synthesizer.py:136-278`), replaced by `BranchGoal` synthesis |
| 7d | Deduplication of the structural triplicates: 4 divergent `_JAVA_KEYWORDS` sets, 3 divergent method-declaration regexes (`targeting.py:12`, `recipe_synthesizer.py:26`, `branch_analysis.py:65`), 3 copies of the parser-cursor marker tuple (`tatg_adapter.py:258,18771,45071`), 2 JUnit-family ladders |

7a and 7d are also correctness fixes: `recipe_synthesizer.py:26` requires an explicit access modifier and therefore **silently skips every package-private method**, while `_caller_entry_recipe` exists specifically to handle package-private targets.

If 7c fails its gate, that is a publishable result in itself — it quantifies exactly how much of the benchmark the rule engine was carrying — but it means Phases 2–6 are not yet sufficient and the work continues rather than the rules being restored.

---

### Phase 8 — Mutation strengthening *(S8)*

Begins only when mean branch ≥ 0.70. Two iterations, per the existing protocol.

Known issues to address first, all in `objectives.py`:
- `_mutation_reachability_operators` (`266-277`) and `_mutation_assertion_operators` (`282-293`) select recipe operators by **substring of the PIT mutator class name** (`"condition"`, `"boundary"`, `"math"`, `"increment"`).
- The gate at `346-350` silently drops every mutator whose name lacks those four tokens — `NullReturnValsMutator`, `EmptyObjectReturnValsMutator`, `RemoveCall` and others never produce constraints.

Replace with: derive the mutation obligation from the **PIT-reported mutated line and its source expression**, expressed as a `BranchGoal` with a differential `Oracle` (`f(x)` under original ≠ `f(x)` under mutant). Same IR, same probe, no mutator-name strings.

---

## Part 4 — Experimental protocol

### 4.1 Tuning subset S1 (28 methods, 19 projects, ~4 h)

Stratified by current branch bucket, project family, and dependency character. Runs at the production configuration: **2 generation iterations + 1 mutation iteration**, `--force`, `qwen/qwen3.6-35b-a3b` at `http://127.0.0.1:11435`.

**Tail, < 0.50 (10)** — the coverage we need to win
`Rotation.getAngles` (Math-2f, 0.386) · `NumberUtils.createNumber` (Lang-4f, 0.420) · `FilteringParserDelegate.nextToken` (JacksonCore-26f, 0.414) · `Metaphone.metaphone` (Codec-18f, 0.493) · `AbstractPatriciaTrie.put` (commons-collections, 0.450) · `RuleCompiler.processMatchExpression` (ruler, 0.148) · `AbstractXMLOutputProcessor.printDocument` (jdom2, 0.433) · `$Gson$Types.resolve` (gson, 0.000) · `SlidingWindowEventCompactor.getCompactionEvents` (adk-java, 0.409) · `JCurlCommandProcessor.resolveCurlCommand` (jquick-curl, 0.250)

**Middle, 0.50–0.70 (10)** — where +11 pp is most achievable
`Flat3Map.get` (0.531) · `JsonReader.readEscapeCharacter` (gson, 0.516) · `DoubleMetaphone.handleG` (commons-codec, 0.540) · `ToXmlGenerator.writeString` (JacksonXml-5f, 0.583) · `DefaultRequestHandler.onMessageSend` (a2a-java/server-common, 0.552) · `SubmissionSummary.copyFrom` (batch-processing-gateway, 0.500) · `CSVParser.createHeaders` (commons-csv, 0.632) · `SAXHandler.startElement` (jdom2, 0.654) · `ByteMachine.addMatchReferences` (ruler, 0.538) · `ServerSentEventParser.processLine` (a2a-java/http-client, 0.688)

**Regression guard, ≥ 0.70 (6)** — must not drop
`GaussNewtonOptimizer.doOptimize` (0.885) · `Rule.pattern` (commons-codec, 0.925) · `Entities.escape` (Jsoup-93f, 1.000) · `JsonReader.doPeek` (gson, 0.712) · `Element.getNamespacesInScope` (jdom2, 0.864) · `AntPathMatcher.isMatch` (windward, 0.972)

**Infrastructure (2)** — currently unmeasured
`PatternOptionBuilder.getValueType` (commons-cli, `project_native_preflight_failed`) · `Contents.rearrangeEventsForLatestFunctionResponse` (adk-java, `compile-type-mismatch`)

Six of these carry ≥ 5 discarded coverage-positive tests, which is what makes the subset sensitive to S3.

### 4.2 Held-out generality set H1 (8 methods) — never used for tuning

`DoubleMetaphone.doubleMetaphone` (0.000) · `AbstractPatriciaTrie.higherEntry` (0.333) · `Ruler.matches` (0.448) · `XPathHelper.getSingleStep` (0.600) · `JsonReader.peekNumber` (0.638) · `PlanNotebook.reviseCurrentPlan` (agentscope-java, 0.321) · `ApplicationSubmissionHelper.getExecutorSpec` (0.559) · `FakeValuesService.bothify` (datafaker, 0.643)

H1 runs **once per accepted strategy**, after S1 acceptance, never before. A strategy that wins on S1 and does nothing on H1 was fitted to S1.

### 4.3 Acceptance rule

A strategy is **accepted** only if all of the following hold on S1, paired per method against the frozen Phase-0 baseline:

| # | Criterion |
|---|---|
| 1 | mean Δbranch ≥ **+2.0 pp** |
| 2 | Wilcoxon signed-rank **p < 0.10** over the 28 paired differences |
| 3 | **no method regresses by more than 3 pp**, and at most **1** method regresses at all |
| 4 | zero new compile failures or runtime-validation failures |
| 5 | wall-clock increase ≤ **25 %** |
| 6 | mutation score on the subset **not lower** than baseline |
| 7 | `measured_fraction ≥ 0.95` (otherwise the comparison is void, not negative) |

Then, before promotion:

| # | Generality criterion |
|---|---|
| 8 | mean Δbranch on **H1** ≥ 60 % of the S1 effect |
| 9 | **Identifier-obfuscation check**: on 3 methods, copy the project, mechanically rename the class, target method, and instance fields to opaque identifiers (`C1`, `m1`, `f1`…), re-run. Branch coverage must not drop by more than 2 pp. A drop means the mechanism is reading names, not semantics — this directly falsifies the paper's central claim and the strategy is rejected regardless of criteria 1–8. |

### 4.4 Anti-plateau discipline

These rules exist because 755 archive directories show where the time went.

1. **Two refinement attempts per strategy, then abandon.** If a strategy fails acceptance, it gets at most two revisions. A third failure means it is retired and the next strategy in the S-ranking starts. Do not tune a losing strategy.
2. **No per-target rescue runs.** No new `branch-<projectname>-*` experiments. If a single method needs a mechanism no other method needs, the mechanism is wrong.
3. **No new entries in `coverage_obligations.py`'s category list.** Any new capability enters as an `Achievement.kind`.
4. **Every strategy must state its expected mechanism before the run** — which advisory-reason counter or discard counter it should move — and the run must move that counter. A coverage gain with no movement in the claimed mechanism is a confound, not a result.
5. **Budget is a control arm, not a strategy.** Run a `baseline @ 3 iterations` arm once, report it separately, and never fold it into a method claim.

### 4.5 Full-benchmark verification

Trigger: accumulated subset mean Δbranch ≥ **+8 pp** over the frozen baseline with all gates passed.

Run all 141 at 2+2 (2 generation, then 2 mutation once branch clears), `--force`, `--post-run-evaluate`, ~20 h. Then merge with `merge_tatg_eval_archive.py` and require:
- mean branch coverage ≥ **0.70**
- `measured_fraction ≥ 0.95`
- no method regressing more than 3 pp vs. the `branch-recovery-20260804-10` reference
- number of methods ≥ 0.70 up from 60

Only then start Phase 8.

### 4.6 Timeline

| Round | Content | Wall clock |
|---|---|---|
| R0 | Phase 0 + two baseline re-runs | ~9 h |
| R1 | Phase 1 probe executor + smoke | ~4 h |
| R2–R3 | Phase 2 (main lever), one revision | ~8 h |
| R4 | Phase 3 | ~4 h |
| R5 | Phase 4 | ~4 h |
| R6 | Phase 5 | ~4 h |
| R7 | Phase 6 | ~4 h |
| R8 | H1 + obfuscation checks on accepted set | ~3 h |
| R9 | Full 141 verification | ~20 h |
| R10–R12 | Phase 7 ablations 7a–7d | ~16 h |
| R13+ | Phase 8 mutation | — |

---

## Part 5 — Code touchpoints

| Area | File:line | Action |
|---|---|---|
| `hard_eligible` conjunction | `branch_analysis.py:20042-20067` | decompose into per-clause status |
| confidence cap to 0.5 | `branch_analysis.py:20032-20033` | delete (double-penalty with `20065`) |
| oracle concreteness blacklist | `branch_analysis.py:17414-17462` | replace with compile/execute check |
| required-state all-or-nothing | `branch_analysis.py:19905-19912` | per-fact |
| ancestor control path | `branch_analysis.py:19929-19934` | Phase 5 path-condition |
| external deps all-or-nothing | `branch_analysis.py:19871-19878` | per-dependency |
| unverified setup actions | `branch_analysis.py:18405, 18765` | per-action `probe_required` |
| grounding gates | `branch_analysis.py:23400-23513, 23516-23557` | probe-backed |
| `external_collaborator` exclusions | `branch_analysis.py:20766, 22139-22147, 24600-24608` | admit |
| `unknown` exclusion | `branch_analysis.py:20644` | admit when subject resolves |
| derived-root drop | `branch_analysis.py:11277` | `probe_required` |
| unconditional demotions | `branch_analysis.py:33645-33663, 34528-34549` | remove for probe-proven rows |
| sketch caps | `branch_analysis.py:20534, 20687, 20696, 20711, 21608` | scale with probe budget |
| analysis cache key | `branch_analysis.py:15702-15745` | include ledger digest |
| obligation category engine | `coverage_obligations.py:2792-3682` | Phase 7c |
| AWS ruler JSON payloads | `coverage_obligations.py:1346-1682` | Phase 7b |
| live benchmark literals | `coverage_obligations.py:667,731,1252,1268,2109`; `recipe_synthesizer.py:45,127,1968`; `tatg_adapter.py:6531,10235` | Phase 7a |
| recipe categories | `recipe_synthesizer.py:136-278` | Phase 7c |
| package-private blind spot | `recipe_synthesizer.py:26-39` | Phase 7d fix |
| sketch/obligation fork | `tatg_adapter.py:44830-44836` | unify on `BranchGoal` |
| required-O-id selection & caps | `tatg_adapter.py:42962-43050` | scale with admitted goals |
| render caps | `tatg_adapter.py:12905-12916` | scale |
| repair-mode ladder | `repair.py:1555-1613` | insert oracle-recovery arm |
| PIT mutator name ladder | `objectives.py:266-350` | Phase 8, derive from mutated source line |
| new | `src/tatg/probe.py` | Phase 1 |
| new | `scripts/compare_subset_runs.py` | Phase 0 |
| suite registry | `scripts/run_all_hard_methods_batch.py:4382` | add `tatg-s1-tuning`, `tatg-h1-holdout` |

---

## Part 6 — Risks

| Risk | Mitigation |
|---|---|
| Probing blows the per-target time budget | hard caps (§Phase 1), probe budget ≤ 25 % of generation timeout, aggressive caching; acceptance criterion 5 enforces ≤ 25 % wall-clock growth |
| Loosening gates admits unsound tests → coverage up, mutation down | acceptance criterion 6 runs mutation on every subset arm; S3 has explicit tautology guards |
| Observed-value oracles become tautologies | provenance tagging, exclusion from mutation adequacy until a mutant is killed, 30 % cap per target |
| Removing Track A collapses the benchmark | Phase 7 runs as four separate ablations, each gated at ≤ 1 pp drop; a failure is a measurement, not a rollback trigger |
| Baseline noise masks real effects | Phase 0 is a hard prerequisite with a two-run reproducibility gate |
| The 70 % mean is reached mostly by extra iterations | budget control arm reported separately (§4.4.5); method claims are made only against the 2-iteration baseline |
| Some branches are genuinely unreachable via public API | `probe_infeasible` marking (Phase 4) gives a defensible, evidence-backed infeasibility set for the paper rather than an unexplained gap |

---

## Part 7 — Decision needed before implementation starts

None blocking. Two worth confirming as we go:

1. **Probe isolation strategy.** The plan reuses `mvn test -Dtest=<Probe>` for fidelity. A faster route is a persistent JVM with a preloaded classpath (~3 s vs ~15 s per probe), at the cost of divergence from the real validation path. Recommendation: start with Maven for correctness, add the fast path only if Phase 1's exit criteria are missed.
2. **Whether `probe_infeasible` methods are excluded from the reported mean.** Recommendation: no — report the mean over all 141 and publish the infeasible set separately, which is more defensible than an adjusted mean.
