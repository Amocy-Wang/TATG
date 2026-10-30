# TATG — Weaknesses of the Current Version

Evidence base: `tatg-eval-archives/branch-recovery-20260804-10/merged_replay_rows.json` (141 targets),
691 branch-analysis detail blocks across `branch-*/` raw artifacts, and the full run artifact +
prompt log for `Ruler.matches` (`branch-ruler-projection-gate-2plus1-20260803`).

Mean branch coverage **0.6444**, median 0.6667, 60 of 135 measurable targets at or above 0.70.

The weaknesses below are ordered by how much of the 0.70 gap each one explains. The first is
arithmetic and was not visible in any summary metric.

---

## W1 — A hard arithmetic ceiling on tests per target

`_method_patch_json_test_budget` (`tatg_adapter.py:2452`) returns `min(4, requested)`. It is applied
on the LM Studio structured-output path, which is the configuration used for every benchmark run:

```python
return max(1, min(4, int(max_tests or 1)))
```

That value flows into the response JSON schema, the system prompt, and the user prompt, so the
model is asked for **at most 4 tests per LLM call**, and each round makes exactly one call.
`generate_test_for_target` had no accumulation across LLM responses — `generated_code` is fully
overwritten on every attempt, and the only `.append` inside the 1 900-line attempt loop is a
diagnostic list.

With the 2-generation-round protocol:

```
4 tests / call  ×  1 call / round  ×  2 rounds  =  8 tests per target, maximum
```

Measured marginal yield is **5.93 branches covered per retained test (median 4.00)**, and later
tests overlap earlier ones heavily. Taking a conservative marginal yield of ~3:

| method size | branches needed for 0.70 | max reachable at 8 tests | verdict |
|---|---:|---:|---|
| 15 branches | ~11 | ~24 | comfortable |
| 40 branches | ~28 | ~24 | marginal |
| 67 branches (`Ruler.matches`) | ~47 | ~24 | **impossible** |
| 88 branches (`RuleCompiler.processMatchExpression`) | ~62 | ~24 | **impossible** |

Confirmed directly in the prompt log for `Ruler.matches` (67 branches, 37 uncovered):

```
entry 1  phase=generate  chars=59165   new_tests_count = 4
entry 2  phase=generate  chars=69836   new_tests_count = 4
```

Eight tests requested for a 37-branch gap. The method finished at 0.448.

The cap itself is well-motivated — local models do pack sibling tests into one JSON item when the
schema is large, and the docstring says so. The defect is that a *per-response* limit was also
serving as the *per-round* limit.

---

## W2 — The test budget is uncorrelated with how much there is to cover

Across the 106 targets with a measured branch total:

```
corr(total_branches, retained_tests)  = +0.016     ← effectively zero
corr(total_branches, generated_tests) = +0.112
corr(total_branches, branch_coverage) = -0.300
corr(retained_tests, branch_coverage) = +0.263
```

Binned:

| total branches | n | mean retained tests | mean generated | mean branch coverage |
|---|---:|---:|---:|---:|
| 0–10 | 4 | 2.50 | 3.75 | 0.604 |
| 10–20 | 19 | 6.53 | 8.37 | 0.689 |
| 20–40 | 59 | 5.51 | 13.54 | 0.728 |
| 40–80 | 20 | 6.45 | 12.10 | 0.646 |
| **80+** | 4 | **4.50** | 11.50 | **0.332** |

The largest methods receive *fewer* tests than mid-sized ones and score half as well. Coverage
correlates positively with test count (+0.263) and negatively with method size (−0.300), while the
allocator ignores both.

The gap is not large in absolute terms. Closing every sub-70 target to exactly 0.70 needs an
estimated **+266 tests across 51 targets — a median of 2.2 more each**, against a current median of
3 retained tests on those targets. Roughly doubling the test count on the failing tail would close
most of it.

---

## W3 — Prompt size throttles the budget in the wrong direction

`_latency_safe_new_tests_budget` (`tatg_adapter.py:174`) caps the test count purely by projected
prompt characters: `≥32 000 → 6`, `≥38 000 → 5`, `≥44 000 → 4`.

Prompt size is a proxy for method complexity — more source, more obligations, more anchors. So the
rule reads: *the more complex the method, the fewer tests it may have.* Both `Ruler.matches`
prompts (59 k and 70 k chars) sat far above the top threshold.

The same pressure damages the feedback itself. Round 2 of `Ruler.matches` contains, twice:

```
[coverage feedback chunks omitted: omitted_chunks=1]
```

`[Coverage Feedback For This Generation Round]` is the only section that names the branches still
uncovered, and it has an 18 000-char sub-budget (`TATG_PROMPT_MAX_COVERAGE_FEEDBACK_CHARS`). On
large methods it overflows and chunks are dropped, while ~20 advisory scaffolding sections
(`Prompt Module Guide`, `Assertion-Quality Guidance`, `Fixture Isolation Rules`,
`Observed Construction Recipes`, …) are retained.

Three effects compound on exactly the methods that need help: fewer test slots, less coverage
feedback, and more scaffolding crowding the context.

---

## W4 — The LLM's branch reasoning is rejected before it can bind

Across 691 branch-analysis detail blocks:

```
raw LLM branch rows   2988
validated rows        3332
hard_eligible rows      75      →  2.5 %
```

In the merged 141-target run, `hard_eligible_count` is **0 in all 19 detail blocks**.

`hard_eligible` (`branch_analysis.py:20042-20067`) is a 21-clause conjunction whose clauses are
themselves all-or-nothing quantifiers over lists of up to 8 facts. Rejection volume:

| advisory reason | count | gate |
|---|---:|---|
| `required_state_not_established` | 2637 | `19905` — all-or-nothing over required-state facts |
| `unmodeled_ancestor_control_path` | 1335 | `19929` — every enclosing predicate must be modelled *and* set up |
| `observable_oracle_not_concrete` | 1331 | `17414` — an **English-phrase blacklist** |
| `external_dependency_not_established` | 790 | `19871` — all-or-nothing |
| `confidence_below_threshold` | 604 | `20065` — and `confidence` is already capped to 0.5 at `20032` whenever no required-state fact verified, so it double-fires |
| `unverified_setup_actions` | 544 | `18405/18765` — one bad line out of 16 kills the row |
| `anchor_diversity_recovery_requires_candidate_replay` | 368 | `33645` — **unconditional** demotion regardless of validation |

Consequence: the O-ids that generation is contractually required to satisfy come from the
deterministic 20-category rule engine in `coverage_obligations.py:2792-3682`, not from the semantic
analysis. The paper's claim is that LLM reasoning replaced the hard-coded rules; in the shipped
pipeline the rules are load-bearing and the reasoning is advisory.

`_observable_oracle_is_concrete` deserves separate mention: an English blacklist rejecting LLM
output on 1331 occasions is itself one of the hard-coded heuristics the work set out to remove.

---

## W5 — Obligations are emitted but not satisfied

Of the rows recording obligation accounting, mean `coverage_obligation_rate` = **0.008**, with
totals such as `(7 declared, 0 satisfied)`, `(14, 1)`, `(16, 0)`, `(13, 0)`.

Both tracks fail to convert: the semantic track never binds (W4), and the rule-engine track emits
volume that generation cannot discharge — partly because of W1/W2 (only 4 slots for up to 16
declared obligations) and partly because the obligations describe *what* to cover without an
executable path to get there.

---

## W6 — Behavioural monoculture: many tests, one shape

`Ruler.matches` ended with 14 tests. Every one has the identical four-line body:

```java
String event = "{...}";
String rule  = "{...}";
boolean result = Ruler.matches(event, rule);
assertEquals(true/false, result);
```

Only the JSON literal varies — `prefix`, `numeric`, `anything-but`, `equals-ignore-case`,
`wildcard`. This is precisely the shape produced by the hard-coded event-ruler JSON row generator
at `coverage_obligations.py:1552-1644`, which emits verbatim rule payloads for that operator list.

Per-candidate coverage from the run's `candidate_audit`:

| test | branches covered / 67 |
|---|---:|
| `matches_validBaselineExactMatch` | 2 |
| `matches_numericRangeBoundaryMatch` | 7 |
| `matches_anythingButNumericNonMatch` | 4 |
| `matches_anythingButIgnoreCaseTextualNonMatch` | 3 |
| `matches_O4_AnythingButTextualNonMatch` | 3 |
| `matches_O1_PrefixTextualMatch` | 3 |

Each test explores one dimension — the rule operator — on a single top-level path. None constructs
a `Machine`, adds rules incrementally, or drives internal state, which is where the remaining 37
branches live. The suite is wide and flat where it needs to be deep.

Two of the new tests are near-duplicates of pre-existing ones
(`matches_validBaselineExactMatch` vs `matches_exactTextualValue_returnsTrue`).

---

## W7 — Coverage-positive tests are discarded

**815 candidate target-positive tests, 749 retained, 234 coverage-positive failures** on 53 targets.
Failure category of the best failing target-positive candidate, where recorded:

| category | count |
|---|---:|
| `oracle-mismatch` | 29 |
| `runtime-validation-failure` | 4 |
| `state-sequence-or-oracle-mismatch` | 3 |
| other | 5 |

Worst offenders: `Rotation.getAngles` 30 candidates → 13 retained (17 discarded, final 0.386);
`CSVParser.createHeaders` 20 → 7; `JsonRuleCompiler.writeRules` 35 → 22.

These tests **executed the branch**. They were deleted because the assertion value was wrong, and
nothing in the pipeline attempts to recover an oracle from the observed value.

A second discard channel is visible in the same audit: `matches_O2_GuardTextualFalse` **passed**,
was **target-positive**, and was still `promotion rejected` with
`closes_baseline_gap: false` / `exploratory_target_hit_retained`. Passing tests that reach the
target are dropped for not closing a *specific* gap.

---

## W8 — Domain-shaped input errors are never repaired

`matches_O3_GuardNumberTrue` failed on malformed rule JSON:

```java
String rule = "{\"num\":[{\"numeric\":\"=\",5}]}";     // generated
String rule = "{\"num\":[{\"numeric\":[\"=\",5]}]}";   // correct
```

A one-token encoding error in a domain-specific input format. It produced `branch_coverage 0.0`,
`target_positive false`, `promotion not_promoted_validation_failed`, and consumed one of the four
slots for that round. There is no cheap execute-and-observe step that would have caught it and let
the model retry against the actual error.

---

## W9 — The feedback loop saturates at round 3

Marginal branch gain per additional iteration:

| iteration | n | mean Δbranch | positive | zero |
|---|---:|---:|---:|---:|
| 2 | 92 | **+0.1676** | 74 | 18 |
| 3 | 67 | **+0.0750** | 37 | 30 |
| 4 | 18 | **+0.0085** | 4 | **14** |

By phase: `regular_generation` +0.1645, `regular_generation_feedback` +0.0848 (37 of 75 exactly
zero), `mutation_extra` +0.0405.

The cause is mechanical. The branch analysis is cached by content hash
(`_analysis_cache_key`, `branch_analysis.py:15702`, `_BRANCH_ANALYSIS_SCHEMA_VERSION = 50`) and the
gates are deterministic, so round *N+1* sees the same source, produces the same rows, hits the same
21 clauses, and renders a near-identical prompt. The existence of a
`tatg_structural_duplicate_retry_by_label` path is the symptom.

Note for attribution: a third generation round is worth roughly +7.5 pp on targets that need it.
That is budget, not method, and must be reported as a control arm rather than folded into a method
claim.

---

## W10 — The primary metric is not measured

`branch_measurement_status` over the 141: **`coverage_unmeasured` 112, `measured` 29,
`not_applicable_no_branches` 3**. `coverage_status = no_json` on 48 rows. Reported coverage mostly
derives from `streaming_candidate_combination:*` snapshots (157 + 146 occurrences) rather than a
final measured JaCoCo report.

At that noise level a +3 pp effect cannot be separated from run-to-run variance. This plausibly
explains why several past optimizations read as ambiguous, and it makes any accept/reject decision
unreliable until fixed.

---

## W11 — Structural gaps that cap the ceiling

- **`external_collaborator` is a dead-end dependency kind.** Excluded from `scaffold_delta_candidate`
  (`20766`), `_semantic_feasibility_candidate_rows` (`22140`) and
  `_source_visible_semantic_sandbox_contract_is_complete` (`24600`). It can never reach a
  replay/sandbox proof — 790 rejections. This kind dominates `adk-java`, `a2a-java`,
  `batch-processing-gateway`, `agentscope-java`.
- **`unknown` drops out of everything** except the weakest advisory path (`20644`).
- **`local_or_unknown` derived roots** hit a hard `return False, "derived_state_root_kind_not_supported"`
  (`11277`).
- **Benchmark-specific literals on the live path**: the commons-math
  `getRowDimension|getColumnDimension` regex in **4 copies** (`recipe_synthesizer.py:45,127`;
  `coverage_obligations.py:667,731`); the commons-lang `NumberUtils` radix/suffix detector
  (`coverage_obligations.py:1252,1268`); Jackson's private `_inputBuffer` field name emitted into
  prompts for arbitrary projects (`recipe_synthesizer.py:1968`); Anthropic-SDK accessors inside
  `_DATA_FLOW_METHODS` (`coverage_obligations.py:2109`); `text`/`content` builder-setter ranking
  (`tatg_adapter.py:10235`); `@curlcommand` (`tatg_adapter.py:6531`). A further ~340 lines of
  verbatim AWS event-ruler rule JSON sit at `coverage_obligations.py:1552-1682` behind a
  constant-`False` gate — disabled, but present in the artifact.
- **Structural triplication**: 4 divergent `_JAVA_KEYWORDS` sets, 3 divergent method-declaration
  regexes (`targeting.py:12`, `recipe_synthesizer.py:26`, `branch_analysis.py:65`), 3 copies of the
  parser-cursor marker tuple, 2 JUnit-family ladders. `recipe_synthesizer.py:26` requires an explicit
  access modifier and therefore **silently skips every package-private method**, while
  `_caller_entry_recipe` exists specifically to handle package-private targets.

---

## Summary

| # | Weakness | Type | Explains |
|---|---|---|---|
| W1 | 4-test-per-call cap is also the per-round cap | arithmetic | large-method failure |
| W2 | Budget uncorrelated with branch count | allocation | 80+ bin at 0.332 |
| W3 | Prompt size throttles budget and drops feedback | inverted heuristic | compounding on hard targets |
| W4 | 97.5 % of LLM branch reasoning demoted | over-strict gate | rules carry the benchmark |
| W5 | Obligation satisfaction 0.008 | conversion | neither track converts |
| W6 | One Arrange/Act shape per target | monoculture | shallow, wide suites |
| W7 | 234 coverage-positive tests discarded | validation | thrown-away coverage |
| W8 | Domain input errors unrecoverable | no execute-observe | wasted slots |
| W9 | Feedback saturates at round 3 | determinism + cache | the plateau |
| W10 | 112/141 unmeasured | measurement | unreliable decisions |
| W11 | Dead-end kinds, benchmark literals, triplication | generality | paper risk |

The single sentence: **coverage is not limited by what the LLM can infer, but by how few tests it is
allowed to write, how much of its inference is vetoed before use, and how many of the tests that do
reach a branch are then deleted.**
