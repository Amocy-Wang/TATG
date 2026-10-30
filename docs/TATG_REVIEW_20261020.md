# TATG infrastructure review and 10-target validation

## Release scope

The starting main revision was `9ff24de46fd3a7bffc562247f8a4ed66382f2551`
(`tatg-stable-20261012`). This review retains the narrowly scoped correctness
fixes below. The release tag is `tatg-infra-reviewed-20261020`.

The 10-target run establishes that the retained version completes the tested
workflow and improves its aggregate coverage in this sample. It does **not**
establish a general performance improvement or absence of per-target regression.
The stricter strategy comparator returned `reject`; its result is preserved
below. Acceptance is limited to the reviewed infrastructure fixes, with the
known performance uncertainty recorded explicitly.

| Retained change | Failure addressed | Preserved validation requirement |
| --- | --- | --- |
| Synchronize repaired source aliases | Final suite merging could select stale source after the repaired candidate passed validation. | Use the revalidated replay source consistently in final/generated/full-source fields. |
| Reject unbalanced Java candidate braces | A class declaration allowed an incomplete compilation unit through the early format check. | Use the existing comment/literal masker; retain support for dollar-named Java classes. |
| Separate deep-path evidence from baseline claims | A deep-path candidate could fail role integrity solely because it lacked a baseline B-id. | Require target-positive coverage and the selected semantic anchor or coverage gain; explicit skeleton contradictions remain rejected. |
| Reject bare null for primitive constructor parameters | A source-known primitive parameter was treated as compatible with an unknown/null argument. | Keep reference-type uncertainty permissive; reject only the source-proven primitive/null contradiction. |
| Correct mutation objective auditing | Maven `.m2` paths were counted as mutation objective `M2` in structural prompt audits. | Recognize uppercase mutation objective IDs without changing generated prompt content. |
| Document subject-root configuration | The batch example omitted the local dataset root configuration. | Supply paths through environment variables. |

These are generic checks and lifecycle fixes. The subject names below are
validation fixtures, not production dispatch rules. No CAT/MUTGEN prompt
semantics, global prompt order, or no-clip default was changed. Some shared CLI
helpers are affected; live validation in this review used TATG structural rounds.

## Validation configuration

- Backend: TATG.
- Provider: local LM Studio, model `qwen/qwen3.6-35b-a3b`.
- Budget: two structural generation iterations and two repair iterations.
- Mutation extra: disabled; mutation score was not measured in this experiment.
- Sample: 10 method/descriptor targets, not the full 141-selector benchmark.
- Each result uses that run's final measured target coverage, with no
  best-of-history replacement or mixing of experiment versions.
- Final retained code: `python -m pytest -q` completed with **3902 passed,
  34 skipped** in 149.53 seconds. `git diff --check` also passed.

## Paired live results

All coverage values are percentages; deltas are percentage points.

| Target | Main line | Retained line | Main branch | Retained branch | Branch delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base64.decode | 19.44 | 77.78 | 20.83 | 70.83 | +50.00 |
| CSVParser.nextRecord | 86.21 | 86.21 | 77.78 | 77.78 | 0.00 |
| DefaultRequestHandler.onMessageSend | 48.94 | 48.94 | 39.66 | 39.66 | 0.00 |
| FilteringParserDelegate.nextToken | 34.17 | 31.67 | 28.28 | 26.26 | -2.02 |
| Flat3MapNew.get | 90.00 | 80.00 | 68.75 | 50.00 | -18.75 |
| FromXmlParser.nextTextValue | 36.76 | 27.94 | 35.71 | 28.57 | -7.14 |
| JsonRuleCompiler.processMatchExpression | 30.84 | 30.84 | 23.86 | 23.86 | 0.00 |
| NumberUtils.createNumber | 55.00 | 52.50 | 39.77 | 35.23 | -4.55 |
| Rotation.getAngles | 90.65 | 89.72 | 68.57 | 67.14 | -1.43 |
| SAXHandler.startElement | 65.67 | 65.67 | 50.00 | 50.00 | 0.00 |
| **Mean** | **55.77** | **59.13** | **45.32** | **46.93** | **+1.61** |

Both runs measured all 10 targets. Total target E2E time fell from 6856.12 to
6534.16 seconds (-4.70%); mean E2E fell from 685.61 to 653.42 seconds.
One target improved in branch coverage, four were unchanged, and five declined.
The median paired branch delta was -0.71 points.

The strict comparator required mean branch gain >=2 points and Wilcoxon
`p < 0.1`. The observed gain was 1.61 points and `p=0.34545`, so the overall
verdict was **reject for a new performance strategy**. Measurement completeness,
runtime, and the comparator's existing 19-point single-target noise threshold
passed. That threshold is a diagnostic convention based on earlier sampling
experiments, not proof that this release cannot regress. In particular,
Flat3MapNew and FromXmlParser need repeat validation before a broad
non-regression claim.

Base64 provides the clearest lifecycle evidence: the repaired-source alias fix
has a deterministic regression test and repeated target-level gains. A separate
focused run reached 86.11% line / 79.17% branch; the final 10-target run reached
77.78% / 70.83%. These supporting runs are not pooled into the table above.

## Prompt review

The retained run contains 61 logged prompts. All 10 first-generation prompts
include the generation control receipt, target overload binding, source-backed
baseline skeleton obligation, and executable baseline schema.

First-generation user prompts range from 48,032 to 78,673 characters, with a
median of 59,370.5 characters. Character counts are not tokenizer counts.
There were zero top-level truncations and zero structural mutation objective
or mutation marker audit hits after correcting the `.m2` false positive.

Fourteen prompt entries contain 16 lower-priority section truncation markers:
10 in `Visible Type Construction APIs` and six in `Target Execution Receipt`.
The reviewed high-priority sections were intact. Long appendices and their
truncation remain an investigation item; this release does not establish that
their content is always sufficient.

## Excluded experiments

| Experiment | Observed outcome | Decision |
| --- | --- | --- |
| Broad prompt reorder / direct-anchor changes | No validated acceptance result supporting inclusion. | Reverted; not shipped. |
| Deterministic source-callee scope ordering | The tested candidate measured 9/10 targets and mean branch fell 2.52 points. SAX branch also fell 26.92 points. | Reverted; no causal claim that ordering alone caused the decline. |
| Audit-backfill sibling-test filtering | The v2 candidate measured 9/10 targets; mean branch fell 1.32 points and the worst decline was 29.32 points. | Reverted together with its added test. |

Only the retained v1 results support this release. Results from rejected
experiments do not replace its individual target rows.

## Remaining work

1. `source_callee_scope` still renders a set-derived list. Its order can vary
   across processes and is a reproducibility risk. Any replacement needs
   prompt-equivalence checks and another live comparison.
2. Model sampling variance remains material. Existing null comparisons are
   documented in `TATG_MEASUREMENT_NOISE.md`; those measurements do not prove
   that every regression here is sampling noise.
3. The shallow JsonRuleCompiler result and variable parser/map results still
   need candidate/input evidence review. This release is not a deep-path
   generation improvement.
4. DefaultRequestHandler sometimes loses final measured coverage in rejected
   runs after setup/combination failures. The retained run matched its baseline,
   but repeated lifecycle validation is still warranted.
5. Mutation-stage quality, full-141 performance, and shared-helper effects on
   other backends were not established by this structural TATG sample.

## Evidence provenance

Raw prompts, source paths, generated tests, and logs remain in the local
evaluation archive outside the repository. The run IDs below identify them
without publishing machine-specific paths or private environment data.

- Baseline: `tatg-review10-baseline-main-20261019-181500`.
- Retained version: `tatg-review10-final-v1-20261019-235332`.
- Focused repaired-source validation: `tatg-review-base64-narrow-20261019-220500`.
- Rejected ordering experiment: `tatg-review10-candidate-narrow-v2-20261019-213500`.
- Rejected sibling-filter experiment: `tatg-review10-final-v2-20261020-021056`.

The retained archive includes `hard_method_report.csv/json/md`,
`comparison.json`, per-target JSON/partial files, prompt logs, candidate audits,
generated tests, and JaCoCo artifacts.

SHA-256 checksums:

- Baseline `hard_method_report.csv`:
  `2bd1b365787b7351d4573382c01d046f073c2ad3192512ace35953537c082455`.
- Retained `hard_method_report.csv`:
  `dade3ef3a2e630f194e119e616bde976f1e3a062db7d4e44d3c3fc683e25f334`.
- Validated eight-file diff against starting main, before adding this document
  (`git diff --binary`):
  `416f4e9c841548d9a318d92febad2ef9ea3a965bba7057ed4d88e2270c458287`.
