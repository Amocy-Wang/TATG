# The S1 acceptance rule cannot be satisfied by any candidate

Measured 2026-08-08 from the S1 baseline arm
(`tatg-eval-archives/s1-focus-batching-20260807/baseline`).

This is the same class of defect as the old "measured fraction >= 95%" gate: a
threshold that no candidate can pass, which would silently auto-reject every
strategy before its coverage numbers were read.

## How it was measured

The baseline arm ran the 34-descriptor S1 suite, and then — through a bug in the
retry driver, `--validation-suite` silently overriding `--methods` — began a
**second full pass under identical configuration**. That accident is the most
useful data in the run: it is a *null A/B*, the same code compared against
itself, and it measures how much of a per-method delta is pure sampling noise.

17 targets carry a branch number in both passes.

## The noise floor

| statistic | value |
|---|---:|
| mean \|Δ\| per target | **7.07 pp** |
| median \|Δ\| | 6.52 pp |
| max \|Δ\| | 20.0 pp |
| sd of signed Δ | **9.31 pp** |
| targets moving > 3 pp | **12 / 17** |
| targets moving > 10 pp | 5 / 17 |
| identical in both passes | 5 / 17 |

Worst movers, with no code change whatsoever:

```
PatternOptionBuilder.getValueType   0.600 -> 0.800   (+20.0pp)
DoubleMetaphone.handleG             0.400 -> 0.260   (-14.0pp)
CSVParser.createHeaders             0.579 -> 0.447   (-13.2pp)
FilteringParserDelegate.nextToken   0.444 -> 0.323   (-12.1pp)
Entities.escape                     0.676 -> 0.794   (+11.8pp)
```

## What that does to each gate

With sd = 9.31 pp per target:

| paired n | SE of mean Δ | min detectable effect (α .05, power .80) |
|---:|---:|---:|
| 17 | 2.26 pp | 6.3 pp |
| 22 | 1.99 pp | 5.6 pp |
| 34 | 1.60 pp | 4.5 pp |

- **`mean Δbranch >= +2.0pp`** — at n=22 that is 1.01 SE. The rule asks for an
  effect the design cannot distinguish from zero. The *smallest* effect S1 can
  actually resolve is ~4.5 pp, and only if every target is measured in both arms.
- **`no method regressing > 3pp` and `at most 1 regressing at all`** — on pure
  noise, **7 of 17** targets regress by more than 3 pp. A candidate identical to
  the baseline fails this gate outright. It is not a quality bar; it is a
  coin-flip that rejects roughly everything.

## Root cause

Generation is sampled and irreproducible:

- `_llm_temperature()` (`tatg_adapter.py:46294`) defaults to **0.2**.
- `openai_compat.py:103` builds the payload with **no `seed`**, so no run can be
  reproduced even in principle.

Branch analysis is already deterministic (`temperature: 0.0` throughout
`branch_analysis.py`), so the entire noise budget comes from the generation call.
LM Studio accepts a `seed` parameter — verified against the live endpoint.

## Seeding does not fix it — tested 2026-08-08, and it failed

`TATG_LLM_SEED` is implemented and threaded into the payload, but **the
stack does not honour it**. Five runs of `PatternOptionBuilder.getValueType`
(the highest-variance target in the null A/B, and the fastest):

| arm | code sha256 | verdict |
|---|---|---|
| seed 1234 | `52494aaff68c0997` | |
| seed 1234 | `c2013bbab197853f` | **differs — same seed, different output** |
| seed 999 | `936290b3545f3771` | differs (as expected) |
| no seed | `72f1e15ab0260355` | |
| no seed | `645707885a5c2127` | differs from its twin |

The same seed produced different generated code, so reproducibility is not
available from the sampler. Either LM Studio ignores `seed` for this model, or
the nondeterminism lives outside sampling (batched inference reordering
floating-point reductions, or concurrency in the pipeline). The env var is kept
because vLLM and other backends do honour `seed`, but **it buys nothing on the
current setup and must not be reported as a mitigation.**

## What is actually available

1. **Repeated runs are the only lever left.** Noise on a per-target mean falls
   as sd/sqrt(k), so k=5 takes sd 9.31 -> ~4.2 pp and the n=34 MDE from 4.5 pp
   to ~2.0 pp. That is 5x the wall-clock: about 25 h per arm at S1 size.
2. **Per-method regression gates re-derived from the measured null** rather than
   asserted at 3 pp — done, in `compare_subset_runs.py`.
3. **Treat any single-run S1 delta below ~5 pp as unresolved**, whichever
   direction it points.
## Greedy decoding does not fix it either — tested 2026-08-08

`TATG_LLM_TEMPERATURE=0`, three runs of the same target:

```
run 0  branch=1.0  gen=8  sha=1d7997d759329d0c
run 1  branch=1.0  gen=8  sha=8644ae050fb4f569
run 2  branch=1.0  gen=8  sha=eb0620acb6960b64
```

Three runs, three different generated-code hashes. Neither seeding nor greedy
decoding makes a run reproducible, so **no configuration change buys
reproducibility** and repeated runs are the only remaining lever.

### Cause isolated — the pipeline is deterministic, the server is not

Two runs with prompt logs preserved (separate output directories, since
`--force` on a shared directory wipes them), comparing the iteration-1 generate
prompt — the one built *before* any LLM output can influence it:

```
system prompt : IDENTICAL   1336 chars    sha 1f95d841acf07fe4
user   prompt : IDENTICAL  32865 chars    sha 7de2950bf5f9bf08
```

Byte-identical. The candidate explanation that prompt construction was varying —
timestamps, set/filesystem ordering, unstable pre-analysis — is **falsified**.
TATG builds the same question every time and gets a different answer.

So the nondeterminism is entirely in the model server: same prompt, same seed,
temperature 0, different output. Nothing in this repo can remove it, and no
further configuration search is worth running.

(Scope: this pins the iteration-1 prompt. Later rounds legitimately differ,
because their coverage feedback depends on which tests the previous round
produced — that is downstream of the server's nondeterminism, not a second
independent source.)

## Variance is not uniform, and it is config-dependent

`PatternOptionBuilder.getValueType` swung 0.600 -> 0.800 between the two
baseline passes, but scored **1.000 on six consecutive observations** under the
candidate configuration (one in-suite, five isolated). The 9.31 pp sd is a
suite-wide average, not a property each target carries: some targets are stable
and some are not, and which is which depends on the configuration. Per-target
variance should be measured per arm rather than assumed constant.

## The S1 result, read against this noise floor

Run 2026-08-08, baseline (pass 1, 34 descriptors) vs candidate (34 descriptors),
21 measured in both arms.

| gate | value | reading |
|---|---:|---|
| mean Δbranch | **+1.53pp** (+1.90pp on measured pairs) | 0.94 SE — indistinguishable from zero |
| Wilcoxon p | 0.5566 | no effect detected |
| worst regression | −26.9pp (`SAXHandler.startElement`) | the only target outside the 2sd noise band |
| regressing methods | 8 of 21 (38%) | the null regressed 41% — *at* chance |
| wall-clock | 4.48h → 5.72h, **+27.7%** | systematic, not noise |
| measured fraction | 64.7% → 67.6% | slightly improved |

**Verdict: REJECT** — and, unlike the pre-calibration gates, an honest one.

### The mechanism moved backwards

The anti-plateau rule says to name the counter that should move and check it.
It moved the wrong way:

| counter | baseline | candidate | |
|---|---:|---:|---|
| generated tests (34 targets) | 186 | **165** | −11.3% |
| targets with more tests | — | 5 | vs 15 with *fewer* |
| feedback-round LLM calls | 65 | **89** | +37% |
| rounds issuing >1 LLM call | 22 | 25 | +3 |

The candidate spent **37% more LLM calls to produce 11% fewer tests**. The gap
budget was designed to raise the test count on large methods and did the
opposite. `corr(Δtests, Δbranch) = +0.13` — the extra spend did not convert.

Two further corrections to the working model:

- **Multi-call rounds are not caused by round batching.** The baseline, with
  `TATG_ROUND_BATCHING=0`, already issues >1 call on 22 rounds. Batching added 3.
  Attributing multi-call behaviour to this change was wrong.
- **The focused-prompt collapse is rare, not systemic.** The baseline already
  carried coverage feedback on 62 of 65 feedback rounds (95%). The fix took that
  to 89/89, closing a ~5% gap. The single-target smoke run that motivated it
  (`Rotation.getAngles`, 6 150 chars, `coverage_feedback_present=False`) is an
  unrepresentative case, and W9's saturation cannot be explained by it.

## Consequence for the work in flight

The batching and feedback-prompt changes were being evaluated against a rule
that no candidate can pass, using an instrument whose noise is 3-4x the effect
being sought. The S1 result should be read as a *sanity check on direction and
mechanism counters*, not as an accept/reject verdict, until seeding lands.
