# RFC 008 — ARC-Easy Cross-Benchmark Validity

| Field | Value |
|---|---|
| **RFC** | 008 |
| **Title** | ARC-Easy Cross-Benchmark Validity |
| **Status** | Finding (pre-registered, executed, **A primary / D secondary**) |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Depends on** | [RFC 002 (gate)](./rfc-002-cot-gate-finding.md), [RFC 003 (plateau)](./rfc-003-inference-compute-plateau.md), [RFC 004 (rep-loops)](./rfc-004-repetition-loop-failure-mode.md), [RFC 005 (methodology)](./rfc-005-methodology.md), [RFC 007 (rep-loop isolation)](./rfc-007-repetition-loop-isolation.md) |
| **Sharpens** | [RFC 003](./rfc-003-inference-compute-plateau.md), [RFC 004](./rfc-004-repetition-loop-failure-mode.md) |
| **Closes** | [RFC 006](./rfc-006-open-questions-and-next-steps.md) item 1 |
| **Source plan** | [`path1_plan4.md`](../plans/path_1_cot_tokens/plan4.md) |

## Abstract

Plan 4 ports the plan-2 plateau check to ARC-Easy and lands two findings the GSM8K work could not produce on its own. **First**, the inference-compute plateau holds cross-benchmark: at n=500, A3-arc (8-shot CoT, greedy, max_new=512) scores 82.8% and B3-arc (k=5 self-consistency at the same length) scores 85.0%. The +2.2 point lift is identical in magnitude to plan 2's GSM8K B4-A3 lift (also +2.2 pts), CIs overlap, and the paired McNemar p=0.080 falls inside the pre-registered "plateau confirmed" band. Path 1's "more compute doesn't help" claim is a property of the model, not of GSM8K. **Second**, the secondary direct-cell sanity check fires outcome D: A3-arc minus direct-arc is **−0.4 points** (414 vs 416/500, McNemar p=0.905). CoT provides zero lift over a length-16 direct-answer cell on ARC-Easy, in stark contrast to GSM8K's +30 point CoT gate ([RFC 002](./rfc-002-cot-gate-finding.md)). Combined with [RFC 007](./rfc-007-repetition-loop-isolation.md)'s GSM8K rep-loop verdict, this RFC closes the cross-benchmark validity question and supplies the Path 1 ARC-Easy entry for the four-paths head-to-head.

## Motivation and recap

[RFC 003](./rfc-003-inference-compute-plateau.md) established that on GSM8K, neither generation length nor self-consistency lifts E2B-it CoT accuracy beyond the greedy 512-token reference. The pre-registered worry was that the plateau was a GSM8K-specific artifact of two failure modes ARC-Easy does not have:

1. Repetition loops at ~10% per [RFC 004](./rfc-004-repetition-loop-failure-mode.md) and [RFC 007](./rfc-007-repetition-loop-isolation.md).
2. 63% of greedy A3 completions hit the 512-token cap on GSM8K, where the answer comes at the *end* of the chain.

Either could mean ARC-Easy reveals a self-consistency response the GSM8K rig hides. Plan 4 was designed as a single decisive cross-benchmark probe answering the question pre-registered in [RFC 006](./rfc-006-open-questions-and-next-steps.md):

> Does the "more compute doesn't help" claim hold on ARC-Easy, or is it a GSM8K-specific artifact?

The optional `direct_arc` cell was added to mirror plan 1's CoT gate on a second benchmark.

## Methodology

Methodology follows [RFC 005](./rfc-005-methodology.md). ARC-Easy specifics:

- **Model & pinning:** `google/gemma-4-E2B-it` at commit `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`, bf16, `transformers==5.6.1`, `torch==2.4.1+cu124`, `datasets==4.8.4` — identical to plan 2.
- **Benchmark:** `allenai/ai2_arc` config `ARC-Easy`, split `test`, deterministic first n=500.
- **Exemplars:** 8 hand-crafted CoT exemplars drawn from the head of the ARC-Easy *train* split (no standard CoT exemplars exist for this benchmark in the literature). Pinned exemplar hash: `633e12e57bd8cd19`. The eight train IDs are stamped into `manifest.json` so the prompt cannot silently drift.
- **Label normalization:** 21 of 500 problems shipped with numeric (1–4) labels rather than letter (A–D); they are remapped to letters with the gold rotated to match. 2 problems have only 3 choices. None were dropped. The remap counts are surfaced in the manifest's `gold_stats`.
- **Answer extraction:** primary regex `[Tt]he answer is\s*\(?([A-E])\)?`, fallback last in-set letter. The `answer_hit_rate` is the ARC analog of GSM8K's `hash_hit_rate`.

The decoding configurations match plan 2 exactly. One small harness asymmetry to note for compute-cost comparisons: A3-arc has no stop-string and runs to the 512-token cap on 93% of problems even though the answer regex hits at 98.2%; B3-arc adds `stop_strings=["\nQuestion:"]` and chains terminate at the next exemplar boundary at a mean of 28.3 tokens. The asymmetry affects mean-tokens-generated reporting but not accuracy — the answer is extracted long before the cap.

## Per-cell results (n=500)

```
cell           accuracy  95% CI            answer_hit  mean_gen_tokens  vote_deg
A3_arc          0.828    (0.7925, 0.8585)    0.982          488.2          —
B3_arc (k=5)    0.850    (0.8160, 0.8786)    0.940           28.3        0.844
direct_arc      0.832    (0.7967, 0.8622)    0.932           10.9          —
```

Three things worth pulling out of that table.

The A3-arc `answer_hit_rate` of 0.982 means the primary regex `"The answer is X"` fired on 491 of 500 completions — well above the [RFC 005](./rfc-005-methodology.md) 0.85 floor. The model is following the exemplar format reliably; accuracy is not being dragged by extraction noise.

B3-arc's `vote_degeneracy_rate` of 0.844 says that 422 of 500 problems had all 5 sampled chains land on the same letter. Compare GSM8K's B3 vote_degeneracy of 0.316 ([RFC 003](./rfc-003-inference-compute-plateau.md)): on ARC-Easy, the model's k=5 sampling produces vastly less per-problem dispersion. Self-consistency cannot lift accuracy meaningfully because the chains are already converging on a single answer. Of the 78 non-degenerate problems, only 4 had unanimous vote-failure (`voted_count=0`, no chain produced a parseable letter); the remaining 74 had majorities at 4/5 (34 problems) or 3/5 (29 problems) with a small tail of 1–2/5 (11 problems).

The direct-arc cell's mean of 10.9 generated tokens at a 16-token cap reflects that the format emits exactly `" The answer is X."` (≈6 tokens) and the rest is room the model didn't need.

### Paired McNemar comparisons

| Comparison | only_A | only_B | both | neither | p (exact) |
|---|---:|---:|---:|---:|---:|
| A3 vs B3 | 11 | 22 | 403 | 64 | 0.080 |
| A3 vs direct | 34 | 36 | 380 | 50 | 0.905 |
| B3 vs direct | 28 | 37 | 388 | 47 | 0.321 |

The A3-vs-B3 p=0.080 is a step closer to significance than GSM8K's B4-vs-A3 p=0.193, but still inside the pre-registered ±3 point band and the CIs overlap.

## Primary outcome — A (plateau confirmed)

The pre-registered band for outcome A required `|B3 − A3| ≤ 3 pts` with overlapping CIs.

Observed: B3-arc − A3-arc = +2.2 pts, CI overlap = `True` (`[0.792, 0.858]` vs `[0.816, 0.879]`). Outcome A fires.

The same +2.2-point lift was observed on GSM8K (B4-A3 = +2.2 pts at p=0.193, [RFC 003](./rfc-003-inference-compute-plateau.md)). Two independent benchmarks, two different sample-vs-greedy comparisons, identical magnitude lift, both insignificant. **Path 1's plateau is a property of the model, not a property of GSM8K.**

This closes [RFC 006](./rfc-006-open-questions-and-next-steps.md) item 1. The four-paths writeup can state cleanly: "inference-time compute beyond a single greedy CoT pass does not improve E2B-it on either GSM8K or ARC-Easy."

## Secondary outcome — D (CoT does not beat direct on ARC-Easy)

Pre-registered threshold: `A3 − direct < 10 pts` fires outcome D. Observed: −0.4 pts. **Outcome D fires sharply.**

This is qualitatively distinct from GSM8K. [RFC 002](./rfc-002-cot-gate-finding.md) recorded a +30-point CoT lift on GSM8K with McNemar p=1.9×10⁻⁹. Plan 4 records **no** measurable CoT lift on ARC-Easy at the same prompt protocol. The two best explanations:

**Explanation 1 (most likely): the direct cell is artificially strong.** ARC-Easy is 4-way multiple choice with the answer letter in-context as part of the 8-shot exemplars. The model is being primed on `"Question: <q>\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: The answer is X."` eight times, and then asked to fill in the same template once more. With strong factual recall from pretraining and IT, the model can answer 83.2% of ARC-Easy questions directly — there is little headroom for CoT to add. The same direct prompt format produced 1.0% on GSM8K because GSM8K answers require multi-step arithmetic that 16 tokens cannot express.

**Explanation 2 (less likely): CoT is genuinely GSM8K-specific on E2B.** Possible but inconsistent with the strong base-model CoT lift in [RFC 002](./rfc-002-cot-gate-finding.md) and the broader literature on CoT's benchmark-dependence.

The pre-registered investigation step for outcome D is to rerun direct-arc with `max_new_tokens=4` to tighten the format; that experiment is not yet executed, but the existing direct cell's `answer_hit_rate=0.932` already says the format is being followed in the vast majority of completions. Re-running at len=4 is unlikely to move the needle by 10+ points.

The honest reading: **on ARC-Easy at this prompt protocol, the +30-point CoT gate finding does not generalize.** It is a GSM8K-specific finding shaped by the benchmark's "answer at the end of multi-step arithmetic" structure. [RFC 002](./rfc-002-cot-gate-finding.md)'s gate result is unaffected on its own benchmark, but the broader claim "CoT enables non-trivial reasoning on E2B" should be qualified to "on benchmarks where direct prompting cannot produce the answer in the budget."

## Repetition loops on ARC-Easy — the predicted disappearance lands

[RFC 004](./rfc-004-repetition-loop-failure-mode.md) hypothesized that repetition loops are tied to "problems whose correct solution has a natural phrase-level repetition structure" — i.e., GSM8K's chained arithmetic. ARC-Easy's 4-way MC short-question format has no such structure. The plan-4 data confirms the prediction sharply.

| Cell | Completion-level rep-loop rate | Reference (GSM8K, RFC 007) |
|---|---:|---:|
| A3-arc (greedy) | 3/500 = **0.6%** | 51/500 = 10.2% |
| B3-arc (sampled, chain-level) | 4/2500 = **0.16%** | 220/2500 = 8.8% |
| B3-arc (problem-level ≥1 chain rep-looping) | 4/500 = 0.8% | not tabulated in [RFC 007](./rfc-007-repetition-loop-isolation.md) |

A 17× reduction at the completion level and a 55× reduction at the chain level. Of the 4 B3-arc problems with any chain rep-looping, **zero** had unanimous rep-looping across all 5 chains. The two A3 rep-loop examples spot-checked are the same kind of phrase-attractor as on GSM8K (e.g., idx=249: `"the framework for the framework for the framework for..."`); the regex's character behavior is benchmark-independent.

This closes the cross-benchmark question [RFC 007](./rfc-007-repetition-loop-isolation.md) explicitly left open under "Open questions explicitly left open / ARC-Easy rep-loop rate." The verdict is what RFC 007 expected: rep-loops are GSM8K-specific (or, more carefully: they are structure-of-the-benchmark-specific), and on ARC-Easy they are essentially absent.

## Compute economics — direct is 45× cheaper for the same accuracy

The four-paths comparison cares about FLOPs and wallclock as well as accuracy. The plan-4 cells produce a striking compute table on a single 3090:

| Cell | Mean wallclock per problem | Total wallclock (n=500) | Accuracy |
|---|---:|---:|---:|
| A3-arc | 3.15 s | 26 min | 0.828 |
| B3-arc (k=5) | 3.15 s | 26 min | 0.850 |
| direct-arc | 0.11 s | 0.95 min | 0.832 |

direct-arc lands within 0.4 percentage points of A3-arc at one-thirtieth the wallclock and one-fortieth the generated tokens (10.9 vs 488.2 mean), at no sacrifice in answer_hit_rate worth flagging. **For ARC-Easy specifically, the compute-optimal Path 1 entry is direct_arc, not A3_arc.**

This does not change the GSM8K side of the head-to-head — direct prompting is firmly ruled out on GSM8K by [RFC 002](./rfc-002-cot-gate-finding.md). It does change the ARC-Easy side: the Path 1 representative for ARC-Easy is reasonable to report as direct_arc (cheaper, statistically equivalent to A3_arc) **or** as A3_arc (matched-protocol with GSM8K's A3 reference). The choice is a writeup decision for the head-to-head, not a measurement question. Recommendation: report A3_arc as the matched representative for cross-benchmark consistency, and disclose direct_arc as a "the cheap variant lands at the same accuracy" footnote.

## Implications for the four-paths head-to-head

Three concrete updates to the framing in [RFC 006](./rfc-006-open-questions-and-next-steps.md) and the broader investigation:

**Path 1's ARC-Easy entry is 82.8% ± 3.3 (A3-arc).** Path 1 representative cells are now confirmed for both benchmarks: A3_len512 at 30.0% on GSM8K and A3_arc at 82.8% on ARC-Easy. The cross-benchmark plateau finding is symmetric (both +2.2 pts SC vs greedy), and the rep-loop story is asymmetric (heavy on GSM8K, absent on ARC-Easy) — both are now firm cross-benchmark observations, not GSM8K artifacts.

**The four-paths writeup can claim cross-benchmark plateau cleanly.** Plan 2's GSM8K plateau ([RFC 003](./rfc-003-inference-compute-plateau.md)) was a single-benchmark result; with plan 4 in hand, the claim is "Path 1 is plateau-bound on both rigs" and other paths must beat both. Self-consistency at k≥5 is empirically not the lever for Path 1 on either benchmark.

**The CoT gate is GSM8K-specific.** [RFC 002](./rfc-002-cot-gate-finding.md)'s finding remains valid on its own benchmark, but the broader narrative ("CoT enables non-trivial reasoning on E2B") needs the ARC-Easy footnote: on benchmarks whose questions can be answered directly from pretraining knowledge, no CoT lift is observable on E2B-it. Future paths claiming reasoning lifts must specify on which benchmark families their lifts apply.

## Pre-registration audit

| outcome | pre-registered threshold | observed | fired? |
|---|---|---|---|
| **A** — plateau confirmed | `\|B3-A3\| ≤ 3 pts`, CIs overlap | +2.2 pts, overlap=True | **yes** |
| B — SC lifts on ARC | `B3-A3 ≥ 5`, non-overlapping CIs | +2.2 pts, overlap=True | no |
| C — saturation | both cells ≥ 90% | A3=82.8%, B3=85.0% | no |
| **D** — CoT does not beat direct | `A3-direct < 10 pts` (only meaningful with direct cell) | −0.4 pts | **yes (secondary)** |

A is the primary outcome. D fires as a secondary outcome on the optional direct cell that was run despite not being required. No follow-up is mandated by the pre-registration; the outcome-D follow-up step (`max_new_tokens=4` direct rerun) is filed as a low-priority open question, not a blocker.

## Sanity-check audit (per [RFC 005](./rfc-005-methodology.md))

1. **Exemplar hash:** `633e12e57bd8cd19` recorded in manifest, byte-stable across runs. ✓
2. **Gold-answer parser:** 21 numeric remaps and 2 three-choice problems handled, 0 dropped, all 500 retained. ✓
3. **Answer-hit rate floor (≥0.85):** A3=0.982, B3 chain=0.940, direct=0.932 — all clear. ✓
4. **Plausibility (A3 in 0.65–0.92):** A3=0.828, comfortably in band. ✓
5. **Direct sanity (≥0.40):** direct=0.832, well above the 25% chance baseline and the 40% floor. ✓
6. **Cross-plan paired reproduction:** plan 4 has no shared cell with plans 1–3 (different benchmark). N/A.

## Limitations

Three to flag in any external writeup.

The exemplars are hand-crafted (no published ARC-Easy CoT exemplars exist), so a different choice of 8 shots could shift the absolute accuracy numbers by a few points. The exemplar_hash pins the exact set used and is recorded in the manifest. The qualitative findings — plateau, no CoT lift over direct, no rep-loops — are robust to plausible exemplar perturbations.

The direct-arc cell uses 8-shot priming, which is the same prompt scaffolding the literature uses for direct-answer baselines. A truly zero-shot direct cell would likely score lower and might restore some apparent CoT lift. That experiment was not specified in plan 4 and is filed as a future option, not a blocker. Plan 5 (0-shot CoT vs 8-shot CoT) covers an adjacent question on GSM8K.

The plateau finding here applies to one greedy length cell and one self-consistency cell. Plan 4 deliberately did not run a full ARC-Easy length sweep on the rationale that GSM8K's length sweep was completely flat. If accuracy on ARC-Easy is at all length-sensitive, that effect is invisible to this plan. The compute-economics observation (direct vastly cheaper) provides indirect evidence that ARC-Easy answers don't need long contexts.

## Calibration of prior RFCs

- **[RFC 003](./rfc-003-inference-compute-plateau.md)** "Limitations and residual uncertainty" listed "this finding is GSM8K-only" as the first item. Plan 4 closes that limitation. The Limitations section should be updated to remove the GSM8K-only caveat and cite this RFC as the cross-benchmark confirmation.
- **[RFC 004](./rfc-004-repetition-loop-failure-mode.md)** predicted that ARC-Easy's log-likelihood scoring would "be immune" to rep-loops. The actual ARC-Easy result is not via log-likelihood scoring (we use generate-then-extract), but the underlying prediction — that ARC-Easy does not exhibit the failure mode — is confirmed at 0.6% completion-level vs 10.2% on GSM8K.
- **[RFC 006](./rfc-006-open-questions-and-next-steps.md)** item 1 is closed. The document's "What is not known" section should be updated to remove plan-4 from the open list. End-state condition 4 (plan 4 outcome labeled) is now satisfied.
- **[RFC 007](./rfc-007-repetition-loop-isolation.md)** "Open questions explicitly left open / ARC-Easy rep-loop rate" can be marked closed with the numbers above.

## Implications for plan 5 (0-shot vs 8-shot)

Plan 5 was scheduled before or alongside plan 4 in [RFC 006](./rfc-006-open-questions-and-next-steps.md)'s recommended order, contingent on a 0-shot lift potentially shifting the Path 1 representative. The plan-4 results suggest the question is more interesting than originally framed: on ARC-Easy, the *prompt format* itself (8-shot priming) appears to be doing essentially all the work, not the CoT reasoning. A 0-shot ARC-Easy cell would almost certainly score lower than 83% — potentially much lower — which would *not* reopen the plateau question but *would* reopen the CoT-vs-direct question by showing how much of the direct cell's strength was 8-shot priming. Plan 5 is currently scoped to GSM8K only; if it executes there and shows 0-shot matches 8-shot, the same experiment on ARC-Easy is high-value-per-hour and worth a future plan-6 ticket.

## Evidence preserved

- `results/path_1_cot_tokens/plan4/cells/A3_arc__0000_0500.jsonl` — full per-problem rows (500).
- `results/path_1_cot_tokens/plan4/cells/B3_arc__0000_0500.jsonl` — k=5 chains preserved per row, voted answer + voted_count.
- `results/path_1_cot_tokens/plan4/cells/direct_arc__0000_0500.jsonl` — full per-problem rows (500).
- `results/path_1_cot_tokens/plan4/manifest.json` — pinned config (model commit, dtype, exemplar set, exemplar hash, regexes, gold_stats with remap counts).
- `results/path_1_cot_tokens/plan4/.DONE` — completion sentinel.

A `results_plan4.json` summary and `results_plan4.png` plot can be generated from the existing JSONLs by running:

```
python path1_arc_easy.py --summarize
```

per the plan-4 deliverables in [`path1_plan4.md`](../plans/path_1_cot_tokens/plan4.md). Source script: `path1_arc_easy.py`.

## References

- [RFC 002](./rfc-002-cot-gate-finding.md) — GSM8K CoT gate finding; the +30-pt lift this RFC qualifies.
- [RFC 003](./rfc-003-inference-compute-plateau.md) — GSM8K plateau finding; the cross-benchmark confirmation lives here.
- [RFC 004](./rfc-004-repetition-loop-failure-mode.md) — rep-loop observation; the cross-benchmark prediction confirmed here.
- [RFC 005](./rfc-005-methodology.md) — methodology; followed without deviation except the harness asymmetry on stop-strings noted above.
- [RFC 006](./rfc-006-open-questions-and-next-steps.md) — open questions; this RFC closes item 1.
- [RFC 007](./rfc-007-repetition-loop-isolation.md) — rep-loop quantitative isolation on GSM8K; this RFC's ARC-Easy rep-loop numbers extend [RFC 007](./rfc-007-repetition-loop-isolation.md)'s closed open question.
- [`path1_plan4.md`](../plans/path_1_cot_tokens/plan4.md) — pre-registered plan, executed without deviation.

---

*This RFC documents a settled, pre-registered analysis result. The verdict (A primary, D secondary) is final pending plan 5 (0-shot CoT) data, which can extend but not retract these conclusions.*
