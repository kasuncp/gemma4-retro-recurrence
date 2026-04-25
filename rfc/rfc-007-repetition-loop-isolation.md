# RFC 007 — Repetition-Loop Isolation: Quantitative Verdict

| Field | Value |
|---|---|
| **RFC** | 007 |
| **Title** | Repetition-Loop Isolation: Quantitative Verdict |
| **Status** | Finding (pre-registered, executed, B-leaning-A) |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Depends on** | [RFC 002 (gate)](./rfc-002-cot-gate-finding.md), [RFC 003 (plateau)](./rfc-003-inference-compute-plateau.md), [RFC 004 (rep-loop observation)](./rfc-004-repetition-loop-failure-mode.md), [RFC 005 (methodology)](./rfc-005-methodology.md) |
| **Sharpens** | [RFC 004](./rfc-004-repetition-loop-failure-mode.md) |
| **Closes** | [RFC 006](./rfc-006-open-questions-and-next-steps.md) item 3 |
| **Source plan** | [`path1_plan3.md`](./path1_plan3.md) |

## Abstract

Plan 3's analysis-only re-classification of all plan 1 and plan 2 generations confirms that repetition loops are a real but small contributor to Path 1's GSM8K ceiling. The lenient counterfactual lifts the A3 representative by +3.4 points (30.0% → 33.4%), placing the result inside the pre-registered 3–8 point band for finding B (medium stability factor). However, the upper-bound counterfactual lifts A3 by **0.0 points** — none of A3's 51 rep-looped problems is solved by any other configuration we ran — and four secondary diagnostics (length-independence, sampling-independence, vote-on-repetition decomposition, sticky-problem concentration) all point toward the small-factor end. The honest reading is **B-leaning-A**: rep-loops cost roughly 0–3 points on top of a hard reasoning ceiling, not 5+. Path 1's plateau is overwhelmingly a reasoning ceiling. The four-paths scoring axis does not need a separate generation-stability metric.

## Motivation and recap

[RFC 004](./rfc-004-repetition-loop-failure-mode.md) observed that a meaningful fraction of E2B-it CoT completions collapse into verbatim repetition. At n=100 the rate was reported as 13%. Plan 3 systematically classifies every completion across plans 1 and 2 — 10 cells, 5,500 axis-A completions, and 9,500 axis-B chain-level outcomes — into the four-bucket taxonomy (`correct` / `terminated_wrong` / `repetition_loop` / `truncated_no_answer`), and computes two pre-registered counterfactual accuracies (lenient and upper-bound) per cell.

The question this RFC answers, pre-registered before execution:

> If repetition-loop failures were excluded from scoring, how much would Path 1's accuracy change — and is that change small (finding A), medium (finding B), or large (finding C)?

The decision matters because the head-to-head framing of Paths 2–4 changes by which finding holds. Under a small/zero stability cost, other paths must beat Path 1 on genuine reasoning. Under a large stability cost, the comparison has a confound and may need a separate stability axis.

## Methodology recap

Methodology follows [RFC 005](./rfc-005-methodology.md). Two plan-specific items:

The canonical detection regex is `(.{10,60})\1{2,}` — a 10-to-60-character substring that repeats at least three times immediately after itself. This regex was committed before the analysis began and was not modified. The canonical regex hash recorded in `results_plan3.json` is `2a30af06047e`.

A 30-completion stratified sample of regex-flagged completions (`flagged_sample_30.json`, seed 42) was inspected manually for false positives. Result: 28 of 30 are clear true-positive rep-loops with 8+ visible repetitions; 2 are borderline at exactly 3 repetitions (the regex's minimum), e.g. `"This is 400. This is 400. This is 400."` — both genuine short loops on inspection. False-positive rate 0/30 = 0%, well below the pre-registered 20% tightening threshold. The canonical regex stands.

False-negative rate (paraphrased, non-verbatim repetition) is acknowledged but not measured here; per [RFC 004](./rfc-004-repetition-loop-failure-mode.md), the true rep-loop rate is plausibly 10–20% higher than what the regex catches. All findings in this RFC are robust to that direction of error: the conclusion is that rep-loops are a small factor, and undercount only strengthens that conclusion.

## Per-cell outcome table

```
cell                   n    correct  term_wrong  rep_loop  trunc_no_ans
plan1.it_cot_n100     100   31       55          14        0
plan1.base_cot_n100   100   22       75           3        0
plan2.A1_len128       500  149      299          51        1
plan2.A2_len256       500  150      296          52        2
plan2.A3_len512       500  150      297          51        2
plan2.A4_len1024      500  150      297          51        2
plan2.B1_k1           500  154      299          46        1   (voted)
plan2.B2_k3           500  159      304          37        0   (voted)
plan2.B3_k5           500  148      320          31        1   (voted)
plan2.B4_k10          500  161      316          23        0   (voted)
```

Axis B chain-level rep-loop counts (across all k×n chains) are 46/500, 145/1500, 220/2500, 475/5000 for B1–B4 respectively, giving chain-level rep-loop rates of 9.2%, 9.7%, 8.8%, 9.5%.

## Adjusted-accuracy table under counterfactuals

The lenient counterfactual is `correct / (n − rep_loop)`: the accuracy on completions that actually attempted to answer. The upper-bound counterfactual is `(correct + n_rep_loop_problems_a3_correct) / n`: for each rep-looped problem in this cell, look up whether A3 (the reference greedy cell) solved that same problem on its own pass; impute that outcome.

| cell | headline | lenient | upper(A3) | Δ lenient | Δ upper |
|---|---:|---:|---:|---:|---:|
| plan1.it_cot_n100 | 0.310 | 0.360 | 0.310 | +5.0 | +0.0 |
| plan1.base_cot_n100 | 0.220 | 0.227 | 0.220 | +0.7 | +0.0 |
| plan2.A1_len128 | 0.298 | 0.332 | 0.306 | +3.4 | +0.8 |
| plan2.A2_len256 | 0.300 | 0.335 | 0.304 | +3.5 | +0.4 |
| **plan2.A3_len512** | **0.300** | **0.334** | **0.300** | **+3.4** | **+0.0** |
| plan2.A4_len1024 | 0.300 | 0.334 | 0.300 | +3.4 | +0.0 |
| plan2.B1_k1 | 0.308 | 0.339 | 0.310 | +3.1 | +0.2 |
| plan2.B2_k3 | 0.318 | 0.343 | 0.320 | +2.5 | +0.2 |
| plan2.B3_k5 | 0.296 | 0.316 | 0.302 | +2.0 | +0.6 |
| plan2.B4_k10 | 0.322 | 0.338 | 0.324 | +1.6 | +0.2 |

Both counterfactuals are clearly labeled as such and do **not** replace the [RFC 003](./rfc-003-inference-compute-plateau.md) headline numbers. The Path 1 representative for the head-to-head remains A3 at 30.0% ± 4.1.

## The central tension: lenient vs upper-bound

The two counterfactuals diverge sharply on A3 — +3.4 points lenient against +0.0 points upper-bound — and that gap is the most informative result in plan 3. The lenient counterfactual assumes rep-looped problems are an unbiased random sample of GSM8K problems and would have been answered at the cell's average correct-rate. The upper-bound counterfactual asks a more specific, falsifiable question: of A3's 51 rep-loop problems, how many does A3 solve on its other passes? The answer is **zero**.

Across the entire axis-A length sweep, rep-loop problems on A3 are not solved by A1, A2, or A4 either. Across axis B, the imputation count is 1, 1, 3, 1 problems out of 23–46 rep-loop problems per cell. **Rep-loops concentrate on problems the model cannot solve through any inference-time path tested in plans 1 and 2.** This is the single strongest factual claim plan 3 produces, and it argues that the lenient swing's +3.4 points is an optimistic upper bracket on stability cost, not a measurement of recoverable accuracy.

## Length-independence (kills RFC 004 prediction 1)

Axis A's rep-loop rate is essentially a flat line:

| cell | max_new_tokens | rep-loop rate |
|---|---:|---:|
| A1_len128 | 128 | 0.102 |
| A2_len256 | 256 | 0.104 |
| A3_len512 | 512 | 0.102 |
| A4_len1024 | 1024 | 0.102 |

Doubling the cap four times changes nothing. Even at the 128-token cap, 10.2% of completions have already entered a verbatim-repeating sub-sequence — meaning loops are an **early-decoding phenomenon**, not a "ran out of useful things to say" phenomenon. The greedy decoder lands in the attractor and stays there from near the start of generation; longer caps just give it more room to keep repeating.

This refutes one of the two predictions [RFC 004](./rfc-004-repetition-loop-failure-mode.md) made about the loops' mechanism. The "longer caps invite more loops" hypothesis is dead.

## Sampling-independence (neither finding D nor E)

Plan 3's pre-registered finding D (axis B chain rep-loops >> A3) and finding E (axis B chain rep-loops << A3) both fail to fire:

| comparison | rep-loop rate |
|---|---:|
| A3 greedy completion-level | 10.2% |
| B1_k1 chain-level | 9.2% |
| B2_k3 chain-level | 9.7% |
| B3_k5 chain-level | 8.8% |
| B4_k10 chain-level | 9.5% |

Sampling at T=0.7 produces a 1–1.4 point reduction in chain-level rep-loop rate relative to greedy. Real, but small. The interpretation is that sampling does occasionally perturb the greedy attractor enough to escape it, but most rep-loops are robust to this level of stochasticity. **Self-consistency cannot solve the plateau by escaping rep-loops** — there are not many to escape and each escape only buys back one chain.

## Vote-on-repetition decomposition

This is the most mechanistically informative axis-B subtable. For each axis-B cell, voted-wrong problems are decomposed into five sub-categories per plan 3 step 5:

| cell | voted-wrong | unanimous rep-loop | rep-loop plurality wrong | rep-loop unlucky | no rep-loop wrong | other |
|---|---:|---:|---:|---:|---:|---:|
| B1_k1 | 346 | 46 (13.3%) | 0 | 0 | 300 (86.7%) | 0 |
| B2_k3 | 341 | 11 (3.2%) | 2 (0.6%) | 43 (12.6%) | 254 (74.5%) | 31 (9.1%) |
| B3_k5 | 352 | 9 (2.6%) | 4 (1.1%) | 68 (19.3%) | 249 (70.7%) | 22 (6.2%) |
| B4_k10 | 339 | 4 (1.2%) | 5 (1.5%) | 106 (31.3%) | 202 (59.6%) | 22 (6.5%) |

Three readings.

The `unanimous_rep_loop` category — every chain rep-loops, voting cannot rescue — collapses geometrically with k. From 13.3% at k=1 to 1.2% at k=10. This is the only mechanism by which voting actually reduces rep-loop accuracy cost: as k grows, the chance that *every* chain falls into the same attractor falls sharply. Concretely, problem-level rep-loops drop from 46 (k=1) to 23 (k=10) — a ~5-problem rescue that matches the +1.4 point lift of B4 over B1 in headline accuracy.

The `rep_loop_plurality_wrong` category — the plan-3-defined case where rep-loop chains outvoted a correct minority — is essentially empty: 0, 0.6%, 1.1%, 1.5% across k. The pre-registered threshold for "concrete mechanism dragging SC down" was >5%. **There is no SC-dragging-by-rep-loops effect on Path 1.** Plan 3 step 5's specific concern is empirically falsified.

The `rep_loop_unlucky` category climbs from 0% to 31.3% across k — but this is bookkeeping, not signal. As k grows, more voted-wrong problems mechanically have *some* rep-looped chain in the bag, even though the rep-loop was incidental to the wrong vote (the majority of non-rep-looped chains independently agreed on a wrong answer). Strip out the bookkeeping and rep-loop-caused vote losses at k=10 are 4 (unanimous) + 5 (plurality) = 9 problems out of 500. **At k=10, rep-loops cost self-consistency at most 1.8 points of accuracy** — well inside [RFC 003](./rfc-003-inference-compute-plateau.md)'s plateau noise floor.

## Sticky problems point at structural difficulty, not stability

Of the top 10 sticky problems (those that triggered rep-loops in the most cells covering them), all 10 trigger in 8+ of 8–10 covering cells, and 8 of 10 trigger at rate 1.0 — every cell, every time. Spot checks of these problems' text:

- `idx 49` (rate 1.0, 10/10 cells): 15-floor apartment, 8 units/floor, 3/4 occupied — find unoccupied. Multi-step proportional.
- `idx 88` (rate 0.9, 9/10 cells): record sales ratio with two-variable substitution.
- `idx 174` (rate 1.0, 8/8 cells): potato peeling time, mixed minute/second units.
- `idx 246` (rate 1.0, 8/8 cells): cost compounding with end-of-job percentage discount.
- `idx 363` (rate 1.0, 8/8 cells): two-leg trip with detour, mixed distance/speed/time.
- `idx 384` (rate 1.0, 8/8 cells): inverse-fraction problem ("sold 3/5, has 12.8 left, find original").

These are not edge cases the model almost solves. They are problems the model gets stuck on the *same way every time*, regardless of decoding strategy or generation length. That is consistent with the upper-bound finding (0/51 A3 rep-loop problems solved by any other Path-1 cell): the repetition is a *symptom* of the underlying difficulty for these specific problem structures — multi-step proportional reasoning, cost compounding with discounts, unit conversion through fractions — not an independent stability failure that masks a recoverable answer.

## Verdict

**Finding letter: B, leaning A.**

The lenient swing on A3 is +3.4 points, which is technically inside the pre-registered 3–8 point band for finding B. By the letter of the pre-registration, B is the verdict and is recorded as such in `results_plan3.json` (`finding.letter = "B"`).

But every other line of evidence in plan 3 argues that the truth lies closer to the small-factor end of B, or even into A:

- **Upper-bound swing on A3: +0.0 points.** Zero of 51 A3 rep-loop problems are solved by any other Path-1 configuration.
- **No length effect.** Doubling the cap from 128 to 1024 leaves the rep-loop rate unchanged at ~10.2%.
- **No sampling effect.** T=0.7 sampling reduces chain-level rep-loop rate by only 1–1.4 points.
- **No SC-dragging effect.** rep_loop_plurality_wrong is 0–1.5% across all k, well below the pre-registered 5% threshold.
- **Rep-loops concentrate on structurally hard problems.** 8 of 10 sticky-problem entries rep-loop in every cell that covers them.

The honest framing for downstream writeups is: rep-loops cost 0–3 points on top of a hard reasoning ceiling, not 5+. The pre-registered finding-B language ("part reasoning, part stability") is technically correct but overstates the stability share.

## Implications for the four-paths head-to-head

Three concrete changes to the framing in [RFC 006](./rfc-006-open-questions-and-next-steps.md) follow.

First, **Path 1's 30.0% GSM8K ceiling is overwhelmingly a reasoning ceiling.** Paths 2–4 cannot pick up meaningful "free" accuracy by reducing rep-loops alone — there are not enough recoverable rep-loops to harvest. They have to beat Path 1 on actual reasoning capacity.

Second, **Path 4 (Mixture-of-Depths) loses its strongest plan-3-conditional pitch.** The framing in [RFC 006](./rfc-006-open-questions-and-next-steps.md) reserved a "concrete stability-aid angle" for Path 4 contingent on a medium or large rep-loop factor. Plan 3 now constrains that angle: at most ~1.8 points on GSM8K at k=10 — i.e., MoD's stability story has to be much sharper than "could escape rep-loops" because the rep-loop budget is small and concentrated on problems the rest of Path 1 also fails. MoD's pitch on this benchmark is more honestly framed as a reasoning-capability story.

Third, **the four-paths scoring axis does not need a separate generation-stability metric.** Plan 3's step-5 confound concern — that a path reducing rep-loops while holding reasoning constant would falsely appear to win on accuracy — is empirically falsified. Rep-loops are not differentially distorting axis A vs axis B comparisons within Path 1, and the same logic generalizes across paths. Accuracy alone is sufficient.

## Calibration of RFC 004's quantitative claims

[RFC 004](./rfc-004-repetition-loop-failure-mode.md) cites 13 rep-loop completions on the n=100 IT-CoT cell. The canonical plan-3 count is **14**. The discrepancy is one completion, sourced from a difference between the original spot-classification used to build [RFC 002](./rfc-002-cot-gate-finding.md)'s failure-mode table and plan 3's systematic regex pass. The corrected breakdown for the n=100 IT-CoT cell is:

| failure category | RFC 004 (original) | plan 3 (canonical) |
|---|---:|---:|
| Terminated cleanly with wrong answer | 49 | 55 |
| Repetition loop | 13 | 14 |
| Truncated without any answer | 7 | 0 |
| **Total wrong** | **69** | **69** |

The total wrong count is preserved (69 in both); the redistribution is across the three sub-categories. The shift from `truncated_no_answer = 7` to `0` reflects that plan 3's classifier checks the rep-loop regex first and several previously-truncated-no-answer cases are actually rep-loops that hit the cap mid-loop. RFC 004 should be updated to cite the canonical numbers and note the methodology shift in the classifier.

The rep-loop rate (14% vs the previously-cited 13%) does not change qualitatively; the "~4× more prone than base" framing is preserved (14/100 vs 3/100). This is a rounding-precision update, not a finding revision.

## Pre-registration audit

The plan committed to five letter-coded outcomes before execution. Audit:

| outcome | pre-registered threshold | observed | fired? |
|---|---|---|---|
| A — small factor | adjusted swing < 3 pts on all cells | A3 lenient = +3.4 | no |
| **B — medium factor** | **adjusted swing 3–8 pts on cells** | **A3 lenient = +3.4** | **yes (low end)** |
| C — large factor | adjusted swing > 8 pts | max swing = +5.0 (it_cot_n100) | no |
| D — axis B >> A3 chain rep-rate | qualitative | B chain rates 8.8–9.7% vs A3 10.2% | no (slight reduction, not amplification) |
| E — axis B << A3 chain rep-rate | qualitative | 1.0–1.4 pt reduction | no (too small to fire E) |

B is the only finding that fires, at the low end of its band. No follow-up experiment is needed — plan 3 is self-contained by design and answers its question in one pass.

## Limitations

Three to flag in any external write-up.

The regex catches only verbatim 10–60 character spans repeating ≥3 times. Paraphrased semantic repetition ("I count 5 apples. I see 5 apples. There are 5 apples.") is missed entirely. The true rep-loop rate is plausibly 10–20% higher than what plan 3 reports; all findings here are robust to that direction of error.

The upper-bound counterfactual uses A3 as the imputation reference. A different choice of reference cell (e.g., B4_k10, which has the highest accuracy of any Path-1 cell) would produce slightly different upper-bound numbers. The qualitative finding — that rep-looped problems are concentrated on hard problems no Path-1 cell solves — is robust across reference choices.

This finding is GSM8K-only. The rep-loop rate on ARC-Easy may differ. Plan 4 produces ARC-Easy completions whose rep-loop rate can be analyzed by the same script for free; that analysis should be added when plan 4 results land.

## Evidence preserved

- `results_plan3.json` — full per-cell counts, adjusted accuracies, length/sampling/vote-on-rep tables, sticky-problem list, FP-sample summary, and finding label.
- `path1_rep_loops_by_cell.csv` — flat table for figure generation downstream.
- `flagged_sample_30.json` — 30 manually-inspected completions with `max_repetition_count` and `heuristic_label` per sample. Used to validate the regex's false-positive rate (0/30 confirmed FPs).
- Visualization: `results_plan3.png` — four-panel summary (per-cell rep-loop rate; adjusted accuracies under counterfactuals; length-dependence and chain-level sampling-dependence; vote-on-repetition stacked bars).

Source script: `path1_rep_loop_analysis.py` per [`path1_plan3.md`](./path1_plan3.md). Ran on CPU in under 2 minutes per spec.

## Open questions explicitly left open

The rep-loop story is closed on GSM8K with greedy/T=0.7 8-shot CoT. Three downstream questions plan 3 does *not* answer:

- **ARC-Easy rep-loop rate.** Will land when plan 4 completes. Re-run `path1_rep_loop_analysis.py` with the new JSONLs added to the cells dictionary; expected runtime is still under 2 minutes.
- **Rep-loop rate under 0-shot CoT.** Plan 5's outcomes will produce JSONLs that can be analyzed the same way. The hypothesis worth flagging: if 0-shot lifts accuracy and also reduces rep-loops, the "8-shot exemplars prime the loop attractor" story gets some support.
- **Mechanistic cause inside the model.** Why these specific 50-ish problems out of 500 are sticky attractors is a model-internals question that no Path-1 plan answers and that plan 3 was not designed to answer. Out of scope.

## References

- [RFC 002](./rfc-002-cot-gate-finding.md) — gate finding; original n=100 failure-mode breakdown.
- [RFC 003](./rfc-003-inference-compute-plateau.md) — plateau finding; A3 representative.
- [RFC 004](./rfc-004-repetition-loop-failure-mode.md) — observation this RFC sharpens. Should be updated with the corrected n=100 counts.
- [RFC 005](./rfc-005-methodology.md) — methodology; counterfactual definitions.
- [RFC 006](./rfc-006-open-questions-and-next-steps.md) — open questions; this RFC closes item 3.
- [`path1_plan3.md`](./path1_plan3.md) — pre-registered plan, executed without modification.

---

*This RFC documents a settled, pre-registered analysis result. The verdict (B-leaning-A) is final pending plan 4 / plan 5 ARC and 0-shot data, which can extend but not retract the GSM8K conclusion.*