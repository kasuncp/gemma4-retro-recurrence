# Path 1 — plan 4: ARC-Easy cross-benchmark validity

## Context

Plans 1 and 2 established Path 1's picture on GSM8K:

- **Plan 1 gate:** IT-CoT beats IT-direct by +30 points (31% vs 1% at n=100), McNemar p = 1.9 × 10⁻⁹. CoT is alive on E2B.
- **Plan 2 sweep:** Beyond that one-time CoT lift, inference-time compute doesn't help. 8 cells spanning `max_new_tokens ∈ {128, 256, 512, 1024}` and `k ∈ {1, 3, 5, 10}` all land inside a 2.6-point band at 29.6%–32.2%. No Pareto frontier. B4 (k=10) vs A3 (greedy 512) has a +2.2 point lift that fails McNemar significance (p = 0.193).

**Everything we know about Path 1's ceiling is from GSM8K.** The four-paths head-to-head scores on **both** ARC-Easy and GSM8K. Path 2's depth-recurrence findings (loopable valley, KV wall, perplexity-reasoning divergence) were verified on *both* benchmarks, and that cross-benchmark agreement is why we trust them. Path 1's story currently has no such cross-validation.

Plan 4 answers the one question that would falsify Path 1's plateau finding:

> **Does the "more compute doesn't help" claim hold on ARC-Easy, or is it a GSM8K-specific artifact?**

### Why ARC-Easy might behave differently

GSM8K has two failure modes that ARC-Easy does not:

1. **Repetition loops.** 13% of IT-CoT n=100 completions and ~8% of n=500 completions collapse into repetition. ARC-Easy uses log-likelihood scoring over four short answer choices — there is no generation to collapse.
2. **Truncation at the answer boundary.** 63% of A3 GSM8K completions hit the 512-token cap. ARC-Easy scoring is length-independent.

Either of these could mean ARC-Easy shows a self-consistency or length response that GSM8K hides. If it does, Path 1's plateau claim needs a GSM8K-specific asterisk in the four-paths writeup.

## Scope — what this plan is and isn't

**In scope:**
- IT model only (`google/gemma-4-E2B-it`, commit `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` from plan 2).
- ARC-Easy test split, deterministic first `n = 500` problems.
- **One axis-A cell** (A3-equivalent, the Path 1 representative) and **one axis-B cell** (the most promising self-consistency cell from plan 2).
- Same pinned deps as plan 2 (`transformers==5.6.1`, `torch==2.4.1+cu124`, `datasets==4.8.4`).

**Explicitly out of scope:**
- No full axis-A length sweep on ARC-Easy. A3 alone suffices for the plateau check because plan 2's GSM8K length sweep was fully flat — there's no signal to probe for on the other benchmark.
- No k > 5 on ARC-Easy. If B3 (k=5) doesn't lift, B4 (k=10) won't either given plan 2's diminishing returns.
- No base model. Plan 2 established the IT model is the deployment target.
- No GSM8K. Plan 2's GSM8K numbers stand.
- No multimodal probing, no training, no on-device measurement.

## Protocol

ARC-Easy is multiple-choice (4 answer options per question). The standard scoring is log-likelihood over each answer choice given the question, with the argmax choice counted correct. Two cells:

### Cell A3-arc — CoT, length-matched to A3

| Parameter | Value |
|---|---|
| Condition | 8-shot CoT, greedy |
| `max_new_tokens` | 512 |
| Decode | greedy |
| Scoring | generate rationale + extract answer letter (A/B/C/D) |
| n | 500 |

ARC-Easy's typical 8-shot CoT prompt format: 8 fixed exemplars showing `Question: <q>\nA) ...\nB) ...\nC) ...\nD) ...\nAnswer: <reasoning>. The answer is <letter>.` Use the ARC-Easy *train* split for exemplars. Any deterministic 8 drawn from the head of the train split is fine — just pin the exact IDs in the manifest so the result is reproducible.

Answer extraction: regex on `The answer is ([A-D])` first, fall back to last-letter-mentioned. Report `answer_hit_rate` (fraction of completions where the primary regex hits) alongside accuracy — it's the ARC analog of GSM8K's `hash_hit_rate`.

### Cell B3-arc — self-consistency, length-matched to B3

| Parameter | Value |
|---|---|
| Condition | 8-shot CoT, sampled, k=5 majority vote |
| `max_new_tokens` | 512 |
| Decode | `do_sample=True, temperature=0.7, top_p=0.95` |
| k | 5 |
| n | 500 |

Majority vote on the extracted letter across 5 chains. Ties broken by first-chain-wins (deterministic given seed).

Why k=5 not k=10: plan 2 showed B3 and B4 are statistically indistinguishable (p=0.902 paired), and B4 is 2× the compute. If self-consistency is going to show a cross-benchmark difference, it will show at k=5; we don't need k=10 to detect it.

### Also optional: direct-answer sanity cell

One optional cell, `direct-arc`, mirrors plan 1's direct-vs-CoT gate on ARC-Easy:

- 8-shot, answer-only exemplars (stripped of reasoning, terminal `The answer is <letter>.`)
- `max_new_tokens = 16`
- greedy

This is **optional but cheap**. If you're running this plan anyway, add this cell; it costs ~5 minutes and gives you the Path 1 direct-vs-CoT comparison on a second benchmark, which strengthens the blog post's claim about CoT capability in E2B. Skip it only if GPU time is precious.

## Budget

| Cell | Generations | Approx. time (3090, bf16) |
|---|---|---|
| A3-arc | 500 | ~45 min |
| B3-arc | 500 × 5 = 2,500 | ~2.5 hr |
| direct-arc (optional) | 500 | ~5 min |

**Total: ~3.5 hr on a single 3090.** Manageable even on a preempt-prone spot instance; use the same resume-on-restart pattern as plan 2.

## Sanity checks before trusting numbers

1. **Exemplar and prompt template pinned.** Save the exact 8-shot ARC-Easy prompt template and the 8 exemplar question IDs in the manifest. Compute a hash analogous to plan 2's `exemplar_hash` and log it.
2. **Gold-answer parser sanity.** On all 500 gold ARC-Easy answers, confirm `answerKey ∈ {A, B, C, D}`. If any are numbered (ARC-Easy historically had some) — remap to letters and flag in the manifest.
3. **Answer-hit rate floor.** If `answer_hit_rate < 0.80` on A3-arc, the model is not following the prompt format; debug before trusting accuracy.
4. **A3-arc vs published E2B-it ARC-Easy numbers.** The "four paths" visual guide notes ARC-Easy tops out around 90% on decent 2B chat models and that E2B is known to be strong here. A plausible A3-arc accuracy is **75%–85%**. If it comes in below 65%, the prompt format is wrong; above 92%, the benchmark is saturated and won't differentiate cells well (but that's still useful information).
5. **direct-arc is not trivially 25%.** If included, direct-arc should land well above chance (25%) — any value above ~40% is fine. ARC-Easy 4-way choice with 8-shot priming is an easier "direct" regime than GSM8K 8-shot direct was.

## Pre-registered interpretation

Decide **before looking at numbers** what each outcome means.

### Outcome A — Plateau confirmed
B3-arc − A3-arc lift is within ±3 points, CIs overlapping. **Path 1's plateau holds on both benchmarks.** The four-paths writeup can state "inference-time compute doesn't help Path 1 on GSM8K or ARC-Easy" as a clean cross-benchmark claim.

### Outcome B — ARC-Easy shows a self-consistency response GSM8K hides
B3-arc − A3-arc ≥ 5 points with non-overlapping CIs. **Interpret cautiously.** Two mechanisms could explain this, and it matters which:

- **Mechanism 1:** ARC-Easy's log-likelihood scoring sidesteps the repetition-loop failure mode, so self-consistency's benefit shows up more cleanly. This would be consistent with plan 1's failure-mode finding. In this case, the Path 1 entry to the head-to-head should report **both** cells (A3 for GSM8K, B3-arc for ARC-Easy), and the writeup explains that GSM8K's SC-flat result is partly a generation-quality artifact.
- **Mechanism 2:** ARC-Easy has a structurally different SC response (e.g. answer-space calibration). This is less likely but would require pulling the other axis-B cells onto ARC-Easy to locate the effect's magnitude.

Choose the follow-up experiment based on which mechanism fits: if Mechanism 1, the repetition-loop re-analysis in plan 3 covers it. If Mechanism 2, run B2-arc and B4-arc to see the full ARC SC curve.

### Outcome C — ARC-Easy saturates above 90% on both cells
A3-arc and B3-arc both land >90%. **Benchmark ceiling too close for meaningful differentiation.** In this case, ARC-Easy is not going to tell us much about Path 1's plateau — but that itself is informative, and the four-paths writeup should flag that the E2B baseline is already near the ARC-Easy ceiling, which limits the headroom any other path has to demonstrate improvement on this benchmark. Consider adding ARC-Challenge to the shared rig.

### Outcome D — CoT doesn't beat direct on ARC-Easy (only meaningful if direct-arc is run)
A3-arc − direct-arc < 10 points. **Surprising and would reopen the Path 1 gate question.** CoT lifts GSM8K by 30 points but not ARC-Easy? Most likely cause: direct-arc is artificially high because 4-way multiple choice is easy enough that the 8-shot priming gives the answer away. Less likely: CoT is genuinely GSM8K-specific on E2B. Investigate before trusting. A plausible response: run direct-arc with `max_new_tokens=4` to tighten the format constraint.

## Deliverables

Script `path1_arc_easy.py` mirroring `path1_length_and_sc.py`:

1. Runs A3-arc, B3-arc, and optionally direct-arc. Resume-on-restart as with plan 2.
2. Per-cell JSONLs in `results/path_1_cot_tokens/plan3/`.
3. Summary `results_plan3.json` with the same shape as `results_plan2.json`.
4. Prints a short table:

   ```
   ARC-Easy (n=500)
   A3-arc      acc=0.xxx  ci=(..,..)  answer_hit=0.xx  mean_tokens=xx
   B3-arc      acc=0.xxx  ci=(..,..)  vote_deg=..%    mean_tokens=xx
   direct-arc  acc=0.xxx  ci=(..,..)  answer_hit=0.xx  [optional]
   ```

5. Prints the outcome interpretation (A / B / C / D) and any follow-up instruction.

## Report back

Paste the table and the outcome label. Do **not** start head-to-head integration or blog writeup until all three plan 3/4/5 results are reviewed together.