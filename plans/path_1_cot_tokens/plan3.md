# Path 1 — plan 3: repetition-loop isolation (analysis-only, no new generations)

## Context

Plan 1's failure-mode breakdown on n=100 IT-CoT showed 13 of 69 wrong answers collapsed into repetition loops — the model gets stuck emitting the same token pattern until it hits the generation cap. At n=100 that's a minor annotation. At plan 2's n=500 across 8 cells (including 4,500 individual sampled chains from axis B), it's a much bigger surface of data.

Plan 3 answers:

> **If repetition-loop failures were excluded from scoring, how much would Path 1's accuracy change — and does that change vary by cell?**

This is a **pure analysis task** on existing data. No new generations, no GPU time, no sweep. But it answers something plans 1–2 left unresolved: is the 30% GSM8K ceiling a **reasoning** ceiling, or a **reasoning plus generation-stability** ceiling? The two are different in the four-paths story:

- **A reasoning ceiling** is fundamental to the 2B weights. Path 2 (depth-recurrence after healing) or Path 3 (quantized E4B) can only exceed it by genuinely improving reasoning.
- **A reasoning-plus-stability ceiling** has a cheaper fix: if the 2B model knows the answer but its generator sometimes derails into repetition, Paths 2/3/4 could pick up "stability" points without needing more reasoning capacity. Path 4 (Mixture-of-Depths) specifically could help here — per-token compute routing might let the model escape repetition attractors.

If repetition loops account for, say, 5 of Path 1's missing 70 percentage points on GSM8K, that's 5 points the other paths can plausibly recover at low cost. If they account for 0–1 points, the other paths need to beat Path 1 on actual reasoning, which is harder.

## Scope — what this plan is and isn't

**In scope:**
- Re-analysis of **all existing JSONL files** from plans 1 and 2:
  - `it__cot__0000_0100.jsonl` and `base__cot__0000_0100.jsonl` from plan 1.
  - `A1_len128__0000_0500.jsonl` through `A4_len1024__0000_0500.jsonl` from plan 2 axis A.
  - `B1_k1__0000_0500.jsonl` through `B4_k10__0000_0500.jsonl` from plan 2 axis B (operating on individual chains inside the `chains` list, not only the voted result).
- Systematic classification of each completion/chain into one of four outcome buckets (the same ones the earlier SVG and D3 figures used): `correct`, `terminated_wrong`, `repetition_loop`, `truncated_no_answer`.
- Per-cell **adjusted accuracy** under two counterfactuals:
  - *Lenient:* excluding repetition-loop failures from the denominator (what's the accuracy on completions that actually *tried* to answer?).
  - *Upper-bound:* treating repetition-loop failures as if the model had reached the greedy A3 correct-rate on those same problems (what's the accuracy if we replaced rep-loops with the reference cell's outcomes?).
- Chain-level analysis for axis B: are the 4,500 sampled chains' repetition rates the same as their greedy equivalents, or does sampling make it worse?

**Explicitly out of scope:**
- No new generations.
- No retroactive "fixing" of scores that changes Path 1's headline number. The Path 1 representative for the head-to-head stays at A3 30.0%. Plan 3's outputs are supplementary evidence for the writeup, not replacements for plan 2's numbers.
- No repetition-detection method changes mid-analysis. Pin the detection regex up-front.

## Repetition-detection method

The earlier figures used `re.search(r'(.{10,60})\1{2,}', completion)` — "find a substring of 10–60 chars that repeats at least 3 times immediately after itself." That regex is the plan 3 canonical definition and will **not** be changed during this analysis.

Document its limitations explicitly in the writeup:

- **False positives:** legitimate repetition in genuine reasoning ("12 + 12 = 24. 24 - 12 = 12. 12 + 12 = 24." has a 3-char repeat but not a 10-char one; shouldn't trigger). Sample 30 flagged completions manually and report the false-positive rate.
- **False negatives:** non-verbatim repetitions ("I count 5 apples. I see 5 apples. There are 5 apples.") where the repeated idea has varying syntax. The regex misses these. Likely undercount by 10–20%.
- **Language-dependent:** the regex is tuned for English. Not an issue here (GSM8K is English-only) but worth noting.

If the manual sample of 30 shows >20% false positives, tighten the regex to `r'(.{15,80})\1{2,}'` and rerun. Otherwise, the canonical version stands.

## Cells to analyze

| Cell | Source | Unit of analysis | Notes |
|---|---|---|---|
| plan1.it_cot_n100 | `it__cot__0000_0100.jsonl` | completion | Reference for rep-loop rate at n=100, already reported |
| plan1.base_cot_n100 | `base__cot__0000_0100.jsonl` | completion | Reference for the base-vs-IT asymmetry finding |
| plan2.A1, A2, A3, A4 | `A*_len*__0000_0500.jsonl` | completion | Length sweep — does rep-loop rate scale with cap? |
| plan2.B1, B2, B3, B4 | `B*_k*__0000_0500.jsonl` | **chain** (k chains per problem) | Does sampling change the rep-loop rate? |
| plan2.B_voted | same, voted_pred | problem | What fraction of voted-wrong outcomes are "the vote converged on a rep-loop answer"? |

## Analysis steps

### Step 1 — Per-cell outcome table

For each cell, produce a table:

```
cell          n     correct   term_wrong   rep_loop   trunc_no_ans
A1_len128     500   149       xxx          xxx        xxx
A2_len256     500   150       xxx          xxx        xxx
A3_len512     500   150       xxx          xxx        xxx
A4_len1024    500   150       xxx          xxx        xxx
B1_k1         500   154       xxx          xxx        xxx       (voted)
B1_k1 chain   500   xxx       xxx          xxx        xxx       (individual chain)
...
```

For axis B, produce both a problem-level row (voted outcome) and a chain-level row (aggregated across all k × 500 chains).

### Step 2 — Adjusted accuracy under two counterfactuals

**Lenient adjusted accuracy** = `correct / (n - rep_loop)`. This treats rep-loops as "we can't tell what the model thinks" rather than "the model was wrong."

**Upper-bound adjusted accuracy** = `correct + (rep_loop × P(A3 correct on those same problems))`. For each rep-loop problem, look at whether A3 got that problem correct; use that as the imputed outcome. The resulting accuracy is "what if we could have swapped in A3's reasoning for the rep-loop problems?"

Both numbers are clearly labeled counterfactuals, not reported accuracies. Neither replaces the plan 2 headline numbers.

### Step 3 — Length-dependence test

Does `rep_loop` rate rise with `max_new_tokens`? Plot A1 → A4's rep-loop rate as a function of cap. If the rate rises meaningfully, the story is "longer generation invites more rep-loops" (a minor failure mode of the length sweep). If it's flat, rep-loops are a property of specific problems rather than of generation length.

### Step 4 — Sampling-dependence test

Compare per-chain rep-loop rate in axis B to A3's rep-loop rate:

- A3: greedy, rep-loop rate = ? (from step 1)
- B1-chain through B4-chain: sampled, per-chain rep-loop rate across all k × 500 chains.

If sampled chains have a *lower* rep-loop rate than greedy, then sampling is implicitly escaping repetition attractors — which would be a mechanism for why axis B showed *any* lift, even an insignificant one. If they have the *same* or *higher* rate, sampling doesn't help with this failure mode.

### Step 5 — Vote-on-repetition detection

For axis B: of the `n − voted_correct` problems where the vote outcome was wrong, how many had the majority of chains fall into rep-loop? Distinguish:

- **Unanimous rep-loop** (all k chains rep-looped): the problem is genuinely hard for the sampler and voting can't rescue it.
- **Rep-loop-plurality wrong** (majority rep-looped but minority got it right): voting actively hurt — the minority had the answer, but the majority of degenerate chains swamped it.
- **Rep-loop-unlucky** (minority rep-looped, majority terminated-wrong): voting would have been wrong anyway; rep-loops were not decisive.

If "rep-loop-plurality wrong" is >5% of voted-wrong problems, that's a concrete mechanism by which self-consistency is being dragged down by generation stability. Path 4 (Mixture-of-Depths) could plausibly address this; worth flagging explicitly in the writeup.

### Step 6 — Problem-level analysis

Which GSM8K problems (by idx) trigger rep-loops across multiple cells? If a handful of problems trigger rep-loops in 5+ cells, they are "sticky" and may share a structural feature (long chain of arithmetic, multi-step fractions, etc.). List the top 10 sticky problems and inspect their text. This is a qualitative finding for the blog post.

## Deliverables

Script `path1_rep_loop_analysis.py`:

1. No GPU. Runs on CPU. Should complete in under 2 minutes.
2. Loads all existing JSONL files from plans 1 and 2 by fixed paths.
3. Produces `results_plan5.json` containing:
   - Per-cell outcome counts (step 1).
   - Per-cell lenient and upper-bound adjusted accuracies (step 2).
   - Length-dependence and sampling-dependence summary tables (steps 3–4).
   - Vote-on-repetition breakdown for axis B (step 5).
   - Top-10 sticky problem IDs and their text (step 6).
4. Prints a human-readable summary.
5. Emits a CSV `path1_rep_loops_by_cell.csv` for direct consumption by the blog-post figure generator.

## Pre-registered interpretation

Before computing, commit to these thresholds:

### Finding A — Rep-loops are a small factor (<3 points of ceiling)
Lenient and upper-bound adjusted accuracies are within 3 points of the headline number across all cells. **Conclusion: Path 1's ceiling is genuinely a reasoning ceiling.** The "generation-stability" hypothesis is falsified; other paths must beat Path 1 on actual reasoning.

### Finding B — Rep-loops are a medium factor (3–8 points)
Adjusted accuracies open up by 3–8 points. **Conclusion: Path 1's ceiling is part reasoning, part stability.** The four-paths writeup should note this explicitly as a mechanism by which Paths 2–4 could plausibly beat Path 1 without needing more reasoning capacity. Especially relevant to framing Path 4 (Mixture-of-Depths), whose mechanism is compute-allocation-per-token — a plausible stability aid.

### Finding C — Rep-loops are a large factor (>8 points)
Adjusted accuracies open up by more than 8 points. **Conclusion: the plan 2 ceiling is significantly inflated by stability failures**, and the four-paths comparison has a confound: a path that reduces rep-loops while holding reasoning constant would look like a reasoning win. Reframe the four-paths scoring axis to control for this — consider adding a generation-stability metric alongside accuracy in the head-to-head rig.

### Finding D — Axis B chain-level rep-loop rate is >> A3's
Sampled chains are markedly more rep-loop-prone than greedy. **Supports plan 2's SC-flat finding by explaining the mechanism.** Worth a dedicated section in the blog post.

### Finding E — Axis B chain-level rep-loop rate is << A3's
Sampled chains are *less* rep-loop-prone than greedy. **Surprising and important.** Would mean sampling is implicitly escaping stability attractors. Even if self-consistency didn't lift on GSM8K for other reasons, this finding applies to Path 4 (Mixture-of-Depths), which may get a free stability benefit from its routing.

## Report back

The deliverable is the analysis script's printed summary plus the outcome label (A / B / C / D / E) and top-10 sticky-problem list. No follow-up experiments needed — this plan is self-contained by design.