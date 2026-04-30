# Path 1 — experiment 9: what does C2 lose vs A3? (analysis-only, no new generations)

## Context

The McNemar paired stats from Experiment 5 showed:

| Comparison | only_C2 | only_A3 | both | neither |
|---|---|---|---|---|
| C2 vs A3 | 234 | **26** | 124 | 116 |

C2 wins by 234 problems, A3 wins by 26. The 234 are the headline. The **26** are interesting in the other direction: there are problems the 8-shot Wei et al. exemplars *do* help with, that the plain zero-shot prompt misses.

If the exemplars are pure noise, that count would be near zero. 26 of 500 (5.2%) is small but not noise. Something specific is going on, and it's worth knowing what — both for completeness on Path 1 and for hedge cases when prompt-engineering is reopened in later phases.

Experiment 9 answers:

> **What is special about the 26 problems where 8-shot Wei et al. CoT (A3) succeeds and zero-shot plain (C2) fails?**

## Scope — what this plan is and isn't

**In scope:**
- Pure analysis on existing JSONL files. No GPU, no new generations.
  - `results/path_1_cot_tokens/plan2/cells/A3_len512__0000_0500.jsonl`
  - `results/path_1_cot_tokens/plan5/cells/C2_zeroshot_plain__0000_0500.jsonl`
- For each of the 26 A3-only-correct problems: extract the question text, the gold answer, A3's completion, and C2's completion.
- Manual qualitative classification of why C2 failed and A3 succeeded:
  - **Format failure** — C2 produced an arithmetically correct answer but in a form the integer extractor missed (decimal, fraction, units)
  - **Reasoning failure** — C2's chain-of-thought went off-track at a specific step
  - **Premature termination** — C2 stopped generating before producing a final number
  - **Repetition loop** — C2 derailed into rep-loop
  - **Lucky A3** — A3's final answer is correct but its reasoning is also flawed (i.e., A3 got it right by accident)
- Quantitative summary: which category dominates the 26.
- Comparison to a control sample: 26 problems where both A3 and C2 are correct, sampled deterministically. Used to characterize what "normal" success looks like and ensure failure-category tags are calibrated.

**Explicitly out of scope:**
- No new GPU runs. No re-generation. No re-prompting.
- No statistical re-test of Experiment 5's headline (that stands).
- No re-classification of the 234 only_C2 problems — that's the success direction; we already know C2 wins more often.

## Method

### Step 1 — paired data extraction

Load both JSONL files keyed by `idx`. For idxs 0–499, identify the sets:

- `S_only_A3 = { idx : A3.correct == 1 ∧ C2.correct == 0 }` — should have |S| = 26
- `S_both = { idx : A3.correct == 1 ∧ C2.correct == 1 }` — should have |S| = 124

Sample a deterministic 26 from `S_both` for the control comparison (e.g., first 26 by sorted idx).

### Step 2 — automatic categorization

For each problem in `S_only_A3` and the control set, compute:

- `c2_has_number_answer`: does C2's completion contain a number? (any integer-valued substring)
- `c2_repetition_flag`: regex `(.{10,60})\1{2,}` on completion (same as Experiment 3)
- `c2_truncated`: did `n_gen_tokens` hit the 512 cap?
- `c2_extracted_pred`: the model's parsed prediction; compare to gold to detect "format failure" (extracted ≠ gold but a reasonable transformation matches)
- `a3_lucky_flag`: does A3's CoT chain contain an obviously wrong intermediate step but reach the right final answer? (heuristic — manual confirm on flagged cases)

### Step 3 — manual qualitative pass

For each of the 26 A3-only-correct problems, the analyst (the human or me) reads:

- The question text
- The gold answer + reasoning
- C2's completion (full)
- A3's completion (full)

And tags one of:
- `format_failure` — C2 reasoned correctly to the right number; the extractor missed it
- `reasoning_failure_arithmetic` — C2 made an arithmetic error mid-chain
- `reasoning_failure_setup` — C2 misunderstood the problem setup
- `reasoning_failure_step_skip` — C2 skipped a load-bearing step
- `truncated` — C2 hit the token cap before finalizing
- `repetition_loop` — C2 derailed
- `lucky_a3` — A3's reasoning is also flawed but its final answer matches gold by coincidence

The control sample (26 random `S_both`) is tagged with whatever C2 did successfully; this helps calibrate "normal."

### Step 4 — quantitative summary

Build a category histogram for both groups. The interesting comparison:

- If `format_failure` dominates → the harness extractor is the bottleneck on those 26; tighten the regex; this is a free win
- If `reasoning_failure_*` dominates → the exemplars are giving A3 a real reasoning scaffold for a specific problem class
- If `lucky_a3` dominates → the 26 isn't a real signal, just noise around the McNemar boundary

### Step 5 — qualitative summary

Pick 3 representative examples from the largest category. Quote the question text, A3's completion, C2's completion. This is the writeup-ready evidence.

## Budget

- Scripted analysis (steps 1, 2, 4): ~30 min
- Manual qualitative pass (step 3): ~2 hours for 52 problems × 5 min each
- Writeup (step 5): ~1 hour

**Total: ~4 hours of analyst time. Zero GPU.**

## Pre-registered interpretation

### Finding A — Format failure dominates (≥ 12 of 26)
**The integer extractor is suppressing C2 wins.** Tighten the C2 regex (e.g., handle decimal answers, fractions, comma-separated numbers); re-score Experiment 5 and Experiment 6 on the tighter extractor; the C2 advantage may grow further.

### Finding B — Reasoning failure dominates (≥ 12 of 26)
**The exemplars are doing real reasoning work for a specific problem class.** Worth one follow-up: identify the class (multi-step ratios? unit conversions? back-substitution?) and decide whether the four-paths comparison should include a per-class breakout.

### Finding C — Lucky A3 dominates (≥ 8 of 26)
**The 26 is statistical noise around the McNemar boundary.** Path 1's prompt-format finding stands cleanly; no follow-up needed.

### Finding D — Mixed / no dominant category
**The 26 is a heterogeneous mix of small effects.** Document the distribution; no follow-up. Flag in the four-paths comparison that a small fraction of GSM8K problems benefit from exemplar priming.

## Deliverables

Script `path1_c2_vs_a3_inspection.py`:

1. CPU-only, runs in seconds.
2. Reads both jsonl files, computes the four McNemar buckets, exports `S_only_A3` and the control `S_both` sample to `results/path_1_cot_tokens/plan9/inspection_set.json` with all relevant fields per problem.
3. Runs Step-2 automatic flags.
4. Prints a categorization template for the analyst to fill in.

Final outputs:
- `results/path_1_cot_tokens/plan9/inspection_set.json` (52 problems with full text + automatic flags)
- `results/path_1_cot_tokens/plan9/categorization.csv` (analyst-filled per-problem category)
- `results/path_1_cot_tokens/plan9/results_plan9.json` (summary histogram + finding label)

## Report back

The 52-problem categorization CSV and the finding label (A / B / C / D). If A or B fires, the implication for Experiment 6 / 7 / blog post should be noted directly in the report.

## What this closes

This is the only "free" experiment in the Path 1 closeout — no GPU, no phone hardware. Worth running before Experiment 6 because if Finding A fires (format-failure dominant), the extractor needs tightening before Experiment 6 measures anything else, otherwise the same suppression will hide further C2 wins.
