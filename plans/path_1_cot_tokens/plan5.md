# Path 1 — plan 5: 0-shot vs 8-shot CoT — is the ceiling artificial?

## Context

Plans 1 and 2 fixed the prompt format at **8-shot Wei-et-al exemplars**, and within that format, plateau: GSM8K accuracy sits at 30% regardless of generation length or self-consistency. That's a clean finding — but it has one uncomfortable asymmetry. Every cell we've measured uses the same 8 exemplars. We've probed every axis of inference compute *within* the 8-shot format. We haven't probed the format itself.

Plan 5 asks:

> **Is 30% a reasoning ceiling on Gemma 4 E2B-it, or an 8-shot Wei-et-al ceiling?**

If 0-shot CoT lifts accuracy materially above 30%, then the plateau in plan 2 is a prompt-format artifact and plan 2's "plateau" story needs qualification. If 0-shot matches or trails 8-shot, the ceiling is genuine and Path 1's story is strengthened.

### Why this is a plausible thing to test

Two priors push in opposite directions:

- **Pro:** Instruction-tuned models often outperform on 0-shot reasoning because their training includes explicit step-by-step demonstrations. 8-shot exemplars can anchor the model toward the exemplars' style rather than the specific question, and the Wei et al. exemplars are from 2022 — before IT post-training ubiquity. For IT models, Gemini's own reports and Llama-3 evals have shown 0-shot sometimes matching or beating 8-shot on GSM8K.
- **Con:** The community-standard best-effort GSM8K number for 2B-class models uses 8-shot CoT. If 0-shot were reliably better, it would be the standard.

Neither prior is strong. The question is empirical, and we can answer it cheaply.

## Scope — what this plan is and isn't

**In scope:**
- IT model only (`google/gemma-4-E2B-it`, same pinned commit as plan 2).
- GSM8K, first `n = 500` test problems — same deterministic slice as plans 1 and 2.
- **Two new cells** at greedy decode, 512-token cap, both scored against the same 500 problems as A3:
  - `C1_zeroshot_simple`: 0-shot + `"Let's think step by step."` prefix.
  - `C2_zeroshot_plain`: 0-shot, no prefix, just the question.
- A **paired comparison** against A3 from plan 2 (re-use A3's completions, do not re-run).

**Explicitly out of scope:**
- No other prompt variants (chain-of-verification, "let's verify step by step", self-refine, reflection, etc.). Those are a different leg — plan 1 explicitly scoped them out.
- No length or self-consistency sweep on these new cells. Plan 2's GSM8K-length-flatness finding applies: if the mean of C1/C2 differs from A3, the question is the mean, not the shape.
- No ARC-Easy in this plan. Plan 4 handles cross-benchmark validity.
- Not a full re-run of plan 1. We're probing one specific confound in plan 2's setup.

## Why two 0-shot cells instead of one

The two 0-shot conditions isolate different things:

- **C1 (`"Let's think step by step."`)** is the canonical 0-shot CoT prompt from the literature. Its presence tests whether *CoT-style prompting alone*, without exemplars, is enough to unlock the model's reasoning.
- **C2 (plain 0-shot, no prefix)** is the baseline for C1. It tests whether the IT model reasons step-by-step spontaneously when given a math word problem, without any prompt engineering.

If C2 scores near C1, the IT model is doing CoT on its own and the "Let's think step by step" prefix adds nothing on top. If C2 is meaningfully worse than C1 (which is what the original Kojima et al. paper found for the base PaLM model), the prefix is doing real work. Both numbers are informative; neither alone is sufficient.

If compute is tight, drop C2 and keep C1. C1 is the cell that decides the plan's main question.

## Protocol

### Shared across both cells

- Model: `google/gemma-4-E2B-it`, pinned commit from plan 2.
- Apply `tokenizer.apply_chat_template` with a single user turn (the question, with or without the prefix). This matters: the IT model expects the chat template; without it, the 0-shot cells would underperform for reasons unrelated to the prompt content.
- Greedy decode, `max_new_tokens = 512`, `do_sample = False`, `temperature = 0`.
- Answer extraction: same `####`-first, last-integer-fallback as plan 1 — but note that 0-shot completions may not include `####`. Record and report the `hash_hit_rate` per cell separately.
- Same seed (`torch.manual_seed(0)` at script start).

### Cell specifications

| Cell | User message |
|---|---|
| C1_zeroshot_simple | `"{question}\n\nLet's think step by step."` |
| C2_zeroshot_plain | `"{question}"` |

Where `{question}` is the raw GSM8K problem text.

### Reference comparison — A3 from plan 2

Do **not** re-run A3. Re-use `A3_len512__0000_0500.jsonl` from plan 2's checkpoint dir. All three cells (A3, C1, C2) operate on the same 500 problems, the same model, and the same greedy decode settings — the only variable is the prompt format. This is the cleanest possible paired comparison.

## Budget

| Cell | Generations | Approx. time |
|---|---|---|
| C1_zeroshot_simple | 500 | ~45 min on 3090 |
| C2_zeroshot_plain | 500 | ~45 min on 3090 |

**Total: ~1.5 hr for both cells.** Small.

## Sanity checks

1. **Chat-template verification.** Print the full tokenized prompt for problem 0 in both cells. Confirm it includes the IT model's user/assistant turn markers (e.g., `<start_of_turn>user` / `<start_of_turn>model`). If it doesn't, you're not actually doing 0-shot IT — you're doing raw-text 0-shot, which will tank accuracy for reasons unrelated to the plan question.
2. **Hash-hit rate comparison.** 0-shot GSM8K often produces answers in prose form ("The answer is 72.") rather than `#### 72`. A lower `hash_hit_rate` for C1/C2 vs A3 is *expected* and does not invalidate results — as long as the fallback integer extractor is hitting. Report both numbers. If `hash_hit_rate < 0.3` for either cell but `fallback_hit_rate > 0.8`, the fallback is carrying the load and the accuracy is trustworthy.
3. **Sample completion inspection.** Before running full 500, generate on the first 10 problems and eyeball the completions. Look for: does the model reason step-by-step in C1? Does it answer directly in C2, or does it reason anyway? The qualitative pattern is often informative before the numbers come in.
4. **Seed determinism.** Greedy + same model + same prompt should be bit-identical across runs. Restart the script mid-run and confirm the first 10 completions match byte-for-byte.

## Pre-registered interpretation

Define before looking at numbers.

### Outcome A — 8-shot is optimal
C1 and C2 both land within ±3 points of A3's 30.0%. **Confirmed: the 30% ceiling is a reasoning ceiling, not a prompt-format artifact.** Plan 2's plateau story is strengthened; the Path 1 writeup can state "we tested 0-shot (with and without CoT prefix) and 8-shot CoT and all three land at the same ceiling, so we believe ~30% is the model's ability on GSM8K at greedy inference."

### Outcome B — 0-shot beats 8-shot meaningfully
C1 beats A3 by ≥ 5 points with non-overlapping CIs. **The plan 2 ceiling is partly artificial.** The Path 1 representative for the head-to-head switches from A3 to C1. Plan 2's generation-length and self-consistency sweeps should be rerun on C1's prompt format, because the flatness at 30% might not be flatness at C1's new baseline. Cost of follow-up: ~8 hours (a plan-2 re-run on C1's prompt). Budget this in before declaring the new ceiling.

### Outcome C — 0-shot underperforms substantially
C1 trails A3 by ≥ 5 points. **Expected direction, confirms 8-shot is doing real work.** The Path 1 story is unchanged; plan 5 is a confirming null. Worth noting in the writeup for completeness.

### Outcome D — 0-shot plain (C2) is much worse than 0-shot simple (C1), but C1 matches A3
Suggests the `"Let's think step by step."` prefix is essentially a minimal-cost replacement for 8-shot exemplars. Interesting for the writeup but doesn't change Path 1's representative cell.

### Outcome E — 0-shot plain (C2) ≈ 0-shot simple (C1) ≈ A3
Tells you the IT model is reasoning step-by-step on its own, with or without scaffolding. Doesn't change Path 1's representative (all three produce the same accuracy so pick the cheapest — which is C2, the plain prompt with no prefix and no exemplars). This would actually be a **compute-efficiency win** for Path 1: if the plain question is as good as 8 exemplars, you save the exemplar tokens on every problem, which is a real speedup for deployment.

## Deliverables

Script `path1_zero_shot.py`:

1. Two cells, resume-on-restart.
2. Per-cell JSONLs in `results/path_1_cot_tokens/plan4/`.
3. Loads A3 from plan 2's results dir by path (pinned to `results/path_1_cot_tokens/round2_partial/A3_len512__0000_0500.jsonl` or wherever it lives).
4. Emits `results_plan4.json` containing C1, C2, and the paired A3 numbers.
5. Prints:

   ```
   Cell                     acc       CI               hash   fallback   mean_tok
   A3_len512 (reference)    0.xxx     (..,..)          0.81   0.99       403
   C1_zeroshot_simple       0.xxx     (..,..)          0.xx   0.xx       xxx
   C2_zeroshot_plain        0.xxx     (..,..)          0.xx   0.xx       xxx

   Paired comparisons vs A3 (McNemar):
     C1 vs A3: only_C1=xx, only_A3=xx, p=0.xxx
     C2 vs A3: only_C2=xx, only_A3=xx, p=0.xxx
     C1 vs C2: only_C1=xx, only_C2=xx, p=0.xxx
   ```

6. Prints outcome label (A / B / C / D / E) and any follow-up instruction.

## Report back

Paste the table and outcome label. The follow-up, if any, will depend on which outcome fires.