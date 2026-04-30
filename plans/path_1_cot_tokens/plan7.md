# Path 1 — experiment 7: cross-benchmark validity on harder reasoning

## Context

Experiment 4 ran an ARC-Easy cross-benchmark check using A3-style 8-shot CoT and found two things: (a) the plateau pattern survived (k=5 SC at 85.0% beats greedy CoT at 82.8% by only 2.2 pp, inside the noise band), and (b) ARC-Easy was *saturated* — direct prompting matched CoT (83.2% vs 82.8%), leaving no room for any inference-time strategy to differentiate.

Experiment 5 then revealed C2 (zero-shot plain via chat template) beats A3 by 41.6 pp on GSM8K. The cross-benchmark question reopens:

> **Does C2's prompt-format advantage and the test-time-compute plateau hold on harder reasoning benchmarks where the model is *not* already saturated?**

If yes, Path 1's representative cell ports cleanly to the four-paths head-to-head across all benchmarks. If no, the head-to-head needs a per-benchmark Path 1 cell — and that itself is an interesting finding about how much the prompt-format effect generalizes.

## Scope — what this plan is and isn't

**In scope:**
- IT model only (`google/gemma-4-E2B-it`), same pinned commit as Experiments 2, 5, 6.
- Three harder reasoning benchmarks:
  - **ARC-Challenge** (n = 500, deterministic head of test split). Multiple choice, harder than ARC-Easy.
  - **MATH** (n = 200, deterministic head of test split). Free-form numeric / expression answers, much harder than GSM8K.
  - **BBH-lite** — a curated 500-problem subset of Big-Bench Hard tasks with mixed answer formats (multiple choice + free-form). Pin task subset and IDs in manifest.
- Three cells per benchmark, mirroring the structure of Experiment 4:
  - **C2-greedy** (zero-shot plain, chat template, max_new_tokens = 512, greedy)
  - **A3-style** (8-shot CoT in the format Experiment 2 used) — for paired comparison, same problems
  - **Direct** (8-shot direct, max_new_tokens = 16) — for saturation check
- Wilson 95% CIs on every cell. Paired McNemar between C2 vs A3-style for the prompt-effect generalization test.

**Explicitly out of scope:**
- No length × k crossed sweep on these benchmarks. Single C2-greedy cell; if Experiment 6 showed length or SC helps, *also* run C2 with that winner — otherwise just C2-greedy.
- No new prompt-engineering experiments. The three cells are the same three protocols Experiment 4 used, with C2 swapped in.
- No additional benchmarks beyond the three above. MMLU is a separate plan if needed.
- No on-device measurement. Experiment 8.

## Environment

Same pinned deps as Experiments 2, 5, 6.

Model: `google/gemma-4-E2B-it`, same commit.

Hardware: single 3090 with ≥ 16 GB VRAM, bfloat16.

## Protocol

### Per-benchmark setup

For each benchmark, the three cells share the same n problems (deterministic head of the test split, IDs pinned in manifest), the same model, the same answer extractor.

**Answer extractors:**
- ARC-Challenge: regex on `The answer is ([A-D])` with last-letter fallback (same as Experiment 4 ARC-Easy).
- MATH: regex on `\boxed{(.*?)}` with last-number-fallback. Match against gold answer with `sympy.simplify` for expression equivalence (the canonical MATH eval). Borrow the standard MATH harness from `lm-evaluation-harness` for parity.
- BBH-lite: per-task. For multiple choice tasks, regex on the answer letter. For free-form, exact match after stripping whitespace. Pin per-task extractor logic in the script.

**Prompt templates:**
- C2-greedy: `tokenizer.apply_chat_template` with single user turn = the problem text.
- A3-style: 8-shot CoT in the same Wei et al.-derived format Experiment 2 used. Per-benchmark exemplars:
  - ARC-Challenge: borrow the 8 ARC-Easy exemplars from Experiment 4 (same domain).
  - MATH: 8-shot CoT with handcrafted rationales, drawn from the MATH train split. Pin IDs.
  - BBH-lite: per-task 3-shot CoT (BBH standard).
- Direct: 8-shot direct (no rationale), per-benchmark exemplars matching A3-style.

### Cell specifications

| Benchmark | Cells | n per cell |
|---|---|---|
| ARC-Challenge | C2-greedy, A3-style, Direct | 500 |
| MATH | C2-greedy, A3-style, Direct | 200 |
| BBH-lite | C2-greedy, A3-style, Direct | 500 |

**Total: 9 cells, ~3,600 generations.**

If Experiment 6 produced a non-greedy winner (Outcome B/C/D), add a fourth cell per benchmark using that winner instead of greedy. Then total: 12 cells, ~5,000–10,000 generations depending on k.

## Budget

| Cell type | Approx wall time per 500 problems |
|---|---|
| C2-greedy (mean ~280 tokens) | ~45 min |
| A3-style (mean ~400 tokens, longer prompt) | ~75 min |
| Direct (16 tokens) | ~5 min |

Per benchmark: ~125 min. Three benchmarks: ~6.5 hours.

**Total: ~7 hours on a single 3090** for the base 9 cells. Add ~3 hours per non-greedy cell from the Experiment 6 winner (if applicable).

## Sanity checks

1. **Gold answer parser sanity.** For each benchmark, confirm the gold-answer extractor matches the harness reference on all loaded problems. Drop or remap any malformed gold answers; flag count in manifest.
2. **Answer-hit rate floor.** For C2-greedy on each benchmark, `fallback_hit_rate ≥ 0.85` is required; if below, the model isn't following the implicit format and accuracy is unreliable.
3. **Reproduction of Experiment 4.** Re-load Experiment 4's `A3_arc__0000_0500.jsonl`. The A3-style ARC-Easy cell from Experiment 4 should still report 82.8% if re-scored with this plan's extractor. If not, parser drift; debug.
4. **MATH harness parity.** Run the standard `lm-evaluation-harness` MATH-500 eval on a single problem and confirm same answer extraction. If different, document the deviation in the manifest.
5. **BBH-lite task selection.** Pin the 500-problem subset by task name + index. Document in manifest. If borrowing from a published BBH-lite definition, cite the source.

## Pre-registered interpretation

Per-benchmark, decide thresholds before looking at numbers.

### Outcome A — C2 advantage holds on harder benchmarks
On all three benchmarks, C2-greedy ≥ A3-style + 10 pp with non-overlapping CIs. **The Wei-et-al.-suppression effect generalizes — it's about IT models meeting old base-model exemplars, not about GSM8K specifically.** Path 1's representative cell ports cleanly: C2 (or the Experiment 6 winner) is the cell for the four-paths head-to-head, on every benchmark.

### Outcome B — C2 advantage is GSM8K-specific
C2-greedy ≈ A3-style on at least one harder benchmark (within ±3 pp). **The prompt-format effect is benchmark-specific.** The four-paths head-to-head needs a per-benchmark Path 1 cell. Worth examining what the harder benchmarks have in common that GSM8K lacks (or vice versa) — likely something about answer-format complexity or the role of arithmetic chain-of-thought.

### Outcome C — C2 advantage *grows* on harder benchmarks
C2-greedy ≥ A3-style + 20 pp on at least one benchmark. **The exemplar-suppression effect compounds with task difficulty** — possibly because harder problems need more "freedom" from the exemplar pattern to reason cleanly. Strengthens the Experiment 5 finding and adds urgency to Path 1's prompt-format claim.

### Outcome D — Direct ≈ CoT on a benchmark
On any benchmark where direct ≥ CoT-greedy − 5 pp, that benchmark is saturated for E2B-it and won't differentiate Path 1 from any other path. Flag in the manifest. Worth knowing for the four-paths head-to-head: those benchmarks are de-emphasized.

### Outcome E — C2 underperforms A3-style somewhere
C2-greedy + 5 pp < A3-style on at least one benchmark. **The 8-shot exemplars are doing real work on this benchmark.** Specific to that benchmark, A3-style is the right Path 1 cell. Document and flag — this is the most interesting outcome scientifically.

## Deliverables

Script `path1_harder_benchmarks.py`:

1. Runs all 9 cells (or 12 if Experiment 6 winner is non-greedy) end-to-end on one GPU, resumable.
2. Per-cell JSONLs in `results/path_1_cot_tokens/plan7/cells/`.
3. `results_plan7.json` with per-benchmark accuracy, CI, paired McNemar against A3-style and Direct, mean tokens, mean FLOPs.
4. `path1_cross_benchmark.csv` with one row per cell (cell, benchmark, prompt, decode, accuracy, CI, n).
5. Prints per-benchmark table and outcome label.

## Report back

Paste the per-benchmark table and the outcome label per benchmark. If any outcome differs across benchmarks, document which.

## What this closes

If Outcome A fires across all three benchmarks, Path 1 is benchmark-portable and the four-paths head-to-head can use C2 (or Experiment 6 winner) as the single cell. If Outcome B or E fires, the head-to-head's Path 1 cell is per-benchmark. Either way, this is the last cross-benchmark question for Path 1; after this, only on-device measurement (Experiment 8) remains.
