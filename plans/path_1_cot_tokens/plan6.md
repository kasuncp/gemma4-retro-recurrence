# Path 1 — experiment 6: length and self-consistency on top of C2

## Context

Experiment 2 measured a flat plateau: eight cells across `max_new_tokens ∈ {128, 256, 512, 1024}` and `k ∈ {1, 3, 5, 10}` all landed inside a 2.6-point band at 29.6%–32.2% on GSM8K. Conclusion at the time: "more compute doesn't help."

Experiment 5 then showed the bottleneck was the prompt, not compute. Switching from 8-shot Wei et al. to a plain zero-shot prompt under the IT model's chat template (cell **C2**) lifts greedy accuracy to **71.6%** — at *less* compute per problem (276 vs 403 generated tokens, 69 vs 747 prompt tokens).

This invalidates Experiment 2's conclusion as a Path 1 ceiling claim. The plateau may have been a function of the prompt format, not the compute axis. Letting a model that already hits 71.6% greedy run multiple sampled chains, or run longer, may unlock another 5–10 points — or it may not.

Experiment 6 answers:

> **Does the length-and-self-consistency plateau hold on top of C2's prompt format, or does test-time compute scaling come back to life when the prompt is no longer the bottleneck?**

This is the experiment that decides Path 1's true ceiling for the four-paths head-to-head.

## Scope — what this plan is and isn't

**In scope:**
- IT model only (`google/gemma-4-E2B-it`), same pinned commit as Experiments 2 and 5.
- GSM8K, deterministic first `n = 500` test problems — same slice as Experiments 2 and 5.
- C2 prompt format from Experiment 5: plain question via `tokenizer.apply_chat_template`, no exemplars, `temperature = 0` for greedy cells.
- Two sweep axes, run independently:
  - **Axis A, generation length:** `max_new_tokens ∈ {128, 256, 512, 1024}` at greedy. The 512 cell is C2 from Experiment 5 — re-use, do not re-run.
  - **Axis B, self-consistency:** `k ∈ {1, 3, 5, 10}` chains at `temperature = 0.7`, `top_p = 0.95`, `max_new_tokens = 512`. Majority-vote on parsed integer answer.
- Wall-clock + analytical FLOPs per cell. Same accounting as Experiment 2.
- Pareto plot of accuracy vs FLOPs per problem on the C2 prompt baseline.

**Explicitly out of scope:**
- No length × k crossed grid. Independent one-dimensional sweeps, mirroring Experiment 2.
- No temperature sweep — fixed at 0.7 for SC cells.
- No `k > 10` and no `max_new_tokens > 1024`.
- No cross-benchmark (ARC-Easy / MATH / BBH) — that's Experiment 7.
- No on-device measurement — that's Experiment 8.
- No re-prompt-engineering. C2 is fixed; only the compute knobs vary.

## Environment

Same pinned deps as Experiments 2 and 5:

```
transformers==5.6.1
torch==2.4.1+cu124
datasets==4.8.4
accelerate
```

Model: `google/gemma-4-E2B-it` at the commit from `results/path_1_cot_tokens/plan1_preview_n100/commits.json`.

Hardware: one GPU with ≥ 16 GB VRAM, bfloat16. Single 3090 sufficient.

## Protocol

### Cells

| Cell | Prompt | Decode | `max_new_tokens` | k |
|---|---|---|---|---|
| **C2-A1**  | C2 plain | greedy | 128  | 1 |
| **C2-A2**  | C2 plain | greedy | 256  | 1 |
| **C2-A3** *(reference, re-used from Exp 5)* | C2 plain | greedy | 512 | 1 |
| **C2-A4**  | C2 plain | greedy | 1024 | 1 |
| **C2-B1**  | C2 plain | sampled (T=0.7, p=0.95) | 512 | 1 |
| **C2-B2**  | C2 plain | sampled | 512 | 3 |
| **C2-B3**  | C2 plain | sampled | 512 | 5 |
| **C2-B4**  | C2 plain | sampled | 512 | 10 |

The C2-A3 cell is the existing C2 result from Experiment 5 (`results/path_1_cot_tokens/plan5/cells/C2_zeroshot_plain__0000_0500.jsonl`). Re-use the file by path; do not re-run. **Sanity-check on read**: confirm idx coverage, accuracy = 71.6%, mean gen tokens ≈ 276.

### Shared across cells

- `n = 500` first GSM8K test problems (same deterministic slice as Experiments 2 and 5).
- Apply `tokenizer.apply_chat_template` with a single user turn = the question text, no system message.
- Answer extraction: `####`-first regex, last-integer-fallback. Note: hash-hit rate will be near zero (the zero-shot model does not emit `#### N` markers); the fallback regex will carry the load. Report both rates.
- Seed: `torch.manual_seed(0)` + `torch.cuda.manual_seed_all(0)` + `random.seed(0)` set once at script start.
- Resume-on-restart with append-JSONL pattern under `results/path_1_cot_tokens/plan6/cells/`.

## Budget

Mean gen tokens for C2-A3 was 276 — about 30% shorter than A3's 403. Per-cell compute should be slightly cheaper than Experiment 2's per-cell.

| Cell | Generations | Approx. wall time (3090, bf16) |
|---|---|---|
| C2-A1 (128 tok)  | 500     | ~30 min |
| C2-A2 (256 tok)  | 500     | ~40 min |
| C2-A4 (1024 tok) | 500     | ~80 min |
| C2-B1 (k=1)      | 500     | ~45 min |
| C2-B2 (k=3)      | 1,500   | ~120 min |
| C2-B3 (k=5)      | 2,500   | ~180 min |
| C2-B4 (k=10)     | 5,000   | ~360 min |

**Total: ~13 hours on a single 3090** (slightly above Experiment 2's 8 hours because more new cells; C2-A3 is already done). On a 4090 expect ~9 hours.

## Sanity checks

1. **C2-A3 reproduction.** Before scoring new cells, re-load the C2 jsonl from Experiment 5 and confirm: 500 idxs, 358 correct (71.6%), `accuracy_ci95 ≈ (0.675, 0.754)`. If this fails, the manifest has drifted.
2. **Chat template byte-equivalence.** Print the full tokenized prompt for problem 0 and confirm it byte-matches the C2 prompt from Experiment 5's first cell jsonl. If different, you're not actually re-using the C2 prompt format.
3. **Hash-hit rate floor.** Expect near-zero `hash_hit_rate` (zero-shot does not emit `####`). Confirm `fallback_hit_rate ≥ 0.95` on each new cell. If fallback drops below 0.85, the integer extractor is misfiring on prose answers; debug before trusting accuracy.
4. **Vote-degeneracy canary.** For axis B, fraction of problems where all k chains voted the same answer. Expected: 0.50–0.75 at T=0.7. If > 0.95, sampling is effectively deterministic; bug or temperature too low.
5. **C2-B1 reproducibility given seed.** Run the first 10 problems twice (restart between). Generations must match byte-for-byte.

## Pre-registered interpretation

Decide thresholds **before** looking at the new numbers.

### Outcome A — Plateau confirmed on C2
All cells within ±3 pp of C2-A3's 71.6%. **Path 1's ceiling is locked at 71.6%** for the four-paths head-to-head. The Experiment 2 plateau finding now generalizes: more inference compute does not buy more accuracy on this model on GSM8K, *regardless* of the prompt format. Path 1's representative cell stays C2 at 512 tokens greedy.

### Outcome B — Length helps further on C2
C2-A4 (1024 tok) ≥ C2-A3 + 5 pp with non-overlapping CIs. **Length was suppressed by the bad prompt; on C2 it works.** Path 1's representative cell becomes C2-A4. Investigate whether `max_new_tokens = 2048` is worth one follow-up cell.

### Outcome C — Self-consistency helps on C2
C2-B3 or C2-B4 ≥ C2-A3 + 5 pp with non-overlapping CIs, and the lift is greater than length's. **SC was suppressed by the bad prompt; on C2 it's the lever.** Path 1's representative cell becomes the best B cell. The compute cost is k×, but if the lift is ≥ 5 pp, it earns its place.

### Outcome D — Both length and SC help (compounding)
Both A4 and B3/B4 individually beat C2-A3 by ≥ 5 pp, and the magnitudes are roughly additive. Worth one follow-up cell that combines: SC k=5 at `max_new_tokens = 1024`. The combined cell is then Path 1's representative.

### Outcome E — SC hurts (the surprise direction)
C2-B3 or C2-B4 < C2-A3 by ≥ 3 pp. Plausible mechanism: zero-shot prompts on IT models produce diverse but lower-quality individual chains; voting across diverse-but-wrong chains converges on a wrong answer. If this fires, the fix is to bias sampling: lower T (0.3, 0.5) or use top-k decoding instead of top-p. One follow-up cell.

## Deliverables

Script `path1_c2_length_and_sc.py` (mirroring `path1_length_and_sc.py`):

1. Runs both axes end-to-end in bf16 on one GPU, resumable.
2. Re-uses C2-A3 jsonl from Experiment 5 by path (do not re-run, do not re-write).
3. Per-cell JSONLs in `results/path_1_cot_tokens/plan6/cells/`.
4. `results_plan6.json` with the same shape as `results_plan2.json` but C2-prefixed cell names.
5. `path1_c2_pareto.csv` for direct consumption by the four-paths comparison rig.
6. Prints both sweep tables and the outcome label (A / B / C / D / E).

## Report back

Paste the two sweep tables, the outcome label, and the new Path 1 representative cell. Update the blog post's conclusion section with the new ceiling number and the follow-up status.

## What this closes

If Outcome A or B fires, Path 1 is closed for the four-paths head-to-head — pending Experiment 7 (cross-benchmark on harder reasoning) and Experiment 8 (phone-class measurement). If Outcome C, D, or E fires, the four-paths comparison's Path 1 number changes, and the blog post's conclusion table needs to be updated.
