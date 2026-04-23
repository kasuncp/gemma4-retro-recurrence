# Path 1 — plan 2: length sweep and self-consistency on `it:cot`

## Context

Plan 1 answered the gate question:

> Does chain-of-thought in-context beat direct answering on GSM8K, at 8-shot matched prompting?

**Verdict: GREEN.** At n=100, IT-CoT scored 31% vs. IT-direct 1%, paired 30-0 discordance, McNemar p = 1.9 × 10⁻⁹, CIs non-overlapping. Path 1 is alive on E2B.

Plan 2 answers the next question, which is what Path 1 is actually for in the four-paths head-to-head:

> **Given that CoT helps, how much more accuracy can we buy by spending more inference-time compute on the same prompt — and what's the shape of the compute-vs-accuracy curve?**

Two mechanisms to sweep:
1. **Generation length.** Does letting the model reason longer than 512 tokens help? Is 512 already past saturation? Where is the knee?
2. **Self-consistency.** Sample k chains at temperature > 0 and majority-vote the parsed answers. Does voting push accuracy beyond the greedy ceiling?

These are the two levers Path 1 has left. Plan 1 already ran deliberation/reflection out of scope, and plan 2 stays there — `Let's think step by step` and similar prompts are not in this leg's measurement rig.

Plan 2's output is a Pareto frontier of accuracy vs. compute on the IT model, which is the data point Path 1 contributes to the eventual four-paths head-to-head.

## Scope — what this plan is and isn't

**In scope:**
- IT model only (`google/gemma-4-E2B-it`). The base model is descoped after plan 1 — it showed a real but smaller CoT lift (22% vs. 9%, McNemar p = 0.015) and is not the deployment target for the four-paths investigation.
- GSM8K, first n=500 problems of the test split. Deterministic slice, same 8 Wei-et-al exemplars as plan 1.
- Two sweep axes, run independently (not crossed):
  - **Axis A, generation length:** `max_new_tokens ∈ {128, 256, 512, 1024}`, greedy decode.
  - **Axis B, self-consistency:** `k ∈ {1, 3, 5, 10}` chains at `max_new_tokens = 512`, `temperature = 0.7`, `top_p = 0.95`, majority-vote the parsed answers.
- FLOPs/token and wall-clock tokens/sec recorded per cell. Peak VRAM per cell.
- A single Pareto plot at the end: accuracy on y-axis, compute (analytical FLOPs per problem) on x-axis.

**Explicitly out of scope (do NOT add these):**
- No temperature sweep. One temperature (0.7) for all self-consistency cells. Tuning T is a second-order optimization not worth plan 2's budget.
- No `k > 10`. Beyond 10 samples the diminishing-returns curve is well-documented and compute is better spent elsewhere.
- No length × k crossed grid. The two sweeps are independent one-dimensional probes. A crossed grid is 4 × 4 = 16 cells at up to 10k generations per cell; plan 2 is not that.
- No deliberation / reflection / "verify your answer" prompts. Plan 1 gated that out; reintroducing it would muddle the comparison.
- No ARC-Easy in this plan. ARC-Easy on IT-CoT is essentially baseline-capability-plus-noise (plan 1 did not need it and the shared rig will pick it up once plan 2 finishes). If plan 2 produces a clear length/k winner, ARC-Easy gets run on that single winning configuration as a handoff to the shared rig, not as an axis here.
- No on-device measurement. Path 1's on-device promotion, if any, happens after the four-paths head-to-head, not inside this plan.
- No training. No adapters. No model surgery.

If the script grows past ~350 lines or needs more than one file, stop and reconsider.

## Environment

Same pinned deps as plan 1:

```
transformers==5.6.1
torch==2.4.1+cu124
datasets==4.8.4
accelerate
```

Model: `google/gemma-4-E2B-it` at commit `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` (pinned from plan 1's `commits.json`).

Hardware: one GPU with ≥16 GB VRAM, bfloat16.

Expected wall time, single GPU: see budget table below. Plan for ~8 hours total across the two sweeps.

## Protocol

### Shared across both axes
- `n = 500` first GSM8K test problems (same deterministic slice as plan 1's manifest).
- 8-shot CoT with the Wei et al. exemplars (same `exemplar_hash = a33e6d90c6844317` from plan 1 — must match byte-for-byte; unit-test the prompt-builder against plan 1's).
- Seed: `torch.manual_seed(0)` + `torch.cuda.manual_seed_all(0)` + `random.seed(0)` set once at script start. Required for sampling determinism.
- Answer extraction: `####`-first, last-integer-fallback, same as plan 1.
- Record `hash_hit_rate` per cell. Any cell with `hash < 0.85` needs a note in the writeup.

### Axis A — generation length sweep

| Cell | `max_new_tokens` | Decode | k |
|------|------------------|--------|---|
| A1 | 128 | greedy | 1 |
| A2 | 256 | greedy | 1 |
| A3 | 512 | greedy | 1 |
| A4 | 1024 | greedy | 1 |

A3 at 512 tokens is plan 1's IT-CoT cell. Its `results_gate.json` numbers are the ground truth; A3 must reproduce them token-for-token on the shared 100 problems (sanity check 2 below). If it doesn't, the harness has drifted and the sweep is invalid.

### Axis B — self-consistency sweep

| Cell | k | Decode | `max_new_tokens` |
|------|---|--------|------------------|
| B1 | 1 | sampled (T=0.7, p=0.95) | 512 |
| B2 | 3 | sampled (T=0.7, p=0.95) | 512 |
| B3 | 5 | sampled (T=0.7, p=0.95) | 512 |
| B4 | 10 | sampled (T=0.7, p=0.95) | 512 |

B1 is *not* the same as A3 — A3 is greedy, B1 is a single temperature-0.7 sample. Both are useful: A3 is the deterministic baseline, B1 is the "single sample at the sampling temperature" reference so B3/B5/B10's vote lift is isolated from the greedy→sampled shift.

Majority vote: plurality of parsed-integer answers across the k chains. Ties broken by first-chain-wins (deterministic given seed).

## Budget table

Assuming IT-CoT at 512 tokens takes ~15 seconds per problem at bf16 on one 3090 (from plan 1 runtime):

| Axis | Cells | Generations per cell | Total generations | Wall time est. |
|------|-------|----------------------|-------------------|----------------|
| A | 4 | 500 | 2,000 | ~4 hr (A4 at 1024 tokens is 2× A3) |
| B | 4 | 500 × k | 500 + 1500 + 2500 + 5000 = 9,500 | ~4 hr (most cells are 512-token sampled) |

**Total: ~11,500 generations, ~8 hr wall time on one 3090.** Roughly 3× plan 1's compute. If the budget is tight, drop B4 (k=10) first — the diminishing-returns slope between k=5 and k=10 is small, and B3 already tells us if self-consistency is working.

## Implementation steps

### Step 1 — Script structure

Single file `path1_length_and_sc.py` at repo root, mirroring `path1_cot_gate.py`. Top-level sections:

1. Pinned config (model id + commit, dataset slice, exemplars, seeds).
2. `load_gsm8k_problems()` and `build_prompt()` — imported verbatim from plan 1's script to eliminate drift.
3. `run_axis_a()` — iterates lengths, greedy decode, writes per-cell JSONL.
4. `run_axis_b()` — iterates k values, sampled decode, runs all k chains per problem in a single batched call where possible, writes per-cell JSONL with `chains` field.
5. `score_axis_b()` — post-hoc majority vote per problem per k.
6. `compute_and_record_cost()` — analytical FLOPs per token + wall-clock.
7. `write_summary()` — emits `results_plan2.json`.

### Step 2 — Resume-on-restart

Plan 1's script didn't have robust resume. Plan 2 *must* — 8 hours on spot RunPod is enough time to get preempted. Use the same append-JSONL pattern as the round-5 checkpoint dir from the depth-recurrence project:

- One JSONL per cell, in `results/path_1_cot_tokens/round2_partial/`.
- Each line: `{"idx": int, "completion": str, "pred": int|None, "correct": bool, "hash_hit": bool, "n_gen_tokens": int}`.
  For axis B: `{"idx": int, "chains": [{"completion": ..., "pred": ...}, ...], "voted_pred": int|None, "correct": bool, "k": int}`.
- On start, read existing JSONL, skip already-completed idxs.
- Append + fsync per problem.
- Manifest check: refuse to resume if the current run's manifest differs from the existing manifest in the checkpoint dir.

### Step 3 — Sampling implementation

```python
@torch.no_grad()
def sample_k_chains(tok, mdl, prompt, k, max_new, T=0.7, top_p=0.95):
    ids = tok(prompt, return_tensors="pt").to("cuda")
    input_len = ids.input_ids.shape[1]
    # Batch the k chains in a single generate call with num_return_sequences=k
    gen = mdl.generate(
        **ids,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=T,
        top_p=top_p,
        num_return_sequences=k,
        pad_token_id=tok.eos_token_id,
    )
    return [tok.decode(gen[i, input_len:], skip_special_tokens=True) for i in range(k)]
```

`num_return_sequences=k` is the right primitive — HF's `generate` will batch the k chains efficiently under a single forward pass-and-cache. Do *not* run k separate `generate` calls in a loop; that's 2-3× slower at k=10.

### Step 4 — FLOPs/token accounting

Analytical, not measured. Use the standard `6 × N_params × tokens` approximation for decoder-only inference, scaled by active params (2.3B for E2B), plus cache attention overhead:

```python
N_ACTIVE = 2.3e9          # Gemma 4 E2B active parameters
HEAD_DIM, N_HEADS, N_LAYERS = 256, 8, 35   # confirm from model.config at runtime
def flops_per_token(prompt_len, completion_len):
    # Feed-forward + projection: 2 * N_active * (prompt_len + completion_len)
    ff = 2 * N_ACTIVE * (prompt_len + completion_len)
    # Attention over growing context (quadratic in prompt+completion):
    # each token attends over ~(prompt_len + pos) tokens for N_LAYERS, N_HEADS, HEAD_DIM
    # This is an approximation — exact if kernels are attention-dominant, which at 2B is not the case.
    attn = N_LAYERS * N_HEADS * HEAD_DIM * (prompt_len * completion_len + completion_len**2 / 2)
    return ff + 4 * attn
```

Per-problem cost for axis A is `flops_per_token(prompt, completion_len)`. For axis B, it's `k × flops_per_token(prompt, completion_len)` — self-consistency pays k× the inference cost for one answer.

Record actual `n_gen_tokens` per chain (not the cap) and compute per-problem FLOPs from observed lengths.

### Step 5 — Summary output

`results/path_1_cot_tokens/results_plan2.json`:

```json
{
  "config": { /* same shape as plan 1's results_gate.json */ },
  "axis_a": {
    "A1_len128":  { "accuracy": 0.xxx, "ci95": [lo, hi], "hash_hit_rate": ..., "mean_flops_per_problem": ..., "mean_gen_tokens": ... },
    "A2_len256":  { ... },
    "A3_len512":  { ... },
    "A4_len1024": { ... }
  },
  "axis_b": {
    "B1_k1":  { "accuracy": ..., "ci95": [...], "mean_flops_per_problem": ..., ... },
    "B2_k3":  { ... },
    "B3_k5":  { ... },
    "B4_k10": { ... }
  },
  "sanity_checks": {
    "a3_reproduces_plan1": { "matches": int, "total": 100 },
    "b_majority_vote_degenerate_check": { /* per-cell: fraction of problems where all k chains voted the same */ }
  },
  "verdict": { "label": "...", "reason": "..." }
}
```

## Sanity checks

Before trusting numbers:

1. **Exemplar hash matches plan 1.** Compute the hash of the prompt-builder output on a fixed test question. Must equal `a33e6d90c6844317`. If not, the prompt is drifting and the 30-point gate result no longer applies as the reference baseline.
2. **A3 reproduces plan 1's IT-CoT on the shared 100 problems.** A3 at n=500 includes the first 100 problems plan 1 already scored. On that shared 100, A3's generations must match plan 1's `it__cot__0000_0100.jsonl` token-for-token. If they don't, seeds, dtype, or transformers version drifted between plan 1 and plan 2 and cross-plan comparisons are suspect.
3. **B1 is reproducible given the seed.** Run B1 on the first 10 problems twice (restart in between). Generations must match byte-for-byte. If not, sampling isn't actually seeded; fix before sweeping.
4. **Hash-hit rate floor.** Any axis-A or axis-B cell with `hash_hit < 0.85` warrants inspection — plan 1's IT-CoT had 0.82, and at longer lengths we expect it to improve (more room to finish the `####`), not degrade.
5. **Vote-degeneracy canary.** For each axis-B cell, report the fraction of problems where all k chains voted the same answer. If that fraction is > 0.95 at k=5 (i.e. the sampling is effectively deterministic), the temperature or top_p is too low and self-consistency is a no-op. Expected: 0.50–0.75 for a well-functioning sampler at T=0.7.

## Pre-registered decision rule

Decide **before looking at numbers** what counts as a meaningful win on each axis.

### Axis A — generation length

The question: does letting E2B think longer buy meaningful accuracy?

- **KNEE-AT-512** (most likely given plan 1's n=100 failure modes showing 7/69 truncations): A3 and A4 within 3 points, A1 and A2 below A3 by >5 points. Conclusion: 512 is at or past saturation. Default to 512 for the head-to-head. Do not pay for A4's doubled compute.
- **MONOTONIC-LIFT**: A4 ≥ A3 + 5 points with non-overlapping CIs. Conclusion: length matters; investigate whether `max_new_tokens = 2048` is worth one follow-up cell. Otherwise use 1024 for the head-to-head.
- **SATURATED-EARLIER**: A2 and A3 within 2 points. Conclusion: 256 is enough; use 256 for the head-to-head and save 2× compute on every downstream eval. Worth investigating whether the 13% repetition-loop rate from plan 1's failure-mode breakdown is dragging longer cells down.

### Axis B — self-consistency

The question: does voting across samples buy more than length alone?

- **SC-WINS**: B3 or B4 beats A3 (greedy 512) by ≥ 8 points with non-overlapping CIs. Conclusion: self-consistency is Path 1's main lever. The compute cost is k×, but if the accuracy lift is ≥ 8 points, Path 1 is properly in the head-to-head. Promote the best-performing B cell to the shared rig.
- **SC-FLAT**: B3 within 3 points of A3. Conclusion: self-consistency on E2B-it is ineffective, likely because the temperature-0.7 samples are low-diversity or because the repetition-loop failure mode from plan 1 (13% of IT-CoT completions) drags the vote. One follow-up permitted: rerun B3 at T=1.0 on the same 500 problems. If still flat, self-consistency is ruled out for Path 1.
- **SC-HURTS**: B3 < A3 by ≥ 3 points. Conclusion: the sampler is producing worse-than-greedy individual chains, and voting doesn't recover. Almost certainly a temperature-too-high artifact or a sampling-bug. Debug before drawing conclusions.

### Compute-adjusted decision (for the Pareto plot)

The Pareto plot is the real output. Any cell that is **not Pareto-dominated** (i.e. no other cell gets higher accuracy at lower FLOPs/problem) is a candidate for the head-to-head. Report all Pareto-optimal cells. In practice we expect 2–3: probably A3 (cheap greedy), maybe A4 (if MONOTONIC-LIFT), and the best axis-B cell if SC-WINS.

The compute axis of the Pareto plot uses **mean FLOPs per problem**, not wall-clock, so Path 1's numbers are directly comparable to Path 2 and Path 3's analytical FLOPs in the head-to-head.

## Risks and known failure modes

- **Repetition-loop regression at longer lengths.** Plan 1 saw 13/69 IT-CoT wrong answers collapse into repetition at `max_new_tokens=512`. At 1024 the absolute count will likely grow. If A4 is *worse* than A3, it's almost certainly this. Flag the repetition rate per cell in the results.
- **Self-consistency voting on degenerate chains.** If 10 chains all collapse into the same repetition loop at T=0.7 (entirely plausible — plan 1 showed IT is more prone than base), the majority vote is on garbage. The vote-degeneracy canary will catch this.
- **GPU memory with `num_return_sequences=k`.** At k=10 and 512 tokens, the KV cache is 10× bigger than a single-chain call. On a 24 GB 3090 this should fit but is not free; if OOM, fall back to a loop over `k` with a shared prompt cache.
- **Seed reproducibility across transformers versions.** `transformers==5.6.1` is pinned; sampling seeds are stable within a version but not across. Do not upgrade mid-run.
- **The 8-shot exemplars in sampled decoding.** Long prompts with sampled completions can occasionally emit an exemplar-like continuation at problem-end (seen rarely in plan 1's IT-CoT). Add a stop sequence on `"\nQuestion:"` to truncate these cleanly — plan 1 didn't need it but plan 2's longer sampled completions make it worth having.

## Deliverable

A single Python script `path1_length_and_sc.py` that:

1. Runs both sweeps end-to-end in bf16 on one GPU, resumable.
2. Writes per-cell JSONLs and `results_plan2.json` as described.
3. Prints the two sweep tables:

   ```
   Axis A — generation length (greedy, n=500)
   A1  len=128    acc=0.xxx  ci=(..,..)  hash=0.xx  mean_tokens=xx
   A2  len=256    acc=0.xxx  ...
   A3  len=512    acc=0.xxx  ...                                    <- plan 1 reference
   A4  len=1024   acc=0.xxx  ...

   Axis B — self-consistency (T=0.7, len=512, n=500)
   B1  k=1        acc=0.xxx  ci=(..,..)  vote_deg=...
   B2  k=3        acc=0.xxx  ...
   B3  k=5        acc=0.xxx  ...
   B4  k=10       acc=0.xxx  ...
   ```

4. Prints the Axis-A and Axis-B verdicts and the Pareto-frontier cells.
5. Emits a CSV `path1_pareto.csv` with columns `(cell, axis, accuracy, ci_lo, ci_hi, mean_flops_per_problem, mean_wallclock_sec)` for direct consumption by the four-paths head-to-head rig.

## Report back

When done, paste:
1. Both sweep tables.
2. The verdict strings for each axis.
3. The list of Pareto-frontier cells.

Do **not** start on head-to-head integration or write the Path 1 blog post until the plan-2 results are reviewed.

## What comes after plan 2

If plan 2 produces at least one Pareto-optimal cell clearly above greedy A3, Path 1 enters the four-paths head-to-head with that cell as its representative. If all plan-2 cells cluster within CI of A3, Path 1's representative is simply A3 at 512 tokens — the honest "CoT helps, but spending more compute at inference doesn't help more" finding, which is itself a valuable datum for the head-to-head.

Plan 3 of Path 1, if it happens, would be a single follow-up experiment on the winning configuration — ARC-Easy at n=500 using the same cell. That's the handoff to the shared rig and is not part of plan 2's scope.