# Round 5 Plan — Reasoning Eval v2

**Project:** Retrofitted recurrence on Gemma 4 E2B
**Status:** Ready for implementation
**Depends on:** Round 4 harness (`results_round4_reasoning.json` format)

---

## Goals

1. **Fix the GSM8K baseline** so math reasoning becomes a real signal. Round 4 baseline was 4.8% — a harness bug, not a model result.
2. **Follow the one interesting lead from round 4** — narrow blocks degrade less than wide ones — by sweeping block width and start position at fixed r.
3. **Pilot one strategy change** (PLE injected only on iteration 1) to test whether PLE re-injection is part of the damage.

---

## Part 1 — Fix the GSM8K baseline

The round 4 GSM8K baseline of 4.8% is not a model result. Diagnostic evidence from `results_round4_reasoning.json`:

- **155 of 238** wrong baseline answers hit `max_gen_tokens=256` mid-arithmetic.
- Remaining wrong answers are producing "Question: … Answer: …" textbook-style continuations instead of answering the asked question.

### Required fixes, in order

1. **Raise `max_gen_tokens` from 256 → 512.**
   - GSM8K chain-of-thought routinely needs 200–400 tokens; 256 truncates most attempts.
   - Do not go to 1024 — runaway repetition loops would eat 4× compute for no benefit, and 512 is enough for degenerate generations to self-terminate.

2. **Use the instruction-tuned chat template properly.**
   - The "Question: … Answer: …" continuation pattern strongly suggests the prompt is being fed as raw text rather than through `tokenizer.apply_chat_template` with E2B's IT turn format.
   - **Confirm which model variant is loaded** (E2B vs E2B-it) — the round 4 config says `google/gemma-4-E2B`, which is the base model. Two valid fixes:
     - **(preferred)** Switch to `google/gemma-4-E2B-it` and use `apply_chat_template`.
     - **(fallback)** Stay on base E2B but add a stop sequence on `"\nQuestion:"` so textbook continuations self-terminate, and confirm the few-shot format matches what the base model expects.

3. **Add an 8-shot CoT prompt for GSM8K.**
   - Use the exemplars from Wei et al. (original CoT paper) — this is what every published Gemma GSM8K number uses, so it is the only way the baseline will be comparable to published numbers.
   - Zero-shot GSM8K numbers for 2B-class models are misleading and not what we want to compare against.

### Validation gate (MANDATORY)

Before running the full round 5 sweep:

- Run baseline-only on **50 GSM8K problems**.
- Confirm accuracy is in the **40–55% range** for Gemma 4 E2B.
- **If it is not, stop.** The harness still has a bug and the full sweep is wasted compute.

No round 5 config results are interpretable until this gate passes.

---

## Part 2 — Narrow-block width sweep (primary experiment)

ARC-Easy is the trustworthy benchmark (baseline 83.5% in round 4 matches expectations). The one signal from round 4: A-r8 `[15,19]` scored 40% vs ~25% for wider blocks. Round 5 nails this down with a width sweep at **fixed r=8** and **fixed start layer 15**.

### Configs

| Config   | Block     | Width | r | Notes                        |
|----------|-----------|-------|---|------------------------------|
| baseline | —         | —     | 1 | Reference                    |
| W2-r8    | [15, 16]  | 2     | 8 | New                          |
| W3-r8    | [15, 17]  | 3     | 8 | New                          |
| W4-r8    | [15, 18]  | 4     | 8 | New                          |
| W5-r8    | [15, 19]  | 5     | 8 | **Rerun of round 4 A-r8**    |
| W5-r1    | [15, 19]  | 5     | 1 | **Sanity check, must = baseline** |

### Two mandatory sanity checks

- **W5-r1 must match baseline token-for-token** on every problem. Same as the D-r1 check in round 4. If it does not, the loop/cache/PLE plumbing has regressed and the sweep is invalid.
- **W5-r8 must reproduce round 4 A-r8 within ±2% accuracy.** A-r8 scored 40.0% on ARC-Easy in round 4. If W5-r8 is not within [38%, 42%], something non-deterministic changed between rounds (seed, dtype, tokenization, generation config, library version) and prior cross-round comparisons are suspect.

### Prediction being tested

Accuracy increases monotonically as width decreases, approaching baseline as width shrinks.

**Possible outcomes and what they mean:**

- **Monotonic increase** → width is a dominant factor; round 6 goes deep on width 1–2 and sweeps r.
- **Flat across widths** → recurrence is broken independent of block size; PLE re-injection or hybrid-attention interference is the likely culprit. Pivot to strategy changes rather than more block sweeps.
- **Non-monotonic** → some specific layer boundary is pathological; inspect which layer is being crossed.

---

## Part 3 — Start-position sweep (secondary)

Second question worth answering in the same run: does the block start elsewhere in the stack produce different damage? Gemma 4 E2B's last layer is mandated global-attention, and late layers have shared-KV entanglement, so looping late layers may be structurally worse than middle.

### Configs

At fixed **width 3** and **r=8**:

| Config   | Block     |
|----------|-----------|
| S10-W3   | [10, 12]  |
| S15-W3   | [15, 17]  | *(same run as W3-r8 above — one run covers both)* |
| S20-W3   | [20, 22]  |

**Important:** Before finalizing indices, read the layer count from the round 4 `inspection` dump. Round 4 inspection confirms `Gemma4TextDecoderLayer` and a PLE kwarg of `per_layer_input`, but verify the total layer count before picking `[20,22]` — adjust if the stack is shorter than expected or if `[20,22]` straddles the shared-KV boundary.

### Interpretation

- **S10 ≫ S20** → late-layer KV sharing is the problem; future work stays in the middle.
- **S20 ≫ S10** → the opposite; middle-layer PLE is the problem.
- **Similar** → start position is not a dominant factor; drop from future sweeps.

### Budget note

If GPU budget is tight, drop the two new start-position configs (S10-W3, S20-W3). The width sweep is the primary experiment and the most likely to produce a clean signal.

---

## Part 4 — PLE ablation (strategy pilot)

Currently the PLE is injected on every iteration of the recurrent loop (iterations 1…r all get the same `per_layer_input`). The PLE is designed to be a per-layer embedding fed once — re-feeding it on every iteration is arguably semantically wrong.

### Config

| Config       | Block     | r | PLE strategy             |
|--------------|-----------|---|--------------------------|
| W5-r8-noPLE  | [15, 19]  | 8 | Iter 1 only; `None` after |

### Implementation detail

- On iteration 1, pass `per_layer_input=<the PLE tensor>` as usual.
- On iterations 2…r, pass `per_layer_input=None`.
- If the `Gemma4TextDecoderLayer.forward` does not tolerate `None` for that kwarg, use a zero tensor of the same shape/dtype/device — but verify behavior matches by inspecting the layer source at `transformers/models/gemma4/modeling_gemma4.py` line 1354 (from round 4 inspection).

### Comparison

Compared against W5-r8 (PLE every iter), this isolates whether PLE re-injection is a meaningful part of the damage.

- **W5-r8-noPLE meaningfully better than W5-r8** → real finding; reshapes round 6 to rework the wrapper and re-run round 4's full sweep with the fix.
- **Same or worse** → rule the PLE hypothesis out; move on.

---

## Total config count and budget

Full round 5 = **1 baseline + 5 width-sweep + 2 new start-position + 1 PLE ablation = 9 configs**.

Minimum viable round 5 (if GPU-constrained) = 1 baseline + 5 width-sweep + 1 PLE ablation = **7 configs**, dropping the start-position sweep.

At round 4 runtime per config this is tractable on the same hardware.

---

## Reporting format

Keep the same JSON output format as round 4 (`per_config` per-example list, `summary` table, `agreement_matrix`, `sanity_checks`). Two additions required:

1. **Truncation rate per config** — fraction of generations that hit `max_gen_tokens` without producing a parseable answer. This is the metric that would have caught the round 4 baseline problem immediately. Add a field to each per-config summary row: `"truncation_rate": <float>`.

2. **Cross-round delta check in `sanity_checks`** — add:
   ```json
   "w5_r8_vs_round4_a_r8": {
       "round4_accuracy": 0.400,
       "round5_accuracy": <observed>,
       "delta": <observed - 0.400>,
       "within_tolerance": <abs(delta) <= 0.02>
   }
   ```

3. **W5-r1 sanity check in `sanity_checks`**, same format as round 4's `gsm8k_d_r1_vs_baseline`:
   ```json
   "w5_r1_vs_baseline_arc": {"matches": <int>, "total": 200},
   "w5_r1_vs_baseline_gsm8k": {"matches": <int>, "total": 250}
   ```

---

## Exit criteria — what round 5 should conclude

One of three outcomes, each pointing somewhere different for round 6:

### Outcome A — Width matters, narrow wins
Accuracy increases monotonically as width shrinks on ARC-Easy.
→ **Round 6:** go deep on width-1 and width-2 configs, sweep r, add ARC-Challenge and HellaSwag to see if narrow-block recurrence actually helps anywhere (not just degrades less).

### Outcome B — PLE ablation wins
W5-r8-noPLE ≫ W5-r8 by a meaningful margin (>5 points on ARC-Easy).
→ **Round 6:** rework the wrapper to handle PLE correctly across iterations (iter-1-only, or scaled decay, or learned gate) and re-run round 4's full original sweep with the fix.

### Outcome C — Everything collapses similarly
All configs land in the 25–40% range on ARC-Easy regardless of width, start, or PLE strategy.
→ **Round 6: pivot.** Recurrence on E2B is structurally hostile regardless of block choice. Two options:
  - Different strategy: recurrence only over the FFN sub-block, skipping attention.
  - Different base model where the paper's assumptions actually hold (plain pre-norm Llama-style stack, no PLE, uniform attention).

---

## Open questions for the user before implementation

1. **Model variant:** Switch to `google/gemma-4-E2B-it` for the baseline fix, or stay on base and add stop sequences? (Preferred: IT variant.)
2. **Budget:** Run all 9 configs, or the 7-config minimum viable set?
3. **CoT exemplars:** Use Wei et al.'s original 8-shot, or a different prompt set the project has settled on?