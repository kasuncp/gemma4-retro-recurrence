# 09 — Round 5: Reasoning Eval v2 (planned, not executed)

**Plan:** `plan5.md`
**Status:** Written and ready for implementation, not yet run. There is no
`results_round5_*.json`.

This document summarizes what Round 5 is supposed to establish and why.

## Three things Round 5 does

1. **Fix the GSM8K baseline** so math becomes a real signal instead of a
   harness artifact. Round 4's 4.8% baseline is a known harness bug (172
   of 238 wrong responses are textbook continuations — see
   `08_round4_reasoning_eval.md`).
2. **Follow the one interesting lead from Round 4** — narrow blocks
   degrade less than wide blocks on ARC-Easy — by sweeping block width
   at fixed r=8.
3. **Pilot one strategy change** — PLE injected only on iteration 1 —
   to test whether repeated PLE injection during recurrence is part of
   the damage.

## Part 1 — Harness fix for GSM8K

### Three required changes, in order

1. **`max_gen_tokens` 256 → 512.** GSM8K chain-of-thought routinely needs
   200–400 tokens. 256 truncates most attempts. 512 allows degenerate
   loops to self-terminate without runaway cost.

2. **Use the instruction-tuned chat template.** Round 4 used
   `google/gemma-4-E2B` (base) with a raw `Question:/Answer:` format,
   which triggers textbook continuation. Two options:
   - **(preferred)** Switch to `google/gemma-4-E2B-it` and use
     `tokenizer.apply_chat_template`.
   - **(fallback)** Stay on base and add a stop sequence on
     `"\nQuestion:"` so textbook continuations self-terminate.

3. **Add 8-shot CoT prompting.** Use Wei et al.'s original 8 exemplars —
   this is what every published Gemma GSM8K number uses, so it's the
   only way the baseline will be comparable to published numbers.

### Mandatory validation gate

Before running the full Round 5 sweep:

- Run baseline-only on 50 GSM8K problems.
- Confirm accuracy is in the **40–55% range** for Gemma 4 E2B. (This
  range is from published Gemma-class expectations.)
- **If not, stop.** Harness still has a bug; full sweep would be wasted
  compute.

No Round 5 config results are interpretable until this gate passes.

## Part 2 — Narrow-block width sweep (primary experiment)

ARC-Easy is the trustworthy benchmark. Fixed r=8, fixed start at layer 15,
varying width.

| Config | Block | Width | r | Notes |
|--------|-------|------:|---|-------|
| baseline | — | — | 1 | Reference |
| W2-r8 | [15, 16] | 2 | 8 | New |
| W3-r8 | [15, 17] | 3 | 8 | New |
| W4-r8 | [15, 18] | 4 | 8 | New |
| W5-r8 | [15, 19] | 5 | 8 | Rerun of Round 4's A-r8 |
| W5-r1 | [15, 19] | 5 | 1 | Sanity — must = baseline |

### Two mandatory cross-round checks

1. **W5-r1 must match baseline token-for-token** on every problem.
2. **W5-r8 must reproduce Round 4's A-r8 to within ±2% accuracy** on
   ARC-Easy. A-r8 scored 40.0%; if W5-r8 is outside [38%, 42%], something
   non-deterministic changed (seed, dtype, tokenization, generation
   config, library version) and prior cross-round comparisons are
   suspect.

### The prediction being tested

Accuracy increases monotonically as width decreases.

### Three possible outcomes

| Outcome | Meaning | Next step |
|---------|---------|-----------|
| **Monotonic increase** (e.g., W2 > W3 > W4 > W5) | Width is the dominant factor | Round 6: go deep on width 1–2, sweep r, add ARC-Challenge + HellaSwag |
| **Flat across widths** | Recurrence broken regardless of block size; PLE or hybrid-attention interference likely | Pivot to strategy changes |
| **Non-monotonic** | Some specific layer boundary is pathological | Inspect which layer is being crossed |

## Part 3 — Start-position sweep (secondary, optional)

Does block start *elsewhere* in the consumer zone produce different
damage? Round 3c Block I (20–27) was catastrophic, but that was at the
usual block width 8. At narrow width 3:

| Config | Block |
|--------|-------|
| S10-W3 | [10, 12] — crosses KV boundary (prediction: catastrophic) |
| S15-W3 | [15, 17] — same as W3-r8 above (one run covers both) |
| S20-W3 | [20, 22] — all-consumer but shifted late |

### Interpretation

- **S10 ≫ S20** → late-layer KV sharing is the real problem; future work
  stays in the middle (contradicts Round 3c's Block I finding, would
  require re-examination).
- **S20 ≫ S10** → middle-layer PLE is the problem (consistent with
  Round 3c).
- **Similar** → start position is not a dominant factor at narrow width.

Drop this part if GPU budget is tight — the width sweep is the primary
experiment.

## Part 4 — PLE ablation (strategy pilot)

| Config | Block | r | PLE strategy |
|--------|-------|---|--------------|
| W5-r8-noPLE | [15, 19] | 8 | Iter 1 only; `None` (or zeros) after |

### Comparison

Directly against W5-r8 (PLE every iter). Isolates whether PLE re-injection
is a meaningful fraction of the damage.

- **W5-r8-noPLE meaningfully better than W5-r8** (say >5 points on
  ARC-Easy) → real finding; Round 6 reworks the wrapper to handle PLE
  correctly across iterations and re-runs Round 4's full sweep with the
  fix.
- **Same or worse** → rule out the PLE-re-injection hypothesis for
  reasoning degradation; move on.

## Total config count and budget

- **Full Round 5:** 1 baseline + 5 width + 2 new start-position + 1 PLE
  ablation = **9 configs**.
- **Minimum viable:** drop start-position sweep → **7 configs**.

## Reporting format additions

Round 5 adds to the existing schema:

1. **`truncation_rate` per config** — fraction of generations hitting
   `max_gen_tokens` without a parseable answer. This is the metric that
   would have caught the Round 4 baseline problem immediately.
2. **Cross-round delta check:** `w5_r8_vs_round4_a_r8` block in
   `sanity_checks` — records Round 4 A-r8 accuracy (0.400), Round 5
   W5-r8 accuracy, their delta, and `within_tolerance` bool.
3. **W5-r1 sanity check**: matches counts vs baseline on both GSM8K
   and ARC-Easy.

## Three exit scenarios and what Round 6 looks like

The plan's three labeled outcomes:

### Outcome A — Width matters, narrow wins

Accuracy increases monotonically as width shrinks on ARC-Easy.
**Round 6:** Go deep on width-1 and width-2 configs, sweep r, add ARC-
Challenge and HellaSwag to see if narrow-block recurrence actually
*helps* anywhere (not just degrades less).

### Outcome B — PLE ablation wins

W5-r8-noPLE dramatically beats W5-r8 (>5 points on ARC-Easy).
**Round 6:** Rework the wrapper to handle PLE correctly across
iterations (iter-1-only, scaled decay, or learned gate) and re-run
Round 4's full original sweep with the fix.

### Outcome C — Everything collapses similarly

All configs land in the 25–40% range on ARC-Easy regardless of width,
start, or PLE strategy.
**Round 6: pivot.** Recurrence on E2B is structurally hostile regardless
of block choice. Two options:

- Try a different recurrence unit — recurrence only over the FFN
  sub-block, skipping attention.
- Try a different base model where the paper's assumptions actually
  hold (plain pre-norm Llama-style stack, no PLE, uniform attention).

## Open questions before execution

The plan ends with three questions for the user:

1. **Model variant:** Switch to `google/gemma-4-E2B-it` for the
   baseline fix, or stay on base and add stop sequences?
2. **Budget:** Run all 9 configs, or the 7-config minimum viable set?
3. **CoT exemplars:** Use Wei et al.'s original 8-shot, or a different
   prompt set the project has settled on?
