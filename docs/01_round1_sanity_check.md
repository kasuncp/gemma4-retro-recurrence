# 01 — Round 1: Single-layer naive loop sanity check

**Plan:** `plan1.md` — "PLE × Loop Recurrence Sanity Check"
**Result file:** `results_round1_fixed.json` (also earlier `results.json`,
near-identical)
**Status:** Complete. Gate passed with flying colors.

## Question

> Does the most naive possible loop — running one middle layer `r` times, with
> PLE applied on every iteration — immediately break the model, or does it
> degrade gracefully?

## Setup

- **Model:** `google/gemma-4-E2B` (base, not `-it`), 35 decoder layers,
  bfloat16.
- **Target:** Layer 17 (middle of 35).
- **Hook:** Replace `decoder_layers[17].forward` with a wrapper that calls the
  original `r` times in a row, preserving tuple return format. PLE handling
  is whatever the unmodified layer does — i.e. full PLE injection on every
  loop iteration.
- **Eval data:** 50 Wikitext-2 test sequences, each up to 512 tokens.
- **Metric:** Mean NLL and perplexity over all non-label-shift tokens.
- **`r` values:** {1, 2, 4, 8}.

## Results

| r | mean NLL | Perplexity | Ratio vs r=1 |
|---|---------:|-----------:|-------------:|
| 1 | 2.52865 | **12.537** | 1.000× |
| 2 | 2.59926 | **13.454** | 1.073× |
| 4 | 3.01942 | **20.480** | 1.634× |
| 8 | 3.75160 | **42.589** | 3.397× |

![Round 1 layer-17 r sweep](figs/fig01_round1_layer17.png)

Hook drift (r=1 ppl minus unmodified ppl): **exactly 0.0**. The r=1 cell is
bitwise-identical to running the unmodified model, confirming the hook
wrapper is a no-op at r=1.

## Interpretation

The plan's interpretation buckets were:

- **r=2 within ~1.2×** → "pretrained weights gracefully tolerate one extra
  loop" — met (1.07×).
- **r=2 ≥ 2×, r=4 ≥ 10×** → "PLE cannot absorb repeated injection; need
  architectural fix" — not met.
- **r=2 fine; r=8 within ~3× of baseline** → "expected and healthy, green
  light for the full probe" — essentially met (3.4×, slightly over 3×).
- **NaN / inf** → none.

**Conclusion from the plan:** *"PLE is a friction, not a wall. Green light to
probe further before committing real training compute."*

The log-linear shape of the degradation (perplexity roughly squares as `r`
doubles) is consistent with each extra loop iteration compounding a small
error in the residual stream. This is the expected failure mode for looping
a layer without training; it is *not* the catastrophic instability that would
force an architectural redesign before retraining.

## Sanity checks performed

The plan required three:

1. ✅ **r=1 matches unmodified model.** Hook drift 0.0, so yes (bitwise).
2. ✅ **Base model, not -it.** Config records `model_id:
   google/gemma-4-E2B`.
3. ✅ **bfloat16 confirmed.** Config records `dtype: bf16`.

## What this round did NOT answer

- Whether PLE injection *policy* matters — vanilla (every iter) was the only
  mode tested. → Round 2a.
- Whether layer 17 is representative or special. → Rounds 2b and 2c.
- Whether block-of-layers looping degrades the same way. → Rounds 3a–3c.
- Whether the perplexity numbers translate to reasoning capability. → Round 4.

## One subtle artifact noted later

Round 1's "vanilla" was taken to mean "PLE applied at full strength on every
iteration because the hook didn't touch it." That turned out to be correct
for this setup (the hook calls `orig_forward` which applies PLE internally),
but it was re-verified in Round 2a addendum 2 using the fixed PLE-intercept
hook — and numbers match. So Round 1 is the real vanilla baseline.
