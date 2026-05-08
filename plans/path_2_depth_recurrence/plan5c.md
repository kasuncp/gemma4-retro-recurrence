# Round 5c — IT-Weight Perplexity Bridge

**Project:** Retrofitted recurrence on Gemma 4 E2B
**Status:** Ready for implementation
**Depends on:** Round 2c (`results_round2c_full_map.json`), Round 3b (`results_round3b_blocks.json`), Round 3c (`results_round3c_extended_blocks.json`).
**Blocks:** Round 6 (`plan6.md`).

---

## Why this round exists

There's a model seam in the project that needs closing before round 6 runs:

| Layer of the study | Model used | Source |
|---|---|---|
| Architectural map (plans 1–3c) | `google/gemma-4-E2B` (base) | Wikitext-2 perplexity |
| Reasoning eval (plans 4, 5, 6) | `google/gemma-4-E2B-it` | GSM8K, ARC-Challenge |

The block-geometry decisions (A, D, G; "anchor at layer 15"; "max width ≈ 10") were established on **base** weights. Plan 6 measures reasoning on **IT** weights. Instruction tuning doesn't change architecture (35 layers, PLE position, KV-sharing pattern at 14/15, attention interleaving), so the map *should* transfer — but "should" isn't measured.

If plan 6 lands in Bucket D or E (severe degradation or loop collapse), without this bridge we can't distinguish:
- (a) IT weights and the prompt fix don't rescue recurrence — a decoding/reasoning failure, or
- (b) the perplexity-tolerance map itself doesn't transfer to IT — a wrong-block failure.

This round is the cheapest possible measurement that distinguishes (a) from (b): re-run perplexity on IT for the *already-chosen* blocks, plus a coarse single-layer overlay, and compare against base.

---

## Scope

**In scope:**
- Add `--mode=it-perplexity-bridge` to `ple_sanity_check.py`.
- Load `google/gemma-4-E2B-it` (instruction-tuned), bf16, `use_cache=False`.
- Same 50 Wikitext-2 raw-text sequences (max_length=512) as rounds 1–3c.
- Compute an unmodified IT baseline first (single number — IT raw-text perplexity is expected to be slightly worse than base because IT shifts the weight distribution toward chat).
- Block sweep: A [15–19], D [15–22], G [15–24] at r ∈ {1, 2, 4, 8}.
- Control: F [25–32] at r=8 (must still break catastrophically — this confirms the architectural map's failure modes also transfer).
- Coarse single-layer overlay: r=8 vanilla on layers {0, 5, 10, 15, 17, 20, 25, 30, 33} (9 cells) — enough to verify the valley is still around 15–19 and the late-layer drift zone still exists.
- Total: ~22 cells. Estimated runtime ~20 minutes.

**Out of scope:**
- No full 35-layer × 3-r re-run. If the bridge reveals the map didn't transfer, *that* triggers a partial round 2c re-run on IT — but only if needed.
- No `once`/`scaled` PLE variants. Round 2a settled this for base; vanilla is the only mode tested.
- No GSM8K, no ARC, no chat-formatted prompts. Wikitext-2 raw text only — apples-to-apples with the base measurements.
- No new hooks. Reuse the fixed block-looping hook from round 3b and the positional-arg PLE handling from round 2a addendum 2.
- No training, no quantization, no on-device.

---

## What "raw text on IT" means and why it's correct

The IT model is trained on chat-formatted dialogue. Wikitext-2 is raw English. Feeding raw text to an IT model still computes a meaningful next-token-prediction loss — IT weights inherit base-model language modeling competence. Expect IT baseline perplexity ≈ 1.0×–1.5× base baseline (12.54), depending on how aggressively the IT post-training shifted weights.

Using raw text (not chat-wrapped) is the right choice because:
- We're testing **architectural tolerance to looping**, not chat-following capability.
- Apples-to-apples comparison with rounds 1–3c requires identical inputs.
- Chat-wrapping would change *which tokens* the loss is computed over (system/turn-marker tokens vs content tokens), confounding the comparison.

If you want chat-wrapped numbers later for sanity, add as a follow-up — not this round.

---

## Configurations

| Config | Block | r | Why |
|---|---|---|---|
| `it-baseline` | — | — | Establish IT raw-text perplexity. Anchor for all ratios in this round. |
| `it-A-r1` | [15,19] | 1 | r=1 regression — must match `it-baseline` bitwise. |
| `it-A-r2`, `it-A-r4`, `it-A-r8` | [15,19] | 2, 4, 8 | Anchor block. Round 3b base ppl at r=8 = 30.1 (ratio 2.4×). |
| `it-D-r1` | [15,22] | 1 | r=1 regression. |
| `it-D-r2`, `it-D-r4`, `it-D-r8` | [15,22] | 2, 4, 8 | Round 3b base ppl at r=8 = 33.3 (ratio 2.7×). |
| `it-G-r2`, `it-G-r4`, `it-G-r8` | [15,24] | 2, 4, 8 | Round 3c base ppl at r=8 = 36.0 (ratio 2.9×). |
| `it-F-r8` | [25,32] | 8 | Control. Round 3b base ppl at r=8 = 338,517 (catastrophic). Must still be catastrophic on IT. |
| `it-singlelayer-{0,5,10,15,17,20,25,30,33}-r8` | layer | 8 | Coarse overlay on round 2c's full map. |

22 cells total. Drop the single-layer overlay to 5 layers (15, 17, 25, 33) if GPU is tight — that still discriminates "valley present / valley absent" and "late drift zone present / absent."

---

## Mandatory regression checks

Run before the main sweep. All must pass.

1. **`it-A-r1` ppl == `it-baseline` ppl** (relative drift < 1e-4).
   r=1 looping must be a no-op. If this fails, the block-looping hook regressed when switching models — debug before proceeding. Same check as round 3b's per-block r=1 regression.

2. **`it-D-r1` ppl == `it-baseline` ppl** (same tolerance).

3. **`it-singlelayer-17-r8` ratio is in the valley range.**
   Round 2c base value at layer 17 r=8 was ~3.4×. Expect IT ratio in [2.5×, 5×]. If it's >10×, the valley shifted — flag immediately.

4. **`it-F-r8` ratio > 100×.**
   F is the catastrophic control. If it's *not* catastrophic on IT, something fundamental about the late-layer drift mechanism changed during instruction tuning — interesting finding, but means the round 2c interpretation also shifted and a bigger re-run is needed. Halt and discuss.

---

## Reporting format

JSON `results_round5c_it_perplexity_bridge.json`:

```json
{
  "config": {
    "model_id": "google/gemma-4-E2B-it",
    "dtype": "bf16",
    "n_sequences": 50,
    "max_length": 512,
    "mode": "it-perplexity-bridge"
  },
  "baselines": {
    "it_baseline_ppl": ...,
    "base_baseline_ppl_round1": 12.5366,
    "it_vs_base_ratio": ...
  },
  "regression_checks": {
    "A_r1_no_op": {"drift": ..., "passed": true},
    "D_r1_no_op": {"drift": ..., "passed": true},
    "L17_r8_in_valley": {"ratio": ..., "passed": true},
    "F_r8_catastrophic": {"ratio": ..., "passed": true}
  },
  "block_cells": [
    {"name": "A-r2", "block": [15,19], "r": 2, "ppl": ..., "ratio_vs_it_baseline": ..., "base_ppl_round3b": 27.3, "transfer_ratio": ...},
    ...
  ],
  "single_layer_cells": [
    {"layer": 15, "r": 8, "ppl": ..., "ratio_vs_it_baseline": ..., "base_ratio_round2c": ...},
    ...
  ]
}
```

The key derived metric is **`transfer_ratio = (it_ppl / it_baseline) / (base_ppl / base_baseline)`**. Values near 1.0 mean the architectural tolerance transferred; values >>1 mean IT weights are more fragile under that block; <<1 means IT is *more* tolerant (unexpected but possible).

Print a final table:

```
=== Round 5c IT-perplexity bridge ===
                          base_ratio   it_ratio   transfer
unmodified                   1.00x       1.00x      —
A  r=2                       2.18x        ?         ?
A  r=4                       2.37x        ?         ?
A  r=8                       2.40x        ?         ?
D  r=2                       2.69x        ?         ?
D  r=4                       2.61x        ?         ?
D  r=8                       2.66x        ?         ?
G  r=2                        ?           ?         ?
G  r=4                        ?           ?         ?
G  r=8                       2.87x        ?         ?
F  r=8 (control)         27,000x         ?         must stay >100x

Single-layer r=8 (coarse):
                           base_ratio   it_ratio
layer  0                   round 2c      ?
layer  5                   round 2c      ?
layer 10                   round 2c      ?
layer 15                   round 2c      ?
layer 17                       3.4x      ?      (valley anchor)
layer 20                   round 2c      ?
layer 25                   round 2c      ?
layer 30                   round 2c      ?
layer 33                   round 2c      ?
```

Pre-fill the `base_ratio` column from `results_round2c_full_map.json` and `results_round3b_blocks.json` so the comparison is on screen.

---

## Interpretation buckets — commit before looking

**Bucket 1 — Tight transfer.**
All block transfer ratios in [0.8, 1.3]. Single-layer overlay shows the same valley shape as round 2c (low ratios around 15–19, high ratios elsewhere). F is still catastrophic. **Plan 6 proceeds as written.** Block geometry decisions are validated for IT. Proceed straight to plan 6.

**Bucket 2 — Quantitative shift, qualitative match.**
Block transfer ratios in [0.5, 2.0] but the *ordering* is preserved (A ≈ D ≈ G ≪ F), and the single-layer valley is still anchored around 15–19. **Plan 6 proceeds, with a noted caveat** that absolute IT-side perplexity ratios are not identical to base. Block choices remain valid.

**Bucket 3 — Valley location shifted.**
Single-layer overlay shows the valley at, e.g., layers 17–22 instead of 15–19, or the drift zone starts earlier (e.g., at layer 22 instead of 25). **Plan 6 blocks need re-picking.** Write a short follow-on plan that re-runs round 2c's full 35-layer × 3-r grid on IT (~210 cells, ~45 min) before plan 6.

**Bucket 4 — Block A breaks catastrophically on IT.**
A r=8 ratio > 100×. Either the block-looping hook regressed when switching models, or IT weights have a fundamentally different fixed-point structure. Halt, debug the hook first (run `it-A-r1` regression carefully, check the layer module class name didn't change between base and IT, verify `apply_chat_template`-related artifacts aren't leaking into the layer-list path). If hook is fine, the architectural feasibility story for IT-direct retrofit is much weaker than for base — discuss before plan 6.

**Bucket 5 — F is not catastrophic on IT.**
F r=8 ratio < 100× (e.g., 5×). Late-layer drift mechanism changed. Interesting; suggests instruction tuning damped some pathological dynamics. **Wider blocks may be viable on IT than on base.** Worth a follow-on plan that explores extending blocks past layer 24 on IT specifically. Plan 6 still proceeds with the existing blocks, but a "block H/I retest" gets queued.

**Bucket 6 — Mixed signal.**
A and D transfer cleanly but G doesn't, or vice versa. Pause and discuss before plan 6.

---

## Exit criteria

- **Bucket 1 or 2** → run plan 6 immediately, no changes.
- **Bucket 3** → write a "round 5d: round-2c re-run on IT" plan; do not run plan 6 until the IT map is established.
- **Bucket 4** → debug; do not run plan 6 with potentially wrong geometry.
- **Bucket 5** → plan 6 runs as written, with a note that wider blocks should be tested in a follow-on.
- **Bucket 6** → discuss before any further plans.

---

## Implementation notes

- The block-looping hook and positional-arg PLE handling work the same on IT — the layer module class is the same (`Gemma4TextDecoderLayer`, per round 4 inspection notes). No code changes to the hook itself.
- The model-loading path may differ if IT uses a multimodal wrapper. Check that `model.model.layers` is still the right path; if not, navigate to the text decoder following round 1's debugging note.
- Reuse the `compute_perplexity` helper from round 1 unchanged.
- Tokenization: use the IT model's own tokenizer (`AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")`). Tokenizers between base and IT are usually identical for Gemma but verify by checking `tokenizer.vocab_size` matches.

---

## Report back

Paste:
- The IT baseline ppl and `it_vs_base_ratio`.
- The 4 regression-check results.
- The block table (3 blocks × 4 r values + F control).
- The single-layer overlay table (9 layers).
- The bucket and the exit-criteria action.

Do not run plan 6 until this is in.
