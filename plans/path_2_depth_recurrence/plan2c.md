# Gemma 4 E2B — Round 2c: Full 35-Layer Looping Tolerance Map

## Context

Round 2b produced two datasets with a gap between them:

- **PLE importance scan**: all 35 layers at r=1. Identified a PLE-inert corridor (layers ~18–32) with a few high-importance spikes (layers 3, 8, 13, 33, 34).
- **Layer-location sweep**: r=4 and r=8 at only three layers (5, 17, 28). Showed that looping tolerance varies enormously across locations: layer 17 degrades 3.4× at r=8, while layers 5 and 28 degrade 62× and 42× respectively.

The three-point layer-location sweep ruled out single-variable explanations for looping tolerance. Layers 17 and 28 share attention type (`sliding_attention`), KV-sharing role (`is_kv_consumer: true`), and near-identical PLE importance (0.004 NLL diff) — yet their r=8 looping ratios differ by 12×. Something structural is going on that three data points can't reveal.

This round maps the full looping-tolerance landscape across all 35 layers, at multiple r values and both PLE modes. The goal is a dataset rich enough to:

1. **Identify the loopable regions** of E2B with confidence (not just single points).
2. **Test which factor explains looping tolerance** — PLE importance, attention type, KV-sharing role, or layer depth.
3. **Determine where `once` beats `vanilla` systematically**, not just at one location.
4. **Inform block-looping placement** for Round 3 (recurrent-block experiments) or the eventual retrofit training.

## Scope

**In scope:**
- Extend `ple_sanity_check.py` with `--mode=full-looping-map`.
- Grid: 35 layers × {vanilla, once} × {r=2, r=4, r=8} = **210 cells**.
- Reuse all existing infrastructure (fixed PLE hook, `use_cache=False`, 50 Wikitext sequences, bf16).
- Include `r=1` vanilla as a per-layer regression check (must match unmodified baseline at every layer).
- Save a single JSON with full per-cell data.
- Print a compact summary + cross-correlations.

**Out of scope:**
- No block-level looping. Still one layer at a time. (That's Round 3.)
- No `scaled` PLE mode. Confirmed indistinguishable from vanilla in round 2a.
- No training.
- No downstream tasks. Wikitext-2 perplexity only.
- No plotting; print tables and compute correlations numerically.

## Experimental design

### Grid

- `layer ∈ {0, 1, ..., 34}` (all 35 layers).
- `ple_mode ∈ {vanilla, once}`.
- `r ∈ {2, 4, 8}`.

Total: 210 cells. Estimated runtime: ~35–45 minutes depending on GPU.

### Why these r values

- **r=2**: mild looping, single extra iteration. A proxy for "inherent layer sensitivity" before compounding dominates. If a layer degrades badly at r=2 already, it's fundamentally not loop-friendly. If a layer tolerates r=2 well but breaks at r=8, the story is about compounding, not the first loop.
- **r=4**: midpoint, best dynamic range. At layer 17, r=4 shows 1.63× degradation (mild); at layer 5, it shows 18× (clearly bad). Signal-rich.
- **r=8**: stress test. The test recurrence the paper typically evaluates at. For layers that break badly here, we get upper-bound data on "how bad does this get."

The `ppl(r=8) / ppl(r=2)` ratio per layer is an informative derived quantity: it measures *how much worse looping gets* as r grows, independent of inherent layer sensitivity. A layer with a low ratio degrades roughly linearly; a high ratio means compounding is catastrophic.

### Regression check

Before the main sweep, at each layer run vanilla at r=1. Verify perplexity matches the unmodified baseline (~12.5366). If any layer's r=1 vanilla disagrees with the baseline, stop and debug before proceeding.

Skip the r=1 `once` check (we've established it's bitwise-identical to vanilla at r=1 by construction).

## Implementation

### Refactor

Add a new mode `full-looping-map` to `ple_sanity_check.py`. Reuse the fixed `make_looped_forward_ple` hook from round 2a addendum 2.

Pseudocode for the main sweep:

```python
def run_full_looping_map(args, model, decoder_layers, inputs):
    n_layers = len(decoder_layers)
    r_values = [2, 4, 8]
    ple_modes = ["vanilla", "once"]

    # Per-layer regression check first
    print("=== Per-layer r=1 regression ===")
    for l in range(n_layers):
        orig_forward = decoder_layers[l].forward
        decoder_layers[l].forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
        )
        _, ppl = compute_perplexity(model, inputs)
        decoder_layers[l].forward = orig_forward
        if not math.isclose(ppl, UNMOD_PPL, rel_tol=1e-3):
            abort(f"Layer {l} r=1 regression failed: {ppl} vs {UNMOD_PPL}")

    # Main 210-cell sweep
    cells = []
    for l in range(n_layers):
        orig_forward = decoder_layers[l].forward
        for ple_mode in ple_modes:
            for r in r_values:
                decoder_layers[l].forward = make_looped_forward_ple(
                    orig_forward, r=r, ple_mode=ple_mode, ple_kwarg=ple_kwarg,
                )
                nll, ppl = compute_perplexity(model, inputs)
                cells.append({
                    "layer": l,
                    "ple_mode": ple_mode,
                    "r": r,
                    "mean_nll": nll,
                    "ppl": ppl,
                })
                print(f"  layer {l:2d}  {ple_mode:7s} r={r}: ppl={ppl:.2f}")
        decoder_layers[l].forward = orig_forward
```

### Layer metadata

For each layer, record alongside the results:
- `attention_type`: `"sliding_attention"` or `"full_attention"` (already logged in round 2b).
- `is_kv_consumer`: boolean (already logged).
- `ple_importance`: the `nll_diff` from round 2b's importance scan. Load it from `results_round2b_importance.json` and merge.

This gives each cell five explanatory variables to cross-correlate against: layer index, attention type, KV role, PLE importance, and the r/mode it was run at.

## Analysis

After the sweep, compute and print:

### Summary table 1: vanilla looping across layers

```
=== Vanilla looping (all 35 layers) ===
layer  attn     kv_cons  PLE_imp   r=2      r=4       r=8      r8/r2
  0   sliding   False    0.0215   XX.XX    XX.XX    XX.XX     X.XX
  1   sliding   False    0.0175   ...
  ...
 34   full      True     0.1137   ...
```

### Summary table 2: once vs vanilla delta

```
=== Once vs vanilla (all 35 layers, r=8) ===
layer  vanilla_ppl   once_ppl    delta_%    helps?
  0     XX.XX         XX.XX      +X.X%      yes/no
  ...
```

### Correlation analysis

For `ppl(r=8, vanilla)` per layer, compute Pearson and Spearman correlations with:
1. `ple_importance` (layer's PLE NLL diff from round 2b).
2. `attention_type` as binary (0=sliding, 1=full).
3. `is_kv_consumer` as binary.
4. `layer_index`.

Report all four correlations. Interpretation depends on which is strongest.

### Top-5 tables

- 5 most loop-friendly layers (lowest r=8 vanilla ppl).
- 5 least loop-friendly layers (highest r=8 vanilla ppl, excluding any that produced NaN/inf).
- 5 layers where `once` most helps (largest ppl improvement over vanilla at r=8).
- 5 layers where `once` most hurts.

## Interpretation — commit before looking

**Scenario A — PLE importance dominates.** r=8 vanilla ppl correlates strongly (|ρ| > 0.6) with PLE importance, and weakly with everything else. Rule becomes: "loop where PLE is inert." Retrofit block placement is a simple PLE-map lookup.

**Scenario B — Attention type dominates.** Global layers behave very differently from sliding layers. Correlation with attention type is strongest. Retrofit must respect attention interleaving — likely picking a block that's a whole period of the 4:1 pattern.

**Scenario C — KV-sharing role dominates.** Consumer layers loop differently from producer layers, regardless of PLE or attention type. Retrofit block must be entirely inside one role (preferably all-consumer or all-non-consumer).

**Scenario D — Layer depth dominates.** Monotonic relationship between index and loopability. Retrofit block placement is simply "deep enough" or "shallow enough."

**Scenario E — Multiple factors combine nontrivially.** No single correlation is dominant. Specific layer-by-layer pattern needs to be memorized and designed around. More complex retrofit design.

**Scenario F — The surprises.** E.g., layers near global-attention boundaries behave specially (the layer-3, layer-8, layer-13 PLE spikes hint at this — each is one layer before a global). If these boundary layers are uniquely fragile or uniquely robust, that's its own design constraint.

**Scenario G — `once` helps systematically in the PLE-inert corridor and hurts elsewhere.** Confirms our hypothesis from round 2b. Retrofit uses `once` only in blocks placed inside the inert corridor.

**Scenario H — `once` doesn't follow any clean pattern.** Drop `once` from the retrofit plan; complexity not worth it.

## Sanity watch

Layer 4 is a global-attention layer and is non-KV-consumer. It's one of the few non-sliding layers we haven't measured. If looping tolerance at layer 4 is wildly different from both layer 5 (sliding) and layer 17 (sliding consumer), that's a strong argument for attention-type being the dominant factor.

Also watch layer 33: it's the sliding layer with the second-highest PLE importance (0.74 NLL diff) and is immediately before the final global. If this layer blows up catastrophically under looping, it confirms PLE-important layers can't be looped. If it's fine, PLE importance and looping tolerance are decoupled.

## Output format

`results_round2c_full_map.json`:

```json
{
  "config": { ... },
  "unmodified": { ... },
  "layer_metadata": [
    {"layer": 0, "attention_type": "sliding_attention", "is_kv_consumer": false, "ple_importance": 0.0215},
    ...
  ],
  "regression_checks": {
    "all_r1_match_baseline": true,
    "max_rel_drift": ...
  },
  "cells": [
    {"layer": 0, "ple_mode": "vanilla", "r": 2, "mean_nll": ..., "ppl": ...},
    ...
  ],
  "analysis": {
    "correlations_r8_vanilla": {
      "ple_importance": {"pearson": ..., "spearman": ...},
      "attention_type": {"pearson": ..., "spearman": ...},
      "is_kv_consumer": {"pearson": ..., "spearman": ...},
      "layer_index": {"pearson": ..., "spearman": ...}
    },
    "top_loopable": [...],
    "top_fragile": [...],
    "once_helps_most": [...],
    "once_hurts_most": [...]
  }
}
```

## Deliverables

1. Updated `ple_sanity_check.py` with `--mode=full-looping-map`.
2. `results_round2c_full_map.json` with 210 cells + metadata + analysis.
3. Printed summary tables (vanilla across layers, once-vs-vanilla delta, correlations, top-5s).
4. A one-paragraph interpretation stating which scenario (A–H) the data supports and the implied retrofit design rule.

## Report back

Paste:
- The vanilla summary table (35 rows).
- The correlation numbers (four correlations with Pearson + Spearman).
- The top-5 loopable / fragile / once-helps / once-hurts tables.
- The interpretation paragraph.

Do not proceed to block-level experiments or any training. Round 3 plan will be written based on the scenario we land in.