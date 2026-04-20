# Gemma 4 E2B — PLE Probe Round 2b

## Context

Round 2a (after fixing the positional-argument bug) showed:

| mode | r=4 | r=8 |
|---|---|---|
| vanilla | 1.63× ppl | 3.40× ppl |
| scaled | 1.63× ppl | 3.40× ppl |
| once | 1.48× ppl | 2.92× ppl |

At layer 17 specifically: `once` (apply PLE on iteration 0 only, skip thereafter) is modestly better than `vanilla` (apply every iteration); `scaled` is indistinguishable from vanilla. The zero-PLE diagnostic at layer 17 showed PLE contributes ~0.4% to perplexity — small but nonzero.

**What this does not tell us:**
- Whether `once` is also better than `vanilla` at other layers. Layer 17 is one point in a 35-layer model.
- Whether each layer's PLE is equally load-bearing, or whether some layers' PLE tables are effectively inert while others carry substantial signal.
- How PLE policy interacts with local-vs-global attention, or with KV sharing.

This round answers two questions:

1. **PLE importance scan**: for each layer, how much does that layer's PLE actually contribute to perplexity? Cheap to measure — vanilla vs zero at r=1, across all 35 layers.
2. **Layer-location × PLE policy sweep**: does the once-beats-vanilla pattern hold at different layer depths? Grid of 3 layer locations × 2 PLE modes × 2 r values.

## Scope

**In scope:**
- Add two new modes to `ple_sanity_check.py`:
  - `--mode=ple-importance-scan`: vanilla vs zero, r=1, for every layer 0..34. 70 cells (35 × 2).
  - `--mode=layer-location`: 3 layers × {vanilla, once} × {r=4, r=8}. 12 cells.
- Use the existing fixed `make_looped_forward_ple` hook with positional-arg handling.
- Results saved to `results_round2b_importance.json` and `results_round2b_location.json`.

**Out of scope:**
- No training.
- No `scaled` mode (round 2a confirmed it's indistinguishable from vanilla — don't waste cells).
- No block-of-layers looping. Still one layer at a time.
- No new metrics beyond perplexity on the same 50 Wikitext sequences.
- No plotting (print tables only).

## Layer choices

For the layer-location sweep, pick three layers chosen to sample different regimes of E2B's architecture:

- **Layer 5**: early. Mostly processes raw token content, closer to the embedding.
- **Layer 17**: middle. Already validated by round 2a — use as the anchor point for comparison with prior data.
- **Layer 28**: late. Likely near or in the KV-consumer region (the model card mentions "the last N layers reuse KV states from earlier layers").

These specific indices are chosen without perfect information about E2B's exact L/G interleaving pattern or KV-sharing map — that would require reading the config more carefully, which can wait. Three well-separated indices is enough for a first look.

If the implementer wants to also log each chosen layer's attention type (local vs global) and whether it's a KV consumer, that's a bonus — but only if it's easy to extract from the loaded model without source diving.

## Experiment 1 — PLE importance scan

### Purpose

Quantify how much PLE at each layer contributes to perplexity. If layer l's PLE contributes ~0%, then "once" and "vanilla" would be identical there (PLE is inert; re-injecting it has no effect because it's a no-op). If layer l's PLE contributes a lot, then policy choice at layer l matters proportionally.

This gives us a per-layer profile that tells us where to be careful with PLE policy when designing the recurrent block.

### Grid

- `layer ∈ {0, 1, 2, ..., 34}` — all 35 layers.
- `ple_mode ∈ {vanilla, zero}`.
- `r = 1` (no looping — we're measuring PLE importance, not recurrence tolerance).

Total: 70 cells. Each cell is a single forward pass over 50 Wikitext sequences. Estimated runtime: ~10–15 minutes.

### Implementation

Reuse `make_looped_forward_ple` with `r=1`. For each layer index, install the hook on that layer with `ple_mode="vanilla"`, measure perplexity, then with `ple_mode="zero"`, measure again. Restore original forward. Move to next layer.

Report per layer: `ppl_vanilla`, `ppl_zero`, `nll_diff = nll_zero - nll_vanilla`. A positive `nll_diff` means zeroing PLE hurt perplexity (PLE was useful); near-zero means PLE at that layer is approximately inert.

### Output table

```
=== PLE importance scan (r=1, all 35 layers) ===
layer    ppl_vanilla   ppl_zero     nll_diff
  0         12.XX        12.XX       +0.XXXX
  1         12.XX        12.XX       +0.XXXX
  ...
 34         12.XX        12.XX       +0.XXXX

Top 5 most PLE-important layers (highest nll_diff):
  layer XX: +0.XXXX
  ...

Top 5 least PLE-important layers (lowest nll_diff):
  layer XX: +0.XXXX
  ...
```

Also save as JSON with each layer's three numbers.

## Experiment 2 — Layer-location × PLE policy

### Purpose

Test whether the "once beats vanilla" finding at layer 17 generalizes across layer depths. If once beats vanilla at all three locations, we commit to `once` as the retrofit default with high confidence. If once loses at some locations, PLE policy is layer-dependent and we need to think more carefully.

### Grid

- `layer ∈ {5, 17, 28}`
- `ple_mode ∈ {vanilla, once}`
- `r ∈ {4, 8}`

Total: 12 cells. Plus 3 r=1 regression cells (one per layer, vanilla) that must match the unmodified baseline.

Estimated runtime: ~5 minutes.

### Regression checks

Before collecting sweep data:

1. At each of the three target layers, run vanilla at r=1. Must match the round-1 unmodified baseline (ppl ≈ 12.53).
2. At each of the three target layers, run once at r=1. Must equal vanilla at r=1 (bitwise — because at r=1, `once` uses scale=1.0 on iteration 0 and never gets to later iterations).

If any regression fails, stop and debug.

### Output table

```
=== Layer-location × PLE policy sweep ===
layer    r     vanilla_ppl   once_ppl    once/vanilla ratio
  5      4        XX.XX        XX.XX          X.XXX
  5      8        XX.XX        XX.XX          X.XXX
 17      4        20.48        18.53          0.905    (reference from round 2a)
 17      8        42.59        36.63          0.860
 28      4        XX.XX        XX.XX          X.XXX
 28      8        XX.XX        XX.XX          X.XXX
```

Also compute, for each (layer, r) pair, the absolute NLL improvement `nll_vanilla - nll_once`.

## Interpretation — commit before looking

**Experiment 1 (importance scan):**

- **Bulk of layers have small `nll_diff` (<0.005); a few stand out high.** Normal expected pattern. The high-importance layers are the ones where PLE policy will matter most. Note which layer ranges are high-importance — this constrains where the recurrent block can go safely.
- **`nll_diff` is ~uniform across all layers.** PLE contributes similarly everywhere; no particular region to avoid. Once/vanilla tradeoff is roughly constant across layer choices.
- **A specific layer has dramatically higher `nll_diff` than all others.** Investigate. May be worth excluding that layer from the recurrent block regardless of policy.
- **Some layers have near-zero `nll_diff`.** Good news — PLE at those layers is approximately inert. Once and vanilla will be indistinguishable there. Those are safe layers to loop.

**Experiment 2 (layer-location × policy):**

- **Once beats vanilla at all three layers, by roughly the same margin.** Strong evidence to commit to once. Retrofit default set.
- **Once beats vanilla at some layers but loses at others.** Policy is layer-dependent. Cross-reference with Experiment 1: does the pattern correlate with PLE importance? If yes, rule becomes "use once where PLE importance is low, vanilla where it's high." If no, something else is going on and we investigate.
- **Once ties vanilla everywhere.** Odd given round 2a showed a clear gap at layer 17. Would suggest the round 2a finding was layer-17-specific, and we should default to vanilla (simpler).
- **Once loses to vanilla at all three layers.** Round 2a's signal was noise. Default to vanilla.
- **Any layer produces NaN or ratio > 10×.** Stop, something is wrong with that layer — likely KV-sharing or attention-type interaction. Report and we discuss.

## Cross-experiment synthesis

After both experiments, provide a one-paragraph synthesis answering:

1. Where in the 35-layer stack does PLE actually carry meaningful signal? (from Exp 1)
2. Does the "once beats vanilla" pattern from round 2a hold at other layer depths? (from Exp 2)
3. Based on the combined evidence, what PLE policy should the retrofit use as default, and are there any layers to avoid placing the recurrent block over?

## Implementation notes

- Both experiments share the same fixed `make_looped_forward_ple` from round 2a addendum 2. Do not modify the hook.
- For the importance scan, you'll iterate over all 35 layers. Make sure each iteration cleanly restores the layer's original forward before moving on, otherwise later measurements will compound.
- Consider adding a `--layers` arg accepting a comma-separated list to make both experiments configurable from the same mode. But don't over-engineer — two separate modes is fine.

## Deliverables

1. `ple_sanity_check.py` updated with `--mode=ple-importance-scan` and `--mode=layer-location` (or a single combined mode with sub-flags — pick whichever is cleaner).
2. `results_round2b_importance.json`: per-layer vanilla vs zero results.
3. `results_round2b_location.json`: the 3 × 2 × 2 sweep.
4. Two printed tables as shown above.
5. One-paragraph cross-experiment synthesis.

## Report back

Paste:
- The importance-scan printed table (or at least the summary rows: top 5 high, top 5 low).
- The layer-location sweep table.
- The synthesis paragraph.

Do not proceed to block-level experiments, KV-sharing tests, or any training work.