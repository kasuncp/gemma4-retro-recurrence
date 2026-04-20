# Gemma 4 E2B — Round 3a: Pair-of-Layers Looping Probe

## Context

Rounds 1–2c established single-layer looping tolerance across all 35 layers of Gemma 4 E2B. Results summary:

- Only layers 15–19 tolerate r=8 looping within 10× baseline perplexity.
- Most other layers catastrophically degrade (ppl 100× to 10⁹× baseline).
- No single architectural factor (PLE, attention type, KV sharing, depth) explains looping tolerance; correlations are all weak.
- `once` PLE mode does not systematically beat `vanilla`. Drop it.

**The key unresolved question:** were the catastrophic single-layer results caused by the layer itself being non-loopable, or by the *downstream cascade* — 34 layers after the loop processing an out-of-distribution hidden state? When we loop one layer, 34 other layers compound the error. When we loop a *block*, inside-block layers re-process each other's outputs, which is what they'd do in the real retrofit.

Before committing compute to full block looping (Round 3b), this probe checks whether looping **pairs** of contiguous layers degrades differently from looping single layers. It's a middle ground:
- Bigger than single-layer (within-block dynamics start to matter).
- Much smaller than a full retrofit block (stays cheap).
- Direct apples-to-apples comparison: pair (L, L+1) vs single L.

If pair-looping degrades markedly more gracefully than single-layer looping, we have strong evidence the "cascade hypothesis" is right and block looping will work. If pair-looping doesn't improve, something structural is fighting us and we need to rethink before investing in training.

## Scope

**In scope:**
- Extend `ple_sanity_check.py` with `--mode=pair-looping-map`.
- Grid: 34 pair starting positions × 3 r values × 1 PLE mode = **102 cells**.
- Pairs = (L, L+1) for L ∈ {0, 1, ..., 33}. Loops pair-as-a-unit r times.
- PLE mode: **vanilla only**. `once` added nothing useful in the single-layer data.
- Reuse all existing infrastructure (use_cache=False, 50 Wikitext sequences, bf16).
- r ∈ {2, 4, 8}.
- Compare directly against round 2c vanilla single-layer results.

**Out of scope:**
- No 3+ layer blocks yet. That's Round 3b, informed by this.
- No training.
- No new metrics — Wikitext-2 perplexity only.
- No `once` variant. We have the layer-by-layer profile already.

## Implementation

### The pair-looping hook

The single-layer hook replaced one layer's `forward` with a function that loops it r times. For a pair, we need to loop *both layers together* — on each iteration, run layer L, then layer L+1, then feed that output back as input to layer L again.

```python
def make_pair_looped_forward(orig_forward_L, orig_forward_L1, r, ple_kwarg):
    """
    Loop two contiguous decoder layers (L, L+1) as a unit, r times.
    
    Approach: replace layer L's forward to do the full pair-loop, 
    and replace layer L+1's forward with a pass-through no-op so the 
    outer model's forward loop doesn't double-apply layer L+1.
    
    The per-layer-input tensors for L and L+1 are captured from the 
    first call's args/kwargs and reused on subsequent iterations at 
    full strength (vanilla PLE policy).
    """
    def looped_L(hidden_states, *args, **kwargs):
        # Extract PLE tensor (positional arg 0 by Gemma 4 convention).
        # The outer model calls this with per_layer_input for layer L.
        ple_L = kwargs.get(ple_kwarg) if ple_kwarg in kwargs else (args[0] if args else None)
        # We DO NOT have per_layer_input for layer L+1 here -- the outer 
        # model will call layer L+1's forward separately. But we've 
        # neutered layer L+1's forward, so we need to grab its PLE 
        # some other way.
        # 
        # CRITICAL: this is the tricky part of the implementation.
        # See "PLE plumbing for L+1" below.
        
        x = hidden_states
        for _ in range(r):
            out_L = orig_forward_L(x, *args, **kwargs)
            x = out_L[0] if isinstance(out_L, tuple) else out_L
            # Now apply layer L+1 with ITS per_layer_input.
            out_L1 = orig_forward_L1(x, ple_L1_tensor, *other_L1_args, **other_L1_kwargs)
            x = out_L1[0] if isinstance(out_L1, tuple) else out_L1
        
        # Return in the format layer L's original forward returned.
        return (x,) if isinstance(out_L, tuple) else x
    
    def passthrough_L1(hidden_states, *args, **kwargs):
        # Layer L+1's computation was already done inside looped_L.
        # Return identity so the outer model's sequential application 
        # doesn't apply layer L+1 a second time.
        return hidden_states if not isinstance(hidden_states, tuple) else hidden_states
    
    return looped_L, passthrough_L1
```

### The PLE plumbing challenge

This is the one subtle implementation point. The outer model's forward does roughly:

```python
for i, layer in enumerate(self.layers):
    hidden_states = layer(hidden_states, per_layer_input=self.compute_ple(i), ...)
```

So layer L gets `per_layer_input` for index L, and layer L+1 gets `per_layer_input` for index L+1. When we install our pair-loop hook at layer L, it receives PLE for layer L in its args. But it *also* needs PLE for layer L+1 to correctly call layer L+1's forward on each iteration.

**Solution:** cache the PLE tensor for layer L+1 at hook-install time.

```python
def install_pair_hook(decoder_layers, L, r, ple_kwarg, model):
    """
    Pre-compute the PLE tensor that layer L+1 will need, then install
    the pair-loop hook.
    
    Tricky bit: per-layer PLE is usually computed fresh each forward 
    pass from input_ids. We can't fully precompute it here without 
    running a forward. Instead, we install a forward pre-hook on 
    layer L+1 that CAPTURES its per_layer_input, and a forward hook 
    on layer L that USES the last captured value.
    """
    captured = {}
    
    def capture_hook(module, args, kwargs):
        # Capture layer L+1's per_layer_input whenever it's called.
        if ple_kwarg in kwargs:
            captured['ple_L1'] = kwargs[ple_kwarg]
            captured['args_L1'] = args
            captured['kwargs_L1'] = {k: v for k, v in kwargs.items() if k != ple_kwarg}
        elif args:
            captured['ple_L1'] = args[0]
            captured['args_L1'] = args[1:]
            captured['kwargs_L1'] = kwargs
        return args, kwargs
    
    # Install the pre-hook on L+1 so we grab its intended per_layer_input
    # on the first forward pass.
    # Then our pair-loop hook on L can use captured['ple_L1'] when 
    # calling orig_forward_L1 inside the loop.
    ...
```

**Simpler alternative:** skip all this and just do the pair-loop at a *higher level* — replace the outer model's forward entirely for the relevant layer range. This is more invasive but avoids the PLE plumbing puzzle.

**Recommended approach:** start with the pre-hook method (capture PLE for L+1). If it gets too hairy, fall back to monkey-patching the outer `Gemma4TextModel.forward` to do the pair-loop at the right point in its layer loop.

### Regression check

At r=1 with pair-looping at (L, L+1), perplexity should match the unmodified baseline exactly. Running "loop the pair once" = "run the pair once" = "run both layers in sequence once," which is just the normal forward pass. Any drift means the hook is broken.

Run this check for 3 representative pairs (e.g., (5,6), (17,18), (28,29)) before the full sweep.

### Main sweep

For each L ∈ {0, ..., 33}, at each r ∈ {2, 4, 8}, compute perplexity with pair-looping. 102 cells total. Estimated runtime: slightly more than round 2c (each cell is a bit more expensive because each iteration does 2 layers of compute), maybe 45–60 minutes.

## Analysis

### Main comparison table

```
=== Pair-looping vs single-layer (r=8, vanilla) ===
   pair      single_L     single_L+1    pair_ppl     improvement
  (0,1)     1099           1129          ???          ???
  (1,2)     1129           1680          ???          ???
  ...
  (15,16)     35             39          ???          ???
  ...
```

For each pair, compute:
- `pair_ppl` at r=8.
- `min_single_ppl` = min(single_L_ppl, single_L+1_ppl) at r=8 (from round 2c).
- `improvement` = `min_single_ppl / pair_ppl`. Values > 1 mean pair is better.
- `geometric_mean_single_ppl` = sqrt(single_L_ppl × single_L+1_ppl).

### Visualizations

After the run, produce a plot similar to round 2c fig 3 (log-ppl vs layer index) but with pair-looping results overlaid on single-layer results. The comparison will be visually obvious: if pair-looping is in a dramatically different (lower) curve, the cascade hypothesis is confirmed.

### Correlations to recompute

For pair-looping r=8 results, recompute correlations with:
- PLE importance at pair (sum of L and L+1).
- Whether pair crosses a global-attention boundary.
- Whether pair straddles the KV-consumer boundary (layer 15 is the first consumer; pair (14,15) is the boundary).
- Pair midpoint index.

## Interpretation buckets

**Bucket 1 — Pair-looping dramatically improves over single-layer in most of the model.**
Most pairs at r=8 land within ~10× baseline even when their constituent single layers were at 100×+. The cascade hypothesis is confirmed. Round 3b (proper block looping) is a strong green light. Retrofit design continues.

**Bucket 2 — Pair-looping helps some regions but not others.**
A subset of pairs (likely spanning layers 15–22 or so) improve dramatically; other regions remain catastrophic. This tells us block size matters *within* certain regions. Retrofit design focuses on those regions.

**Bucket 3 — Pair-looping is essentially identical to single-layer looping.**
The cascade hypothesis is wrong. Gemma 4 E2B is fundamentally fragile to looping beyond the narrow 15–19 window, regardless of block size. Retrofit would need architectural surgery (not just healing training). Consider whether to:
 - Accept a very narrow recurrent block (only 5 layers wide).
 - Pivot approach (e.g. train a small auxiliary recurrent network rather than looping Gemma's own layers).
 - Reconsider model choice.

**Bucket 4 — Pair-looping is systematically WORSE than single-layer looping.**
Unexpected. Would suggest that pairs introduce coherent errors that single layers don't. Stop and investigate — probably a hook bug.

## Output

`results_round3a_pair_looping.json`:

```json
{
  "config": { ... },
  "unmodified": { "ppl": 12.54 },
  "regression_checks": { "pairs_tested": [...], "all_pass": true },
  "pair_cells": [
    {"L": 0, "r": 2, "ppl": ...},
    {"L": 0, "r": 4, "ppl": ...},
    {"L": 0, "r": 8, "ppl": ...},
    ...
  ],
  "comparison": [
    {"L": 0, "L1": 1, "pair_r8_ppl": ..., "single_L_r8_ppl": ..., "single_L1_r8_ppl": ..., "improvement": ...},
    ...
  ],
  "analysis": {
    "n_pairs_within_10x_baseline": ...,
    "expansion_vs_single_layer": ...,  // e.g. "valley grew from 5 layers to N"
    "correlations_r8": { ... }
  }
}
```

Plus a printed table showing the 34 pairs with pair_ppl, single-layer ppl of both components, and the improvement ratio.

## Deliverables

1. Updated `ple_sanity_check.py` with `--mode=pair-looping-map`.
2. `results_round3a_pair_looping.json`.
3. Printed comparison table (34 pairs).
4. Brief numerical summary: how many pairs are within 5×, 10×, 50× baseline.
5. One paragraph identifying which interpretation bucket the data supports.

## Report back

Paste:
- The comparison table (34 pairs).
- Counts: pairs within {5×, 10×, 50×, 100×} baseline at r=8.
- The interpretation paragraph.

Do not proceed to 3+ layer blocks or any training. The Round 3b plan will be written based on this bucket.