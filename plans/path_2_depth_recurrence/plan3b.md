# Gemma 4 E2B — Round 3b: Block-Looping Probe

## Context

Probe progression so far:

| Round | Unit | r=8 valley width | Conclusion |
|---|---|---|---|
| 1, 2a | Single layer 17 | 1 layer | Baseline: 3.4× ppl at r=8 |
| 2c | Single layer, all 35 | 5 layers (15–19) | Valley is narrow; rest fragile |
| 3a | Pairs of contiguous layers | ~4 pairs (15–19) | Pair-looping doesn't rescue the rest |

The pair-looping result is ambiguous for what the retrofit actually cares about. Two layers isn't enough for "within-block re-processing" dynamics to dominate — inside a 2-layer loop, only one layer sees the output of another looped layer. The scenarios that remain live:

- **Scenario X: "Valley is a hard ceiling."** Only layers ~15–19 can ever be looped, regardless of block size. Retrofit must use a tight 4–5 layer recurrent block.
- **Scenario Y: "Valley is a floor."** A 6–10 layer block anchored in the valley and extending outward may behave much better than single-layer/pair probes suggested, because within-block re-processing dominates at that scale.

Round 3b directly tests which scenario holds by looping actual candidate blocks of 4–10 layers, measuring perplexity the same way as prior rounds. The result determines whether we design a narrow or wider recurrent block for training.

## Scope

**In scope:**
- Extend `ple_sanity_check.py` with `--mode=block-looping`.
- Test 6 candidate blocks (listed below) × 3 r values × 1 PLE mode = **18 cells**.
- PLE mode: **vanilla only**. `once` was not a systematic win in prior rounds.
- Each block is a contiguous range of layers looped together as a unit r times.
- Reuse all existing infrastructure (use_cache=False, 50 Wikitext sequences, bf16, positional-arg PLE handling).

**Out of scope:**
- No training.
- No downstream tasks (Wikitext-2 perplexity only).
- No non-contiguous blocks.
- No `once` PLE variant.
- No exhaustive block sweep — 6 hand-picked candidates, not all possible blocks.

## Candidate blocks

Chosen to discriminate cleanly between scenarios X and Y. Each block is named by (start, end) inclusive.

| Block | Layers | Width | Rationale |
|---|---|---|---|
| **A: valley-core** | 15–19 | 5 | The valley as-is. Lower bound on block-looping tolerance. |
| **B: valley-narrow** | 15–18 | 4 | Even tighter, avoids the global layer at 19. |
| **C: valley-extend-down** | 12–19 | 8 | Extend into fragile territory below. Tests scenario Y in the "upper-fragile" direction. |
| **D: valley-extend-up** | 15–22 | 8 | Extend into fragile territory above (PLE-inert but fragile in 2c). Tests scenario Y in the "lower-fragile" direction. |
| **E: valley-centered** | 13–22 | 10 | Symmetrically extend in both directions. Paper-style 10-layer block. |
| **F: late-block** | 25–32 | 8 | Control: a block entirely in the PLE-inert but fragile region. If scenario Y is real, this might also work — if not, confirms the valley is privileged. |

Blocks A and B are lower-bound anchors (close to what single/pair looping already showed). Blocks C, D, E directly test the cascade hypothesis at meaningful scale. Block F is a control: if the PLE-inert corridor is indeed loopable once the block is big enough, F should work. If F stays broken, the valley has something special beyond just "less PLE."

## Implementation

### The block-looping hook

This is the trickiest hook we've built. We need to replace a *range* of decoder layers so that the range loops as a unit, while correctly handling:

- Each layer's `per_layer_input` (PLE) tensor, which is different per layer.
- Positional args vs kwargs (already handled in round 2a addendum 2).
- The outer model's sequential layer iteration (which would otherwise run each replaced layer separately).

**Recommended approach — monkey-patch the outer model forward.** The cleanest path is to modify `Gemma4TextModel.forward` (or whichever class runs the layer-by-layer loop) to detect the block range and loop it as a unit. Something like:

```python
def patch_outer_forward(model, block_start, block_end, r):
    """
    Replace the text model's forward so that layers [block_start..block_end]
    are looped as a unit r times instead of being applied once each.
    """
    text_model = find_text_model(model)  # navigate through multimodal wrapper
    orig_forward = text_model.forward

    def patched(self, *args, **kwargs):
        # This replicates orig_forward but replaces the layer loop.
        # Simplest implementation: call orig_forward with a monkey-patched
        # layer list where the block is replaced by a single "block module"
        # that applies the loop.
        ...
    
    text_model.forward = types.MethodType(patched, text_model)
    return orig_forward  # for restoration
```

**Cleaner alternative** if the above is too invasive: replace each layer in the block with a "sentinel" forward that does the right thing based on iteration state. Specifically:

```python
def block_looped_forward(block_layers, per_layer_input_cache, r, ple_kwarg):
    """
    Only layer block_layers[0] does actual work -- it runs the full 
    pair-loop internally. Layers block_layers[1:] are replaced with 
    passthroughs so the outer model's sequential application doesn't 
    double-apply them.
    """
    captured_ples = {}  # populated by pre-hooks on each block layer
    
    def looped_first(hidden_states, *args, **kwargs):
        # Extract this layer's PLE (positional arg 0 for Gemma 4).
        ple_first = args[0] if args else kwargs.get(ple_kwarg)
        other_args = args[1:] if args else ()
        other_kwargs = {k: v for k, v in kwargs.items() if k != ple_kwarg}
        
        x = hidden_states
        for iteration in range(r):
            # Apply each layer in the block, with its own PLE.
            for i, layer in enumerate(block_layers):
                ple_i = ple_first if i == 0 else captured_ples[i]
                # Call the layer's original forward (cached before patching)
                out = orig_forwards[i](x, ple_i, *other_args, **other_kwargs)
                x = out[0] if isinstance(out, tuple) else out
        return (x,) if isinstance(out, tuple) else x
    
    def passthrough(hidden_states, *args, **kwargs):
        # Layer already applied inside looped_first.
        return (hidden_states,) if needs_tuple else hidden_states
    
    return looped_first, passthrough
```

The trick for capturing PLEs for block_layers[1..]: install a forward pre-hook on each that captures `kwargs[ple_kwarg]` (or `args[0]`) into the shared `captured_ples` dict. The pre-hooks fire when the outer model's forward first calls those layers, so PLEs get captured on the first pass. But since we've turned block_layers[1..] into passthroughs, their "first call" is when the outer model tries to apply them — which happens *after* looped_first has already done the real work.

**There's a timing ordering issue here that needs care:**

- On the first forward pass, the outer model calls: `layer[block_start](x, ple_first, ...)` → our `looped_first` runs. At this moment, we need the PLEs for block_layers[1..], but the outer model hasn't called them yet and hasn't computed their PLEs.
- Solution: the outer model's `Gemma4TextModel.forward` almost certainly pre-computes *all* PLEs up-front (as a big tensor), then indexes into it for each layer. If so, we can intercept that pre-computed tensor and extract the per-layer slices we need at hook-install time rather than at call time.

### Step 0 (REQUIRED before coding): inspect how PLEs are computed

Before writing the hook, read the `Gemma4TextModel.forward` (or equivalent) source and determine:

1. Where are PLE tensors computed? (Expected: early in the outer forward, before the layer loop.)
2. Are they stored as a single big tensor `[num_layers, batch, seq, hidden]`, or separately per layer?
3. Can we access the precomputed PLE tensor from the hook, or must we recompute?

Based on the signature we already know (`per_layer_input` is a positional arg that the outer forward passes to each layer), the most likely structure is: outer forward computes a big PLE tensor, slices it for each layer, passes the slice positionally. If so, the cleanest hook is:

```python
def capture_all_ples(text_model_forward):
    """Wrap the outer forward to capture PLE tensor on first call."""
    captured = {}
    orig = text_model_forward
    def wrapped(*args, **kwargs):
        # Call orig but intercept the PLE computation.
        # Implementation detail: may need to monkey-patch a sub-method
        # like self._compute_per_layer_inputs or similar.
        ...
```

If the inspection reveals a messier structure, describe what you found and we'll adjust before implementing.

### Fallback if the hook gets too complex

Worst case, we can just rewrite `Gemma4TextModel.forward` as a new method that does the block loop. Copy the source, modify the layer loop, install the new forward. This is heavy-handed but definitely works. Use as fallback if the pre-hook approach has subtle timing issues.

### Regression check

At r=1 with any block, perplexity must match the unmodified baseline bitwise. "Loop block once" = "apply block layers sequentially once" = normal forward pass. Run this check for at least 2 blocks (one small, one large) before the main sweep.

## Experiment design

For each of the 6 blocks × 3 r values, compute perplexity on the standard 50 Wikitext sequences.

Expected runtime: each cell is more expensive than single-layer looping (block contains multiple layers and we loop the whole thing), but the grid is small. Estimated total: 20–30 minutes.

## Analysis

### Comparison against single-layer baseline

For each block, also compute:
- `block_r8_ppl`
- `max_single_ppl_in_block` = worst single-layer ppl at r=8 over layers in the block (from round 2c)
- `min_single_ppl_in_block` = best single-layer ppl at r=8 over layers in the block
- `improvement_over_worst` = max_single_ppl / block_ppl (>1 = block rescues the worst layer)

A block that "rescues" its weakest layer is direct evidence for the cascade hypothesis.

### Table to print

```
=== Block-looping results at r=8 ===
Block     Layers    Width  block_ppl  worst_single_in_block  ratio
A         15-19     5      ???        ???                    ???
B         15-18     4      ???        ???                    ???
C         12-19     8      ???        ???                    ???
D         15-22     8      ???        ???                    ???
E         13-22     10     ???        ???                    ???
F         25-32     8      ???        ???                    ???
```

Also include r=2 and r=4 numbers in the JSON output (for understanding compounding) but the r=8 table is the main deliverable.

## Interpretation buckets

**Bucket 1 — Valley blocks work, extensions don't.**
Block A (15–19) at r=8 is within ~5× baseline; blocks C, D, E all degrade by 50×+. This is **scenario X confirmed**: valley is a hard ceiling. Retrofit uses a 4–5 layer recurrent block. Accept this constraint and move to training design.

**Bucket 2 — Valley-anchored extensions work; distant blocks don't.**
Blocks A, C, D, E are all within ~10× baseline; block F (25–32) is catastrophic. This is **scenario Y confirmed for valley-anchored blocks** but not for distant regions. Retrofit uses the widest valley-anchored block that still holds (likely E: 13–22). Good outcome — more compute per recurrence than scenario X.

**Bucket 3 — Even the distant block works.**
All 6 blocks within ~10× baseline, including F. The cascade hypothesis is fully confirmed — within-block re-processing dominates at this scale regardless of location. Retrofit has maximum flexibility for block placement. Strongest possible green light.

**Bucket 4 — Nothing works.**
Even block A degrades 50×+ at r=8. This contradicts all prior single/pair results in the valley — likely a hook bug. Stop and debug. Expected to be rare because blocks A and B are so close to the single/pair setups we've already validated.

**Bucket 5 — Unexpected pattern.**
E.g., block C (extending down) works but D (extending up) doesn't; or block F is better than E; or blocks show non-monotone behavior in width. This means something structural we haven't identified is at play. Pause, paste results, and we discuss before further plans.

## Deliverables

1. Updated `ple_sanity_check.py` with `--mode=block-looping`.
2. `results_round3b_blocks.json` with all 18 cells + comparison analytics.
3. Printed block-comparison table at r=8.
4. A one-paragraph interpretation stating which bucket the data lands in and the implied retrofit block design.
5. The hook inspection notes from step 0 (how PLEs are computed in the outer forward).

## Report back

Paste:
- The step-0 inspection notes (how PLEs are plumbed at the outer-model level).
- The 6-block comparison table.
- The interpretation paragraph identifying the scenario bucket.

Do not proceed to training or to more blocks without checking in. Depending on the bucket, the next plan is either retrofit training design (Buckets 1, 2, 3) or debugging (Bucket 4) or a discussion (Bucket 5).