# Gemma 4 E2B — PLE Probe Round 2a (PLE variants only)

## Context

The 30-minute sanity check passed cleanly. Looping a single middle decoder layer (layer 17) `r` times in pretrained Gemma 4 E2B produces smooth log-linear perplexity degradation:

| r | perplexity | ratio vs r=1 |
|---|---|---|
| 1 | 12.53 | 1.00× |
| 2 | 13.45 | 1.07× |
| 4 | 20.48 | 1.63× |
| 8 | 42.57 | 3.40× |

Hook drift was exactly 0.0. No NaN, no cliff. PLE is a *friction*, not a *wall*. Green light to probe further before committing real training compute.

This round answers ONE question: **which PLE injection policy tolerates looping best?**

The round-1 sanity check applied PLE on every iteration at full strength (call this "vanilla"). There are two natural alternatives worth testing:

- **scaled**: apply PLE every iteration but multiply by `1/r`, so total injection budget matches a single pass.
- **once**: apply PLE only on iteration 0, skip it for iterations 1..r-1.

Whichever wins at high `r` is the PLE strategy we'll build the full retrofit around.

A follow-up probe (Round 2b, to be specified after this one's results come in) will test layer-location sensitivity. Do NOT start on that here.

## Scope — what this probe is and isn't

**In scope:**
- Extend the existing `ple_sanity_check.py`. Do NOT start over. Reuse its loading, eval, and hook infrastructure.
- Add one new experiment mode: `--mode=ple-variants`.
- Hold `TARGET_LAYER=17` fixed.
- Compare vanilla / scaled / once at `r ∈ {1, 4, 8}`. Nine cells total.
- Output JSON + a printed table.

**Explicitly out of scope:**
- No training. No weight updates.
- No layer-location experiment. That's Round 2b.
- No block-of-layers looping. Still one layer.
- No downstream tasks (still Wikitext-2 perplexity only).
- No architectural variants beyond the three PLE policies. Do NOT invent new ones.
- No plotting.

If this takes more than ~100 lines added to the existing script, something has gone wrong — stop and ask.

## The critical implementation challenge: intercepting PLE

The round-1 sanity check didn't touch PLE — it just wrapped the layer's `forward` and called it `r` times. That works because "vanilla" behaviour is "apply PLE every time `forward` is called," which is what happens when you loop `forward`.

For this probe we need finer control: on each iteration, *decide* whether PLE is applied and at what scale. This requires understanding *where* PLE is applied inside Gemma 4's decoder layer.

### Step 0 (REQUIRED before coding) — Inspect the Gemma 4 source

Locate the Gemma 4 modeling file in the installed `transformers` version:

```bash
python -c "import transformers, os; print(os.path.dirname(transformers.__file__))"
```

Then find the Gemma 4 modeling file under `models/gemma4/` (or whatever Gemma 4 is called in that version — check `transformers/models/` for a gemma-related directory). Read the `Gemma4DecoderLayer.forward` method (or equivalent class name).

Look for the line that applies PLE — it will add a per-layer-indexed embedding vector to the residual stream. Candidates:

```python
hidden_states = hidden_states + self.per_layer_embedding(input_ids)
# or
hidden_states = hidden_states + per_layer_input
# or
hidden_states = hidden_states + self.ple_proj(ple_tensor)
```

**Before writing any code, document:**
1. Exact file path.
2. Exact line(s) applying PLE.
3. Whether PLE is computed inside the layer's forward, or passed in as a kwarg/positional arg.
4. Whether PLE is scalar-multiplied anywhere already (e.g. a constant scaling factor) — if so, we need to multiply our `ple_scale` in addition, not replace it.

This inspection determines which strategy below applies.

### Strategy A — PLE applied inside `forward` (expected case)

If PLE is added inline via something like `h = h + ple_term`, monkey-patch the decoder layer class's `forward` method to take an extra kwarg `ple_scale: float = 1.0` that multiplies the PLE term. Example patched line:

```python
hidden_states = hidden_states + ple_scale * self.per_layer_embedding(input_ids)
```

The default `ple_scale=1.0` means all unmodified calls behave identically to the original. Only our looped hook passes a different value.

### Strategy B — PLE applied in outer model forward

If PLE is added in `Gemma4Model.forward` outside the per-layer forward (e.g., the outer loop does `h = h + ple_all[i]; h = layer_i(h)`), then our round-1 hook fundamentally could not control PLE. In that case, **stop coding and flag this immediately**, because:

- Round 1's "vanilla" was actually the "once" variant in disguise (PLE injected once before the loop, then not again during looped iterations).
- The 3.4× perplexity at r=8 happened *without* repeated PLE injection.
- This materially changes the interpretation and the retrofit design.

Do not try to work around Strategy B in this round. Report back with the finding so we can rethink.

### Strategy C — PLE fused into something weird

If the source shows PLE is fused into an attention projection, the MLP, or something else that isn't a simple residual add, describe what you see and stop. Don't invent a workaround.

## Implementation steps

### Step 1 — Refactor the existing script

Modify `ple_sanity_check.py`:

1. Keep all existing env/model-loading/eval helpers unchanged.
2. Add `--mode` argument with choices `["original", "ple-variants"]`, default `"original"` for backward compat.
3. Route `main()` to one of two experiment functions based on mode.
4. `python ple_sanity_check.py` with no args must still reproduce round-1 numbers.

### Step 2 — Patch the decoder layer class

Assuming Strategy A:

```python
import types

def patch_ple_scale(decoder_layer_class):
    """Monkey-patch the forward to accept a ple_scale kwarg."""
    orig_forward = decoder_layer_class.forward

    def patched_forward(self, hidden_states, *args, ple_scale=1.0, **kwargs):
        # Inspection step 0 tells us exactly which line to modify.
        # If PLE is self.per_layer_embedding(input_ids), the cleanest
        # patch is to override that method temporarily — but the
        # simplest is to subclass and rewrite forward to pass ple_scale
        # into the PLE-add line.
        ...

    decoder_layer_class.forward = patched_forward
```

**Exact patching approach depends on step 0's findings.** If the PLE logic is complex, it may be cleaner to copy the original `forward` method's source, modify the one PLE line, and replace the method. Document which approach was chosen.

### Step 3 — Looping hook with PLE mode

```python
def make_looped_forward_ple(orig_forward, r, ple_mode):
    """
    ple_mode in {"vanilla", "scaled", "once"}.
    Requires orig_forward to accept a ple_scale kwarg.
    """
    def looped(hidden_states, *args, **kwargs):
        out = None
        for i in range(r):
            if ple_mode == "vanilla":
                scale = 1.0
            elif ple_mode == "scaled":
                scale = 1.0 / r
            elif ple_mode == "once":
                scale = 1.0 if i == 0 else 0.0
            else:
                raise ValueError(ple_mode)
            out = orig_forward(hidden_states, *args, ple_scale=scale, **kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped
```

### Step 4 — Regression sanity checks before the main run

Run these three checks and confirm they pass before collecting experiment data:

1. **Vanilla at r=1 matches round 1's r=1** (ppl ≈ 12.53). Confirms the PLE patch doesn't break the base forward pass.
2. **Scaled at r=1 equals vanilla at r=1** exactly. Because `1/r = 1` when `r=1`, these must be bitwise identical.
3. **Once at r=1 equals vanilla at r=1** exactly. Same reasoning.

If any check fails, stop and debug. Do not report experiment numbers on a broken patch.

### Step 5 — Run the 9-cell experiment

- `TARGET_LAYER = 17`
- `ple_mode ∈ {vanilla, scaled, once}`
- `r ∈ {1, 4, 8}`
- Same 50 Wikitext-2 test sequences as round 1.

Expected runtime: ~5–10 minutes.

### Step 6 — Output

JSON file `results_round2a.json`:

```json
{
  "config": { ... },
  "round1_baseline": { "unmodified_ppl": 12.53165455614523 },
  "regression_checks": {
    "vanilla_r1_matches_round1": true,
    "scaled_r1_equals_vanilla_r1": true,
    "once_r1_equals_vanilla_r1": true
  },
  "cells": [
    {"ple_mode": "vanilla", "r": 1, "mean_nll": ..., "ppl": ...},
    {"ple_mode": "vanilla", "r": 4, "mean_nll": ..., "ppl": ...},
    {"ple_mode": "vanilla", "r": 8, "mean_nll": ..., "ppl": ...},
    {"ple_mode": "scaled",  "r": 1, ...},
    {"ple_mode": "scaled",  "r": 4, ...},
    {"ple_mode": "scaled",  "r": 8, ...},
    {"ple_mode": "once",    "r": 1, ...},
    {"ple_mode": "once",    "r": 4, ...},
    {"ple_mode": "once",    "r": 8, ...}
  ]
}
```

Printed table:

```
=== PLE variants (layer 17) ===
                r=1       r=4       r=8
vanilla       12.53     20.48     42.57
scaled        12.53     ??.??     ??.??
once          12.53     ??.??     ??.??
```

## Interpretation buckets — commit before looking at numbers

- **Bucket 1: Scaled and once both clearly beat vanilla at r=8** (e.g., ratio <2× vs vanilla's 3.4×). → The right PLE strategy for retrofit is "dilute the injection." Adopt whichever of scaled/once is better. Scaled is simpler (no conditional); once is cheaper (skips a lookup).
- **Bucket 2: Vanilla wins or ties.** → The pretrained model genuinely benefits from per-pass PLE signal; diluting hurts more than over-injecting. Keep vanilla, rely on training to teach the model to handle accumulation.
- **Bucket 3: All three within noise of each other at r=8.** → PLE injection policy doesn't matter much at inference time; pick the simplest (once) for implementation cleanness.
- **Bucket 4: Once dramatically worse than vanilla/scaled.** → PLE carries real computational value that each iteration needs, not just layer-identification info. Argues strongly against "strip PLE from recurrent block" variants of the retrofit design.
- **Bucket 5: Results disagree with Bucket 1–4 expectations entirely** (e.g., scaled explodes, once is best, something weird). → Stop, paste numbers, let's discuss before drawing conclusions.

## Deliverables

1. Updated `ple_sanity_check.py` with `--mode=ple-variants` flag, backward compatible with round 1.
2. A short write-up at the top of the new code (or a separate note) documenting what Step 0 found: file path, PLE line, and which strategy (A/B/C) applied.
3. `results_round2a.json`.
4. The printed 3×3 table.
5. A one-paragraph interpretation stating which bucket the results fall into and the implied PLE strategy for the full retrofit.

## Report back

After running, paste:
- The Step 0 inspection notes (file, line, strategy).
- The 3×3 table.
- The interpretation paragraph.

Do not proceed to layer-location experiments, block-of-layers looping, or training. The Round 2b plan will be written after seeing these results.