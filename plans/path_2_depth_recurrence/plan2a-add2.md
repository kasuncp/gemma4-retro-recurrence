# Gemma 4 E2B — Round 2a Addendum 2: Fix PLE Hook and Rerun

## Context

The zero-PLE diagnostic returned `patch_status: BROKEN`. `zero` at r=1 produced bitwise-identical perplexity to `vanilla` at r=1, meaning the `ple_scale` multiplication never actually executed.

**Root cause (identified):** In `make_looped_forward_ple`, the hook reads PLE from `kwargs.get(ple_kwarg, None)`. But the decoder layer's signature is:

```
(self, hidden_states, per_layer_input: torch.Tensor = None, shared_kv_states, ...)
```

`per_layer_input` is the **second positional parameter**. The outer `Gemma4Model.forward` almost certainly passes it positionally, not as a kwarg. So `per_layer_input` arrives in the hook's `*args` tuple, not in `**kwargs`. The `kwargs.get(...)` returns `None`, the scaling branch is guarded out, and every iteration re-passes the original unscaled tensor via `*args`.

Consequence: **every `ple_mode` has been running vanilla behaviour.** Round 2a's three-way tie is an artifact.

**What is still valid from earlier rounds:** round 1's perplexity numbers (12.53 / 13.45 / 20.48 / 42.57) reflect genuine vanilla behaviour — PLE re-injected at full strength on every iteration. The round-1 interpretation stands. Only round 2a's variant comparison is invalid.

## Scope

**In scope:**
- Fix `make_looped_forward_ple` so it correctly intercepts `per_layer_input` whether it arrives positionally or as a kwarg.
- Verify the fix with the zero-PLE diagnostic (expect `zero r=1` ppl ≠ `vanilla r=1` ppl).
- Rerun the full round-2a 9-cell grid with the fix in place.
- Save results to `results_round2a_fixed.json` (new file; do not overwrite the buggy one).

**Out of scope:**
- Do not refactor anything else.
- Do not add new `ple_mode` values.
- Do not touch round 1 or the layer-location experiment.
- Do not "simplify" by refactoring the looped forward beyond what the bug fix requires.

## The fix

Replace `make_looped_forward_ple` with a version that handles both argument-passing conventions. Recommended implementation:

```python
def make_looped_forward_ple(orig_forward, r, ple_mode, ple_kwarg):
    """
    Hook that loops a decoder layer r times with controlled PLE re-injection.

    Handles both positional and kwarg passing of `per_layer_input`.
    The Gemma 4 decoder layer signature is:
        forward(self, hidden_states, per_layer_input=None, shared_kv_states=None, ...)
    and Gemma4Model.forward passes per_layer_input positionally. So we must
    look for it in args[0] (first positional after hidden_states is split off
    by the *args capture) *or* in kwargs[ple_kwarg].
    """
    def looped(hidden_states, *args, **kwargs):
        # Determine where per_layer_input actually arrived.
        if ple_kwarg in kwargs:
            original_ple = kwargs[ple_kwarg]
            ple_location = "kwarg"
        elif len(args) >= 1:
            # Signature says per_layer_input is the 2nd positional (first after
            # hidden_states). Our *args captures everything after hidden_states,
            # so args[0] is per_layer_input.
            original_ple = args[0]
            ple_location = "positional"
        else:
            original_ple = None
            ple_location = "missing"

        out = None
        for i in range(r):
            if ple_mode == "vanilla":
                scale = 1.0
            elif ple_mode == "scaled":
                scale = 1.0 / r
            elif ple_mode == "once":
                scale = 1.0 if i == 0 else 0.0
            elif ple_mode == "zero":
                scale = 0.0
            else:
                raise ValueError(f"unknown ple_mode={ple_mode!r}")

            # Build call args with scaled PLE.
            if original_ple is None or scale == 1.0:
                # Pass through unchanged (preserves bitwise identity at scale=1).
                call_args = args
                call_kwargs = kwargs
            else:
                scaled_ple = original_ple * scale if original_ple is not None else None
                if ple_location == "kwarg":
                    call_args = args
                    call_kwargs = dict(kwargs)
                    call_kwargs[ple_kwarg] = scaled_ple
                elif ple_location == "positional":
                    call_args = (scaled_ple,) + args[1:]
                    call_kwargs = kwargs
                else:
                    call_args = args
                    call_kwargs = kwargs

            out = orig_forward(hidden_states, *call_args, **call_kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped
```

**Critical detail:** the `ple_location` must be *logged once* at the top of each experiment run so we can confirm from the output which path was taken. Add this as a print in the calling code:

```python
# Inside run_ple_variants_mode, before the main sweep:
# Do a dry-run forward to detect where PLE arrives.
print(f"PLE location detected: (see first hook call)")
# ... or add a one-time print inside the hook itself.
```

Simpler: add a print inside `looped` that fires only on the very first call (guarded by a mutable flag or by a `dry_run=True` invocation before the timed sweep).

## Verification sequence

Run these in order. **Stop immediately if any step fails.**

### Step 1 — Rerun the zero diagnostic with the fix

```bash
python ple_sanity_check.py --mode ple-variants --only-diagnostic
```

Expected output:

```
vanilla r=1: ppl = 12.536588...
zero    r=1: ppl = X.XXXX      (DIFFERENT from vanilla)
patch_status: WORKING
```

If `patch_status: WORKING` → the hook now actually intercepts PLE. Continue.
If still `BROKEN` → stop and report. The positional/kwarg theory was wrong; something else is the cause.

### Step 2 — Rerun round-2a regression checks

As part of the full `--mode ple-variants` run, the existing regression checks will verify:

- vanilla r=1 ppl ≈ round-1 baseline (12.53)
- scaled r=1 == vanilla r=1 (bitwise)
- once r=1 == vanilla r=1 (bitwise)

All three must still pass. At r=1, `scale=1.0` for vanilla/scaled/once (and our hook passes args through unchanged when `scale==1.0`), so bitwise identity at r=1 is preserved.

### Step 3 — Rerun the full 9-cell grid

```bash
python ple_sanity_check.py --mode ple-variants --output-json results_round2a_fixed.json
```

**Expected signature of a working fix:**
- At r=1: all three modes identical (by construction — regression checks).
- At r=4 and r=8: at least some mode differs from others (the whole point of the experiment).
- If all three still match bitwise at r=4 and r=8, something else is wrong. Stop and report.

## Output format

`results_round2a_fixed.json` — same schema as `results_round2a.json`, with an added `addendum2` field:

```json
{
  "addendum2": {
    "ple_location": "positional" | "kwarg",
    "zero_diagnostic_passed": true,
    "fix_applied": "handle_positional_ple_args"
  },
  "config": { ... },
  "cells": [ ... ]
}
```

Also print the standard 3×3 table at the end.

## Deliverables

1. Updated `ple_sanity_check.py` with the fixed `make_looped_forward_ple`.
2. Output of the zero diagnostic showing `patch_status: WORKING`.
3. `results_round2a_fixed.json` with real variant-comparison data.
4. The 3×3 table printed to stdout.
5. A one-line note stating whether `ple_location` was `"positional"` or `"kwarg"`.

## Report back

Paste:
- The zero-diagnostic output (vanilla r=1 ppl, zero r=1 ppl, WORKING status).
- The 3×3 table.
- The `ple_location` finding.

Do not proceed to interpretation or layer-location experiments — I'll do interpretation from the numbers.