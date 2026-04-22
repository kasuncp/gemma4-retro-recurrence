# 02 — Round 2a: PLE injection policy at layer 17

**Plan:** `plan2a.md` — "PLE Probe Round 2a (PLE variants only)"
+ two addenda (`plan2a-add1.md`, `plan2a-add2.md`)
**Result files:**
- `results_round2a.json` (original, broken)
- `results_round2a_run2.json` (repeat of original, still broken)
- `results_round2a_addendum.json` (zero-PLE diagnostic, BROKEN verdict)
- `results_round2a_addendum_v2.json` (zero-PLE after hook fix, WORKING)
- `results_round2a_fixed.json` (full 9-cell grid with fixed hook)

**Status:** Complete with a noteworthy debugging episode.

## Question

At layer 17, which PLE injection policy tolerates looping best?

- **vanilla:** apply PLE every iteration at full strength
- **scaled:** apply PLE every iteration, multiplied by 1/r
- **once:** apply PLE on iteration 0 only, skip on 1..r-1

## Step 0 inspection

Located `Gemma4TextDecoderLayer` in
`transformers/models/gemma4/modeling_gemma4.py` line 1354:

```python
(self, hidden_states, per_layer_input=None, shared_kv_states=None,
 position_embeddings=None, attention_mask=None, position_ids=None,
 past_key_values=None, **kwargs)
```

Inspection verdict: **Strategy A** — PLE is applied inline within
`forward` and can be controlled by passing a different `per_layer_input`
tensor to each iteration.

## The bug and the fix (important)

### What went wrong

The first run of Round 2a produced this result:

| ple_mode | r=1 | r=4 | r=8 |
|----------|-----|-----|-----|
| vanilla | 12.5366 | 20.4795 | 42.5893 |
| scaled  | 12.5366 | 20.4795 | 42.5893 |
| once    | 12.5366 | 20.4795 | 42.5893 |

All three modes produced **bitwise-identical** numbers to 16 decimal places.

### Diagnostic (addendum 1)

To distinguish "the patch is silently broken" from "PLE policy genuinely
doesn't matter at this layer", a 4th mode `zero` was added — always pass
`ple_scale=0.0`. Expected: different from vanilla at r=1. Observed: identical
to vanilla at r=1 (`nll_difference = 0.0`). Verdict: **patch BROKEN**.

### Root cause (addendum 2)

The hook was reading `per_layer_input` from `kwargs.get(ple_kwarg)`, but the
outer `Gemma4Model.forward` passes `per_layer_input` **positionally** (it's
the second positional parameter in the signature). So the PLE tensor arrived
in `args`, not `kwargs`, the hook's scaling branch never fired, and every
iteration re-passed the unscaled tensor via `*args`.

### Fix

Handle both argument-passing conventions explicitly:

```python
if ple_kwarg in kwargs:
    original_ple = kwargs[ple_kwarg]; ple_location = "kwarg"
elif len(args) >= 1:
    original_ple = args[0]; ple_location = "positional"  # ← E2B's case
else:
    original_ple = None; ple_location = "missing"
```

The fixed run recorded `ple_location: positional` and `zero_diagnostic_passed:
True` (zero r=1 ppl = 12.587 vs vanilla r=1 ppl = 12.537, NLL diff
+0.004035 — small but nonzero).

### What this means for earlier rounds

**Round 1's numbers are still valid.** Round 1 didn't touch PLE at all — it
just wrapped `forward` and called it r times, which means PLE gets injected
naturally by the unmodified forward on each iteration. That's exactly
"vanilla" behavior. Only Round 2a's *variant comparison* was invalid before
the fix.

## Results (fixed hook)

| ple_mode | r=1 | r=4 | r=8 |
|----------|-----:|-----:|-----:|
| vanilla | 12.5366 | 20.4795 | 42.5893 |
| scaled  | 12.5366 | 20.4711 | 42.5565 |
| once    | 12.5366 | **18.5349** | **36.6270** |

Ratios vs vanilla at the same `r`:

| ple_mode | r=4 ratio | r=8 ratio |
|----------|----------:|----------:|
| scaled | 0.9996 (≈vanilla) | 0.9992 (≈vanilla) |
| once   | **0.9050** | **0.8600** |

![Round 2a PLE policies at layer 17](figs/fig02_round2a_ple_policies.png)

## Interpretation

From the plan's buckets:

- **Bucket 1 (scaled and once both beat vanilla at r=8):** half-met. `once`
  clearly beats vanilla (−14% ppl at r=8); `scaled` is indistinguishable
  from vanilla.
- **Bucket 2 (vanilla wins):** no.
- **Bucket 3 (all three within noise):** no, `once` is clearly different.
- **Bucket 4 (once dramatically worse):** no, the opposite.

**Plan 2b's summary table**, which triggered Round 2b, puts the finding as:

> At layer 17 specifically: `once` (apply PLE on iteration 0 only, skip
> thereafter) is modestly better than `vanilla` (apply every iteration);
> `scaled` is indistinguishable from vanilla. The zero-PLE diagnostic at
> layer 17 showed PLE contributes ~0.4% to perplexity — small but nonzero.

"Small but nonzero" is important: a mode comparison can only be as
discriminating as the underlying signal is large. At layer 17 the PLE
contribution is tiny (0.4% of NLL), which bounds how much `once` vs `vanilla`
*could* differ there. The 14% ppl improvement at r=8 is therefore not about
reducing PLE injection per se — it's about reducing *repeated* PLE injection,
which must mean the accumulated re-injection is a real fraction of the
looping damage.

## Regression checks

The fixed-hook run records all regression checks passing:

- `vanilla_r1_matches_round1: True` (12.5366)
- `scaled_r1_equals_vanilla_r1: True` (bitwise at r=1 by construction,
  because 1/r = 1)
- `once_r1_equals_vanilla_r1: True` (bitwise at r=1, because `once` uses
  scale=1 on iter 0 and never reaches iter 1..r-1 when r=1)
- `zero_r1_differs_from_vanilla_r1: True` (0.004035 NLL difference, above
  the 1e-6 threshold)

## What this round did NOT answer

- Whether the `once > vanilla` pattern at layer 17 generalizes. It was tested
  at three layers in Round 2b location sweep and the answer is **no — it's
  layer-dependent**.
- Whether PLE policy interacts with the loopability of individual layers
  across the full stack. → Round 2c includes both modes × all 35 layers.
- **The PLE hook fix is load-bearing for every subsequent round.** All
  Round 2b/2c/3a/3b/3c/4 results use the positional-arg-aware hook.

## Files in the project directory

| File | Role |
|------|------|
| `results_round2a.json` | Original broken run. Do not use for variant comparison. |
| `results_round2a_run2.json` | Repeat of the broken run. Same numbers. |
| `results_round2a_addendum.json` | Zero-PLE diagnostic — confirmed BROKEN |
| `results_round2a_addendum_v2.json` | Zero-PLE diagnostic after fix — WORKING |
| `results_round2a_fixed.json` | **Authoritative** 9-cell result |
