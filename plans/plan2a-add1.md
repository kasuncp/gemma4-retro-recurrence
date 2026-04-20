# Gemma 4 E2B — Round 2a Addendum: Zero-PLE Diagnostic

## Context

Round 2a produced a surprising result: `vanilla`, `scaled`, and `once` modes produced **bitwise-identical** perplexities at every `r ∈ {1, 4, 8}`. Mean NLL values agree to 16 decimal places.

This could mean one of two very different things:

1. **The patch worked correctly, and PLE injection policy genuinely doesn't affect perplexity at layer 17.** The perplexity degradation at r>1 is caused entirely by looping attention+MLP, not by re-injecting PLE. In this case PLE is a much smaller concern for retrofitting than we feared.

2. **The patch doesn't actually intercept PLE.** All three "modes" are in fact running the same computation, and `ple_scale` is being ignored. The identical numbers are an artifact of a broken intervention.

We cannot distinguish these from the round-2a data alone, because both produce identical numbers across modes.

This addendum runs **one diagnostic cell** that distinguishes them.

## The diagnostic

Add a fourth `ple_mode` called `zero`: always pass `ple_scale=0.0` on every iteration, starting from iteration 0. This should completely zero out the PLE contribution at layer 17 for the entire forward pass.

Run it at **r=1 only**. One cell. One number.

### Interpretation

- **If `zero` at r=1 produces different perplexity from vanilla at r=1** (any difference, even small) → the patch is working. PLE at layer 17 does contribute to the forward pass, and the three-way tie in round 2a is a real finding: *PLE injection policy doesn't affect perplexity under our looping setup*. Proceed to Round 2b with this knowledge.

- **If `zero` at r=1 produces identical perplexity to vanilla at r=1** → the patch isn't intercepting PLE at all. Round 2a's results are inconclusive because the intervention never took effect. We need to re-examine the patching approach before trusting any PLE variant results.

A difference of even 0.001 in mean NLL is enough to confirm the patch works. We're looking for "any nonzero change" vs "zero change."

## Scope

**In scope:**
- Add `zero` to the allowed `ple_mode` values.
- Run exactly one cell: `ple_mode="zero"`, `r=1`, `TARGET_LAYER=17`.
- Also rerun `vanilla` at `r=1` in the same script execution, for a direct comparison on the same random state / evaluation batch.
- Print both numbers side by side and report whether they differ.
- Update `results_round2a.json` (or create `results_round2a_addendum.json`) with the new cell.

**Out of scope:**
- Do NOT run `zero` at other r values.
- Do NOT re-run all of round 2a.
- Do NOT investigate the patching code further until we know whether the patch is working.
- Do NOT proceed to Round 2b.

This should take under 2 minutes of compute.

## Implementation

The `make_looped_forward_ple` helper already has the branching structure. Add one clause:

```python
def make_looped_forward_ple(orig_forward, r, ple_mode):
    def looped(hidden_states, *args, **kwargs):
        out = None
        for i in range(r):
            if ple_mode == "vanilla":
                scale = 1.0
            elif ple_mode == "scaled":
                scale = 1.0 / r
            elif ple_mode == "once":
                scale = 1.0 if i == 0 else 0.0
            elif ple_mode == "zero":
                scale = 0.0  # NEW: always zero, from iteration 0
            else:
                raise ValueError(ple_mode)
            out = orig_forward(hidden_states, *args, ple_scale=scale, **kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped
```

Expose it via the same `--mode=ple-variants` flag, just run only the `{vanilla, zero} × {r=1}` pair. Simplest approach: add an `--only-diagnostic` flag to `ple_sanity_check.py` that limits the grid to those two cells.

## Report back format

Print exactly:

```
=== Zero-PLE diagnostic ===
vanilla r=1:  ppl = 12.536588219504221
zero    r=1:  ppl = XX.XXXXXXXXXXXXX

Difference: YY.YYYYYY  (absolute NLL difference)
Patch status: WORKING / BROKEN
```

Where:
- "WORKING" if the difference in mean NLL is greater than 1e-6.
- "BROKEN" if the difference is exactly 0 (or below 1e-6, indicating numerical noise only).

That's the entire deliverable for this addendum.

## Do not proceed to

- Round 2b (layer location).
- Any further PLE variant experiments.
- Any training.

Stop after this diagnostic and report back.