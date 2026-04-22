# 05 — Round 3a: Pair-of-layers looping probe

**Plan:** `plan3a.md`
**Result file:** `results_round3a_pair_looping.json`
**Status:** Complete but data is flagged as likely contaminated by a hook
bug. Interpretation should be tentative.

## Question

When we looped a single layer, the downstream cascade (34 layers processing
a perturbed hidden state) might have been the actual cause of the
catastrophic degradation, not the looped layer itself. If we loop a pair
(L, L+1) as a unit, inside-pair re-processing starts to matter. Does the
valley expand?

## Design

- 34 pair starting positions L ∈ {0, ..., 33}, pair = (L, L+1)
- r ∈ {2, 4, 8}
- vanilla PLE only
- 102 cells total
- Regression check: r=1 at 3 test pairs (5,6), (17,18), (28,29) must match
  baseline. All passed (rel_drift 0.0).

## Hook implementation

This was the trickiest hook in the project. The challenge: the outer model
normally iterates over all 35 layers sequentially. To loop *both* L and L+1
as a unit r times, the hook must:

- Apply both layers' forwards with their respective PLE tensors on each
  iteration
- Neuter L+1's forward on subsequent outer-model calls so it doesn't get
  applied a second time
- Capture each layer's `per_layer_input` correctly (they're *different*
  tensors per layer)

The plan sketched two approaches: (a) pre-hooks to capture each layer's
PLE, then a loop inside L's forward that calls both; or (b) monkey-patch
the outer `Gemma4TextModel.forward`. The actual implementation path isn't
recorded in the result JSON, but the interpretation_hint field flags a
likely bug (see below).

## Results — pair-looping vs single-layer at r=8

| Pair (L, L+1) | single L ppl | single L+1 ppl | pair ppl | pair vs min(single) |
|---|---:|---:|---:|---:|
| (0,1) | 1099 | 1129 | 8696 | 0.13× (worse) |
| (1,2) | 1129 | 1680 | 1309 | 0.86× |
| (2,3) | 1680 | 855 | 1205 | 0.71× |
| (3,4) | 855 | 3096 | 963 | 0.89× |
| (4,5) | 3096 | 774 | 920 | 0.84× |
| (5,6) | 774 | 576 | 676 | 0.85× |
| (6,7) | 576 | 543 | 972 | 0.56× |
| (7,8) | 543 | 2045 | 1802 | 0.30× |
| (8,9) | 2045 | 5104 | 643 | 3.18× (better) |
| (9,10) | 5104 | 5218 | 2971 | 1.72× (better) |
| (10,11) | 5218 | 5.6e7 | 2.4e8 | ~0 |
| (11,12) | 5.6e7 | 81,336 | 1.65e10 | ~0 |
| (12,13) | 81,336 | 261,567 | 126,868 | 0.64× |
| (13,14) | 261,567 | 1873 | 132,733 | 0.01× |
| (14,15) | 1873 | **35.3** | 1156 | 0.03× |
| **(15,16)** | **35.3** | **38.8** | **31.9** | **1.11× (better)** |
| (16,17) | 38.8 | 42.6 | 51.2 | 0.76× |
| (17,18) | 42.6 | 108.1 | 112.5 | 0.38× |
| (18,19) | 108.1 | 103.7 | 134.7 | 0.77× |
| (19,20) | 103.7 | 417 | 254 | 0.41× |
| (20,21) | 417 | 454 | 560 | 0.74× |
| (21,22) | 454 | 976 | 757 | 0.60× |
| (22,23) | 976 | 592 | 740 | 0.80× |
| (23,24) | 592 | 4391 | 1166 | 0.51× |
| (24,25) | 4391 | 1143 | 5976 | 0.19× |
| (25,26) | 1143 | 860 | 12,094 | 0.07× |
| (26,27) | 860 | 480 | 4622 | 0.10× |
| (27,28) | 480 | 524 | 3543 | 0.14× |
| (28,29) | 524 | 2435 | 11,584 | 0.05× |
| (29,30) | 2435 | 490 | 12,604 | 0.04× |
| (30,31) | 490 | 36,895 | 802,628 | 0.001× |
| (31,32) | 36,895 | 56,541 | 7.1e6 | 0.005× |
| (32,33) | 56,541 | 5.5e9 | 7.3e8 | ~0 |
| (33,34) | 5.5e9 | 2.9e20 | 19,253 | huge but meaningless |

![Round 3a: Pair-looping vs single-layer](figs/fig06_round3a_pair_vs_single.png)

## Counts by tolerance

At r=8, how many of 34 pairs stay within baseline multiples of the
unmodified ppl (12.53)?

| Threshold | Pairs passing | Single layers passing (for comparison) |
|-----------|---------------|----------------------------------------|
| 5× baseline (<63) | 2 | 3 |
| 10× (<125) | 3 | 5 |
| 50× (<627) | 6 | 6 |
| 100× (<1253) | 16 | 7 |

## Correlation analysis at r=8

| Factor | Pearson ρ | Spearman ρ |
|---|---:|---:|
| PLE importance sum (L + L+1) | −0.05 | +0.15 |
| Crosses an attention-type boundary | −0.15 | −0.08 |
| Crosses the KV boundary | −0.03 | −0.06 |
| Pair midpoint index | −0.09 | +0.30 |

All weak — no factor explains pair-looping tolerance either.

## Interpretation — and the suspected hook bug

The JSON file records:

> **interpretation_hint:** Bucket 4 (WORSE — investigate). 3/34 pairs vs
> 5/35 single layers [within 10×]. Pair-looping is systematically worse;
> this is unexpected and likely a hook bug. Stop and investigate.

What the plan had predicted:

- **Bucket 1 (pairs rescue most of the stack)** — confirmed cascade
  hypothesis, full green light for block looping.
- **Bucket 3 (pairs ≈ single layers)** — E2B is fundamentally fragile.
- **Bucket 4 (pairs dramatically worse)** — unexpected, likely a hook bug.

The data shows only pair (15,16) clearly beating its constituent singles
(31.9 vs min 35.3). Two others — (8,9) → 643 and (9,10) → 2971 — beat the
min of their singles but remain fragile in absolute terms. Everywhere else,
pair-looping is equal or worse.

**What this does and doesn't conclude:**

The "pair-looping is systematically worse" signal does not fit any
theoretical prediction — if pair-looping is well-implemented, it should
behave like a miniature block loop, which Round 3b showed works fine for
valley-anchored blocks. The finding that pairs (15,16), (16,17), (17,18)
all degrade worse than their constituent single layers is especially
suspicious given Round 3b confirmed the valley supports 4–10 layer blocks.

The most likely explanation is a timing/plumbing issue in how
`per_layer_input` was captured for L+1. If the capture happened out of
order, L+1 got the wrong PLE tensor on every iteration, producing
*different* broken behavior per pair rather than consistent behavior.

**What this means for decisions:** Round 3a was treated as uninformative
for retrofit design. The project moved directly to Round 3b (proper block
looping) rather than trying to fix 3a. That was a sound call — 3b
delivered clean, regression-passing results using a different hook
implementation.

## What this round established

1. **Pair-looping via this specific hook doesn't behave like a smaller
   version of block-looping.** Round 3b's block-looping hook at width 4
   (block B, layers 15–18) produces 29.7 at r=8, while this round's pair
   (15,16) produces 31.9 and pair (17,18) produces 112.5. Those shouldn't
   diverge if the hooks were equivalent.

2. **The single clean result** — pair (15,16) beating its components —
   is consistent with the valley-rescue story even if everything else is
   suspect.

3. **The decision to proceed to Round 3b rather than debug 3a** was
   correct in hindsight. Further pair-looping work would have been
   progress on a data artifact.
