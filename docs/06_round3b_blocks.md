# 06 — Round 3b: Block-looping probe (6 candidate blocks)

**Plan:** `plan3b.md`
**Result file:** `results_round3b_blocks.json`
**Status:** Complete with clean regression passes. Results are trustworthy.

## Setup

Six candidate blocks tested at r ∈ {2, 4, 8} with vanilla PLE. 18 cells
total. Each block is a contiguous range of layers looped together as a
unit.

Regression: at r=1, blocks must reproduce baseline ppl exactly. Tested at
blocks B (15–18) and E (13–22) — both passed with drift 0.0.

### The 6 blocks

| Block | Layers | Width | Design rationale |
|-------|--------|-------|------------------|
| A: valley-core | 15–19 | 5 | The valley as-is — lower-bound anchor |
| B: valley-narrow | 15–18 | 4 | Even tighter, avoids the global at 19 |
| C: valley-extend-down | 12–19 | 8 | Extend into fragile territory below |
| D: valley-extend-up | 15–22 | 8 | Extend into fragile territory above |
| E: valley-centered | 13–22 | 10 | Symmetric extension |
| F: late-block | 25–32 | 8 | Control in the PLE-inert late region |

## Results at r=8

| Block | Layers | Width | r=2 ppl | r=4 ppl | r=8 ppl | Status |
|-------|--------|-------|--------:|--------:|--------:|--------|
| **A** | 15–19 | 5 | 27.3 | 29.7 | **30.1** | viable |
| **B** | 15–18 | 4 | 25.8 | 28.7 | **29.7** | viable |
| **D** | 15–22 | 8 | 33.7 | 32.7 | **33.3** | viable |
| C | 12–19 | 8 | 102,534 | 152,039 | 209,050 | broken |
| E | 13–22 | 10 | 175,479 | 1,013,914 | 846,403 | broken |
| F | 25–32 | 8 | 37.1 | 13,850 | 338,517 | broken (drift) |

![Round 3b + 3c blocks at r=8](figs/fig07_round3bc_blocks.png)

## The three clean verdicts

### 1. The KV boundary is a hard wall

Blocks C (12–19) and E (13–22) both **cross** the L14→L15 KV
producer→consumer boundary. Both explode catastrophically even at r=2 (the
mildest loop tested):

- C at r=2: ppl = 102,534. The model is already incoherent from the first
  extra iteration.
- E at r=2: ppl = 175,479.

By contrast, blocks A, B, D are entirely inside the consumer region and
stay under 40 ppl at r=2.

This is the strongest architectural constraint the project has discovered.
It wasn't directly measurable in earlier single-layer rounds — L14 alone at
r=8 is 1873 ppl, which is bad but nothing like the boundary-crossing
failure. The boundary matters specifically when a block loops across it,
because each iteration re-generates L14-era KV state that downstream
consumers weren't built for.

### 2. Valley-anchored extension works in one direction

- Block D (15–22, width 8, extending *up* from the valley): r=8 ppl = 33.3.
  Essentially flat vs block A's 30.1. A 3-layer extension into the fragile
  region is absorbed — the block rescues its weakest layer dramatically.
  `improvement_over_worst_single` = 29.3× (worst single in block was L22
  at 976 ppl).
- Block C (12–19, width 8, extending *down* from the valley): broken,
  because going down crosses the KV boundary.

So "valley-anchored extension works" is *not* symmetric. The block must
start at the boundary itself (L15) and extend upward.

### 3. The late-block control (F) drifts under iteration

Block F (25–32) is the plan's control for testing whether the valley is
special or whether any 8-layer all-consumer block can be looped. The
answer: the valley is special.

F's behavior is diagnostic:

- r=2: 37.1 ppl (nearly fine, comparable to valley blocks)
- r=4: 13,850 ppl (catastrophic drift)
- r=8: 338,517 ppl (further drift)

Unlike blocks C/E (which break from r=2), F looks okay at r=2 and then
collapses as r grows. This is a different failure mode — **slow drift under
iteration** rather than first-iteration collapse. It means the block
doesn't have a stable fixed point; each iteration nudges the hidden state
further off the pretrained manifold.

## Plan's scenario attribution

The JSON records:

> **interpretation_hint:** Bucket 5 (unexpected pattern) — the results do
> not fit scenarios X or Y cleanly (e.g., non-monotone in width, or one
> valley anchor catastrophic but not the other).

The non-symmetry of valley extension (D works, C doesn't) was the
surprise. Once decomposed into "the KV boundary is a wall", the pattern
fits cleanly — which is what Round 3c pinned down.

## Improvement-over-worst-single

| Block | worst single in block (r=8) | block ppl (r=8) | improvement |
|-------|----------------------------:|----------------:|------------:|
| A | 108 (L18) | 30.1 | 3.6× |
| B | 108 (L18) | 29.7 | 3.6× |
| C | 261,567 (L13) | 209,050 | 1.3× (irrelevant — block is broken) |
| D | 976 (L22) | 33.3 | **29.3×** |
| E | 261,567 (L13) | 846,403 | 0.3× (block broken, worse than worst single) |
| F | 56,541 (L32) | 338,517 | 0.17× (worse than worst single) |

Block D is the real win: it takes a block containing layer L22 (which at
r=8 solo is ~976 ppl) and reduces its contribution to 33 ppl by allowing
L15–L21 to absorb L22's damage through within-block re-processing. This
**is the cascade hypothesis working as predicted**, just on a narrower
scope than the plan had guessed.

## What this round established

1. **The KV producer→consumer boundary (L14/L15) is a hard wall** for
   block-looping. Blocks must be entirely on the consumer side.
2. **Valley-anchored blocks** (starting at L15) can extend *upward* into
   the fragile region successfully — block D (15–22, width 8) is
   essentially as good as block A (15–19, width 5).
3. **The valley is not just "less PLE importance" or "smaller
   perplexity"** — it's specifically the region between the KV boundary
   and where slow drift starts. Block F's all-consumer block outside the
   valley drifts under iteration.
4. **Block C's broken behavior tells us you can't "bridge" the KV
   boundary** by making the block wider — the moment L14 is in the block,
   things break.

## Next-step implications

Round 3b leaves two live questions:

1. **How far up can the block extend?** D (15–22) works; E (13–22) is
   broken because of the bottom end; the top end isn't yet bounded.
2. **Must the block anchor at L15, or does "any 8-layer all-consumer
   block" work as long as it stays in a drift-free subregion?** Block F is
   a single counterexample — more points needed.

Round 3c picks up both questions with three more blocks.
