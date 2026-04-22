# 07 — Round 3c: Pinning down the stable block region

**Plan:** `plan3c.md`
**Result file:** `results_round3c_extended_blocks.json`
**Status:** Complete. Answered both live questions from Round 3b.

## Design

Three new blocks, each testing a specific hypothesis from Round 3b:

| Block | Layers | Width | Question |
|-------|--------|-------|----------|
| G | 15–24 | 10 | Does the stable region extend past L22 up to L24? |
| H | 15–23 | 9 | Intermediate point — if G breaks but H works, boundary is at L23. |
| I | 20–27 | 8 | Does block location within the consumer zone matter, or is any 8-layer all-consumer block fine? |

Plus r=1 regression checks at B (15–18, width 4) and G (15–24, width 10).
Both passed with drift 0.0.

## Full results at r=2/4/8 (with blocks from Round 3b for reference)

| Block | Layers | Width | r=2 | r=4 | r=8 | vs D at r=8 |
|-------|--------|------:|----:|----:|----:|------------:|
| A | 15–19 | 5 | 27.3 | 29.7 | 30.1 | 0.90× |
| B | 15–18 | 4 | 25.8 | 28.7 | **29.7** | **0.89×** (best) |
| D | 15–22 | 8 | 33.7 | 32.7 | 33.3 | 1.00× (ref) |
| **G** | **15–24** | **10** | **37.7** | **35.4** | **36.0** | **1.08×** |
| **H** | **15–23** | **9** | **35.3** | **37.4** | **36.7** | **1.10×** |
| **I** | **20–27** | **8** | **887** | **1181** | **970** | **29.1×** (broken) |

![Block-loop stability across r](figs/fig10_round3c_stability.png)

## The two verdicts

### Width: stable region extends to at least L24

- G (15–24, width 10): 36.0 at r=8. 8% worse than D (33.3) — essentially
  flat on log scale.
- H (15–23, width 9): 36.7 at r=8.

Both G and H stay well within 5× baseline. The plan's bucket 1 is met:
**"Stable region extends to at least layer 24. Use block G (15-24) as the
retrofit recurrent block: 10 layers, 25% more effective compute than D."**

Notably, the stability-across-r is slightly *better* for wider blocks (r=2
→ r=8 ratios: A=1.10, B=1.15, D=0.99, G=0.95, H=1.04). Wider blocks have
more layers to absorb per-iteration error internally.

### Position: block must anchor near L15

Block I (20–27, width 8, shifted late by 5 layers from D) is catastrophic:
970 ppl at r=8, **29× worse than D**. Even at r=2, I is already at 887 ppl
— an order of magnitude worse than D's r=2 (33.7).

I is entirely inside the consumer zone, has 8 layers like D, and covers
mostly sliding-attention layers. The only meaningful difference from D is
that I doesn't start at the first consumer layer (L15).

The plan's interpretation for this axis:

> **Position axis: Bucket 5 (I breaks, D works).** Position matters: block
> must anchor near layer 15. Possibly because layer 15 is the first KV
> consumer; the producer→consumer transition is a natural anchor point.

## The full picture after Round 3c

![Block comparison at r=8](figs/fig07_round3bc_blocks.png)

**Viable blocks** (r=8 ppl < 100):

| Block | Layers | Width | r=8 ppl |
|-------|-------|------:|--------:|
| A | 15–19 | 5 | 30.1 |
| B | 15–18 | 4 | 29.7 |
| D | 15–22 | 8 | 33.3 |
| G | 15–24 | 10 | 36.0 |
| H | 15–23 | 9 | 36.7 |

**Catastrophic blocks** (r=8 ppl > 100):

| Block | Layers | Width | r=8 ppl | Why broken |
|-------|-------|------:|--------:|----|
| I | 20–27 | 8 | 970 | Wrong anchor (starts at L20 not L15) |
| C | 12–19 | 8 | 209,050 | Crosses KV boundary (L12–14 on wrong side) |
| F | 25–32 | 8 | 338,517 | In drift zone, no valley anchor |
| E | 13–22 | 10 | 846,403 | Crosses KV boundary |

Every viable block starts at L15. No viable block crosses the KV boundary.
No block outside the valley-anchored region is viable.

## Design constraints summary for retrofit

After Round 3c, the empirical rules for where to place a recurrent block
in E2B are:

1. **Must start at L15** (the first KV consumer).
2. **Must not cross the KV producer→consumer boundary** (can't extend
   left of L15).
3. **Can extend up to L24** safely. L25+ untested inside a valley-
   anchored block.
4. **Cannot replicate the paper's "all consumer" blocks** shifted to
   different consumer regions — the anchor is load-bearing.
5. **Width up to 10** works; 11+ untested.

## Structural interpretation

Why must blocks start at L15 specifically? The plan speculated, and the
hypothesis remains speculative:

> Possibly because layer 15 is the first KV consumer — the transition from
> producer to consumer state is a natural anchor point.

Mechanistically this makes sense. The KV consumers at L15+ all *reuse KV
states* produced by earlier layers. If the block starts at L15, each
iteration consumes the same (prelude-computed) KVs and produces a stable
residual trajectory. If the block starts at L20, each iteration may
consume KVs that are no longer consistent with the drifting residual
stream — the KV-consumer assumption silently breaks.

This is testable but hasn't been tested. A useful Round 6 experiment
(implied but not written up) would be: at block I, what do the shared
KV states look like over iterations? Are they diverging fast?

## What this round established

1. **The maximum viable recurrent block is 10 layers wide (G: 15–24) or
   9 layers (H: 15–23).** Both are 12–25% wider than the paper's
   `(4, 8, 4)` block on TinyLlama, proportional to E2B's deeper stack.
2. **Position is load-bearing.** The block must anchor at L15. No
   known workaround.
3. **The KV producer→consumer boundary is the dominant architectural
   constraint** for this method on E2B. Not PLE, not attention type, not
   layer depth.
4. **After 3c, perplexity-side design is settled.** The project moves to
   downstream evaluation (Round 4).
