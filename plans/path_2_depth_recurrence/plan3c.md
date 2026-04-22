# Gemma 4 E2B — Round 3c: Pin Down the Stable Block Region

## Context

Round 3b tested 6 candidate blocks and gave us three viable winners plus three clear failures:

| Block | Layers | Width | r=8 ppl | Status |
|---|---|---|---|---|
| A: valley-core | 15–19 | 5 | 30.1 | viable |
| B: valley-narrow | 15–18 | 4 | 29.7 | viable |
| **D: valley-extend-up** | **15–22** | **8** | **33.3** | **widest viable so far** |
| C: valley-extend-down | 12–19 | 8 | 209,050 | broken (crosses KV boundary) |
| E: valley-centered | 13–22 | 10 | 846,403 | broken (crosses KV boundary) |
| F: late-block | 25–32 | 8 | 338,517 | broken (slow drift under iteration) |

Two hypotheses emerged from this:

1. **The KV producer/consumer boundary at layer 14/15 is a hard wall.** Any block crossing it (C, E) breaks catastrophically from the very first extra iteration, because looping re-generates KVs that downstream consumers weren't built for.
2. **Within the consumer region (layers 15–34), there's a "stable fixed-point subregion" and a "slow-drift subregion."** Block D (15–22) is in the stable zone; block F (25–32) is in the drift zone. The exact boundary between them is unknown — could be at layer 23, 24, or 25.

This probe pins down the *maximum viable block width* by testing three additional blocks:

- **Block G (15–24)**: 10 layers wide, all KV-consumer. Tests if the stable region extends past layer 22 up to 24. If this works, we have a 25% wider recurrent block than D.
- **Block H (15–23)**: 9 layers wide, all KV-consumer. Intermediate point — if G breaks but H works, we know the stable zone ends at layer 23.
- **Block I (20–27)**: 8 layers wide, all KV-consumer but *shifted later* than D. Tests whether the stable subregion has a lower bound at layer 15 (i.e., must anchor at 15) or whether any 8-layer all-consumer block works.

## Scope

**In scope:**
- Extend `ple_sanity_check.py`'s block-looping mode to accept these three new blocks via the `--blocks` flag, or run with a new default block set.
- 3 blocks × 3 r values = **9 cells**. Vanilla PLE only.
- Reuse all round 3b infrastructure.
- Compare directly against blocks A, B, D from round 3b as references.

**Out of scope:**
- No multi-block experiments. That's the follow-up Round 4.
- No training.
- No new PLE modes.
- No blocks that cross the KV boundary (we've already confirmed those break).
- No blocks larger than 10 layers (would require testing block geometries that cross the sliding-to-global attention pattern more than once, complicating interpretation).

## Experiment design

| Block | Layers | Width | Contains globals at | Rationale |
|---|---|---|---|---|
| G | 15–24 | 10 | 19, 24 | Stretch D upward; crosses two global-attention layers instead of D's one |
| H | 15–23 | 9 | 19 | Intermediate between D (8) and G (10); only one global |
| I | 20–27 | 8 | 24 | Same width as D (8), shifted +5 layers — tests whether position matters within the consumer zone |

Regression checks (r=1 must match unmodified baseline ppl=12.5366 bitwise) for at least one block before the main sweep.

Expected runtime: 9 cells at ~15 seconds each = a few minutes at most.

## Analysis

### Primary comparison table

```
=== Round 3c results at r=8 ===
Block    Layers    Width   r=2     r=4     r=8     vs block D (r=8)
A        15-19     5       27.3    29.7    30.1    0.9x  (reference)
B        15-18     4       25.8    28.7    29.7    0.9x  (reference)
D        15-22     8       33.7    32.7    33.3    1.0x  (reference)
G        15-24    10       ???     ???     ???     ???
H        15-23     9       ???     ???     ???     ???
I        20-27     8       ???     ???     ???     ???
```

### Secondary: stability across r

For each block, compute `r8/r2` ratio. Block D's ratio is 0.99 (near-flat). A ratio under 1.2 means the block is stable; over 2.0 means it's compounding; over 10 means catastrophic drift.

## Interpretation buckets

**Bucket 1 — G works (ppl < 50 at r=8).**
Stable region extends to at least layer 24. Use block G as the retrofit recurrent block: 10 layers wide, 25% more effective compute than D. Worth pushing further to test layers 25+ (block J: 15–25, 11 layers), unless G already spans all the way to layer 24.

**Bucket 2 — G breaks, H works (ppl < 50 at r=8).**
Stable region ends at layer 23. Use block H: 9 layers wide, 12% more than D. The layer-23→24 boundary matters for some reason — worth noting but probably not worth investigating further; proceed with H.

**Bucket 3 — G and H both break; D remains the maximum.**
Stable region ends at layer 22 exactly. Use D. The single-layer probes at layers 23 and 24 (both had low PLE importance, should have been safe) were hiding some interaction that block-looping reveals. This would be a surprise given layer 23's single-layer ppl was 592 and 24 was 4391 — both bad on their own, but I expected blocks to rescue them. If they don't, there's an unseen constraint.

**Bucket 4 — I works (ppl < 50 at r=8).**
Position within the consumer zone doesn't strictly matter; any 8-layer all-consumer block is viable as long as it avoids the drift zone. This suggests "the stable subregion" is actually broader than 15–22, and we simply chose a good starting point. Worth testing one more shifted-late block to confirm (e.g., block J: 22–29).

**Bucket 5 — I breaks but D works.**
Position matters: must anchor at layer 15 (or close to it). Something specific to layer 15 is what makes the block stable. Possibly related to layer 15 being the first KV consumer — the transition from producer to consumer state is a natural anchor point.

**Bucket 6 — Unexpected combinations.**
E.g., G works but H doesn't; or I works but G doesn't; or all three partially work at r=2 but break at r=8. Pause and discuss.

## Deliverables

1. `results_round3c_extended_blocks.json` with 9 cells plus comparison table.
2. Printed comparison table (6 blocks: A, B, D from round 3b as references + G, H, I from this round).
3. A one-paragraph interpretation stating the bucket and the resulting maximum-viable-block choice.
4. If any block in Bucket 1 or 4 is significantly better than D, flag it as the new retrofit baseline.

## Report back

Paste the comparison table and the interpretation paragraph. Based on the bucket, next step is either Round 4 (multi-block probe, if single-block is settled) or a narrower follow-up if an edge case appears.