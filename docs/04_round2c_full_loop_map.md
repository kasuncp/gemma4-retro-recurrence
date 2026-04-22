# 04 — Round 2c: Full 35-layer loop-tolerance map

**Plan:** `plan2c.md`
**Result file:** `results_round2c_full_map.json` (210 cells, 64 KB)
**Status:** Complete. This is the most important characterization round of
the project — its findings shape every subsequent decision.

## Design

- Grid: 35 layers × {vanilla, once} × {r=2, 4, 8} = **210 cells**
- PLE modes: vanilla, once (no scaled — Round 2a showed it's identical to
  vanilla)
- Regression check: at every layer, r=1 vanilla ppl must match the
  unmodified baseline bitwise. **All 35 layers passed** with `max_rel_drift: 0.0`.

## The full vanilla table

| L | attn | KV | PLE imp | r=2 ppl | r=4 ppl | r=8 ppl | r8/r2 |
|---|------|:--:|--------:|--------:|--------:|--------:|------:|
|  0 | sliding |   | +0.0215 | 1909 | 1095 | 1099 | 0.6 |
|  1 | sliding |   | +0.0175 | 394 | 913 | 1129 | 2.9 |
|  2 | sliding |   | +0.0263 | 52 | 717 | 1680 | 32.4 |
|  3 | sliding |   | **+1.8302** | 267 | 688 | 855 | 3.2 |
|  4 | full    |   | +0.0274 | 102 | 784 | 3096 | 30.3 |
|  5 | sliding |   | +0.0110 | 27 | 228 | 774 | 28.9 |
|  6 | sliding |   | +0.0111 | 17 | 182 | 576 | 32.9 |
|  7 | sliding |   | +0.0121 | 15 | 65 | 543 | 36.2 |
|  8 | sliding |   | **+0.6453** | 62 | 1987 | 2045 | 33.1 |
|  9 | full    |   | +0.0121 | 29 | 369 | 5104 | 175.6 |
| 10 | sliding |   | +0.0058 | 25 | 816 | 5218 | 209.3 |
| 11 | sliding |   | +0.0371 | **89,336** | **34M** | **56M** | 630 |
| 12 | sliding |   | +0.0548 | 1242 | 129,393 | 81,336 | 65.5 |
| 13 | sliding |   | **+0.3471** | 107,616 | 204,664 | 261,567 | 2.4 |
| 14 | full    |   | +0.0136 | 202 | 3163 | 1873 | 9.3 |
| **15** | sliding | ✓ | +0.0624 | **18.8** | **30.8** | **35.3** | **1.9** |
| **16** | sliding | ✓ | +0.0134 | **13.6** | **20.7** | **38.8** | **2.8** |
| **17** | sliding | ✓ | +0.0040 | **13.5** | **20.5** | **42.6** | **3.2** |
| **18** | sliding | ✓ | +0.0162 | **14.2** | **35.9** | **108.1** | 7.6 |
| **19** | full    | ✓ | +0.0032 | **13.7** | **30.2** | **103.7** | 7.6 |
| 20 | sliding | ✓ | +0.0020 | 15.4 | 71.0 | 417 | 27.1 |
| 21 | sliding | ✓ | +0.0038 | 14.3 | 35.9 | 454 | 31.8 |
| 22 | sliding | ✓ | −0.0022 | 14.6 | 45.6 | 976 | 66.6 |
| 23 | sliding | ✓ | +0.0764 | 18.8 | 118.9 | 592 | 31.5 |
| 24 | full    | ✓ | +0.0248 | 18.1 | 140.9 | 4391 | 242.8 |
| 25 | sliding | ✓ | +0.0054 | 15.4 | 51.6 | 1143 | 74.3 |
| 26 | sliding | ✓ | +0.0036 | 14.1 | 32.4 | 860 | 61.2 |
| 27 | sliding | ✓ | +0.0039 | 13.9 | 28.6 | 480 | 34.6 |
| 28 | sliding | ✓ | +0.0045 | 14.4 | 29.5 | 524 | 36.4 |
| 29 | full    | ✓ | +0.0028 | 14.3 | 47.5 | 2435 | 170.1 |
| 30 | sliding | ✓ | +0.0058 | 13.4 | 23.7 | 490 | 36.6 |
| 31 | sliding | ✓ | +0.0037 | 13.7 | 46.8 | 36,895 | 2692 |
| 32 | sliding | ✓ | +0.0098 | 14.1 | 47.7 | 56,541 | 3998 |
| 33 | sliding | ✓ | **+0.7405** | 53 | 45,222 | **5.5e+9** | 1e+8 |
| 34 | full    | ✓ | +0.1137 | **1.8e+18** | **1.2e+19** | **2.9e+20** | 163 |

Bold rows: the "valley" — layers that stay within 10× baseline at r=8.

![Round 2c full loop tolerance map](figs/fig04_round2c_loop_tolerance.png)

## Top 5 loopable layers (lowest r=8 ppl, vanilla)

| Rank | Layer | ppl at r=8 |
|---|---|---:|
| 1 | 15 | 35.3 |
| 2 | 16 | 38.8 |
| 3 | 17 | 42.6 |
| 4 | 19 | 103.7 |
| 5 | 18 | 108.1 |

## Top 5 fragile layers

| Rank | Layer | ppl at r=8 |
|---|---|---:|
| 1 | 34 | 2.9 × 10²⁰ |
| 2 | 33 | 5.5 × 10⁹ |
| 3 | 11 | 5.6 × 10⁷ |
| 4 | 13 | 2.6 × 10⁵ |
| 5 | 12 | 8.1 × 10⁴ |

## Structural observations

### The valley

Layers **15–19** are qualitatively different from everything else. At r=8:

- L15: 35.3, L16: 38.8, L17: 42.6, L18: 108.1, L19: 103.7 (all under 110)
- L20: 417, L21: 454 (already 5–10× worse)
- L14: 1873 (50× worse than L15)

The valley has a **sharp left edge** (L14 → L15 is a 50× ppl drop) and a
**gradual right edge** (L19 → L20 is 4×, L20 → L21 is ~flat).

### The KV boundary

The left edge of the valley aligns exactly with the KV
producer→consumer boundary. `is_kv_consumer` flips from `False` to `True`
between layer 14 and layer 15. This is very unlikely to be coincidence.
Round 3b confirmed it by explicitly testing blocks that cross the boundary
— they all break catastrophically.

### What does NOT predict loopability

The analysis section records Pearson and Spearman correlations between r=8
vanilla ppl and four candidate factors:

| Factor | Pearson ρ | Spearman ρ |
|--------|----------:|-----------:|
| PLE importance | −0.003 | +0.392 |
| Attention type (full=1, sliding=0) | +0.343 | +0.262 |
| is_kv_consumer (1/0) | +0.149 | −0.349 |
| Layer index | +0.289 | +0.072 |

None of these is strong. The best Spearman (+0.392) is with PLE importance,
but with a direction that's counterintuitive: higher PLE importance
correlates with *worse* loop tolerance only through the non-consumer region
(where L3, L8, L13 spike on both dimensions). Inside the consumer region,
PLE importance is uniformly tiny but loopability varies enormously. The
apparent correlation is a spurious consequence of the KV boundary
coincidence.

**Translation:** no single architectural property predicts loopability.
Whatever makes layer 11 blow up to 5.6 × 10⁷ while layer 15 stays at 35 is
not captured by the four variables we recorded.

### r=8 / r=2 ratios

This ratio measures compounding rate — how much worse looping gets as r
grows, independent of how bad the first extra loop is:

- **Valley** (L15–L19): ratio 1.9–7.6. These layers have stable fixed-point
  behavior; extra iterations don't spiral.
- **Late consumer region** (L20–L30): ratio 27–170. Layers that look okay
  at r=2 (all under 20) explode by r=8.
- **L31, L32, L33**: ratios 2,700 / 4,000 / 100M. These are "slow drift"
  layers that look fine at r=2 but diverge explosively.
- **L34**: ppl is already 10¹⁸ at r=2, so the r8/r2 ratio of 163 doesn't
  mean the layer is stable — it means it's broken from the first loop and
  then keeps breaking.

## once-vs-vanilla at r=8

![Once vs vanilla delta at r=8](figs/fig05_round2c_once_vs_vanilla.png)

**Top 5 where `once` helps most at r=8** (absolute ppl delta):

| Layer | vanilla ppl | once ppl | absolute delta |
|---|---:|---:|---:|
| 33 | 5.5e9 | 2.3e8 | 5.2e9 |
| 13 | 2.6e5 | 1.5e5 | 1.1e5 |
| 31 | 36,895 | 26,375 | 10,520 |
| 10 | 5,218 | 2,343 | 2,876 |
| 14 | 1,873 | 769 | 1,103 |

**Top 5 where `once` hurts most:**

| Layer | vanilla ppl | once ppl | absolute delta |
|---|---:|---:|---:|
| 34 | 2.9e20 | 8.9e21 | −8.6e21 |
| 11 | 5.6e7 | 3.4e8 | −2.8e8 |
| 12 | 81,336 | 159,115 | −77,779 |
| 9 | 5,104 | 23,679 | −18,574 |
| 1 | 1,129 | 2,414 | −1,285 |

### No systematic once-vs-vanilla pattern

Out of 35 layers, `once` helps (vs vanilla) at 16 layers and hurts at 19
layers at r=8. The "helps" and "hurts" groups don't cluster in any
architecturally-meaningful way:

- At PLE-important layers (L3, L8, L13, L33), `once` helps at L8, L13, L33
  but hurts at L3.
- Inside the valley, `once` helps at L15, L16, L17, L18 but hurts at L19
  (the one global-attention layer inside the valley).

The plan's interpretation:

> "Once does not follow any clean pattern. Drop `once` from the retrofit
> plan; complexity not worth it."

This is why Rounds 3a, 3b, and 3c all test vanilla only.

## Sanity watches

The plan named two specific predictions to watch:

1. **Layer 4** (global, non-KV-consumer). Its r=8 ppl is 3096, which is
   worse than neighbors L5 (774), L6 (576), L7 (543) but better than L8
   (2045). It's wildly different from valley layers — no single factor
   explains that, consistent with the "no dominant correlate" finding.

2. **Layer 33** (sliding, 2nd-highest PLE importance, one before final
   global). Prediction was "if this blows up, PLE-important layers are
   un-loopable; if it's fine, PLE importance and loop tolerance are
   decoupled." It blew up — 5.5 × 10⁹ at r=8. Consistent with the plan's
   first hypothesis, though the data doesn't rule out alternatives.

## Interpretation

The plan had 8 scenario buckets. The data fits **Scenario E — multiple
factors combine nontrivially** most closely. Key takeaways:

1. The valley (15–19) is the safe zone. It's 5 layers wide and bounded by
   the KV boundary on the left.
2. The KV boundary is the most load-bearing architectural feature for this
   method. Everything changes on crossing it.
3. The last two layers (33, 34) are unusable in any looping setup. L33 has
   PLE signal so large that re-injection accumulates; L34 is a full-
   attention layer dedicated to final output prediction.
4. Most of the stack (layers 0–14, 20–34 excluding the valley) is not
   loopable via single-layer probing — but that doesn't rule out block-
   looping working in some of those regions. That's Rounds 3a–3c.

## Next-step implications

Round 3a (pair-looping) was written to ask: "are the catastrophic results
caused by the layer itself, or by the 34 downstream layers processing an
out-of-distribution hidden state?" The 5-layer valley looked suspiciously
narrow in a way that could either be a ceiling (only these layers can ever
be looped) or a floor (they're the only places where downstream cascade
damage happens to be small, and block-looping will rescue neighboring
fragile layers). Rounds 3b/3c answered this decisively — the valley is the
floor, but the KV boundary is an absolute ceiling.
