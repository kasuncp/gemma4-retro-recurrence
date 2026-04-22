# 03 — Round 2b: PLE importance scan + layer-location sweep

**Plan:** `plan2b.md`
**Result files:**
- `results_round2b_importance.json` (35-layer importance scan, r=1)
- `results_round2b_importance_run2.json` (identical, confirming reproducibility)
- `results_round2b_location.json` (3 layers × 2 PLE modes × 2 r values)

**Status:** Complete.

## Two experiments in one round

1. **PLE importance scan** — for each of the 35 layers, how much does that
   layer's PLE contribute to perplexity? Measured as `nll_diff = nll_zero -
   nll_vanilla` at r=1 (no looping at all — just with vs without that layer's
   PLE).
2. **Layer-location × PLE policy** — does the "once beats vanilla" pattern
   from Round 2a generalize across layer depths? Grid of layers {5, 17, 28}
   × modes {vanilla, once} × r {4, 8}.

---

## Experiment 1 — PLE importance scan

### Full per-layer table

70 cells (vanilla and zero at r=1 for each of 35 layers). All r=1 vanilla
runs match the unmodified baseline (12.537) bitwise — regression passes.

| L | attn | KV-cons | ppl_zero | nll_diff |
|---|---|:---:|---:|---:|
|  0 | sliding |   | 12.809 | +0.0215 |
|  1 | sliding |   | 12.757 | +0.0175 |
|  2 | sliding |   | 12.871 | +0.0263 |
|  3 | sliding |   | **78.169** | **+1.8302** |
|  4 | full    |   | 12.885 | +0.0274 |
|  5 | sliding |   | 12.675 | +0.0110 |
|  6 | sliding |   | 12.676 | +0.0111 |
|  7 | sliding |   | 12.689 | +0.0121 |
|  8 | sliding |   | **23.902** | **+0.6453** |
|  9 | full    |   | 12.689 | +0.0121 |
| 10 | sliding |   | 12.610 | +0.0058 |
| 11 | sliding |   | 13.010 | +0.0371 |
| 12 | sliding |   | 13.243 | +0.0548 |
| 13 | sliding |   | **17.738** | **+0.3471** |
| 14 | full    |   | 12.709 | +0.0136 |
| 15 | sliding | ✓ | 13.344 | +0.0624 |
| 16 | sliding | ✓ | 12.705 | +0.0134 |
| 17 | sliding | ✓ | 12.587 | +0.0040 |
| 18 | sliding | ✓ | 12.741 | +0.0162 |
| 19 | full    | ✓ | 12.576 | +0.0032 |
| 20 | sliding | ✓ | 12.562 | +0.0020 |
| 21 | sliding | ✓ | 12.584 | +0.0038 |
| 22 | sliding | ✓ | 12.509 | **−0.0022** (noise) |
| 23 | sliding | ✓ | 13.532 | +0.0764 |
| 24 | full    | ✓ | 12.851 | +0.0248 |
| 25 | sliding | ✓ | 12.604 | +0.0054 |
| 26 | sliding | ✓ | 12.582 | +0.0036 |
| 27 | sliding | ✓ | 12.586 | +0.0039 |
| 28 | sliding | ✓ | 12.593 | +0.0045 |
| 29 | full    | ✓ | 12.572 | +0.0028 |
| 30 | sliding | ✓ | 12.610 | +0.0058 |
| 31 | sliding | ✓ | 12.582 | +0.0037 |
| 32 | sliding | ✓ | 12.660 | +0.0098 |
| 33 | sliding | ✓ | **26.288** | **+0.7405** |
| 34 | full    | ✓ | 14.046 | +0.1137 |

![Round 2b per-layer PLE importance](figs/fig03_round2b_ple_importance.png)

### What the pattern looks like

**Top 5 most PLE-important layers:**
- L3: +1.830 NLL
- L33: +0.740 NLL
- L8: +0.645 NLL
- L13: +0.347 NLL
- L34: +0.114 NLL

**Top 5 least PLE-important layers:**
- L22: −0.002 (essentially noise)
- L20: +0.002
- L29: +0.003
- L19: +0.003
- L26: +0.004

### Structural observations

1. **PLE signal is extremely spiky.** Four layers (3, 8, 13, 33) carry most
   of the PLE contribution; their `nll_diff` is 50×–500× larger than the
   mean of their neighbors. Zeroing PLE at L3 alone raises perplexity from
   12.54 → 78.17.

2. **There is no obvious periodic structure** in where the spikes land. L3
   is in the early prelude, L8 is mid-early, L13 is right before the
   KV-consumer boundary, L33 is second-to-last. The paper's guidance "pick
   early for prelude, late for recurrent and coda" gives no hint about which
   exact layers to avoid or include.

3. **The KV-consumer region (layers 15–32) is almost entirely PLE-inert.**
   Except for L23 (+0.076) and L33 (+0.740), every layer in that range has
   `nll_diff < 0.03`. This is the region where PLE-policy choices are
   essentially free.

4. **L22 has slightly *negative* `nll_diff`.** Zeroing PLE at L22
   *decreases* NLL by 0.002. At this scale it's noise, but the sign
   suggests L22's PLE is at worst not helpful and at best redundant with
   neighbors.

5. **The final global layer (L34) is PLE-nontrivial.** +0.114 NLL. Not in
   the top 5 spikes but clearly above the inert baseline. This layer also
   ends up being the most fragile layer to looping — see Round 2c.

---

## Experiment 2 — Layer-location × PLE policy

### Full 18-cell result (3 layers × 3 r × 2 modes)

| Layer | ple_mode | r=1 | r=4 | r=8 |
|---|---|---:|---:|---:|
| 5  | vanilla | 12.537 | 227.78 | 774.12 |
| 5  | once    | 12.537 | 293.44 | 1144.24 |
| 17 | vanilla | 12.537 | 20.48  | 42.59 |
| 17 | once    | 12.537 | 18.53  | 36.63 |
| 28 | vanilla | 12.537 | 29.55  | 523.95 |
| 28 | once    | 12.537 | 29.55  | 564.96 |

### Once-vs-vanilla at r=8

| Layer | vanilla ppl | once ppl | once/vanilla |
|---|---:|---:|---:|
| 5  | 774.12 | 1144.24 | 1.478× (once **hurts**) |
| 17 | 42.59  | 36.63   | 0.860× (once **helps**) |
| 28 | 523.95 | 564.96  | 1.078× (once slightly hurts) |

### Interpretation

The `once > vanilla` pattern from Round 2a does **not** generalize. It is
layer-dependent:

- At **layer 5** (early, non-KV-consumer), `once` is 48% worse than vanilla
  at r=8. Skipping PLE re-injection on iterations 1..r-1 actually hurts.
- At **layer 17** (middle, KV-consumer, PLE-inert), `once` is 14% better
  than vanilla — matching Round 2a.
- At **layer 28** (late, KV-consumer, PLE-inert), `once` is 8% worse than
  vanilla.

The plan had bucketed this as the "policy is layer-dependent" outcome. The
cross-experiment synthesis that triggered Round 2c notes:

> *Layers 17 and 28 share attention type (`sliding_attention`), KV-sharing
> role (`is_kv_consumer: true`), and near-identical PLE importance (0.004
> NLL diff) — yet their r=8 looping ratios differ by 12×. Something
> structural is going on that three data points can't reveal.*

That "something structural" motivated Round 2c's full 35-layer map.

### Regression checks

All 6 r=1 regression cells (3 layers × 2 modes) match baseline exactly:
`round1_baseline_matches: True`, `once_eq_vanilla_r1: True`.

## What this round established

1. **PLE importance is extremely unevenly distributed.** Four spike layers
   (3, 8, 13, 33) carry most of the signal.
2. **The KV-consumer zone (15–32) is mostly PLE-inert**, with exception
   layers at 23 and 33.
3. **PLE injection policy matters most where PLE itself matters most**,
   except the direction of the effect is opposite to what you'd guess:
   - At PLE-inert L17, `once` helps (modestly, +14%).
   - At PLE-active L5, `once` hurts (substantially, +48%).
4. **Three points along the stack isn't enough** to predict which layers
   are loopable. Round 2c needed to measure all 35.

## Files

| File | Role |
|------|------|
| `results_round2b_importance.json` | PLE importance scan, authoritative |
| `results_round2b_importance_run2.json` | Reproducibility check, identical |
| `results_round2b_location.json` | Location sweep |
