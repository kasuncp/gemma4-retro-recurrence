# Path 2 v2 — Phase 3: Per-Layer r-Sweep Diagnostic

**Project:** Retrofitted recurrence on Gemma 4 E2B
**Status:** Ready for implementation (Phase 2 complete; halt verdict — see `results/path_2_depth_recurrence_v2/phase2/phase2_summary.json`)
**Depends on:** `phase2_layer_map.md` (per-layer r=8 reasoning map; pre-flight L17-r1 token match passed)
**Blocks:** Phase 4 (one of F1 retrofit / F2 healing / F3 anti-loop / F4 sub-block pivot — *which one* is the output of this phase)
**Pod budget:** ~3 pod-hours on a single RTX 4090 (~$3 at $0.69/h spot)

---

## Why Phase 3 exists

Phase 2 swept all 35 decoder layers at r=8 (maximum stress) on the IT
model and put every layer in the **broken** bucket. Its halt decision
routed straight to "Phase 6 retrofit training", but the per-cell data
shows two distinct failure modes that imply *different* training pivots:

- **Loop-pathology cluster** (L15–L34, mostly sliding-attention KV
  consumers): `loop_rate ∈ [0.44, 1.00]`, `mean_gen_tokens` saturating
  at 512 — generation enters a fixed point and never escapes.
- **EOS-collapse cluster** (L00, L03, L11, L28, L31): `loop_rate < 0.10`
  but `mean_gen_tokens ∈ [2, 32]` — recurrence destroys the residual so
  thoroughly the model emits EOS within a few tokens.

`phase0.md`'s F1 retrofit-training trigger was *"any pretrained-only
config preserves reasoning within 3 pp of baseline-C2"*. Phase 2 fired
the F1 trigger's negation, so F1 isn't directly actionable. Three
post-Phase-5 directions are still on the table — F2 healing, F3
anti-loop decoding, F4 sub-block pivot — and committing to any of them
without first disambiguating the failure mode is the same bet-without-a-gate
mistake Phase 0 D7 was written to prevent.

Phase 3 runs **the cheapest probe that distinguishes these failure
modes**: an r-sweep at r ∈ {2, 4, 6} on the four most diagnostic layers
from Phase 2. The output is a Phase 4 launch decision, the same way
Phase 2's output was a Phase 3 launch decision. No retrofit training
happens until Phase 4.

This phase produces no architectural decisions on its own. It produces
a **routing signal**: which of F1/F2/F3/F4 the next phase implements,
backed by an empirical measurement of how loop pathology, EOS collapse,
and accuracy each scale with r.

---

## Scope

**In scope** (12 main cells + 1 reused-from-Phase-2 anchor per layer,
~3 h on a 4090):

| Group | Cells | Config | Benchmark | N | Gate (per cell) |
|---|---|---|---|---|---|
| r-sweep | `L17-r2`, `L17-r4`, `L17-r6` | (17,17), C2 | GSM8K | 50 | bucket assignment via `phase2_gates.classify_layer` |
| r-sweep | `L28-r2`, `L28-r4`, `L28-r6` | (28,28), C2 | GSM8K | 50 | same |
| r-sweep | `L31-r2`, `L31-r4`, `L31-r6` | (31,31), C2 | GSM8K | 50 | same |
| r-sweep (control) | `L21-r2`, `L21-r4`, `L21-r6` | (21,21), C2 | GSM8K | 50 | same |
| Reused | `L17-r8`, `L28-r8`, `L31-r8`, `L21-r8` | from Phase 2 | GSM8K | 50 | already on disk; re-read for the curve |
| Reused | `L17-r1` token match | from Phase 2 pre-flight | GSM8K | 20 | already passed (20/20) — confirms hook works at single-layer width; no rerun |

**Phase 4 launch gate** (computed *after* the 12 r-sweep cells finish,
combining each layer's 4-point r curve including the r=8 reuse):

| Outcome | Phase 4 action |
|---|---|
| Any (layer, r) lands in **preserved** bucket | Launch **Phase 4 = F1 retrofit (LoRA)** anchored on that (layer, r). The F1 gate from `phase0.md` is now satisfied. |
| Any (layer, r) lands in **degraded** AND loop_rate ≤ 0.30 | Launch **Phase 4 = F2 healing-then-retrofit** anchored on that (layer, r). The healing recipe addresses the residual gap. |
| All cells **broken** AND loop_rate decreases monotonically with decreasing r AND any cell has loop_rate < 0.30 at r=2 | Launch **Phase 4 = F3 anti-loop decoding**. Pathology is r-driven; runtime r-annealing or loop-detector aborts may rescue without retraining. |
| All cells **broken** AND loop_rate is approximately flat across r (no cell drops below 0.30 even at r=2) | Launch **Phase 4 = F4 sub-block pivot** (FFN-only / attention-only loops on L17 ± 2). Pathology is structural in the full layer; isolate which sub-block carries it. |
| EOS-collapse layer (L31) shows `mean_gen_tokens` recovers ≥ 200 at r=2 *and* loop_rate stays < 0.30 | Document as a sub-finding; the EOS-collapse mechanism is r-driven. Does NOT change the launch decision unless the layer also lands in degraded/preserved. |

**Out of scope:**

- Block looping. Phase 2 chose single-layer-first per D12; Phase 3 stays
  single-layer for the same reason — a block at r=2 confounds "lower r
  rescues this layer" with "the block dilutes per-layer stress".
  Block r-sweeps are Phase 4's job *if* F1 or F2 fires.
- New layers. Phase 2 already gave us the per-layer ranking; rerunning
  the layer scan at r=2 (35 cells × ~5 min ≈ 3 h on its own) is not
  worth the spend before the 4-layer probe answers the routing question.
- ARC / BBH. Same rationale as Phase 2 — GSM8K is the discriminating
  benchmark for this question; ARC's prompt-format inversion (Path 1
  plan 7) and BBH's saturation pattern would muddy the loop-vs-r signal.
- 8-shot CoT controls. Cross-round bridging is settled (Phase 1 1B-r5
  passed); we don't re-run controls per phase.
- PLE strategy ablations. Same as Phase 2 — `every-iter` only.
- N=200 confirmation. N=50 is sufficient to triage (Wilson 95 % CI
  ±13 pp); Phase 4 picks one (layer, r) and runs N=200 on that single
  config as part of its sanity gate.
- Retrofit training itself. That's Phase 4. Phase 3's only job is to
  pick *which* Phase 4 plan to write.

If a Phase 3 cell would require a new prompt builder, a new extractor,
a new generation kwarg, or a new hook code path beyond what Phases 0–2
already shipped, stop and re-scope. The point of this phase is to *vary
r* on the existing harness.

---

## Why these four layers

Selected from Phase 2's 35-layer table by least-pathological-among-KV-consumers,
plus one worst-case control:

| Layer | Phase 2 (r=8) acc / loop / trunc / mean_tok | Attn type | Tertile | Why selected |
|---|---|---|---|---|
| **L17** | 0.04 / 0.46 / 0.04 / 501 | sliding | mid | Lowest loop_rate among mid KV-consumers; the W5 block centroid (round 5 anchor); pre-flight target already validated |
| **L28** | 0.02 / 0.44 / 0.02 / 190 | sliding | late | Second-lowest loop; late-tertile representative; intermediate `mean_gen_tokens` straddles loop-and-EOS modes |
| **L31** | 0.00 / 0.32 / 0.02 / 183 | sliding | late | Lowest loop_rate of all 35 layers; **but** `mean_gen_tokens ≈ 183` suggests EOS-collapse — does lower r let it generate longer? |
| **L21** | 0.00 / 1.00 / 0.12 / 512 | sliding | mid | **Worst-case control.** Loop_rate = 1.00 at r=8 — does r=2 tame even guaranteed loopers? If yes → F3 fireable. If no → loop pathology is structural at this layer regardless of r. |

**Architectural diversity is constrained by Phase 2's data.** All four
selected layers happen to be sliding-attention KV-consumers because
Phase 2's full-attention layers (L04, L09, L14, L19, L24, L29, L34) all
showed worse pathology than the corresponding sliding-attention
neighbours. We accept this — testing a full-attention layer here would
double the cell count without changing the routing decision (the F4
plan, if triggered, will revisit attention vs FFN at the sub-block
level).

**L17 vs L18.** L18 had `loop=0.58` vs L17's `0.46` and is otherwise
near-identical. We pick L17 to keep continuity with Phase 2's pre-flight
and the W5 anchor. If L17 recovers at lower r, Phase 4's F1/F2 plan
will N=200-confirm L17 *and* re-test L18 for block extension.

**L11 was considered and rejected.** Phase 2: `loop=0.06 trunc=0.00
mean_tok=31`. Pure EOS-collapse, no loop pathology. Including L11
wouldn't disambiguate F3 vs F4 — it'd only re-measure EOS-collapse,
which is already characterised. L31 covers the EOS-collapse axis with
the bonus of being late-tertile and KV-consumer.

---

## Cell-by-cell specification

### r-sweep cells — `L<NN>-r<R>`

**Symmetry note:** every cell shares the same spec; only `(layer, r)`
varies. Section describes one cell; the runner handles all 12.

**Config (per cell):** `L<NN>-r<R>` (block `(L, L)`, r ∈ {2, 4, 6},
every-iter PLE, C2 prompt).
**Model:** `google/gemma-4-E2B-it`.
**Benchmark:** GSM8K, N=50, seed=42 (the same first-50 problems Phase 2
used; subset of Phase 1's 1A N=200).
**Generation:** `max_new_tokens=512`, greedy, `use_cache=False`,
`pad_token_id=eos`. Same as Phase 2; do not change.

**Per-cell bucket** (computed by `phase2_gates.classify_layer`, reused
verbatim — no new gates module):

| Bucket | Criterion |
|---|---|
| **preserved** | `acc ≥ 0.65` AND `loop < 0.50` AND `trunc < 0.30` |
| **degraded** | `0.40 ≤ acc < 0.65` AND `loop < 0.50` AND `trunc < 0.30` |
| **broken** | `acc < 0.40` OR `loop ≥ 0.50` OR `trunc ≥ 0.30` |

**Reusing Phase 2's gate is intentional.** A bucket that Phase 2
considered "broken" at r=8 should remain "broken" at r=4 unless the
rescue is real. New thresholds tuned to Phase 3 data would be result-
fitting.

**Per-cell secondary metrics** (recorded but not gated, all from
Phase 2's schema):

| Metric | Why it matters in Phase 3 |
|---|---|
| `mean_gen_tokens` | The EOS-collapse axis. L31 at r=2 with `mean_tok ≥ 200` and `loop < 0.30` falsifies the "single-layer recurrence destroys the residual stream" reading |
| `accuracy_legacy` | Phase 4's F1/F2 retrofit plan compares against the round-5 8-shot anchor (54.8 %); legacy column gives the bridge for free |
| `pathology_flag` | Same Phase 2 trigger (`loop > 0.5 OR trunc > 0.5`) — used by the plotting code to flag bars in the figure |

**Per-cell secondary signal — the loop-progression curve:**

For each layer, the four-point sequence
`loop_rate(r=2), loop_rate(r=4), loop_rate(r=6), loop_rate(r=8)` is
recorded as a list in the summary. Phase 4's launch logic uses
**monotonicity** (Spearman rank correlation between r and loop_rate
≥ 0.8) as a stricter test of "loop pathology is r-driven" than just
checking the r=2 value.

**On failure (per-cell):** failures are bucket assignments, not halts.
Phase 3 runs all 12 cells regardless; the launch decision uses the joint
distribution.

**The exception:** if any r=2 cell produces NaN/Inf in `accuracy_smart_v2`
or `mean_gen_tokens`, halt that cell only and inspect the JSONL — Phase 2
never produced NaN at r=8, so a r=2 NaN is a hook regression that needs
investigating before the curve is interpreted.

**Wall budget (per cell):**
- r=2: ~5 min (per-token cost ≈ 14 s × (1 + 1/35) ≈ 14.4 s; 50 problems
  × ~6 s avg with EOS happening earlier than r=8 → ~5 min)
- r=4: ~7 min
- r=6: ~9 min
- r=8 (already on disk): 0 min
- Total: ~12 cells × ~7 min avg = ~85 min, plus ~1 model load (~30 s)
  shared across all cells.

---

## Implementation deliverables

### 1. CONFIGS extension (~10 lines, in `probes/eval_v3.py`)

12 new entries, generated programmatically (mirrors Phase 2's pattern):

```python
# Phase 3: r-sweep on selected layers from Phase 2's least-pathological set.
# r=8 entries (L17-r8, L21-r8, L28-r8, L31-r8) already exist from Phase 2.
PHASE3_LAYERS = [17, 21, 28, 31]
PHASE3_R_VALUES = [2, 4, 6]
for _L in PHASE3_LAYERS:
    for _R in PHASE3_R_VALUES:
        CONFIGS[f"L{_L:02d}-r{_R}"] = {
            "prompt": "C2",
            "block": (_L, _L),
            "r": _R,
            "ple_strategy": "every-iter",
            "notes": f"phase 3 r-sweep: L{_L} at r={_R}",
        }
```

### 2. New runner (~200 lines, `experiments/path2_v2_phase3.py`)

Same skeleton as `experiments/path2_v2_phase2.py`. Sequentially shells
out to `path2_v2_eval.py` per cell so a crash in one cell doesn't kill
the rest. Modes:

- (default) — 12 cells, then aggregation that **also reads the four
  r=8 JSONLs from Phase 2**.
- `--summarize-only` — read existing JSONL (own + phase2's), recompute
  the curves, no GPU.
- `--skip-cell L<NN>-r<R>` — skip individual cells (repeatable).
- `--shard <i>/<n>` — for parallel pods. Cell index `k` runs only when
  `k % n == i`. With n=2, wall drops to ~45 min/pod. Probably not worth
  it for a 90-min job.

The runner declares cells as a list of dicts with `name, kind, config,
benchmark, n, model_id`, identical to Phase 2's `CELLS` shape.

**Phase 2 JSONL pickup:** the aggregator reads
`results/path_2_depth_recurrence_v2/phase2/gsm8k__L<NN>-r8.summary.json`
for each `L ∈ PHASE3_LAYERS` and merges those summaries into Phase 3's
`cells` dict under the same key (`L<NN>-r8`). If a summary is missing,
log a warning and continue — the curve will be 3-point instead of
4-point for that layer; the routing logic still works.

### 3. New gate module (~80 lines, `probes/phase3_gates.py`)

Pure-Python evaluator. Returns `(launch_mode, anchor, rationale)`
where `launch_mode ∈ {f1_retrofit, f2_healing, f3_anti_loop, f4_subblock,
halt}` and `anchor` is the (layer, r) tuple that triggered it (or `None`
for f3/f4/halt).

```python
from probes.phase2_gates import classify_layer  # reused

LAYER_R_PRESERVED = ("preserved",)
LAYER_R_DEGRADED  = ("degraded",)
ANTI_LOOP_LOOP_MAX_AT_R2 = 0.30
ANTI_LOOP_MONOTONE_RHO    = 0.80   # Spearman r vs loop_rate

def classify_r_curve(per_cell_summaries: dict[tuple[int, int], dict]) -> dict:
    """per_cell_summaries: {(layer, r): summary_dict_with_at_least_acc/loop/trunc}.

    Returns:
      {"mode": "f1_retrofit"|"f2_healing"|"f3_anti_loop"|"f4_subblock"|"halt",
       "anchor": (layer, r) or None,
       "rationale": str,
       "per_cell_buckets": {(layer, r): bucket_str}}
    """
    buckets = {key: classify_layer(s)[0] for key, s in per_cell_summaries.items()}

    # 1. Any preserved cell -> F1
    for key, b in buckets.items():
        if b == "preserved":
            return {"mode": "f1_retrofit", "anchor": key, "rationale": ...,
                    "per_cell_buckets": buckets}

    # 2. Any degraded cell with loop <= 0.30 -> F2
    for key, b in buckets.items():
        if b == "degraded" and per_cell_summaries[key]["loop_rate"] <= 0.30:
            return {"mode": "f2_healing", "anchor": key, "rationale": ...,
                    "per_cell_buckets": buckets}

    # 3. Loop monotonicity check per layer
    layer_keys = sorted({L for (L, _) in per_cell_summaries})
    for L in layer_keys:
        rs = sorted(r for (LL, r) in per_cell_summaries if LL == L)
        if len(rs) < 3:  # need at least 3 r values for a curve
            continue
        loops = [per_cell_summaries[(L, r)]["loop_rate"] for r in rs]
        rho = spearman(rs, loops)
        loop_at_r2 = per_cell_summaries.get((L, 2), {}).get("loop_rate", 1.0)
        if rho >= ANTI_LOOP_MONOTONE_RHO and loop_at_r2 < ANTI_LOOP_LOOP_MAX_AT_R2:
            return {"mode": "f3_anti_loop", "anchor": (L, 2), "rationale": ...,
                    "per_cell_buckets": buckets}

    # 4. All cells broken AND no monotone-r-driven loop signal -> F4
    return {"mode": "f4_subblock", "anchor": None, "rationale": ...,
            "per_cell_buckets": buckets}
```

`spearman` is a 5-line implementation (rank both inputs, Pearson on
ranks). No scipy import. Tests cover the three boundary cases:
preserved-fires-first, degraded-with-low-loop-fires-second,
monotone-loop-fires-third.

### 4. Plot + CSV (~140 lines, `experiments/path2_v2_r_sweep_plot.py`)

Two artefacts under the phase 3 results dir:

- `phase3_r_sweep.png` — matplotlib figure, 2×2 grid of small multiples
  (one panel per layer). Each panel: x-axis = r ∈ {2, 4, 6, 8}, three
  lines (acc, loop_rate, trunc_rate) with the bucket boundary at
  `LAYER_PRESERVED_ACC_MIN = 0.65` drawn as a horizontal reference.
  Each datapoint annotated with bucket colour (preserved=green,
  degraded=amber, broken=red). Title cites N=50 and the Wilson 95 % CI
  reminder.
- `phase3_r_sweep.csv` — one row per (layer, r) cell:
  `layer, r, accuracy_smart_v2, accuracy_legacy, loop_rate,
  truncation_rate, mean_gen_tokens, attention_type, is_kv_consumer,
  depth_tertile, bucket`. Phase 4 reads this if F1/F2 fires.

Per the user's CLAUDE.md memory, PNGs land alongside JSON under
`results/path_2_depth_recurrence_v2/phase3/`.

### 5. No new prompt / extractor / hook code

This is a re-execution of the Phase 2 harness with a different `r`. If
the runner discovers it needs *anything* beyond Phase 2's surface area,
the implementation is wrong — stop and reconcile.

---

## Output / report shape

After the 12 cells finish (and the 4 Phase 2 r=8 summaries are merged),
`experiments/path2_v2_phase3.py` writes
`results/path_2_depth_recurrence_v2/phase3/phase3_summary.json`:

```json
{
  "phase": "phase3-r-sweep-diagnostic",
  "model_ids": {"it": "google/gemma-4-E2B-it"},
  "depends_on": {
    "phase2_summary": "results/path_2_depth_recurrence_v2/phase2/phase2_summary.json",
    "preflight_passed": true
  },
  "cells": {
    "L17-r2": {"summary": {...}, "metadata": {...}, "bucket": "broken",
               "bucket_reason": "acc=0.04 loop=0.30 trunc=0.04"},
    ...
    "L17-r8": {"summary": {...}, "from_phase2": true, "bucket": "broken",
               "bucket_reason": "acc=0.04 loop=0.46 trunc=0.04"},
    ...
  },
  "r_curves": {
    "L17": {
      "r":          [2,    4,    6,    8],
      "accuracy":   [0.04, 0.04, 0.02, 0.04],
      "loop_rate":  [0.30, 0.38, 0.42, 0.46],
      "trunc_rate": [0.02, 0.02, 0.04, 0.04],
      "mean_tok":   [380,  445,  490,  501],
      "loop_rho_vs_r": 0.95
    },
    "L21": {...},
    "L28": {...},
    "L31": {...}
  },
  "phase4_launch": {
    "mode": "f1_retrofit" | "f2_healing" | "f3_anti_loop" | "f4_subblock",
    "anchor": ["L17", 2],
    "rationale": "L17-r2 lands in degraded with loop_rate=0.18 — F2 healing-then-retrofit recipe applies.",
    "per_cell_buckets": {"L17-r2": "degraded", ...}
  },
  "wall_seconds_total": XXXX
}
```

Console table at the end of the run:

```
=== Phase 3 r-sweep (N=50/cell, C2 prompt, IT model) ===
cell      acc    loop   trunc  mean_tok  bucket
L17-r2    0.06   0.18   0.02   380       broken
L17-r4    0.04   0.32   0.04   445       broken
L17-r6    0.02   0.40   0.04   490       broken
L17-r8*   0.04   0.46   0.04   501       broken    (* from phase 2)
L21-r2    0.00   0.92   0.10   512       broken
...
L31-r2    0.00   0.10   0.00   245       broken
...

=== Per-layer loop-vs-r monotonicity (Spearman rho) ===
L17: rho=+0.95   loop@r2=0.18  -> r-driven loop pathology
L21: rho=+0.30   loop@r2=0.92  -> structural loop (r doesn't help)
L28: rho=+0.80   loop@r2=0.20  -> r-driven loop pathology
L31: rho=+0.40   loop@r2=0.10  -> low loop already; EOS-collapse axis

=== Phase 4 launch decision ===
mode: f3_anti_loop
anchor: (L17, 2)
rationale: 2/4 layers (L17, L28) show r-driven loop pathology with
           loop_rate < 0.30 at r=2 and Spearman rho >= 0.80; no cell
           lands in preserved or degraded. Anti-loop runtime decoding
           (r-annealing or loop-detector aborts) is the cheapest test
           before committing training compute.
```

`phase3_r_sweep.png` and `phase3_r_sweep.csv` ship alongside.

---

## Phase 4 launch decision logic — written before looking

The four launch modes correspond directly to `phase0.md`'s F1, F2, F3,
F4 future directions. Pre-committing to which signal triggers which
plan prevents result-fitting:

### `f1_retrofit` (matches `phase0.md` F1)
**Trigger:** any (layer, r) cell with `acc ≥ 0.65 AND loop < 0.50 AND
trunc < 0.30`.
**Phase 4 = full retrofit-training plan.** LoRA on the (layer, r)
anchor; same recipe as McLeish et al. 2025; budget ~24 H100-h. Pre-commit
to "no improvement after training = ship the pretrained config" gate
(2 pp lift on GSM8K minimum, per F1's existing definition).

### `f2_healing` (matches `phase0.md` F2)
**Trigger:** any (layer, r) cell with `0.40 ≤ acc < 0.65 AND loop ≤ 0.30
AND trunc < 0.30`. (Tighter loop threshold than the standard `degraded`
bucket because F2's healing recipe assumes generation pathology is
absent before training begins.)
**Phase 4 = healing-then-retrofit plan.** Healing-LM steps before retrofit;
two ablations: healing length (100 / 500 / 2000 steps) and healing data
mix (raw text only vs raw + CoT). Same anchor as F1. Budget ~30 H100-h.

### `f3_anti_loop` (matches `phase0.md` F3)
**Trigger:** every cell broken, but for at least one layer: Spearman ρ
between r and `loop_rate` ≥ 0.80 *and* `loop_rate(r=2) < 0.30`.
**Phase 4 = anti-loop runtime decoding plan.** No training. Implement
r-annealing during decoding (`r(t) = r_max` for prompt + first ~50
tokens, decay to 1 over next 100 tokens, then 1) and a runtime
loop-detector abort. Re-run Phase 2's 35-layer scan with the modified
decoder. Budget ~6 GPU-h.

### `f4_subblock` (matches `phase0.md` F4)
**Trigger:** every cell broken AND no layer satisfies the F3 monotonicity
+ low-loop-at-r=2 conditions.
**Phase 4 = FFN-only / attention-only sub-block recurrence plan.** Module-
level loops on L15–L19. Tests whether the recurrent dynamic that *might*
help reasoning lives in attention or feed-forward; isolates the
pathology to one sub-component. Budget ~10 GPU-h.

### `halt` (only if pre-flight regresses)
**Trigger:** `L17-r1` token-match against Phase 2 fails (i.e., the hook
broke between phases).
**Action:** halt and fix the hook. Do not interpret r-sweep cells.

The launch logic is deterministic and pure-Python — no manual judgement
between Phase 3 and Phase 4 unless a result genuinely surprises us
(e.g., a layer's `mean_gen_tokens` drops below 50 across all r values,
which `phase2_layer_map.md` already says is forensics-worthy but not a
halt).

---

## Compute budget

| Step | Cells | N | Wall (4090) | Notes |
|---|---|---|---|---|
| r-sweep (r=2) | 4 | 50 | ~20 min | per-token cost ≈ 14.4 s; ~50 × ~5 s/problem |
| r-sweep (r=4) | 4 | 50 | ~28 min | per-token cost ≈ 15.5 s |
| r-sweep (r=6) | 4 | 50 | ~36 min | per-token cost ≈ 16.8 s |
| Aggregation + Phase 2 merge + plots | 0 | — | <1 min | CPU; runs after the pod's GPU work |
| **Total** | | | **~85 min** | budget 3 h with model-download overhead and one mid-run watcher tick |

Cost: ~$3 on a 4090 spot at $0.69/h. The job is small enough that
sharding adds more orchestration overhead than it saves; recommend a
single pod.

If GPU is tight, drop **L21** first — it's the worst-case control,
informative if it succeeds (rare) but expected to fail; the F3 vs F4
decision still works on L17/L28/L31 alone. Cost reduction: ~25 % wall.

If GPU is *very* tight, drop **r=6** entirely — the F3 monotonicity test
is robust on three points (r=2, r=4, r=8) for a Spearman; r=6 is
resolution, not load-bearing. Cost reduction: ~30 % wall on top.

---

## Open questions resolved before Phase 3 launches

| Question | Decision |
|---|---|
| Why r ∈ {2, 4, 6} and not {2, 3, 5, 7}? | **Geometric coverage with low cell count.** 2, 4, 6 plus the reused 8 give a 4-point curve covering the same range as a 5-point {2, 3, 4, 6, 8} sweep at 75 % the cost. The Spearman test cares about ordinality, not granularity. |
| Why include r=2 specifically? | **F3's anti-loop trigger requires it.** "Loop rate at r=2 < 0.30" is the operational definition of "this layer's pathology is r-driven, not structural". Without an r=2 measurement, the F3 vs F4 routing collapses. |
| Why N=50 and not bumping to N=100 since the cell count is small? | **Triage-grade is sufficient.** Phase 4 will N=200-confirm the chosen anchor as part of its own sanity gate. Bumping Phase 3 to N=100 doubles wall for a ±4 pp tighter CI on a categorical decision; not worth it. |
| Reuse Phase 2's r=8 JSONLs or rerun? | **Reuse.** Phase 2 ran on the same harness, same seed, same N, same problems. Re-running is ~$1 of wasted compute and adds no signal. Aggregator reads `phase2/gsm8k__L<NN>-r8.summary.json` directly. If transformers / hook / weights drift between phase 2 and phase 3 commits, that's caught by the `L17-r1` token-match pre-flight (already passed in Phase 2; we don't rerun). |
| What if loop_rate is non-monotonic (e.g., 0.4 → 0.2 → 0.5 → 0.7)? | **Spearman rho captures this cleanly.** Non-monotone curves return ρ < 0.8 and route to F4 (sub-block) — the right call, since non-monotone behaviour suggests the failure isn't a clean loop dynamic and module-level isolation is the next probe. |
| Re-test the EOS-collapse layers (L00, L11, L13)? | **No, with one exception.** L31 covers the EOS-collapse axis as a secondary signal. Including L00/L11/L13 would 1.75× the cell count to characterise a failure mode that doesn't gate the F1–F4 routing decision; they're already documented in Phase 2's per-cell summary and can be re-probed in a Phase 4 follow-up if F4 fires. |
| Use a smarter "least-pathological" selector instead of the four hard-coded layers? | **No.** Hard-coded selection makes the plan reproducible; a "top-K by pathology rank" selector silently changes which layers get tested if Phase 2's data is re-run. The four chosen layers are explicitly listed in the plan so a future reader can see the choice. |
| What if a layer's r=4 cell crashes (OOM, timeout)? | **Skip and continue.** The runner uses the same subprocess-per-cell pattern as Phase 2. Missing cells produce a 3-point curve for that layer, which the Spearman handles natively. Halt only if L17-r2 fails (the load-bearing cell for the F3 trigger). |
| Should we also test on L18 (block-adjacent to L17)? | **No, not in Phase 3.** Adding L18 makes the plan creep toward a block sweep. If F1 or F2 fires on L17, Phase 4's plan will N=200-confirm L17 *and* re-test L18 as part of its block-extension scope. If F3 or F4 fires, L18 testing happens inside that plan. |
| Is the F3 trigger threshold (loop@r=2 < 0.30) too lenient? | **Calibrated against Phase 2's 1D smoke cell.** Phase 1 1D defined "structural collapse" at loop > 0.95 (very generous because r=8 on the worst block); Phase 2's per-cell broken threshold tightened to 0.50. F3's 0.30 is "below the broken threshold by a margin large enough that the pathology is plausibly fixable by runtime intervention". Lower than 0.30 (e.g., 0.10) would be result-fitting. |

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| All 12 cells stay broken at the same loop_rates as r=8 (no rescue at lower r) | medium | Routes cleanly to F4 sub-block pivot; no halt needed. The result is informative — it says loop pathology is structural, not r-driven. |
| Hook regresses between Phase 2 and Phase 3 commits | low | The `L17-r1` token-match cell from Phase 2 is the pre-flight; if any code path that touches `install_block_loop_hooks` changes, re-run that single 5-min cell first and confirm 20/20 match before launching Phase 3. (Plan: include an explicit `--preflight-rerun` flag on the runner; off by default but cheap to invoke.) |
| Spearman ρ is unstable on a 4-point series | medium | The decision boundary is ρ ≥ 0.80, which on 4 points means "essentially monotone with at most one tiny inversion". The complement case (ρ < 0.80) routes to F4, which is the cautious default — false-negative on F3 just means we test sub-blocks before runtime fixes, costing an extra ~10 GPU-h. Tolerable. |
| L21 (the worst-case control) succeeds at r=2 — opens an unexpected path | low | Wonderful. Update the launch decision: F3 trigger fires on L21, anchor (21, 2). The fact that the worst-case loop layer became tractable at r=2 is the single most informative outcome this phase can produce. |
| Wall time blows past budget | low | Per-cell wall is well-understood (Phase 2 measured 14 s/problem at r=8 single-layer; r=2 should be ~half). If wall spikes >1.5× budget, halt and investigate before continuing. The Phase 2 v1 emergency-exit pattern (`max_hours=3`) catches this. |
| Phase 2 r=8 JSONL is missing or corrupted on the pod | low | `git pull` on the pod includes the `results/` subtree. If a summary JSON fails to load, the aggregator continues with a 3-point curve for that layer; the launch decision is robust to one missing point. |
| Phase 3 fires F1/F2 but Phase 4's N=200 confirmation contradicts it | medium | This is exactly why Phase 4's first action is N=200 confirmation on the chosen anchor. If N=200 shows acc < 0.40 (i.e., the N=50 hit was statistical noise), Phase 4 reverts to the next-best routing decision (F3 if it had a candidate, else F4) without rerunning Phase 3. Document this fallback in Phase 4's plan. |

---

## Implementation checklist (chronological)

1. Add the 12 CONFIGS entries to `probes/eval_v3.py` (programmatic
   block; mirrors Phase 2's pattern).
2. Add `probes/phase3_gates.py` with `classify_r_curve` + thresholds +
   a 5-line `spearman` helper.
3. Write `experiments/path2_v2_phase3.py` runner (clone
   `path2_v2_phase2.py`; replace cell list; add Phase 2 JSONL pickup
   in the aggregator).
4. Write `experiments/path2_v2_r_sweep_plot.py` (matplotlib 2×2 small
   multiples + CSV writer; CPU-only; mirror the Phase 2 plot module's
   structure).
5. Add tests in `tests/path2_v2/`:
   - `test_phase3_gates.py` — F1/F2/F3/F4 trigger boundaries on
     synthetic per-cell summaries (six cases: preserved-fires-first,
     degraded-with-low-loop-fires-second, monotone-loop-fires-F3,
     all-broken-flat-fires-F4, missing-cell-handled, NaN-cell-halts)
   - `test_phase3_runner.py` — fixture-based smoke test of the runner
     and Phase 2 JSONL pickup (writes a fake `phase2/gsm8k__L17-r8.summary.json`,
     asserts the aggregator merges it into Phase 3's `cells` dict)
   - `test_r_sweep_plot.py` — write a fake CSV, render the PNG,
     assert PNG file exists and is non-empty
6. Create `experiment_path2_v2_phase3.yaml`: set
   `flags: "--script path2-v2-phase3"`,
   `result_dir: results/path_2_depth_recurrence_v2/phase3`,
   `cap_usd: 3.0`, `emergency_usd: 0.5`, `max_hours: 3`,
   `git ref: path_2_phase_3`. Add `path2-v2-phase3` to `scripts/run.sh`'s
   five EXPERIMENT_* arrays (mirrors how `path2-v2-phase2` is wired).
7. Run the full CPU test suite locally; expect all green.
8. Push the branch, spin up a pod, watch.
9. After GPU completes: pull, regenerate the small-multiples figure
   locally if needed, commit `results/path_2_depth_recurrence_v2/phase3/`.
10. Read `phase4_launch.mode` from `phase3_summary.json`. Open the
    corresponding Phase 4 plan stub:
    - `f1_retrofit` → `phase4_retrofit_training.md`
    - `f2_healing` → `phase4_healing_then_retrofit.md`
    - `f3_anti_loop` → `phase4_anti_loop_decoding.md`
    - `f4_subblock` → `phase4_subblock_pivot.md`

---

## What Phase 4 inherits from Phase 3

Regardless of which mode fires, Phase 4 inherits:

- **`phase3_r_sweep.csv`** — the 4-point r curve per layer with bucket
  assignments. F1/F2 plans use the anchor row directly; F3/F4 plans
  use it as architectural context.
- **`phase3_summary.json#phase4_launch`** — the launch mode + anchor.
  This is what each Phase 4 plan opens with: "Phase 3 selected
  (`L17`, `r=2`) under mode `f2_healing`; this plan implements F2
  against that anchor."
- **A reusable cached IT model load** if Phase 4 launches on the same
  pod (only relevant for F3 and F4, since F1 and F2 will move to an H100).
- **Empirical falsification of the alternatives.** When Phase 4 = F3
  fires, the writeup can cite the specific (layer, r) cells that
  *didn't* land in preserved/degraded. The "why not F1 directly?"
  question has a numeric answer.

---

## What Phase 3 explicitly does NOT inherit from prior phases

- **Phase 2's bucket gate is reused as-is.** New thresholds calibrated
  to Phase 3 data would be result-fitting; the Phase 0 D7 principle
  (commit gates before looking) applies recursively across phases.
- **Phase 1's 1A baseline (acc=0.755) is the comparison anchor for
  the `preserved` bucket.** Same rationale as Phase 2 — the IT model's
  ceiling under C2 + smart_v2 extractor at r=1 is the only meaningful
  reference.
- **Round 2c's perplexity ranking is irrelevant here.** Phase 0 D12
  established that perplexity-tolerant ≠ reasoning-tolerant; lowering r
  doesn't change which axis we measure on.
- **Round 5's W5 block (=[15,19]) result is not a Phase 3 prior.** L17
  is selected because of *Phase 2's* per-layer evidence (lowest
  loop_rate among mid KV-consumers), not because of round 5's block
  centroid. The two happen to agree; the agreement is informational,
  not load-bearing.

---

## Exit criteria

- Phase 3 finishes all 12 cells (or skips at most 2 due to crashes
  with a documented reason).
- `phase3_summary.json` is written and committed alongside the CSV
  and PNG.
- `phase4_launch.mode ∈ {f1_retrofit, f2_healing, f3_anti_loop,
  f4_subblock}` (never `halt` unless the L17-r1 pre-flight regresses).
- The corresponding Phase 4 plan file is opened on the branch (just
  the file with the launch decision header; full plan body is the
  next session's work).

In all cases, the answer to "what's the next compute commitment?" is
unambiguous after this phase, with an empirical justification that fits
in two sentences.
