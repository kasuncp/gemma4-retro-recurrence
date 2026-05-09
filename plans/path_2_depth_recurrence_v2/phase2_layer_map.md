# Path 2 v2 — Phase 2: Per-Layer Reasoning Map

**Project:** Retrofitted recurrence on Gemma 4 E2B
**Status:** Ready for implementation (Phase 1 complete; all six gates pass on N=200 / N=50 / N=50 / N=20 / N=10 / 50-seq Wikitext run)
**Depends on:** `phase0.md` (harness, extractors, prompts, eval_v3), `phase1_hook_sanity.md` (verified hook + bridges)
**Blocks:** Phase 3 (`phase3_block_sweep.md`)
**Pod budget:** ~8 pod-hours on a single RTX 4090 (~$5.50 at $0.69/h spot)

---

## Why Phase 2 exists

Round 2c produced a 35-layer × {r=2, r=4, r=8} **perplexity** map and showed
that some layers tolerate looping under base-model perplexity. Round 4/5
then showed that perplexity-tolerant ≠ reasoning-tolerant *on the
instruction-tuned model*. We don't have a layer-level reasoning map; we
went straight to blocks in round 3 and that proved premature.

Phase 2 produces the reasoning analog of round 2c: for each of the 35
decoder layers, loop *that layer alone* at r=8 on the IT model and
measure GSM8K accuracy. The result is a single ranked table:
"which single layers preserve generation accuracy under maximum stress."

Phase 3's block sweep is *guided by* this map, not derived from base
perplexity. If Phase 2 says only layers 15–19 preserve reasoning, Phase 3
doesn't waste cells on blocks anchored at layer 5 or 28.

This phase produces no architectural decisions on its own. It produces
a localisation: a list of "safe layers" Phase 3 starts blocks from, and
a map of where reasoning collapses for the qualitative writeup.

---

## Scope

**In scope** (35 main cells + 1 structural pre-flight, ~7 h on a 4090):

| Group | Cells | Config | Benchmark | N | Gate (per cell) |
|---|---|---|---|---|---|
| Pre-flight | `L17-r1` token-match vs `baseline-C2` | (17,17), r=1 | GSM8K | 20 | per-problem completion strings byte-equal (assert hook works at single-layer; mirrors Phase 1's 1C) |
| Layer map | `L00-r8` … `L34-r8` (35 cells) | (L,L), r=8, C2 prompt | GSM8K | 50 | bucket assignment: **preserved** / **degraded** / **broken** (defined below) |

**Phase 3 launch gate** (computed *after* the 35 cells finish):

| Outcome | Phase 3 action |
|---|---|
| ≥ 3 layers in **preserved** bucket | Launch Phase 3 normally; block sweep starts anchored at preserved layers |
| 1–2 **preserved** layers | Launch Phase 3 in **narrow mode**: only blocks anchored at preserved layers ± 2 neighbours |
| 0 **preserved** layers, ≥ 3 **degraded** | Launch Phase 3 with a relaxed bucket gate (use degraded as anchors) and document recurrence as a partial-preservation regime in the writeup |
| 0 preserved AND 0 degraded | **HALT.** Recurrence is fundamentally fragile on IT at r=8. Skip Phase 3 / 4 and route directly to Phase 6 (retrofit training) — re-scope under bucket C/D in `phase0.md` |

**Out of scope:**

- Block looping. That's Phase 3 — and it depends on Phase 2's output.
- r-sweep. That's Phase 4. Phase 2 is r=8 only.
- 8-shot CoT cells (Phase 1 has both bridges).
- ARC / BBH (Path 1 plan 7's prompt-format finding makes ARC noisy under
  C2; we localise reasoning collapse on GSM8K first).
- PLE strategy ablations (Phase 4's `iter1-only` vs `every-iter` sweep
  belongs after the layer is known).
- Per-layer N=200 confirmation (mentioned as a possible Phase 2.5 if a
  bucket boundary is contested; not in this plan).

If a Phase 2 cell would require a new prompt builder, a new extractor,
or a new generation kwarg beyond what Phases 0/1 already shipped, stop
and re-scope. The point of this phase is to *use* the validated harness.

---

## Implementation deliverables

### 1. CONFIGS extension (~10 lines, in `probes/eval_v3.py`)

35 single-layer configs, generated programmatically so we don't paste 35
near-identical dicts. Append after the existing manual entries:

```python
# Phase 2: single-layer reasoning probes. One config per decoder layer
# at r=8 with the C2 prompt; structural-pre-flight uses L17-r1 instead.
for _L in range(35):
    CONFIGS[f"L{_L:02d}-r8"] = {
        "prompt": "C2",
        "block": (_L, _L),                # start == end → single layer
        "r": 8,
        "ple_strategy": "every-iter",
        "notes": f"phase 2: single-layer r=8 probe at decoder layer {_L}",
    }

CONFIGS["L17-r1"] = {                     # phase 2 pre-flight only
    "prompt": "C2",
    "block": (17, 17),
    "r": 1,
    "ple_strategy": "every-iter",
    "notes": "phase 2 pre-flight: single-layer r=1 must token-match baseline-C2.",
}
```

`install_block_loop_hooks` must accept `start == end`. If it doesn't,
the pre-flight will fail loudly *before* the 35-cell sweep starts; fix
the hook there, don't paper over it.

### 2. New runner (~250 lines, `experiments/path2_v2_phase2.py`)

Same skeleton as `experiments/path2_v2_phase1.py`. Sequentially shells
out to `path2_v2_eval.py` per cell so a crash in one layer doesn't kill
the rest. Modes:

- (default) — pre-flight, then 35 layer cells, then aggregation.
- `--summarize-only` — read existing JSONL, recompute the map, no GPU.
- `--skip-cell L<NN>` — skip individual layers (repeatable; useful when
  one layer crashes and you want to fix that layer without re-running 34).
- `--screen-mode` — *optional* fast-path: run all 35 at N=20 first, then
  re-run the top 10 at N=50. Saves ~40 % wall but doubles plan complexity.
  **Default off**; flag exists for the case where compute is tight.
- `--shard <i>/<n>` — for parallel pods: cell `L<k>` runs only when
  `k % n == i`. Three pods at `--shard 0/3 1/3 2/3` cuts wall time to
  ~2.5 h.

Cells declared as a list of dicts (`name, kind, config, benchmark, n,
model_id`), same shape as Phase 1's `CELLS`. Pre-flight is `name="P0_L17_r1_token_match"`;
layer cells are `name="L<NN>"`.

### 3. New gates module (~80 lines, `probes/phase2_gates.py`)

Pure-Python gate evaluator. Each per-cell gate returns `(bucket, reason)`
where bucket ∈ {`preserved`, `degraded`, `broken`}.

```python
# Bands chosen against the 1A anchor (acc_smart_v2 = 0.755 on N=200).
# "Preserved" = within ~10 pp of the unmodified baseline; that's the
# threshold above which Phase 4's r-sweep can hope to recover the
# remaining gap with retrofit training.
LAYER_PRESERVED_ACC_MIN = 0.65            # ~10 pp below 1A anchor
LAYER_DEGRADED_ACC_MIN  = 0.40            # >40 % means "model still
                                          #   reasons, just worse"
LAYER_BROKEN_LOOP_MAX   = 0.50            # ≥50 % loop_rate is collapse
LAYER_BROKEN_TRUNC_MAX  = 0.30            # uses Phase-1 truncation
                                          #   metric (cap-hit AND no
                                          #   extractable answer)

def classify_layer(s: dict) -> tuple[str, str]:
    if s.get("n_problems", 0) == 0:
        return "broken", "no rows"
    acc   = s["accuracy_smart_v2"]
    loop  = s["loop_rate"]
    trunc = s["truncation_rate"]
    if loop >= LAYER_BROKEN_LOOP_MAX or trunc >= LAYER_BROKEN_TRUNC_MAX:
        return "broken", f"loop={loop:.2f} trunc={trunc:.2f}"
    if acc >= LAYER_PRESERVED_ACC_MIN:
        return "preserved", f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}"
    if acc >= LAYER_DEGRADED_ACC_MIN:
        return "degraded", f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}"
    return "broken", f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}"
```

Pre-flight gate is a copy of `gate_1C` (token-match): exit code 0 from
`path2_v2_token_match.py` against the L17-r1 vs `baseline-C2` JSONLs.

### 4. Heat map + CSV (~120 lines, `experiments/path2_v2_layer_map_plot.py`)

Emits two artefacts under the phase 2 results dir:

- `phase2_layer_map.png` — matplotlib figure, x-axis = layer index 0–34,
  three stacked bars per layer (acc, loop_rate, trunc_rate). Bucket
  colour-codes the layer index label below the bar. A horizontal line at
  `LAYER_PRESERVED_ACC_MIN` marks the bucket boundary. Title cites N=50
  and notes the Wilson 95 % CI is roughly ±13 pp per layer.
- `phase2_layer_map.csv` — one row per layer: `layer, accuracy_smart_v2,
  accuracy_legacy, loop_rate, truncation_rate, mean_gen_tokens,
  attention_type, is_kv_consumer, depth_tertile, base_ppl_round2c,
  bucket`. This file is what Phase 3 reads as input.

Per the user's CLAUDE.md (`.claude` memory), PNGs land alongside JSON
under `results/path_2_depth_recurrence_v2/phase2/`.

### 5. Per-cell base-perplexity bridge (no new module; ~30 lines in eval_v3)

Phase 0 D12 says base perplexity is "free during the same forward pass"
and should be a per-cell secondary signal. Phase 2 is the first place
this is actually used as a sanity check.

Wire `summarise_gsm8k` (or a sibling) to optionally compute Wikitext-2
perplexity once per cell. Implementation:

- After the cell's GSM8K run finishes (model still loaded, hook still
  installed), run `compute_perplexity(model, prepare_inputs(tokenizer,
  20, 256))` and record `base_ppl` in the cell's summary JSON.
- Phase 2's aggregator reads round 2c's JSON, looks up that layer's
  vanilla-r=8 perplexity, and computes `delta_ppl = our_ppl - r2c_ppl`.
  If `|delta_ppl| > 1.0` for any layer, log a warning (not a halt) —
  this catches transformers/version drift between rounds.

This is **opt-in via `--with-ppl-bridge`** on the runner. Default off
because the extra ~5 min/cell is meaningful (35 cells × 5 min ≈ 3 h).
Recommend running it once at the start of Phase 2 with the flag, then
not re-running unless something looks off.

---

## Cell-by-cell specification

### Pre-flight — `P0_L17_r1_token_match`

**Goal:** confirm the hook works when `start == end`. The plan in
`probes/hooks.py:install_block_loop_hooks` was originally written for
multi-layer blocks; Phase 1's 1C cell tested width-5 (15..19), not
width-1. A single-layer block with `r=1` must still produce
byte-identical output to the unhooked baseline.

**Configs:** `L17-r1` and `baseline-C2` (already on disk from Phase 1's 1A).
**Benchmark:** GSM8K, N=20, seed=42 (same first 20 problems as Phase 1's
1C cell).

**Test:**

```bash
python experiments/path2_v2_eval.py --phase layer-map \
    --config L17-r1 --benchmark gsm8k --n 20 --seed 42 \
    --max-new-tokens 512 --model-id google/gemma-4-E2B-it --dtype bf16 \
    --output-dir results/path_2_depth_recurrence_v2/phase2

python experiments/path2_v2_token_match.py \
    --left  results/path_2_depth_recurrence_v2/phase1/gsm8k__baseline-C2.jsonl \
    --right results/path_2_depth_recurrence_v2/phase2/gsm8k__L17-r1.jsonl \
    --limit 20
```

**Gate:** exit 0 (`MATCH 20/20`).

**On failure:** `install_block_loop_hooks` is mishandling the
`start == end` case. Likely culprits:

1. The hook installs both a pre- and post-hook on the start layer and
   the post-hook on the end layer; with `start == end` these collide,
   and the "is this the inner replay" guard double-fires.
2. The PLE re-injection slot isn't preserved when there's only one
   layer in the loop body (the "every-iter" strategy shares the
   per-layer input across the loop body's first layer only).

Fix the hook before running the 35-layer sweep — a 1-pp accuracy drop
across 35 cells is invisible noise; a structural hook bug is not.

**Wall budget:** ~5 min (20 problems, baseline already cached).

---

### Layer cells — `L00-r8` … `L34-r8`

**Symmetry note:** every cell shares the same spec; only the layer index
varies. This section describes one cell and the aggregator handles all 35.

**Config (per cell L):** `L<NN>-r8` (block `(L, L)`, r=8, every-iter PLE,
C2 prompt).
**Model:** `google/gemma-4-E2B-it`.
**Benchmark:** GSM8K, N=50, seed=42 (the first 50 of the seed=42 shuffle —
same problems Phase 1's 1A used as its first 50).
**Generation:** `max_new_tokens=512`, greedy, `use_cache=False`,
`pad_token_id=eos`. Same as 1A; do not change.

**Anchor:** Phase 1's 1A measured `accuracy_smart_v2 = 0.755` (CI
[0.69, 0.81]) at r=1 on N=200 of the same problems. A r=8 single-layer
loop that *preserves* reasoning would land within ~10 pp of that. A
loop that *breaks* reasoning collapses below 0.40 or pathologises
(loop_rate ≥ 0.50 or truncation_rate ≥ 0.30 under the Phase-1 metric).

**Per-cell bucket** (computed by `phase2_gates.classify_layer`):

| Bucket | Criterion |
|---|---|
| **preserved** | `acc ≥ 0.65` AND `loop < 0.50` AND `trunc < 0.30` |
| **degraded** | `0.40 ≤ acc < 0.65` AND `loop < 0.50` AND `trunc < 0.30` |
| **broken** | `acc < 0.40` OR `loop ≥ 0.50` OR `trunc ≥ 0.30` |

**Why these bands** (intentional, not arbitrary):

- N=50 with binary outcome at p ≈ 0.7 has a Wilson 95 % CI of about
  ±13 pp. A 10-pp drop from the 1A anchor is noticeable but within
  small-sample noise; we don't pretend to rank within the preserved
  bucket. Phase 3 / 4 uses N=200 to rank.
- `loop ≥ 0.50` is the same threshold Phase 1's 1D cell uses for
  "structural collapse" (1D is generous at < 0.95 because it's r=8 on
  the *worst* block; here we're at r=8 on a *single layer*, so 50 % is
  the appropriate red line).
- `trunc ≥ 0.30` uses the Phase-1 redefinition (cap-hit AND no
  extractable answer). The 1A baseline measured 0/200; anything ≥ 30 %
  here means the harness *itself* failed to surface answers, not that
  the model rambled.

**Per-cell secondary metrics** (recorded but not gated):

| Metric | Why |
|---|---|
| `accuracy_legacy` | informational; cross-references Path 1's last-int extractor |
| `mean_gen_tokens` | catches the "model EOSes early everywhere" failure that hides under high accuracy on trivial answers |
| `pathology_flag` | summary-level pathology trigger (truncation > 0.5 OR loop > 0.5) |
| `base_ppl` *(if `--with-ppl-bridge`)* | round-2c bridge; warns if `|delta_ppl| > 1.0` vs `results_round2c_full_map.json` |

**On failure (per-cell):** failures are bucket assignments, not halts.
Phase 2 finishes all 35 cells regardless. The Phase 3 launch decision
uses the count of preserved cells.

The exception: if **every** layer's pre-flight assertion fails (i.e., the
hook is broken at single-layer width), halt and fix the hook.

**Wall budget (per cell):** ~12 min on a 4090 with `use_cache=False` and
r=8 on a single layer. Per-token cost is ~14 s × (1 + 7/35) ≈ 17 s.
50 problems × ~17 s/problem ≈ 14 min. The 35 cells share one model
load (~30 s), so the runner amortises that.

---

## Output / report shape

After all 36 cells finish, `experiments/path2_v2_phase2.py` writes
`results/path_2_depth_recurrence_v2/phase2/phase2_summary.json`:

```json
{
  "phase": "phase2-layer-reasoning-map",
  "model_ids": {"it": "google/gemma-4-E2B-it"},
  "preflight": {
    "L17_r1_token_match": {
      "shared": 20, "matches": 20, "passed": true
    }
  },
  "cells": {
    "L00": {
      "summary": {"n_problems": 50, "accuracy_smart_v2": 0.X, ...},
      "metadata": {"attention_type": "...", "is_kv_consumer": true,
                   "depth_tertile": "early"},
      "base_ppl": 12.83,                 // present iff --with-ppl-bridge
      "round2c_base_ppl": 12.85,         // looked up from r2c JSON
      "delta_ppl": -0.02,
      "bucket": "preserved",
      "bucket_reason": "acc=0.71 loop=0.00 trunc=0.00"
    },
    ...
    "L34": {...}
  },
  "buckets": {
    "preserved": ["L15", "L17", "L19", ...],
    "degraded":  ["L05", "L08", ...],
    "broken":    ["L00", "L01", ..., "L33", "L34"]
  },
  "phase3_launch": {
    "mode": "normal" | "narrow" | "relaxed" | "halt",
    "anchors": ["L15", "L17", "L19"],     // layers Phase 3 should
                                            //   start blocks from
    "rationale": "..."
  },
  "wall_seconds_total": XXX
}
```

Console table at the end of the run:

```
=== Phase 2 layer reasoning map (N=50/layer, r=8, C2 prompt, IT model) ===
layer  acc    loop   trunc  mean_tok  attn        kv_cons  bucket
L00    0.10   0.84   0.34   501       sliding     False    broken
L01    0.18   0.62   0.28   478       sliding     False    broken
...
L15    0.71   0.00   0.00   245       full        True     preserved
L16    0.69   0.02   0.00   251       full        True     preserved
L17    0.73   0.00   0.00   238       full        True     preserved
...
L33    0.08   0.91   0.40   512       sliding     False    broken
L34    0.06   0.93   0.42   512       sliding     False    broken

=== Bucket counts ===
preserved: 5  ([L15, L16, L17, L19, L22])
degraded:  4  ([L05, L08, L18, L20])
broken:    26

=== Phase 3 launch decision ===
mode: normal
anchors: [L15, L16, L17, L19, L22]
rationale: ≥3 preserved layers; Phase 3 block sweep can anchor at preserved layers.
```

`phase2_layer_map.png` (heat map) and `phase2_layer_map.csv` (Phase 3
input) ship alongside.

---

## Exit criteria

- **mode = normal** (≥ 3 preserved) → write a 1-line update to
  `phase0.md`'s "design decisions" section confirming D12 (single-layer
  before block) was the right choice empirically. Launch Phase 3.
- **mode = narrow** (1–2 preserved) → write the same update *and* a
  scope reduction note for Phase 3 (block-sweep grid shrinks from "all
  starts × all widths" to "preserved-layer ± 2 starts × widths 2–6").
- **mode = relaxed** (0 preserved, ≥ 3 degraded) → write a Phase 3 plan
  amendment that lowers the bucket gate. The reasoning collapse becomes
  a finding in its own right; document before launching Phase 3.
- **mode = halt** (0 preserved AND 0 degraded) → STOP path 2. Recurrence
  on the IT model at r=8 is fundamentally fragile per-layer; block
  geometry can't rescue it. Route directly to Phase 6 (retrofit
  training, bucket C/D path in `phase0.md`). Document and update the
  roadmap before any further compute is spent.

In all cases, `phase2_summary.json` and the heat map PNG are committed.

---

## Compute budget

| Phase | Cells | N | Wall (4090) | Notes |
|---|---|---|---|---|
| Pre-flight | 1 | 20 | ~5 min | re-uses cached baseline-C2 JSONL from Phase 1 |
| Layer map | 35 | 50 | ~8 h | 14 s/problem at r=1 (measured 1A: 2115s / 150 = 14.1 s); r=8 single-layer adds ~20 % per-token (35 layers + 7 replays = 42 layer-equivalents); 50 × ~17 s × 35 cells ≈ 8.2 h |
| Aggregation + plots | 0 | — | <1 min | CPU; runs after the pod's GPU work |
| **Total** | | | **~8 h** | budget 9 h with model-download overhead and one mid-run watcher tick |

Cost: ~$5.50 on a 4090 spot at $0.69/h. With `--shard` across three
pods, wall time drops to ~3 h at the same total cost (linear scaling,
modulo one model load per pod).

If `--with-ppl-bridge` is set, add ~3 h (5 min/cell × 35 cells) → ~11 h
total. Recommend running with the flag *only* on the first cell or on
suspect cells, not the whole sweep, since round 2c's perplexity numbers
are already on disk and the only thing this catches is harness drift.

---

## Open questions resolved before Phase 2 launches

| Question | Decision |
|---|---|
| Use base or IT for the layer map? | **IT** (decided in phase0 D12; restated for completeness). Base→IT doesn't transfer (Phase 5c). |
| One r value or a sweep? | **r=8 only.** "Maximum stress" is the right localisation question for Phase 2; r-sweep is Phase 4's job after Phase 3 picks the block. Three r values would 3× compute for no Phase-3 actionable signal. |
| N per layer? | **N=50.** Wilson CI ±13 pp is generous but sufficient for 3-bucket categorisation. Fine ranking inside the preserved bucket happens at Phase 3's N=200. |
| Include the round-2c perplexity bridge per cell? | **Opt-in via `--with-ppl-bridge`.** Phase 0 said yes; in practice the cost (~3 h) only buys early detection of harness drift, which is unlikely if Phase 1's 1E gate passes. Default off. |
| Run a per-layer r=1 token-match for all 35 layers? | **No.** One pre-flight at L17 is enough — the hook is layer-agnostic, and the 1C cell already validated the 5-layer block path. If pre-flight fails we fix the hook; if it passes we trust the abstraction. |
| What about screen-then-confirm (r=4 N=20 first)? | **Flag exists, default off.** Adds plan branching for a ~40 % wall savings on a phase that's already cheap (~$5). Not worth the complexity unless GPU budget tightens. |
| Sharding across pods? | **Yes, via `--shard <i>/<n>`.** Cells are independent; layer index is the natural shard key. Each pod's JSONL output is on disjoint files, so no merge step needed beyond running the aggregator after `git pull`. |
| Use the C2 prompt or the round-5 prompt? | **C2.** Phase 1 confirmed both bridges; C2 is the deployment prompt (zero-shot, chat-templated) and the one Phase 4 will eventually report against. The 8-shot bridges were anchors, not benchmarks. |
| What if the per-cell `mean_gen_tokens` is suspiciously low (e.g., <50)? | **Flag in the report; no automatic halt.** Could indicate the hook is making the model emit EOS early (a real failure mode round 5 saw at r=8). The bucket gate already catches this when it co-occurs with low accuracy or high loop_rate; the standalone signal is for forensics. |

---

## What Phase 3 inherits from Phase 2

If Phase 2 finishes (any non-halt outcome), Phase 3 inherits:

- **`phase2_layer_map.csv`** — one row per layer with bucket, accuracy,
  pathology metrics, and architectural metadata. This is the input to
  the block-anchor selection.
- **`phase2_summary.json`** — provides `phase3_launch.mode` and the
  recommended anchor list. Phase 3's plan reads this and uses the mode
  to pick its sweep grid.
- **A reusable cached IT model load** if Phase 3 launches on the same
  pod: skip the 30 s download.
- **Architectural correlations** (informal, written into the qualitative
  section of the Phase 2 report): "preserved layers cluster in the
  full-attention KV-consumer mid-section" or whatever the data shows.
  Phase 3's block geometry choices use this as priors, not constraints.

---

## What Phase 2 explicitly does NOT inherit from prior rounds

- **Round 2c's per-layer perplexity ranking is not a layer prior.** Phase
  0 D12 was specifically because perplexity-tolerant ≠ reasoning-tolerant.
  We *cross-reference* the perplexity numbers (via `--with-ppl-bridge`)
  but don't seed the layer order from them.
- **Round 3b/3c's "winning" blocks are not layer priors.** Round 3b
  claimed [15, 19] was the winner on perplexity; Phase 2 must
  re-discover whether [15, 19] are individually preserved on
  *reasoning*. If only L17 is preserved and L15/L19 are degraded, Phase
  3's "block of [15, 19]" inherits a degraded layer at each end and the
  block result will reflect that.
- **Path 1's plan-7/plan-8 prompt-format findings are GSM8K-specific
  here.** Phase 2 doesn't run ARC or BBH precisely because those
  benchmarks have prompt-format inversion artifacts that would muddy
  the per-layer signal.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Hook breaks at `start == end` | medium | Pre-flight cell catches this at N=20 before the 35-cell sweep; halt and fix |
| Pod crashes mid-sweep | medium | JSONL append-and-fsync (D11); resume picks up where it left off, no re-run of completed layers |
| Wall-time blows past budget | low–medium | The 1A actual measurement (200 problems × ~10.5 s) was lower than the projected ~14 s/problem; per-token cost is the dominant factor and r=8 single-layer is well-modelled. If wall time spikes >1.5× budget, halt and investigate before continuing |
| Bucket assignments oscillate at the boundary | medium | N=50 is coarse on purpose. The plan does not promise stable boundary calls; document boundary cells (acc ∈ [0.62, 0.68] or [0.37, 0.43]) as "ambiguous" in the writeup and let Phase 3's N=200 resolve them |
| Round 2c base-ppl bridge fires (delta > 1.0) on a layer | low | Means transformers / weights / hook changed since round 2c. Halt that cell only; investigate; the layer's reasoning measurement may still be valid but should be flagged |
| Aggregator runs before any cell completes | low | `--summarize-only` on an empty dir prints `(no JSONL cells)` and exits 0 (already true of `path2_v2_eval.py`'s table printer); Phase 2's aggregator should match |
| Heat map renders badly on a CPU laptop without matplotlib | low | Make the PNG generator optional (skip with `--no-plots` if matplotlib import fails); the CSV is the load-bearing artefact for Phase 3 |

---

## Implementation checklist (chronological)

1. Add the 35 + 1 CONFIGS entries to `probes/eval_v3.py`.
2. Add `probes/phase2_gates.py` with `classify_layer` + bucket constants.
3. Write `experiments/path2_v2_phase2.py` runner (model the structure on
   Phase 1's runner; copy the subprocess + summary loop).
4. Write `experiments/path2_v2_layer_map_plot.py` (matplotlib heat map +
   CSV writer; CPU-only).
5. Add tests in `tests/path2_v2/`:
   - `test_phase2_gates.py` — bucket assignment boundaries
   - `test_phase2_runner.py` — fixture-based smoke test of the runner
     and aggregator (mirrors `test_phase1_runner.py`)
   - `test_layer_map_plot.py` — write a fake CSV, render the PNG, assert
     the PNG file exists and is non-empty
6. Create `experiment_path2_v2_phase2.yaml` (new file): set
   `flags: "--script path2-v2-phase2"`,
   `result_dir: results/path_2_depth_recurrence_v2/phase2`,
   `cap_usd: 7.0`, `emergency_usd: 1.0`, `max_hours: 9`,
   `git ref: path_2_phase_2`. Add `path2-v2-phase2` to
   `scripts/run.sh`'s `EXPERIMENT_KEYS`/`EXPERIMENT_SCRIPTS`/
   `EXPERIMENT_DEFAULTS`/`EXPERIMENT_ROOTS`/`EXPERIMENT_DEPTHS` arrays
   (mirrors how `path2-v2-phase1` is wired today).
7. Run the full CPU test suite locally; expect all green.
8. Push the branch, spin up a pod with the yaml, watch.
9. After GPU completes: pull, regenerate the heat map locally if needed,
   commit `results/path_2_depth_recurrence_v2/phase2/`.
10. Open Phase 3 plan if `phase3_launch.mode != halt`.
