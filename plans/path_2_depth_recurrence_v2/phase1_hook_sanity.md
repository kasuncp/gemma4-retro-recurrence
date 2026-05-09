# Path 2 v2 — Phase 1: Hook Sanity Gate

**Project:** Retrofitted recurrence on Gemma 4 E2B
**Status:** Ready for implementation (Phase 0 complete)
**Depends on:** `phase0.md` (harness, extractors, prompts, eval_v3)
**Blocks:** Phase 2 (`phase2_layer_map.md`)
**Pod budget:** ~1 pod-hour on a single RTX 4090 (~$0.50)

---

## Why Phase 1 exists

This is the gate Path 2 round 4 didn't have. Round 4 went straight from
"hook works in perplexity sweeps" to "let's measure GSM8K with the hook
and 8-shot CoT" and produced a 4.8 % baseline that nobody caught for
five rounds. Phase 1 is **deliberately small and slow to fail**: every
cell has a numeric band that must be hit before the next phase runs,
and any single failure halts the whole route.

Phase 1 produces no research findings. It validates the harness.

---

## Scope

**In scope** (6 cells, ~65 min on a 4090):

| Cell | Model | Config | Benchmark | N | Gate |
|---|---|---|---|---|---|
| **1A** IT C2 baseline | E2B-it | `baseline-C2` | GSM8K | 50 | accuracy_smart_v2 ∈ [65 %, 82 %], loop_rate < 5 %, truncation < 5 % |
| **1B-p1** IT 8-shot Path 1 anchor | E2B-it | `baseline-8shot-control` | GSM8K | 50 | accuracy_smart_v2 ∈ [20 %, 40 %], loop_rate < 25 %, truncation < 20 % |
| **1B-r5** IT 8-shot round 5 anchor | E2B-it | `baseline-8shot-round5` | GSM8K | 50 | accuracy_legacy ∈ [44 %, 64 %], loop_rate < 25 %, truncation < 20 % |
| **1C** Token-match no-op | E2B-it | `W5-r1` vs `baseline-C2` | GSM8K | 20 | per-problem completion strings byte-equal |
| **1D** W5-r8 smoke | E2B-it | `W5-r8` | GSM8K | 10 | process exits 0; OOM-free; loop_rate < 95 % |
| **1E** Base ppl smoke | E2B (base) | hook-on r=1 vs hook-off | Wikitext-2 | 50 seqs | rel drift < 1e-4 |

**Out of scope:**

- No reasoning interpretation. We are not measuring "does recurrence help"
  here. Cell 1D's accuracy is irrelevant; only its pathology rate matters.
- No new benchmarks (ARC, BBH-lite). Phase 3 first opens those.
- No retrofit-training prep, no LoRA, no on-device.
- No new configs beyond the four already in `probes.eval_v3.CONFIGS`.

If any Phase 1 cell would require new code beyond the small token-match
helper specified below, stop and re-scope.

---

## Implementation deliverables

### 1. Token-match helper (~50 lines, CPU only)

Add `experiments/path2_v2_token_match.py`:

```python
"""Compare per-problem completions across two cells.

Given two JSONL paths produced by ``path2_v2_eval.py``, verify that
for every shared ``idx`` the ``completion`` field is byte-equal.
This is the structural correctness gate for ``r=1 hook == no-op``.

Why a separate tool: greedy decoding under bf16 SHOULD be bitwise
deterministic, and the block-loop hook at r=1 SHOULD return the
unmodified output. But subtle bugs (e.g., the hook mutating shared
KV state in a way that doesn't show up in perplexity but does in
generation) only manifest as token-level drift after several
generation steps. We catch those here, before Phase 2 burns 35 cells.

Usage:
    python experiments/path2_v2_token_match.py \\
        --left  results/path_2_depth_recurrence_v2/phase1/gsm8k__baseline-C2.jsonl \\
        --right results/path_2_depth_recurrence_v2/phase1/gsm8k__W5-r1.jsonl

Exit 0 + prints "MATCH N/N" on success;
exit 1 + prints first 3 mismatched idxs on failure.
"""
import argparse, json, sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--left",  required=True, type=Path)
    p.add_argument("--right", required=True, type=Path)
    p.add_argument("--limit", type=int, default=20,
                   help="Compare only the first N shared idxs.")
    args = p.parse_args()

    L = {json.loads(l)["idx"]: json.loads(l) for l in open(args.left)}
    R = {json.loads(l)["idx"]: json.loads(l) for l in open(args.right)}
    shared = sorted(set(L) & set(R))[: args.limit]
    if not shared:
        print(f"FAIL: no shared idxs between {args.left} and {args.right}")
        sys.exit(1)

    mismatches = []
    for i in shared:
        if L[i]["completion"] != R[i]["completion"]:
            mismatches.append(i)

    if not mismatches:
        print(f"MATCH {len(shared)}/{len(shared)}: r=1 is bitwise no-op.")
        sys.exit(0)
    print(f"FAIL: {len(mismatches)}/{len(shared)} idxs differ.")
    for i in mismatches[:3]:
        print(f"  idx={i}")
        print(f"    left:  {L[i]['completion'][:200]!r}")
        print(f"    right: {R[i]['completion'][:200]!r}")
    sys.exit(1)

if __name__ == "__main__":
    main()
```

CPU test: feed it two JSONL files with identical / different rows and
assert exit codes. Add to `tests/path2_v2/test_token_match.py` as a
subprocess test (~6 lines).

### 2. Base-model perplexity smoke (~30 lines)

The existing `--mode original` on `ple_sanity_check.py` already does
this --- we just need a cleaner wrapper that records *both* numbers
(no-hook and r=1 hook) and computes drift. Two options:

- **Reuse:** call `python experiments/ple_sanity_check.py --mode original
  --target-layer 17 --r-values 1` once. The output JSON includes
  baseline + r=1 ppl. Phase 1 reports `drift = abs(r1 - baseline) / baseline`.
- **Add:** a small `--mode hook-smoke` that does just this (~30 lines).

The simpler path is reuse. Phase 1 acceptance just reads the round 1
JSON and asserts `drift < 1e-4`.

### 3. Phase yaml under `experiments/path_2_v2/phase1.yaml`

Drop-in for `runpod.sh go`. The 5 cells are sequential commands a single
session runs; we don't parallelise within Phase 1 because the budget
is small enough that the orchestration cost dominates the savings.

```yaml
# experiment_path2_v2_phase1.yaml
run:
  flags: "--script path2-v2-phase1"   # NEW key, see registry below
  result_dir: "results/path_2_depth_recurrence_v2/phase1"

pod:
  gpu_types: "NVIDIA GeForce RTX 4090"
  gpu_count: 1
  cloud_type: SECURE
  container_disk_gb: 30
  volume_gb: 50

git:
  url: "https://github.com/kasuncp/gemma4-retro-recurrence.git"
  ref: "path_2_v2"

budget:
  cap_usd: 1.5     # ~1 pod-hour at 4090 spot pricing
  emergency_usd: 0.30
  max_hours: 2

watch:
  tick_seconds: 180  # tighter than usual --- short job
```

### 4. `scripts/run.sh` registry entry: `path2-v2-phase1`

Wraps the 5 cells in a single sequential dispatcher. New entry default args:

```
"--script path2-v2-phase1"  ->  experiments/path2_v2_phase1.py
```

`experiments/path2_v2_phase1.py` is a small shell-style runner (~80
lines) that does:

```python
from experiments.path2_v2_eval import _run_one_cell, parse_args  # reused

CELLS = [
    # (config, benchmark, n, model_id, gate_callable)
    ("baseline-C2",             "gsm8k", 50, "google/gemma-4-E2B-it", gate_1A),
    ("baseline-8shot-control",  "gsm8k", 50, "google/gemma-4-E2B-it", gate_1B_path1),
    ("baseline-8shot-round5",   "gsm8k", 50, "google/gemma-4-E2B-it", gate_1B_round5),
    ("W5-r1",                   "gsm8k", 20, "google/gemma-4-E2B-it", gate_1C),
    ("W5-r8",                   "gsm8k", 10, "google/gemma-4-E2B-it", gate_1D),
    # Cell 1E uses ple_sanity_check.py --mode original
]
```

Each gate is a function `(summary_dict) -> (passed: bool, message: str)`.
The runner prints a final table and exits non-zero on any failure.

---

## Cell-by-cell specification

### Cell 1A — IT C2 baseline gate

**Config:** `baseline-C2` (no hook, C2 prompt).
**Model:** `google/gemma-4-E2B-it`.
**Benchmark:** GSM8K, N=50, seed=42.
**Generation:** `max_new_tokens=512`, greedy, `use_cache=False`,
`pad_token_id=eos`.

**Anchor:** Path 1 plan 5/8 measured 71.6 % on N=500 GSM8K with this
exact prompt + model + extractor. Path 1 plan 9 confirmed smart_v2
extraction lifts that to 78.0 %.

**Gates** (all must pass):

| Metric | Band | Why |
|---|---|---|
| `accuracy_smart_v2` | ≥ 0.65 and ≤ 0.82 | ±10 pp around 71.6 % anchor; N=50 95 % CI is roughly ±13 pp, so the band is generous and only catches structural breakage |
| `loop_rate` | < 0.05 | Path 1 measured 0.0 % on the same cell at N=500. Any nonzero means the prompt builder regressed |
| `truncation_rate` | < 0.05 | Path 1 measured ~0 % at 512 tokens; any ≥10 % means EOS isn't firing |
| `parse_rate` | ≥ 0.95 | smart_v2 returned non-None |

**On failure:** the most likely cause is `apply_chat_template` not
being applied (Path 1 plan 5's first-hour bug). Inspect the first
prompt; it must end in `<start_of_turn>model\n`. If it doesn't,
`build_c2_gsm8k` regressed --- run `tests/path2_v2/test_prompts.py`.

**Wall budget:** ~12 min on a 4090 with `use_cache=False` (50 problems
× ~14 s each).

---

### Cell 1B — IT 8-shot bridges (two cells)

Round 5 (legacy = 54.8 %) and Path 1 plan 5 (smart_v2 = 30.0 %) used
**different prompts**, so one prompt cannot bridge both anchors. The
first iteration of this plan tried — observed legacy = 38 % on the
Path-1 prompt, never reaching round 5's number. Phase 1 now runs two
cells, one per anchor.

#### Cell 1B-p1 — Path 1 plan 5 anchor (smart_v2)

**Config:** `baseline-8shot-control` (no hook, single-user-turn
prompt with `Q: ...\nA: ...\n#### N` exemplars; matches
`experiments/path1_cot_gate.py:EXEMPLARS_COT` byte-for-byte).
**Benchmark:** GSM8K, N=50, seed=42 (same problems as 1A).

**Anchor:** Path 1 plan 5 measured smart_v2 = 30.0 % on N=500 with
this exact prompt. `accuracy_legacy` is recorded as informational but
NOT gated — this prompt does not reproduce round 5's 54.8 %, by design.

**Gates:**

| Metric | Band | Why |
|---|---|---|
| `accuracy_smart_v2` | ≥ 0.20 and ≤ 0.40 | ±10 pp around Path 1 plan 5's 30.0 % |
| `loop_rate` | < 0.25 | Path 1 plan 3 saw 10–12 %; we drop the lower bound because `has_repetition_loop` in `probes.extractors` is stricter than Path 1 plan 3's detector |
| `truncation_rate` | < 0.20 | 8-shot CoT runs longer than C2; some truncation expected |

**On failure:** if `accuracy_smart_v2` lands far from 30 %, diff
`probes.prompts.WEI_8SHOT_EXEMPLARS` and `build_8shot_cot_gsm8k`
against `experiments/path1_cot_gate.py:EXEMPLARS_COT` and
`build_chat_prompt`.

#### Cell 1B-r5 — Round 5 anchor (legacy)

**Config:** `baseline-8shot-round5` (no hook, alternating
user/assistant turns + `The answer is N.` marker + stop_strings
`["\nQ:", "\nQuestion:"]`; mirrors `probes.mode_round5._format_gsm8k_prompt_chat`).
**Benchmark:** GSM8K, N=50, seed=42 (same problems as 1A).

**Anchor:** Path 2 round 5 measured `accuracy_legacy` = 54.8 % on N=250
with this exact wrapping. `accuracy_smart_v2` is recorded as
informational but NOT gated — both extractors converge on this prompt
and smart_v2 will likely run higher than the path-1 band.

**Gates:**

| Metric | Band | Why |
|---|---|---|
| `accuracy_legacy` | ≥ 0.44 and ≤ 0.64 | ±10 pp around round 5's 54.8 %. This is the cross-round bridge |
| `loop_rate` | < 0.25 | Same upper bound as 1B-p1; lower bound dropped for the same reason |
| `truncation_rate` | < 0.20 | If this fires, stop_strings are likely not wired through `path2_v2_eval` to `model.generate` |

**On failure:** if `accuracy_legacy` lands far from 54.8 %, the cell
contract has drifted from `probes.mode_round5._format_gsm8k_prompt_chat`.
Three places to diff: (a) `WEI_8SHOT_EXEMPLARS_ROUND5` vs
`mode_round5.WEI_COT_EXEMPLARS`, (b) `build_8shot_cot_gsm8k_round5`
turn structure vs `_format_gsm8k_prompt_chat`, (c) stop_strings
plumbing in `experiments/path2_v2_eval.py` (the config dict carries
`stop_strings`; the eval forwards them as `gen_kwargs["stop_strings"]`
+ `gen_kwargs["tokenizer"]`).

**Wall budget:** ~30 min combined (~15 min per cell; longer prompts →
more tokens generated than C2).

---

### Cell 1C — Token-match no-op

**Configs:** `W5-r1` (block [15,19] r=1, hook installed) AND
`baseline-C2` (no hook). Same C2 prompt, same model, same 20 problems.

**Test:** Run `W5-r1` on the same 20 GSM8K problems Cell 1A used
(idx-aligned via the fixed seed). Then:

```
python experiments/path2_v2_token_match.py \
    --left  results/.../phase1/gsm8k__baseline-C2.jsonl \
    --right results/.../phase1/gsm8k__W5-r1.jsonl \
    --limit 20
```

**Gate:** exit code 0 (every shared idx's completion string is byte-equal).

**Why this is THE Phase 1 cell that decides everything:**

- Phase 0's tests confirm the hook code is structurally correct.
- Round 2c / 3b regression checks confirm `r=1` is identity at the
  perplexity level.
- Neither is sufficient. Generation calls `model.generate()`, which
  invokes forward many times in a tight loop with `use_cache=False`.
  If the hook leaves any state (KV state, hidden state cache, dropout
  RNG, RoPE position state) that perplexity doesn't surface, it'll
  show up here as token-level drift after several generation steps.
- A single mismatched idx halts the route and we debug; if all 20
  match byte-for-byte, the hook is genuinely no-op for downstream
  work.

**On failure:** likely culprits, in order:

1. `install_block_loop_hooks` returns from the post-hook even at r=1
   in a way that mutates the output tuple's secondary elements (e.g.,
   returning `(x,) + output[1:]` when the original returned `output`
   directly without tuple-wrapping for some forward signatures).
2. A pre-hook side-effect that survives between calls: e.g., the
   captured `kwargs` dict isn't being cleared between generation
   steps, and a stale `position_ids` is reused.
3. `model.generate` advancing the cache state inconsistently when
   `use_cache=False` is set after the model was loaded with
   `use_cache=True`.

**Wall budget:** ~5 min (re-runs 20 problems, baseline already cached).

---

### Cell 1D — W5-r8 process smoke

**Config:** `W5-r8` (block [15,19] r=8, every-iter PLE).
**Benchmark:** GSM8K, N=10. Smaller N because this is only "doesn't
crash"; we don't gate on accuracy.

**Gates:**

| Metric | Band | Why |
|---|---|---|
| process exit code | 0 | OOM, hook crash, NaN-during-generation all surface here |
| `n_problems` (rows in JSONL) | 10 | All 10 generations completed |
| `loop_rate` | < 0.95 | Round 5 saw 97-100 % on this exact config but with 8-shot. With C2 we expect lower. < 0.95 is a generous "the cell isn't 100 % pathological" check |
| any single `n_gen_tokens` | > 0 | Catches the case where every generation immediately emits EOS (hook collapsed the activations) |

**Note:** Cell 1D explicitly does NOT gate on accuracy. If recurrence
collapses reasoning at r=8 we want Phase 4 to characterise that, not
Phase 1 to halt on it. Loop rate ≥ 95 % WOULD halt because that
indicates the hook is producing pathological output that no later
phase can interpret.

**Wall budget:** ~10 min (10 problems, but each potentially generates
to the 512 cap if loop collapse is happening; budget for the worst case).

---

### Cell 1E — Base perplexity smoke

**Goal:** confirm the hook code itself hasn't regressed on the
*existing* perplexity test it's known to pass. This is a 30-second
sanity to catch refactoring damage.

**Run:**

```
python experiments/ple_sanity_check.py --mode original \
    --target-layer 17 --r-values 1 \
    --output-json results/path_2_depth_recurrence_v2/phase1/wikitext_smoke.json
```

`--mode original` already computes:
- unmodified base ppl on Wikitext-2 (50 seqs, 512 tokens)
- per-r looped ppl on layer 17 (here, just r=1)

**Gate:** load the JSON, compute
`drift = abs(ppl_r1 - unmodified_ppl) / unmodified_ppl`,
assert `drift < 1e-4`.

**Wall budget:** ~1 min.

---

## Reporting format

`results/path_2_depth_recurrence_v2/phase1/phase1_summary.json`:

```json
{
  "phase": "phase1-hook-sanity",
  "model_ids": {
    "it":   "google/gemma-4-E2B-it",
    "base": "google/gemma-4-E2B"
  },
  "cells": {
    "1A_baseline_C2": {
      "n": 50,
      "accuracy_smart_v2": 0.XXX,
      "accuracy_smart_v2_ci95": [low, high],
      "accuracy_legacy":   0.XXX,
      "loop_rate":         0.XXX,
      "truncation_rate":   0.XXX,
      "parse_rate":        0.XXX,
      "wall_seconds":      XXX,
      "gate_passed":       true|false,
      "gate_message":      "..."
    },
    "1B_baseline_8shot_path1":  {...},
    "1B_baseline_8shot_round5": {...},
    "1C_token_match":   {"shared_idxs": 20, "matches": 20, "first_mismatches": [], "gate_passed": true},
    "1D_W5_r8_smoke":   {...},
    "1E_base_ppl_smoke": {"unmodified_ppl": 12.5366, "r1_ppl": 12.5367, "drift": 1.2e-5, "gate_passed": true}
  },
  "all_gates_passed": true|false,
  "wall_seconds_total": XXX,
  "exit_action": "PROCEED to Phase 2" | "HALT --- <reason>"
}
```

Console table at end of run:

```
=== Phase 1 hook sanity gates ===
cell                    metric                  value     band              status
1A baseline-C2          accuracy_smart_v2       0.720     [0.65, 0.82]      PASS
1A baseline-C2          loop_rate               0.000     < 0.05            PASS
1A baseline-C2          truncation_rate         0.020     < 0.05            PASS
1B-p1 8shot-path1       accuracy_smart_v2       0.300     [0.20, 0.40]      PASS
1B-p1 8shot-path1       accuracy_legacy         0.380     (informational)
1B-p1 8shot-path1       loop_rate               0.000     < 0.25            PASS
1B-p1 8shot-path1       truncation_rate         0.160     < 0.20            PASS
1B-r5 8shot-round5      accuracy_legacy         0.548     [0.44, 0.64]      PASS
1B-r5 8shot-round5      accuracy_smart_v2       0.500     (informational)
1B-r5 8shot-round5      loop_rate               0.040     < 0.25            PASS
1B-r5 8shot-round5      truncation_rate         0.060     < 0.20            PASS
1C token-match          matches                 20/20     == 20/20          PASS
1D W5-r8 smoke          process_exit            0         == 0              PASS
1D W5-r8 smoke          loop_rate               0.700     < 0.95            PASS
1E base ppl drift       rel_drift               1.2e-05   < 1e-04           PASS

ALL GATES PASSED. Proceed to Phase 2.
```

Or on failure:

```
1A baseline-C2          accuracy_smart_v2       0.300     [0.65, 0.82]      FAIL
HALT: Cell 1A failed --- harness has regressed vs Path 1.
Likely cause: chat template not applied. Diff prompt against tests/path2_v2/test_prompts.py.
```

---

## Exit criteria

- **All six cells PASS** → write the 2-line update to `phase0.md`'s
  "design decisions" section confirming D6 (loop_rate first-class) and
  D8 (token-match) held empirically. Proceed to Phase 2.

- **Cell 1A fails on accuracy** → harness regression vs Path 1.
  Most often: `apply_chat_template` not in use, or smart_v2 broken.
  Halt; debug; do not run later cells.

- **Cell 1B-p1 fails on accuracy_smart_v2** → Path 1 plan 5 anchor
  broken. Diff `probes.prompts.WEI_8SHOT_EXEMPLARS` and
  `build_8shot_cot_gsm8k` against `experiments/path1_cot_gate.py:EXEMPLARS_COT`
  and `build_chat_prompt`. Halt.

- **Cell 1B-r5 fails on accuracy_legacy** → round-5 cross-round bridge
  broken. Diff `WEI_8SHOT_EXEMPLARS_ROUND5` and
  `build_8shot_cot_gsm8k_round5` against
  `probes.mode_round5._format_gsm8k_prompt_chat`; verify
  `stop_strings` are wired through `experiments/path2_v2_eval.py` (the
  config carries the field; the eval forwards it to `model.generate`).
  Halt.

- **Cell 1C fails on token-match** → the load-bearing structural
  failure. Hook is mutating something at generation time that didn't
  show up in perplexity. Halt; debug the hook before any other phase.
  This is the failure mode that justifies Phase 1's existence.

- **Cell 1D fails on process exit / loop_rate ≥ 95 %** → recurrence at
  r=8 produces unrecoverable generation pathology even on the C2
  prompt. This is itself a meaningful finding (round 5's collapse was
  NOT entirely prompt-induced). Halt Phase 1, but write up the
  finding before pivoting --- it changes Phase 4's r-sweep design
  (would explore much smaller r).

- **Cell 1E fails** → hook code regressed at the perplexity level.
  Halt; diff against the round 1 result JSON committed under
  `results/path_2_depth_recurrence/results.json`.

---

## Compute budget

| Cell | N | Wall (4090) | Notes |
|---|---|---|---|
| 1A | 200 | ~50 min | bumped from 50 to disambiguate the N=50 truncation rate (3/50 was just over the < 5 % gate) |
| 1B-p1 | 50 | ~15 min | longer 8-shot prompts |
| 1B-r5 | 50 | ~15 min | round-5 prompt is even longer (16 turns) |
| 1C | 20 | ~5 min  | re-runs 20 problems, baseline already cached |
| 1D | 10 | ~10 min | r=8 → 8× per-token compute, plus possible truncation to cap |
| 1E | 50 seqs | ~1 min  | perplexity is cheap |
| **Total** | | **~95 min** | budget 1.75 h with overhead |

Cost: ~$1.10–1.40 on a 4090 spot at $0.69/h.

If 1A passes cleanly at N=200, drop it back to N=50 in the runner so future Phase 1 reruns stay cheap.

---

## Open questions resolved before Phase 2

| Question | Decision |
|---|---|
| Should we test on base E2B beyond Cell 1E? | **No.** Base has no chat template; reasoning eval on base would require the round 4 raw-text-continuation prompt that we already know is broken. The architectural-correctness check is Cell 1E's perplexity drift. |
| Why two 8-shot cells (1B-p1 and 1B-r5)? | **Path 1 plan 5 and round 5 used different prompts.** Path 1 single-turn + `#### N` gave smart_v2 ≈ 30 %; round 5 multi-turn + "The answer is N." + stop_strings gave legacy ≈ 54.8 %. Trying to bridge both with one prompt fails (first iteration produced legacy = 38 %). Each cell anchors against its own prompt and gates only the metric that prompt was tuned for. |
| Do we also run Cell 1D with `iter1-only` PLE? | **No.** That's a Phase 3 ablation. Cell 1D's only job is "hook at r=8 doesn't crash." |
| What N for Cell 1D? | **10.** Lower bound for ≥1 minute of compute (so OOM has time to manifest), upper bound for "still 10 min". Smaller N risks missing rare crash modes that depend on a specific prompt. |
| Can we run Phase 1 on a 3090 instead? | **Yes, slightly slower** — budget ~70 min instead of ~45 min. The 4090 is preferred only because Phases 2/3 will need 4090 throughput anyway. |

---

## What Phase 2 inherits from Phase 1

If Phase 1 passes, Phase 2 inherits:

- **Confirmed harness:** `baseline-C2` accuracy is in band, smart_v2
  works, loop detector works, JSONL append survives.
- **A reusable cached IT model load:** Phase 2 can launch on the same
  pod and skip the 5-min model download.
- **A `baseline-C2` JSONL** at N=50 that Phase 2's per-layer cells will
  compare against. Each Phase 2 cell adds another JSONL alongside it.
- **Confidence that r=1 is bitwise no-op,** which means any per-layer
  cell at r=8 that produces drift is producing real recurrence-induced
  drift, not a hook artifact.

If Phase 1 fails, Phase 2 doesn't start.

---

## Implementation order (as a checklist for the PR)

- [ ] `experiments/path2_v2_token_match.py` (~50 lines)
- [ ] `tests/path2_v2/test_token_match.py` (subprocess test, ~40 lines)
- [ ] `experiments/path2_v2_phase1.py` runner (~80 lines), reuses
       `_run_one_cell` from `path2_v2_eval.py` for cells 1A/1B/1C/1D and
       shells out to `ple_sanity_check.py --mode original` for 1E
- [ ] `scripts/run.sh` registry entry: `path2-v2-phase1`
- [ ] `tests/run/test_dry_run.sh` row for the new key
- [ ] `experiment_path2_v2_phase1.yaml` at repo root
- [ ] Update `plans/path_2_depth_recurrence_v2/README.md` (1-page) once
       it exists, or add a phase-row to `phase0.md`'s status section

Phase 1 PR is small (~250 lines of new code, all CPU-testable).
Phase 2 plan does not get drafted until Phase 1 lands and passes.
