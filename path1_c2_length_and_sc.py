"""Path 1 --- plan 6: length + self-consistency on top of the C2 prompt.

Plan 2 measured a flat plateau for length and SC sweeps under the 8-shot
Wei-et-al prompt format. Plan 5 then showed the prompt was the bottleneck:
the C2 zero-shot plain chat-template prompt lifted greedy accuracy from
the ~30% plateau to 71.6% at less compute per problem.

Plan 6 re-runs plan 2's two one-dimensional probes on the C2 prompt format
to decide whether the length-and-SC plateau is a property of the model or
an artefact of the bad prompt:

    Axis A (length, greedy, IT chat-template, plain question):
        C2_A1_len128    C2_A2_len256    C2_A3_len512    C2_A4_len1024

    Axis B (self-consistency, T=0.7, top_p=0.95, len=512):
        C2_B1_k1        C2_B2_k3        C2_B3_k5        C2_B4_k10

C2_A3_len512 is the existing plan-5 cell and is re-used by reading its
JSONL shard directly --- this script does NOT re-run it. The shard lives
under ``results/path_1_cot_tokens/plan5/cells/C2_zeroshot_plain__0000_0500.jsonl``
and is exposed as the C2_A3_len512 reference cell at --summarize time.

Default --- one GPU, runs all 7 new cells (A1/A2/A4 + B1..B4) sequentially:
    python path1_c2_length_and_sc.py

Quick preview (auto-routes to plan6_preview_n<N>/ to keep full-run shards
intact):
    python path1_c2_length_and_sc.py --n 20

Axis-parallel on two GPUs:
    CUDA_VISIBLE_DEVICES=0 python path1_c2_length_and_sc.py --axis A
    CUDA_VISIBLE_DEVICES=1 python path1_c2_length_and_sc.py --axis B
    python path1_c2_length_and_sc.py --summarize

Pre-registered outcome labels (A/B/C/D/E) are computed during --summarize
and stamped on the figure. See plan6.md for the full interpretation.
"""

import argparse
import csv
import json
import math
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import probes  # noqa: F401  --- triggers HF_HOME redirect before torch loads.
from probes.env import DTYPE_MAP, load_model, print_env

# Borrow shard IO + Wilson CI + GSM8K loader from plan 1 so the row schema
# and problem indexing match the rest of Path 1.
from path1_cot_gate import (
    GOLD_RE,
    FALLBACK_RE,
    append_jsonl,
    load_existing,
    load_problems,
    wilson_ci,
)

# Plan 5 owns the C2 prompt format and the zero-shot extractor. Importing
# them verbatim eliminates drift: any change to the chat template here would
# force a corresponding change in plan 5, which would be caught by sanity
# check #1 (acc reproduction) when the manifest is re-loaded.
from path1_zero_shot import (
    C2_CELL,
    build_chat_prompt,
    extract_zeroshot,
    flops_per_problem,
    verify_model_arch,
    N_ACTIVE_PARAMS,
    N_LAYERS,
    N_HEADS,
    HEAD_DIM,
)


MODEL_KEY = "it"
MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_N = 500
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan6"
CELLS_SUBDIR = "cells"

# Plan 5's C2 cell shard --- re-used as the C2_A3_len512 reference.
C2_REF_DIR = "results/path_1_cot_tokens/plan5"
C2_REF_CELL = "C2_zeroshot_plain"  # plan 5's cell id

# Cell definitions. C2_A3_len512 is virtual: not run by this script, read
# from C2_REF_DIR at summarize time. The other seven cells are generated
# under RESULTS_SUBDIR/cells/.
AXIS_A = [
    ("C2_A1_len128",  128),
    ("C2_A2_len256",  256),
    ("C2_A3_len512",  512),  # virtual --- plan-5 reference, not re-run.
    ("C2_A4_len1024", 1024),
]
AXIS_B = [
    ("C2_B1_k1",  1),
    ("C2_B2_k3",  3),
    ("C2_B3_k5",  5),
    ("C2_B4_k10", 10),
]
AXIS_B_LEN = 512
AXIS_B_TEMPERATURE = 0.7
AXIS_B_TOP_P = 0.95

# Per-problem seed for axis-B sampling so generations are stable across
# restarts. See sanity check #5.
AXIS_B_BASE_SEED = 0

C2_A3_CELL = "C2_A3_len512"  # convenience handle

# Sanity thresholds.
#
# C2-A3 reference (from plan 5's results_plan5.json on the same shard):
#   accuracy 71.6% (358/500), mean_gen_tokens 275.748, mean_prompt_tokens 69.06.
# We assert hard match on the count and a ±0.5-token band on the mean to
# absorb any future re-summarize that aggregates rows in a different order.
C2_A3_EXPECTED_CORRECT = 358
C2_A3_EXPECTED_ACC = 0.716
C2_A3_EXPECTED_MEAN_GEN = 275.748
C2_A3_MEAN_GEN_TOL = 1.0

# Hash-hit rate is expected near zero (zero-shot model does not emit
# `#### N` markers); fallback regex carries the load. Plan 6 sanity check
# #3 wants fallback_hit_rate >= 0.95 on each new cell.
HASH_HIT_FLOOR = 0.30
FALLBACK_HIT_FLOOR = 0.95
FALLBACK_HIT_DEBUG_FLOOR = 0.85

# Vote degeneracy canary: at T=0.7 we expect 0.50--0.75. > 0.95 means
# sampling is effectively deterministic.
VOTE_DEGEN_LOW = 0.50
VOTE_DEGEN_HIGH = 0.95

# Pre-registered outcome thresholds (plan6.md "Pre-registered interpretation").
OUTCOME_NEAR = 0.03  # A: |Cx - C2_A3| within 3pp
OUTCOME_WIN = 0.05   # B/C/D: >= 5pp lift, non-overlapping CIs
OUTCOME_HURT = 0.03  # E: >= 3pp drop


# -----------------------------------------------------------------------------
# args
# -----------------------------------------------------------------------------

def parse_args():
    all_cells = [c for c, _ in AXIS_A] + [c for c, _ in AXIS_B]
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Number of GSM8K test problems to evaluate (default {DEFAULT_N}). "
             f"When set below {DEFAULT_N} and --results-dir is not overridden, "
             f"results auto-route to {RESULTS_SUBDIR}_preview_n<N>/ so full-run "
             f"shards are never overwritten. CIs will be wide --- treat preview "
             f"results as directional.",
    )
    p.add_argument(
        "--axis", choices=["A", "B", "both"], default="both",
        help="Which axis to run. Default both. Use --axis A on one GPU and "
             "--axis B on another for axis-parallel execution.",
    )
    p.add_argument(
        "--cells", nargs="+", choices=all_cells, default=None,
        help="Specific cell names to run (overrides --axis). "
             f"Choices: {', '.join(all_cells)}. C2_A3_len512 is virtual "
             f"(re-uses plan 5's shard) and is silently skipped if listed.",
    )
    p.add_argument(
        "--problems", default=None,
        help="Problem range as START:END (half-open) over the first --n GSM8K "
             "test problems. Default 0:N. Used to shard a single cell across "
             "GPUs (e.g. 0:250 and 250:500 at n=500).",
    )
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Axis-A greedy batch size. Default 1 is single-sequence and "
             "byte-deterministic. Larger values amortize GPU idle time; on a "
             "3090 batch=8 typically lifts throughput 3-5x. Axis-B sampling "
             "ignores this flag (it already batches k chains internally via "
             "num_return_sequences=k).",
    )
    p.add_argument(
        "--results-dir", default=RESULTS_SUBDIR,
        help="Directory for per-cell JSONL shards and final results_plan6.json.",
    )
    p.add_argument(
        "--c2-ref-dir", default=C2_REF_DIR,
        help=f"Directory holding plan 5's C2 shard (default {C2_REF_DIR}). "
             f"This script reads {C2_REF_CELL}__*.jsonl from "
             f"<c2-ref-dir>/{CELLS_SUBDIR}/ as the C2_A3_len512 reference cell. "
             f"At startup the script verifies the shard's row count + "
             f"accuracy match plan 5's published 358/500 = 0.716. Skipped "
             f"with a warning if absent.",
    )
    p.add_argument(
        "--summarize", action="store_true",
        help="Skip model loading; merge all per-cell shards in --results-dir, "
             "merge plan 5's C2 shard as the C2_A3_len512 reference, print "
             "both sweep tables, write results_plan6.json, path1_c2_pareto.csv, "
             "and the Pareto plot.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Before the full run, print one (prompt, completion) per axis "
             "and assert per-problem determinism on the first 3 problems for "
             "one cell per axis. Also runs sanity check #2 (chat-template "
             "byte-equivalence with plan 5's first row) using the loaded "
             "tokenizer.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing JSONL rows for the cells this invocation runs, "
             "overwriting them from scratch. Other cells' shards are untouched.",
    )
    args = p.parse_args()
    if args.n < 1:
        p.error(f"--n must be >= 1, got {args.n}.")
    if args.batch_size < 1:
        p.error(f"--batch-size must be >= 1, got {args.batch_size}.")
    if args.problems is None:
        args.problems = f"0:{args.n}"
    if args.cells is None:
        args.cells = (
            [c for c, _ in AXIS_A] if args.axis == "A" else
            [c for c, _ in AXIS_B] if args.axis == "B" else
            [c for c, _ in AXIS_A] + [c for c, _ in AXIS_B]
        )
    if args.n != DEFAULT_N and args.results_dir == RESULTS_SUBDIR:
        args.results_dir = f"{RESULTS_SUBDIR}_preview_n{args.n}"
    return args


def parse_range(spec, upper):
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"--problems must be START:END, got {spec!r}")
    start, end = int(m.group(1)), int(m.group(2))
    if not (0 <= start < end <= upper):
        raise ValueError(
            f"--problems range {start}:{end} out of bounds [0, {upper}]"
        )
    return start, end


# -----------------------------------------------------------------------------
# manifest
# -----------------------------------------------------------------------------

def manifest(args):
    return {
        "plan": "path_1_cot_tokens/plan6",
        "model_id": MODEL_ID,
        "model_key": MODEL_KEY,
        "dtype": args.dtype,
        "n_total": args.n,
        "axis_a_cells": [{"cell": c, "max_new_tokens": mn,
                           "decode": "greedy",
                           "virtual": (c == C2_A3_CELL)}
                          for c, mn in AXIS_A],
        "axis_b_cells": [{"cell": c, "k": k, "max_new_tokens": AXIS_B_LEN,
                           "decode": "sampled",
                           "temperature": AXIS_B_TEMPERATURE,
                           "top_p": AXIS_B_TOP_P}
                          for c, k in AXIS_B],
        "axis_b_base_seed": AXIS_B_BASE_SEED,
        "uses_chat_template": True,
        "user_template": "{question}",
        "answer_primary_regex": GOLD_RE.pattern,
        "answer_fallback_regex": FALLBACK_RE.pattern,
        "c2_ref": {"dir": str(args.c2_ref_dir), "cell": C2_REF_CELL,
                    "expected_correct": C2_A3_EXPECTED_CORRECT,
                    "expected_acc": C2_A3_EXPECTED_ACC},
    }


def check_manifest(results_dir, args):
    p = Path(results_dir) / "manifest.json"
    desired = manifest(args)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(desired, indent=2))
        return desired
    existing = json.loads(p.read_text())
    # Compare on stable fields. n_total widens automatically across runs.
    stable = ("plan", "model_id", "model_key", "dtype",
              "axis_a_cells", "axis_b_cells", "axis_b_base_seed",
              "uses_chat_template", "user_template",
              "answer_primary_regex", "answer_fallback_regex")
    diffs = [k for k in stable if existing.get(k) != desired.get(k)]
    if diffs:
        print(f"ERROR: manifest at {p} is incompatible with this run:")
        for k in diffs:
            print(f"  {k}: existing={existing.get(k)!r}  requested={desired[k]!r}")
        print(f"  Delete {p.parent} or fix the mismatch.")
        sys.exit(1)
    existing["n_total"] = max(existing.get("n_total", 0), args.n)
    existing["c2_ref"] = desired["c2_ref"]
    p.write_text(json.dumps(existing, indent=2))
    return existing


# -----------------------------------------------------------------------------
# C2-A3 reference loader (plan 5's shard --- read-only)
# -----------------------------------------------------------------------------

def load_c2_ref(c2_ref_dir):
    """Read plan 5's C2 shard as a list of rows. Returns ([], None) if the
    directory or shard is absent --- callers treat that as 'skip with warning'.
    The same loader is used at startup (sanity check #1) and at summarize
    (to populate the C2_A3_len512 cell)."""
    cells = Path(c2_ref_dir) / CELLS_SUBDIR
    if not cells.is_dir():
        return [], None
    shards = sorted(cells.glob(f"{C2_REF_CELL}__*.jsonl"))
    if not shards:
        return [], None
    rows, seen = [], set()
    for s in shards:
        for line in s.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("idx")
            if idx is None or idx in seen:
                continue
            seen.add(idx)
            rows.append(row)
    rows.sort(key=lambda r: r["idx"])
    return rows, [s.name for s in shards]


def slice_rows(rows, n_target):
    """Keep only deterministic first-n rows by idx."""
    return [r for r in rows if int(r.get("idx", -1)) < n_target]


def verify_c2_ref(c2_ref_dir, n_target):
    """Sanity check #1: confirm plan 5's C2 shard is present, has 500 idxs,
    and reproduces the published 358/500 = 71.6% accuracy. Returns a status
    dict for results_plan6.json."""
    rows, shard_names = load_c2_ref(c2_ref_dir)
    if not rows:
        msg = (f"C2 reference shard not found under "
               f"{c2_ref_dir}/{CELLS_SUBDIR}/{C2_REF_CELL}__*.jsonl. "
               f"Sync plan 5's results down or rebootstrap. The C2_A3_len512 "
               f"cell will show 'no rows' in summarize and the outcome label "
               f"will be INCOMPLETE.")
        print(f"WARNING: {msg}")
        return {"skipped": True, "reason": msg, "rows": 0}

    n = len(rows)
    correct = sum(int(r.get("correct", 0)) for r in rows)
    mean_gen = sum(int(r.get("n_gen_tokens", 0)) for r in rows) / max(1, n)
    acc = correct / max(1, n)

    issues = []
    if n != n_target:
        issues.append(f"row count {n} != n_target {n_target}")
    if correct != C2_A3_EXPECTED_CORRECT and n == DEFAULT_N:
        issues.append(f"correct {correct} != expected {C2_A3_EXPECTED_CORRECT}")
    if abs(mean_gen - C2_A3_EXPECTED_MEAN_GEN) > C2_A3_MEAN_GEN_TOL and n == DEFAULT_N:
        issues.append(f"mean_gen_tokens {mean_gen:.2f} drifted from expected "
                      f"{C2_A3_EXPECTED_MEAN_GEN:.2f} (tol +/-{C2_A3_MEAN_GEN_TOL})")
    if issues:
        print("ERROR: plan 5 C2 reference does not reproduce. The C2_A3 "
              "anchor for plan 6 is invalid. Issues:")
        for i in issues:
            print(f"  - {i}")
        print(f"  Plan 5 dir: {c2_ref_dir}")
        print(f"  Shards: {shard_names}")
        sys.exit(1)
    print(f"sanity #1 C2-A3 reproduction: OK --- "
          f"n={n}  correct={correct}  acc={acc:.3f}  "
          f"mean_gen_tokens={mean_gen:.2f}  shards={shard_names}")
    return {
        "skipped": False, "rows": n, "correct": correct, "accuracy": acc,
        "mean_gen_tokens": mean_gen,
        "expected_correct": C2_A3_EXPECTED_CORRECT,
        "expected_acc": C2_A3_EXPECTED_ACC,
        "expected_mean_gen": C2_A3_EXPECTED_MEAN_GEN,
        "shard_files": shard_names,
    }


# -----------------------------------------------------------------------------
# chat-template byte-equivalence (sanity check #2)
# -----------------------------------------------------------------------------

def verify_chat_template(tokenizer, problems, c2_ref_rows):
    """Sanity check #2: build the chat prompt for problem 0 with this
    tokenizer + plan 5's prompt builder, count input tokens, and confirm it
    matches plan 5's row 0 prompt_tokens. A mismatch means the chat template
    or tokenizer has drifted from plan 5 --- the 71.6% C2-A3 anchor would no
    longer be the same prompt format. Returns a status dict.

    Skipped (returns ``{"skipped": True}``) if plan 5's reference rows are
    absent. Hard-fails on mismatch.
    """
    if not c2_ref_rows:
        msg = "plan 5 reference rows absent; cannot run chat-template equivalence"
        print(f"WARNING: sanity #2 chat-template equivalence: SKIPPED --- {msg}")
        return {"skipped": True, "reason": msg}
    if not problems:
        msg = "no problems in range; sanity #2 skipped"
        print(f"NOTE: sanity #2 chat-template equivalence: SKIPPED --- {msg}")
        return {"skipped": True, "reason": msg}

    ex0 = problems[0]
    plan5_row0 = c2_ref_rows[0]
    plan5_tokens = plan5_row0.get("prompt_tokens")
    if plan5_tokens is None:
        msg = "plan 5 row 0 missing prompt_tokens; cannot compare"
        print(f"WARNING: sanity #2 chat-template equivalence: SKIPPED --- {msg}")
        return {"skipped": True, "reason": msg}

    prompt = build_chat_prompt(tokenizer, C2_CELL, ex0["question"])
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    plan6_tokens = int(enc["input_ids"].shape[1])

    if plan6_tokens != plan5_tokens:
        print(f"ERROR: sanity #2 chat-template byte-equivalence FAILED.")
        print(f"  problem 0 prompt_tokens: plan6={plan6_tokens}  plan5={plan5_tokens}")
        print(f"  The chat template, tokenizer, or model commit has drifted "
              f"from plan 5. Plan 6's results would not be on the same C2 "
              f"prompt format; aborting before wasting compute.")
        sys.exit(1)
    print(f"sanity #2 chat-template equivalence: OK --- "
          f"problem 0 prompt_tokens={plan6_tokens} (matches plan 5)")
    return {"skipped": False, "plan5_prompt_tokens": plan5_tokens,
            "plan6_prompt_tokens": plan6_tokens}


# -----------------------------------------------------------------------------
# shard paths + completed-set loading
# -----------------------------------------------------------------------------

def shard_jsonl(results_dir, cell, start, end):
    return Path(results_dir) / CELLS_SUBDIR / f"{cell}__{start:04d}_{end:04d}.jsonl"


# -----------------------------------------------------------------------------
# generation: axis A (greedy) and axis B (sampled with k returns)
# -----------------------------------------------------------------------------

def _greedy_gen_cfg(model):
    cfg = deepcopy(model.generation_config)
    cfg.do_sample = False
    cfg.top_p = None
    cfg.top_k = None
    cfg.temperature = None
    return cfg


def _sampled_gen_cfg(model):
    cfg = deepcopy(model.generation_config)
    cfg.do_sample = True
    cfg.temperature = AXIS_B_TEMPERATURE
    cfg.top_p = AXIS_B_TOP_P
    cfg.top_k = None
    return cfg


def _count_gen_tokens(gen_ids, pad_id):
    """Count real (non-pad) generated tokens. HF generate pads after EOS
    when batching; single-sequence calls return early with no padding."""
    if pad_id is None:
        return int(gen_ids.numel())
    return int((gen_ids != pad_id).sum().item())


def generate_greedy(model, tokenizer, prompt, max_new):
    import torch
    enc = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(model.device)
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc, generation_config=_greedy_gen_cfg(model),
            max_new_tokens=max_new, pad_token_id=pad_id,
        )
    gen_ids = out[0, input_len:]
    completion = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_gen = _count_gen_tokens(gen_ids, pad_id)
    return completion, n_gen, input_len


def generate_greedy_batch(model, tokenizer, prompts, max_new):
    """Batched equivalent of generate_greedy. Returns a list of
    (completion, n_gen, input_len) per input prompt, in order. Uses
    left-padding so generation continues from each prompt's true last token.
    Not bit-identical to single-sequence generate on the same prompt; FP
    matmul reordering can flip a few tokens per problem."""
    import torch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            list(prompts), return_tensors="pt", padding=True,
            add_special_tokens=False,
        ).to(model.device)
    finally:
        tokenizer.padding_side = prev_side
    input_shape = enc["input_ids"].shape[1]
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc, generation_config=_greedy_gen_cfg(model),
            max_new_tokens=max_new, pad_token_id=pad_id,
        )
    gen_ids = out[:, input_shape:]
    results = []
    for row_gen, input_len in zip(gen_ids, input_lens):
        completion = tokenizer.decode(row_gen, skip_special_tokens=True)
        n_gen = _count_gen_tokens(row_gen, pad_id)
        results.append((completion, n_gen, int(input_len)))
    return results


def sample_k_chains(model, tokenizer, prompt, k, max_new, idx):
    """Sample k chains in a single generate() call (num_return_sequences=k),
    seeded per-problem for resume-stable determinism (sanity check #5)."""
    import torch
    torch.manual_seed(AXIS_B_BASE_SEED + idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(AXIS_B_BASE_SEED + idx)
    enc = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(model.device)
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc, generation_config=_sampled_gen_cfg(model),
            max_new_tokens=max_new, pad_token_id=pad_id,
            num_return_sequences=k,
        )
    completions, n_gens = [], []
    for i in range(k):
        gen_ids = out[i, input_len:]
        completions.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        n_gens.append(_count_gen_tokens(gen_ids, pad_id))
    return completions, n_gens, input_len


def majority_vote(preds):
    """Plurality of non-None integer predictions. Ties broken by order
    (first-chain-wins), which is deterministic given the per-problem seed."""
    counts = {}
    order = []
    for p in preds:
        if p is None:
            continue
        if p not in counts:
            counts[p] = 0
            order.append(p)
        counts[p] += 1
    if not order:
        return None, 0
    best = max(order, key=lambda p: (counts[p], -order.index(p)))
    return best, counts[best]


# -----------------------------------------------------------------------------
# per-cell runners
# -----------------------------------------------------------------------------

def run_axis_a_cell(model, tokenizer, cell, max_new, problems, golds, start,
                    end, results_dir, no_resume, batch_size=1):
    if cell == C2_A3_CELL:
        print(f"[{cell}] virtual cell --- re-using plan 5's C2 shard, "
              f"nothing to generate.")
        return None
    path = shard_jsonl(results_dir, cell, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p, g) for i, (p, g) in enumerate(zip(problems, golds))
                 if (start + i) not in done]
    print(f"[{cell}] greedy max_new={max_new}  {start}:{end}  "
          f"{len(done)} done, {len(remaining)} to generate  -> {path.name}  "
          f"batch_size={batch_size}")
    gen_secs = 0.0
    if batch_size <= 1:
        for i, prob, gold in remaining:
            prompt = build_chat_prompt(tokenizer, C2_CELL, prob["question"])
            t0 = time.perf_counter()
            completion, n_gen, prompt_len = generate_greedy(
                model, tokenizer, prompt, max_new,
            )
            dt = time.perf_counter() - t0
            gen_secs += dt
            pred, hashed, fallback = extract_zeroshot(completion)
            append_jsonl(path, {
                "idx": start + i,
                "gold": gold,
                "pred": pred,
                "correct": int(pred is not None and pred == gold),
                "hash_hit": int(hashed),
                "fallback_hit": int(fallback),
                "completion": completion,
                "n_gen_tokens": n_gen,
                "prompt_tokens": prompt_len,
                "gen_secs": dt,
            })
    else:
        for b in range(0, len(remaining), batch_size):
            batch = remaining[b:b + batch_size]
            prompts = [build_chat_prompt(tokenizer, C2_CELL, prob["question"])
                       for _, prob, _ in batch]
            t0 = time.perf_counter()
            per_row = generate_greedy_batch(
                model, tokenizer, prompts, max_new,
            )
            dt = time.perf_counter() - t0
            gen_secs += dt
            per_row_dt = dt / len(batch)
            for (i, _prob, gold), (completion, n_gen, prompt_len) in zip(
                    batch, per_row):
                pred, hashed, fallback = extract_zeroshot(completion)
                append_jsonl(path, {
                    "idx": start + i,
                    "gold": gold,
                    "pred": pred,
                    "correct": int(pred is not None and pred == gold),
                    "hash_hit": int(hashed),
                    "fallback_hit": int(fallback),
                    "completion": completion,
                    "n_gen_tokens": n_gen,
                    "prompt_tokens": prompt_len,
                    "gen_secs": per_row_dt,
                })
    if gen_secs > 0 and remaining:
        print(f"[{cell}] {len(remaining)/gen_secs:.2f} problems/s  "
              f"over {gen_secs:.1f}s")
    return path


def run_axis_b_cell(model, tokenizer, cell, k, problems, golds, start, end,
                    results_dir, no_resume):
    path = shard_jsonl(results_dir, cell, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p, g) for i, (p, g) in enumerate(zip(problems, golds))
                 if (start + i) not in done]
    print(f"[{cell}] sampled k={k} len={AXIS_B_LEN} "
          f"T={AXIS_B_TEMPERATURE} top_p={AXIS_B_TOP_P}  "
          f"{start}:{end}  {len(done)} done, {len(remaining)} to generate  "
          f"-> {path.name}")
    gen_secs = 0.0
    for i, prob, gold in remaining:
        prompt = build_chat_prompt(tokenizer, C2_CELL, prob["question"])
        t0 = time.perf_counter()
        completions, n_gens, prompt_len = sample_k_chains(
            model, tokenizer, prompt, k, AXIS_B_LEN, start + i,
        )
        dt = time.perf_counter() - t0
        gen_secs += dt
        chains = []
        preds = []
        for comp, n in zip(completions, n_gens):
            pred, hashed, fallback = extract_zeroshot(comp)
            chains.append({"completion": comp, "pred": pred,
                           "hash_hit": int(hashed),
                           "fallback_hit": int(fallback),
                           "n_gen_tokens": n})
            preds.append(pred)
        voted, voted_count = majority_vote(preds)
        append_jsonl(path, {
            "idx": start + i,
            "gold": gold,
            "k": k,
            "chains": chains,
            "voted_pred": voted,
            "voted_count": voted_count,
            "vote_degenerate": int(voted_count == k),
            "correct": int(voted is not None and voted == gold),
            "prompt_tokens": prompt_len,
            "gen_secs": dt,
        })
    if gen_secs > 0 and remaining:
        print(f"[{cell}] {len(remaining)/gen_secs:.2f} problems/s  "
              f"over {gen_secs:.1f}s")
    return path


# -----------------------------------------------------------------------------
# smoke: chat-template + sample completions per axis + determinism
# -----------------------------------------------------------------------------

def run_smoke(model, tokenizer, cells_to_run, problems, golds, start):
    """Print one (prompt, completion) per axis and assert per-problem
    determinism on the first 3 problems for one cell per axis. Greedy is
    bit-exact across calls; sampled is bit-exact under the per-problem seed."""
    import torch
    if not problems:
        print("  (no problems in range; smoke skipped)")
        return
    axis_a_to_run = [c for c in cells_to_run
                     if c in dict(AXIS_A) and c != C2_A3_CELL]
    axis_b_to_run = [c for c in cells_to_run if c in dict(AXIS_B)]
    ex0 = problems[0]
    prompt = build_chat_prompt(tokenizer, C2_CELL, ex0["question"])
    head = prompt[:240].replace("\n", "\\n")
    tail = prompt[-160:].replace("\n", "\\n")
    print(f"  SMOKE C2 chat-template head: {head!r}")
    print(f"  SMOKE C2 chat-template tail: {tail!r}")

    if axis_a_to_run:
        cell = axis_a_to_run[0]
        max_new = dict(AXIS_A)[cell]
        comp, n_gen, _ = generate_greedy(model, tokenizer, prompt, max_new)
        pred, hashed, fallback = extract_zeroshot(comp)
        print(f"  SMOKE {cell} (greedy, len={max_new})  idx={start}  "
              f"pred={pred}  gold={golds[0]}  hash={hashed}  fallback={fallback}  "
              f"n_gen={n_gen}")
        print(f"    completion[:240] = {comp[:240]!r}")
        det_ok = True
        for i in range(min(3, len(problems))):
            p = build_chat_prompt(tokenizer, C2_CELL, problems[i]["question"])
            a, _, _ = generate_greedy(model, tokenizer, p, max_new)
            b, _, _ = generate_greedy(model, tokenizer, p, max_new)
            if a != b:
                det_ok = False
                print(f"  DETERMINISM FAIL on {cell} idx={start+i}")
        if not det_ok:
            print(f"ERROR: greedy decoding is not byte-for-byte deterministic.")
            sys.exit(1)
        print(f"  GREEDY DETERMINISM OK ({cell}, first 3 problems)")
    if axis_b_to_run:
        cell = axis_b_to_run[0]
        k = dict(AXIS_B)[cell]
        comps, n_gens, _ = sample_k_chains(
            model, tokenizer, prompt, k, AXIS_B_LEN, start,
        )
        preds = [extract_zeroshot(c)[0] for c in comps]
        voted, voted_count = majority_vote(preds)
        print(f"  SMOKE {cell} (sampled, k={k})  idx={start}  "
              f"chain_preds={preds}  voted={voted}  gold={golds[0]}")
        det_ok = True
        for i in range(min(3, len(problems))):
            p = build_chat_prompt(tokenizer, C2_CELL, problems[i]["question"])
            a, _, _ = sample_k_chains(model, tokenizer, p, k, AXIS_B_LEN,
                                       start + i)
            b, _, _ = sample_k_chains(model, tokenizer, p, k, AXIS_B_LEN,
                                       start + i)
            if a != b:
                det_ok = False
                print(f"  SAMPLING DETERMINISM FAIL on {cell} idx={start+i}")
        if not det_ok:
            print(f"ERROR: per-problem-seeded sampling is not byte-for-byte "
                  f"reproducible. Sanity check #5 broken.")
            sys.exit(1)
        print(f"  SAMPLING DETERMINISM OK ({cell}, first 3 problems)")
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# summarize: merge shards, compute metrics + verdicts, write outputs
# -----------------------------------------------------------------------------

def load_shards(results_dir, cell):
    cells_dir = Path(results_dir) / CELLS_SUBDIR
    shards = sorted(cells_dir.glob(f"{cell}__*.jsonl"))
    rows, seen = [], set()
    for s in shards:
        for line in s.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["idx"] in seen:
                continue
            seen.add(row["idx"])
            rows.append(row)
    rows.sort(key=lambda r: r["idx"])
    return rows


def metrics_for_axis_a(rows):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(int(r.get("correct", 0)) for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    hash_hit = sum(int(r.get("hash_hit", 0)) for r in rows) / n
    no_extract = sum(int(r.get("pred") is None) for r in rows) / n
    fallback_hit = 1.0 - no_extract
    if rows and "fallback_hit" in rows[0]:
        strict_fallback = sum(int(r.get("fallback_hit", 0)) for r in rows) / n
    else:
        strict_fallback = sum(
            int(r.get("hash_hit", 0) == 0 and r.get("pred") is not None)
            for r in rows
        ) / n
    mean_gen = sum(int(r.get("n_gen_tokens", 0)) for r in rows) / n
    prompt_lens = [int(r.get("prompt_tokens", 0)) for r in rows]
    mean_prompt = sum(prompt_lens) / max(1, len(prompt_lens))
    flops = sum(flops_per_problem(int(r.get("prompt_tokens", 0)),
                                   int(r.get("n_gen_tokens", 0)))
                for r in rows) / n
    mean_wall = sum(float(r.get("gen_secs", 0.0)) for r in rows) / n
    return {
        "n": n, "correct": correct, "accuracy": acc,
        "ci95": [lo, hi],
        "hash_hit_rate": hash_hit,
        "fallback_hit_rate": fallback_hit,
        "strict_fallback_hit_rate": strict_fallback,
        "no_extract_rate": no_extract,
        "mean_gen_tokens": mean_gen,
        "mean_prompt_tokens": mean_prompt,
        "mean_flops_per_problem": flops,
        "mean_wallclock_sec": mean_wall,
    }


def metrics_for_axis_b(rows, k):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(int(r.get("correct", 0)) for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    vote_degen = sum(int(r.get("vote_degenerate", 0)) for r in rows) / n
    chain_hash, chain_fallback, chain_gen = [], [], []
    for r in rows:
        for ch in r.get("chains", []):
            chain_hash.append(int(ch.get("hash_hit", 0)))
            chain_fallback.append(int(ch.get("fallback_hit", 0)))
            chain_gen.append(int(ch.get("n_gen_tokens", 0)))
    nchains = len(chain_hash)
    hash_hit = sum(chain_hash) / nchains if nchains else 0.0
    strict_fallback = sum(chain_fallback) / nchains if nchains else 0.0
    no_extract_chains = sum(
        1 for r in rows for ch in r.get("chains", [])
        if ch.get("pred") is None
    ) / max(1, nchains)
    fallback_hit = 1.0 - no_extract_chains
    mean_gen = sum(chain_gen) / nchains if nchains else 0.0
    prompt_len = int(rows[0].get("prompt_tokens", 0))
    flops = sum(
        flops_per_problem(int(r.get("prompt_tokens", prompt_len)),
                           sum(int(ch.get("n_gen_tokens", 0))
                               for ch in r.get("chains", [])) / max(1, k),
                           k=k)
        for r in rows
    ) / n
    mean_wall = sum(float(r.get("gen_secs", 0.0)) for r in rows) / n
    return {
        "n": n, "correct": correct, "accuracy": acc,
        "ci95": [lo, hi],
        "hash_hit_rate": hash_hit,
        "fallback_hit_rate": fallback_hit,
        "strict_fallback_hit_rate": strict_fallback,
        "no_extract_rate": no_extract_chains,
        "vote_degeneracy_rate": vote_degen,
        "mean_gen_tokens": mean_gen,
        "mean_prompt_tokens": prompt_len,
        "mean_flops_per_problem": flops,
        "mean_wallclock_sec": mean_wall,
    }


def warn_sanity_thresholds(axis_a_metrics, axis_b_metrics):
    """Soft warnings on hash_hit / fallback_hit floors and vote-degeneracy
    canary (sanity #3 and #4 in plan6.md). Non-fatal."""
    for name, m in axis_a_metrics.items():
        if m is None or name == C2_A3_CELL:
            continue
        h = m["hash_hit_rate"]
        f = m["fallback_hit_rate"]
        if f < FALLBACK_HIT_DEBUG_FLOOR:
            print(f"WARNING: {name} fallback_hit_rate={f:.2f} < "
                  f"{FALLBACK_HIT_DEBUG_FLOOR} --- integer extractor is "
                  f"misfiring on prose answers; debug before trusting accuracy.")
        elif f < FALLBACK_HIT_FLOOR:
            print(f"NOTE: {name} fallback_hit_rate={f:.2f} below the "
                  f"{FALLBACK_HIT_FLOOR} floor but above the debug floor; "
                  f"extraction is shaky but probably trustworthy.")
        if h < HASH_HIT_FLOOR:
            print(f"NOTE: {name} hash_hit_rate={h:.2f} < {HASH_HIT_FLOOR} "
                  f"(expected for 0-shot prose answers --- fallback carries "
                  f"the load).")
    for name, m in axis_b_metrics.items():
        if m is None:
            continue
        f = m["fallback_hit_rate"]
        if f < FALLBACK_HIT_DEBUG_FLOOR:
            print(f"WARNING: {name} chain fallback_hit_rate={f:.2f} < "
                  f"{FALLBACK_HIT_DEBUG_FLOOR} --- chain extraction is "
                  f"unreliable.")
        vd = m["vote_degeneracy_rate"]
        if vd > VOTE_DEGEN_HIGH:
            print(f"WARNING: {name} vote_degeneracy_rate={vd:.2f} > "
                  f"{VOTE_DEGEN_HIGH} --- sampling is effectively "
                  f"deterministic; bug or temperature too low.")
        elif vd < VOTE_DEGEN_LOW:
            print(f"NOTE: {name} vote_degeneracy_rate={vd:.2f} < "
                  f"{VOTE_DEGEN_LOW} --- chains diverge a lot; may indicate "
                  f"high-entropy sampling.")


def ci_overlap(a_ci, b_ci):
    a_lo, a_hi = a_ci
    b_lo, b_hi = b_ci
    return not (a_lo > b_hi or b_lo > a_hi)


def best_b_cell(axis_b_metrics):
    """Pick the highest-accuracy axis-B cell among B3 / B4 (where the lift
    is supposed to land per plan 6). Returns (name, metrics) or (None, None)
    if neither has rows."""
    candidates = []
    for name in ("C2_B3_k5", "C2_B4_k10"):
        m = axis_b_metrics.get(name)
        if m is not None:
            candidates.append((name, m))
    if not candidates:
        return None, None
    best = max(candidates, key=lambda x: x[1]["accuracy"])
    return best


def determine_outcome(axis_a, axis_b):
    """Pre-registered interpretation from plan6.md.

    Inputs are the per-cell metrics dicts. Returns (label, reason).

    Labels:
      A --- plateau confirmed (all cells within +-3pp of C2-A3)
      B --- length helps (C2-A4 >= C2-A3 + 5pp, non-overlapping CIs)
      C --- SC helps (best B cell >= C2-A3 + 5pp, non-overlapping CIs,
            and the lift is greater than length's)
      D --- both length and SC help (A4 and B3/B4 both beat by >= 5pp)
      E --- SC hurts (best B cell < C2-A3 by >= 3pp)
      INCOMPLETE / AMBIGUOUS --- otherwise
    """
    a3 = axis_a.get(C2_A3_CELL)
    if a3 is None:
        return "INCOMPLETE", ("C2_A3_len512 reference missing; sync plan 5 "
                                "shards down or pass --c2-ref-dir.")

    a4 = axis_a.get("C2_A4_len1024")
    b_name, b_best = best_b_cell(axis_b)

    # Need at least one new axis-A cell or one B cell beyond C2_A3 to call
    # any outcome. Otherwise the plan-6 question hasn't been answered yet.
    new_a_cells = [m for name, m in axis_a.items()
                    if name != C2_A3_CELL and m is not None]
    new_b_cells = [m for m in axis_b.values() if m is not None]
    if not new_a_cells and not new_b_cells:
        return "INCOMPLETE", ("only C2_A3 reference present; no new plan-6 "
                                "cells generated yet. Run axis A and/or "
                                "axis B before --summarize.")

    a3_acc = a3["accuracy"]
    a3_ci = a3["ci95"]
    cells_have_5_or_more = lambda m: m is not None and m["n"] >= 5

    a4_lift = (a4["accuracy"] - a3_acc) if cells_have_5_or_more(a4) else None
    a4_lift_clean = (a4_lift is not None
                      and a4_lift >= OUTCOME_WIN
                      and not ci_overlap(a3_ci, a4["ci95"]))
    b_lift = (b_best["accuracy"] - a3_acc) if cells_have_5_or_more(b_best) else None
    b_lift_clean = (b_lift is not None
                     and b_lift >= OUTCOME_WIN
                     and not ci_overlap(a3_ci, b_best["ci95"]))

    # Outcome E first: SC hurts is the surprise direction.
    if b_lift is not None and b_lift <= -OUTCOME_HURT:
        return "E", (f"{b_name} ({b_best['accuracy']:.3f}) - "
                      f"C2_A3 ({a3_acc:.3f}) = {b_lift:+.3f} <= "
                      f"{-OUTCOME_HURT:+.2f}. Plausibly: zero-shot prompts "
                      f"yield diverse-but-wrong chains; voting converges on "
                      f"a wrong answer. Follow-up: lower T to 0.3-0.5 or "
                      f"switch to top-k sampling.")

    # Outcome D: length + SC compounding.
    if a4_lift_clean and b_lift_clean:
        return "D", (f"Both compounding: A4 lift={a4_lift:+.3f}, "
                      f"{b_name} lift={b_lift:+.3f}. Run one combined cell "
                      f"(SC k=5 at len=1024) and use it as Path 1's "
                      f"representative.")

    # Outcome C: SC helps and beats length.
    if b_lift_clean and (a4_lift is None or b_lift > a4_lift):
        return "C", (f"{b_name} ({b_best['accuracy']:.3f}) - "
                      f"C2_A3 ({a3_acc:.3f}) = {b_lift:+.3f} >= "
                      f"{OUTCOME_WIN:+.2f} with non-overlapping CIs. "
                      f"Self-consistency is the lever once the prompt is "
                      f"good. Path 1's representative becomes {b_name}.")

    # Outcome B: length helps.
    if a4_lift_clean:
        return "B", (f"C2_A4_len1024 ({a4['accuracy']:.3f}) - "
                      f"C2_A3 ({a3_acc:.3f}) = {a4_lift:+.3f} >= "
                      f"{OUTCOME_WIN:+.2f} with non-overlapping CIs. "
                      f"Length was suppressed by the bad prompt; it works on "
                      f"C2. Investigate len=2048 follow-up. Path 1's "
                      f"representative becomes C2_A4_len1024.")

    # Outcome A: plateau confirmed if every cell is within +-3pp of C2-A3.
    all_cells = list(axis_a.items()) + list(axis_b.items())
    bands = []
    band_ok = True
    for name, m in all_cells:
        if m is None or name == C2_A3_CELL:
            continue
        diff = m["accuracy"] - a3_acc
        bands.append(f"{name}={diff:+.3f}")
        if abs(diff) > OUTCOME_NEAR:
            band_ok = False
    if band_ok and bands:
        return "A", (f"All cells within +-{OUTCOME_NEAR:.2f} of "
                      f"C2_A3 ({a3_acc:.3f}). Path 1 ceiling locked at "
                      f"71.6%; more inference compute does not buy more "
                      f"accuracy on this model on GSM8K, regardless of "
                      f"prompt format. Representative stays C2_A3_len512.")

    # Otherwise.
    return "AMBIGUOUS", (
        f"a3={a3_acc:.3f} | a4_lift={a4_lift} clean={a4_lift_clean}, "
        f"{b_name}_lift={b_lift} clean={b_lift_clean}, bands=[{', '.join(bands)}]; "
        f"no plan-6 pre-registered pattern matched."
    )


def fmt_row_a(name, m, extra=""):
    if m is None:
        return f"{name:18s}  (no rows)"
    return (f"{name:18s}  acc={m['accuracy']:.3f}  "
            f"ci=({m['ci95'][0]:.3f},{m['ci95'][1]:.3f})  "
            f"hash={m['hash_hit_rate']:.2f}  "
            f"fb={m['fallback_hit_rate']:.2f}  "
            f"mean_tok={m['mean_gen_tokens']:.1f}{extra}  n={m['n']}")


def fmt_row_b(name, m, extra=""):
    if m is None:
        return f"{name:18s}  (no rows)"
    return (f"{name:18s}  acc={m['accuracy']:.3f}  "
            f"ci=({m['ci95'][0]:.3f},{m['ci95'][1]:.3f})  "
            f"vote_deg={m['vote_degeneracy_rate']:.2f}  "
            f"hash={m['hash_hit_rate']:.2f}  "
            f"fb={m['fallback_hit_rate']:.2f}  "
            f"mean_tok={m['mean_gen_tokens']:.1f}{extra}  n={m['n']}")


def pareto_frontier(cells):
    """Return the subset of (cell_name, acc, flops) tuples that are not
    dominated by any other. Cell X is dominated by Y iff Y has strictly
    higher accuracy AND strictly lower FLOPs."""
    frontier = []
    for name, acc, flops in cells:
        dominated = any(
            other_acc > acc and other_flops < flops
            for other_name, other_acc, other_flops in cells
            if other_name != name
        )
        if not dominated:
            frontier.append((name, acc, flops))
    return sorted(frontier, key=lambda x: x[2])


def summarize(results_dir, c2_ref_dir):
    root = Path(results_dir)
    mpath = root / "manifest.json"
    if not mpath.exists():
        print(f"ERROR: no manifest at {mpath}. Run at least one cell first.")
        sys.exit(1)
    man = json.loads(mpath.read_text())
    n_target = man.get("n_total", DEFAULT_N)

    # Load axis A. C2_A3 is virtual --- pulls from plan 5's dir.
    axis_a_metrics = {}
    c2_ref_rows, c2_ref_shards = load_c2_ref(c2_ref_dir)
    c2_ref_rows = slice_rows(c2_ref_rows, n_target)
    axis_a_metrics[C2_A3_CELL] = (
        metrics_for_axis_a(c2_ref_rows) if c2_ref_rows else None
    )
    for cell, _ in AXIS_A:
        if cell == C2_A3_CELL:
            continue
        rows = load_shards(results_dir, cell)
        axis_a_metrics[cell] = metrics_for_axis_a(rows)

    # Load axis B.
    axis_b_metrics = {}
    for cell, k in AXIS_B:
        rows = load_shards(results_dir, cell)
        axis_b_metrics[cell] = metrics_for_axis_b(rows, k)

    # Tables.
    print()
    print(f"Axis A --- generation length on C2 prompt (greedy, n={n_target})")
    for cell, max_new in AXIS_A:
        tag = f"  len={max_new}"
        if cell == C2_A3_CELL:
            tag += "  <- plan 5 reference"
        print(f"  {fmt_row_a(cell, axis_a_metrics[cell], tag)}")

    print()
    print(f"Axis B --- self-consistency on C2 prompt "
          f"(T={AXIS_B_TEMPERATURE}, top_p={AXIS_B_TOP_P}, "
          f"len={AXIS_B_LEN}, n={n_target})")
    for cell, k in AXIS_B:
        print(f"  {fmt_row_b(cell, axis_b_metrics[cell], f'  k={k}')}")

    warn_sanity_thresholds(axis_a_metrics, axis_b_metrics)

    label, reason = determine_outcome(axis_a_metrics, axis_b_metrics)
    print()
    print(f"OUTCOME: {label} --- {reason}")

    # Pareto frontier across all cells with rows.
    all_cells = []
    for cell, _ in AXIS_A:
        m = axis_a_metrics.get(cell)
        if m is not None:
            all_cells.append((cell, m["accuracy"], m["mean_flops_per_problem"]))
    for cell, _ in AXIS_B:
        m = axis_b_metrics.get(cell)
        if m is not None:
            all_cells.append((cell, m["accuracy"], m["mean_flops_per_problem"]))
    frontier = pareto_frontier(all_cells)
    print()
    print(f"Pareto-optimal cells (sorted by FLOPs):")
    for name, acc, flops in frontier:
        print(f"  {name:18s}  acc={acc:.3f}  flops/problem={flops:.3e}")

    # Representative cell selection mirrors the outcome label.
    rep_cell = None
    if label == "A":
        rep_cell = C2_A3_CELL
    elif label == "B":
        rep_cell = "C2_A4_len1024"
    elif label == "C":
        rep_cell, _ = best_b_cell(axis_b_metrics)
    elif label == "D":
        rep_cell = "combined-followup-needed"

    out = {
        "config": {
            **man,
            "outcome_thresholds": {"near": OUTCOME_NEAR,
                                     "win": OUTCOME_WIN,
                                     "hurt": OUTCOME_HURT},
            "hash_hit_floor": HASH_HIT_FLOOR,
            "fallback_hit_floor": FALLBACK_HIT_FLOOR,
            "vote_degeneracy_window": [VOTE_DEGEN_LOW, VOTE_DEGEN_HIGH],
            "flops_formula": ("ff = 2*N_active*(prompt+completion); "
                              "attn = N_layers*N_heads*head_dim*"
                              "(prompt*completion + completion^2/2); "
                              "total = ff + 4*attn, times k for axis B"),
            "flops_constants": {"N_active": N_ACTIVE_PARAMS,
                                 "N_layers": N_LAYERS,
                                 "N_heads": N_HEADS,
                                 "head_dim": HEAD_DIM},
            "c2_ref": {"dir": str(c2_ref_dir),
                        "cell": C2_REF_CELL,
                        "shard_files": c2_ref_shards},
        },
        "axis_a": axis_a_metrics,
        "axis_b": axis_b_metrics,
        "pareto_frontier": [{"cell": n, "accuracy": a,
                             "mean_flops_per_problem": f}
                            for n, a, f in frontier],
        "outcome": {"label": label, "reason": reason,
                     "representative_cell": rep_cell},
    }
    (root / "results_plan6.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root / 'results_plan6.json'}")

    write_pareto_csv(root, axis_a_metrics, axis_b_metrics)
    plot_path = make_pareto_plot(root, axis_a_metrics, axis_b_metrics,
                                  frontier, n_target, label, reason)
    if plot_path:
        print(f"wrote {plot_path}")


def write_pareto_csv(root, axis_a_metrics, axis_b_metrics):
    path = root / "path1_c2_pareto.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "axis", "accuracy", "ci_lo", "ci_hi",
                    "mean_flops_per_problem", "mean_wallclock_sec",
                    "hash_hit_rate", "fallback_hit_rate",
                    "vote_degeneracy_rate", "n"])
        for cell, _ in AXIS_A:
            m = axis_a_metrics.get(cell)
            if m is None:
                continue
            w.writerow([cell, "A", f"{m['accuracy']:.6f}",
                         f"{m['ci95'][0]:.6f}", f"{m['ci95'][1]:.6f}",
                         f"{m['mean_flops_per_problem']:.6e}",
                         f"{m['mean_wallclock_sec']:.6f}",
                         f"{m['hash_hit_rate']:.6f}",
                         f"{m['fallback_hit_rate']:.6f}",
                         "", m["n"]])
        for cell, _ in AXIS_B:
            m = axis_b_metrics.get(cell)
            if m is None:
                continue
            w.writerow([cell, "B", f"{m['accuracy']:.6f}",
                         f"{m['ci95'][0]:.6f}", f"{m['ci95'][1]:.6f}",
                         f"{m['mean_flops_per_problem']:.6e}",
                         f"{m['mean_wallclock_sec']:.6f}",
                         f"{m['hash_hit_rate']:.6f}",
                         f"{m['fallback_hit_rate']:.6f}",
                         f"{m['vote_degeneracy_rate']:.6f}",
                         m["n"]])
    print(f"wrote {path}")


def make_pareto_plot(root, axis_a_metrics, axis_b_metrics, frontier, n_total,
                      label, reason):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return None
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    frontier_names = {n for n, _, _ in frontier}

    def scatter(metrics_map, cells, marker, color, label_str):
        xs, ys, names = [], [], []
        for cell, _ in cells:
            m = metrics_map.get(cell)
            if m is None:
                continue
            xs.append(m["mean_flops_per_problem"])
            ys.append(m["accuracy"])
            names.append(cell)
            err = [[m["accuracy"] - m["ci95"][0]],
                   [m["ci95"][1] - m["accuracy"]]]
            ax.errorbar([m["mean_flops_per_problem"]], [m["accuracy"]],
                         yerr=err, fmt="none", ecolor=color,
                         alpha=0.5, capsize=3, linewidth=0.8)
        if xs:
            ax.scatter(xs, ys, marker=marker, color=color, s=70,
                        label=label_str, edgecolor="black", linewidth=0.6,
                        zorder=3)
            for x, y, name in zip(xs, ys, names):
                weight = "bold" if name in frontier_names else "normal"
                ax.annotate(name, (x, y), textcoords="offset points",
                             xytext=(6, 4), fontsize=8, fontweight=weight)

    scatter(axis_a_metrics, AXIS_A, "o", "#1f6fb4", "Axis A (length, greedy)")
    scatter(axis_b_metrics, AXIS_B, "s", "#c44e4e",
             "Axis B (self-consistency)")

    if frontier:
        fx = [f for _, _, f in frontier]
        fy = [a for _, a, _ in frontier]
        ax.plot(fx, fy, linestyle="--", color="#333333", alpha=0.4,
                 linewidth=1.2, zorder=2, label="Pareto frontier")

    a3 = axis_a_metrics.get(C2_A3_CELL)
    if a3 is not None:
        ax.axhline(a3["accuracy"], color="#888888", linestyle=":",
                    alpha=0.5, linewidth=1.0)
        ax.annotate(f"C2_A3 = {a3['accuracy']:.3f}",
                     xy=(ax.get_xlim()[1], a3["accuracy"]),
                     xytext=(-6, 4), textcoords="offset points",
                     ha="right", fontsize=8, color="#555555")

    ax.set_xscale("log")
    ax.set_xlabel("mean FLOPs per problem (log)")
    ax.set_ylabel("GSM8K exact-match accuracy")
    ax.set_title(f"Path 1 plan 6 --- C2 prompt: accuracy vs compute "
                  f"(n={n_total})\nOUTCOME: {label}", fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(0.01, 0.01, reason[:240], fontsize=7,
              color="#555555", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = root / "results_plan6.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.summarize:
        summarize(args.results_dir, args.c2_ref_dir)
        return

    print_env()
    start, end = parse_range(args.problems, args.n)
    check_manifest(args.results_dir, args)
    print(f"cells:     {args.cells}")
    print(f"problems:  [{start}, {end}) of n={args.n}")
    if args.n != DEFAULT_N:
        print(f"PREVIEW MODE (n={args.n} < {DEFAULT_N}): routing to "
              f"{args.results_dir}/")

    # Sanity check #1 --- C2-A3 reference (read-only, no GPU).
    c2_ref_rows, _ = load_c2_ref(args.c2_ref_dir)
    if c2_ref_rows:
        verify_c2_ref(args.c2_ref_dir, DEFAULT_N)
    else:
        print(f"WARNING: C2 reference shard not found under "
              f"{args.c2_ref_dir}/{CELLS_SUBDIR}. Sanity check #1 skipped; "
              f"summarize will report C2_A3_len512 as missing and the "
              f"outcome label will be INCOMPLETE.")

    problems, golds = load_problems(start, end, args.n)

    import torch
    dtype = DTYPE_MAP[args.dtype]
    print(f"\nLoading {MODEL_ID} in {args.dtype} ...")
    tokenizer, model = load_model(MODEL_ID, dtype)
    verify_model_arch(model)

    # Sanity check #2 --- chat-template byte-equivalence with plan 5.
    verify_chat_template(tokenizer, problems, c2_ref_rows)

    try:
        if args.smoke:
            run_smoke(model, tokenizer, args.cells, problems, golds, start)
        for cell in args.cells:
            if cell in dict(AXIS_A):
                run_axis_a_cell(
                    model, tokenizer, cell, dict(AXIS_A)[cell],
                    problems, golds, start, end,
                    args.results_dir, args.no_resume,
                    batch_size=args.batch_size,
                )
            elif cell in dict(AXIS_B):
                run_axis_b_cell(
                    model, tokenizer, cell, dict(AXIS_B)[cell],
                    problems, golds, start, end,
                    args.results_dir, args.no_resume,
                )
            else:
                print(f"WARNING: unknown cell {cell!r}; skipping.")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preview_suffix = f" --n {args.n}" if args.n != DEFAULT_N else ""
    print(f"\nDone. Run `python path1_c2_length_and_sc.py --summarize"
          f"{preview_suffix}` once all cells are populated.")


if __name__ == "__main__":
    main()
