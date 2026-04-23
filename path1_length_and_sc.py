"""Path 1 --- plan 2: generation-length and self-consistency sweeps on IT-CoT.

Two independent one-dimensional probes on `google/gemma-4-E2B-it` with the
same 8-shot Wei-et-al exemplars as plan 1:

    Axis A (length sweep, greedy):
        A1 len=128    A2 len=256    A3 len=512    A4 len=1024

    Axis B (self-consistency, T=0.7, top_p=0.95, len=512):
        B1 k=1        B2 k=3        B3 k=5        B4 k=10

Defaults to n=500 GSM8K test problems. Use --n for a runtime-controlled
sample size (previews auto-route to plan2_preview_n<N>/ so full-run shards
aren't overwritten). Supports sharded multi-GPU runs via --cells /
--problems, and a single-pass --summarize to produce results_plan2.json,
path1_pareto.csv, and the Pareto plot.

Full run:
    python path1_length_and_sc.py

Quick direction-check:
    python path1_length_and_sc.py --n 20 --cells A3 B3

Axis-parallel on two GPUs:
    CUDA_VISIBLE_DEVICES=0 python path1_length_and_sc.py --axis A
    CUDA_VISIBLE_DEVICES=1 python path1_length_and_sc.py --axis B
    python path1_length_and_sc.py --summarize

A3 is plan 1's IT-CoT cell at 512 tokens and serves as the reference for
cross-plan sanity checks when plan 1 shards are present on disk.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

import probes  # noqa: F401  --- triggers HF_HOME redirect before torch loads.
from probes.env import DTYPE_MAP, load_model, print_env

# Prompt + parsing are imported verbatim from plan 1 to eliminate drift.
# Sanity check #1 (exemplar hash) compares against the value captured when
# plan 1 was run, so any change to EXEMPLARS here would correctly fail the
# hash check rather than silently re-baseline.
from path1_cot_gate import (
    EXEMPLARS,
    EXEMPLAR_SET_ID,
    build_prompt,
    extract,
    wilson_ci,
    load_problems,
    append_jsonl,
    load_existing,
    exemplar_hash,
)


MODEL_KEY = "it"
MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_N = 500
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan2"
CELLS_SUBDIR = "cells"

# Cell definitions. Cell names are the canonical ids used in JSONL filenames
# and the results_plan2.json keys.
AXIS_A = [
    ("A1_len128",  128),
    ("A2_len256",  256),
    ("A3_len512",  512),
    ("A4_len1024", 1024),
]
AXIS_B = [
    ("B1_k1",  1),
    ("B2_k3",  3),
    ("B3_k5",  5),
    ("B4_k10", 10),
]
AXIS_B_LEN = 512
AXIS_B_TEMPERATURE = 0.7
AXIS_B_TOP_P = 0.95

# Axis-B sampling seed: per-problem seed so generations are stable across
# restarts. See sanity check #3 in the plan.
AXIS_B_BASE_SEED = 0

# Gemma 4 E2B architectural constants used in the analytical FLOPs formula
# (plan step 4). Verified at runtime against model.config --- mismatches
# produce a warning, not a failure, since we still want the sweep to run.
N_ACTIVE_PARAMS = 2.3e9
HEAD_DIM = 256
N_HEADS = 8
N_LAYERS = 35

# Expected exemplar hash under plan 1's prompt builder. If plan 1 never ran
# on this machine, we fall back to verifying stability across plan 2 runs.
PLAN1_EXEMPLAR_HASH = "a33e6d90c6844317"

# Plan 1 shards live under plan1/cells/ with filenames like
# `it__cot__0000_0500.jsonl`. We intersect idx values with A3_len512 on the
# first PLAN1_SHARED_PROBLEMS indices for the drift-canary check. 100 is the
# plan-specified window.
PLAN1_RESULTS_SUBDIR = "results/path_1_cot_tokens/plan1"
PLAN1_SHARED_PROBLEMS = 100


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
             f"shards are never overwritten. Verdict thresholds still apply, "
             f"but CIs will be wide --- treat preview results as directional.",
    )
    p.add_argument(
        "--axis", choices=["A", "B", "both"], default="both",
        help="Which axis to run. Default both. Use --axis A on one GPU and "
             "--axis B on another for axis-parallel execution.",
    )
    p.add_argument(
        "--cells", nargs="+", choices=all_cells, default=None,
        help="Specific cell names to run (overrides --axis). "
             f"Choices: {', '.join(all_cells)}.",
    )
    p.add_argument(
        "--problems", default=None,
        help="Problem range as START:END (half-open) over the first --n GSM8K "
             "test problems. Default 0:N. Used to shard a single cell across "
             "GPUs (e.g. 0:250 and 250:500 at n=500).",
    )
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument(
        "--results-dir", default=RESULTS_SUBDIR,
        help="Directory for per-cell JSONL shards and final results_plan2.json.",
    )
    p.add_argument(
        "--summarize", action="store_true",
        help="Skip model loading; merge all per-cell shards in --results-dir, "
             "print both sweep tables, write results_plan2.json, "
             "path1_pareto.csv, and the Pareto plot.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Before the full run, verify exemplar hash, print one "
             "(prompt, completion) per axis, and assert per-problem "
             "determinism on the first 3 problems for one cell per axis.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing JSONL rows for the cells this invocation runs, "
             "overwriting them from scratch. Other cells' shards are untouched.",
    )
    p.add_argument(
        "--plan1-dir", default=PLAN1_RESULTS_SUBDIR,
        help=f"Directory holding plan 1's shards (default {PLAN1_RESULTS_SUBDIR}). "
             f"Enables sanity check #2: A3_len512 must match plan 1's "
             f"it__cot__* completions byte-for-byte on the first "
             f"{PLAN1_SHARED_PROBLEMS} problems. Skipped with a warning if the "
             f"directory or its shards are missing.",
    )
    args = p.parse_args()
    if args.n < 1:
        p.error(f"--n must be >= 1, got {args.n}.")
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
        "plan": "path_1_cot_tokens/plan2",
        "model_id": MODEL_ID,
        "exemplar_set": EXEMPLAR_SET_ID,
        "exemplar_hash": exemplar_hash(),
        "dtype": args.dtype,
        "n_total": args.n,
        "axis_a_cells": [{"cell": c, "max_new_tokens": mn} for c, mn in AXIS_A],
        "axis_b_cells": [{"cell": c, "k": k, "max_new_tokens": AXIS_B_LEN,
                          "temperature": AXIS_B_TEMPERATURE,
                          "top_p": AXIS_B_TOP_P} for c, k in AXIS_B],
        "axis_b_base_seed": AXIS_B_BASE_SEED,
    }


def check_manifest(results_dir, args):
    p = Path(results_dir) / "manifest.json"
    desired = manifest(args)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(desired, indent=2))
        return desired
    existing = json.loads(p.read_text())
    diffs = [k for k in desired if existing.get(k) != desired[k]]
    if diffs:
        print(f"ERROR: manifest at {p} is incompatible with this run:")
        for k in diffs:
            print(f"  {k}: existing={existing.get(k)!r}  requested={desired[k]!r}")
        print(f"  Delete {p.parent} or fix the mismatch.")
        sys.exit(1)
    return existing


def verify_exemplar_hash():
    h = exemplar_hash()
    if h != PLAN1_EXEMPLAR_HASH:
        print(f"ERROR: exemplar hash mismatch: got {h!r}, expected "
              f"{PLAN1_EXEMPLAR_HASH!r}. The prompt-builder has drifted from "
              f"plan 1; the 30-point gate result no longer applies as the "
              f"reference baseline. Fix EXEMPLARS in path1_cot_gate.py before "
              f"continuing.")
        sys.exit(1)
    print(f"exemplar_hash = {h}  (matches plan 1)")


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
    import torch
    if pad_id is None:
        return int(gen_ids.numel())
    return int((gen_ids != pad_id).sum().item())


def generate_greedy(model, tokenizer, prompt, max_new):
    import torch
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
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


def sample_k_chains(model, tokenizer, prompt, k, max_new, idx):
    """Sample k chains in a single generate() call (num_return_sequences=k),
    seeded per-problem for resume-stable determinism (sanity check #3)."""
    import torch
    torch.manual_seed(AXIS_B_BASE_SEED + idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(AXIS_B_BASE_SEED + idx)
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc, generation_config=_sampled_gen_cfg(model),
            max_new_tokens=max_new, pad_token_id=pad_id,
            num_return_sequences=k,
            stop_strings=["\nQuestion:"], tokenizer=tokenizer,
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
                    end, results_dir, no_resume):
    path = shard_jsonl(results_dir, cell, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p, g) for i, (p, g) in enumerate(zip(problems, golds))
                 if (start + i) not in done]
    print(f"[{cell}] max_new={max_new}  {start}:{end}  "
          f"{len(done)} done, {len(remaining)} to generate  -> {path.name}")
    gen_secs = 0.0
    for i, prob, gold in remaining:
        prompt = build_prompt(EXEMPLARS["cot"], prob["question"])
        t0 = time.perf_counter()
        completion, n_gen, prompt_len = generate_greedy(
            model, tokenizer, prompt, max_new,
        )
        dt = time.perf_counter() - t0
        gen_secs += dt
        pred, hashed = extract(completion)
        append_jsonl(path, {
            "idx": start + i,
            "gold": gold,
            "pred": pred,
            "correct": int(pred is not None and pred == gold),
            "hash_hit": int(hashed),
            "completion": completion,
            "n_gen_tokens": n_gen,
            "prompt_tokens": prompt_len,
            "gen_secs": dt,
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
    print(f"[{cell}] k={k} len={AXIS_B_LEN} T={AXIS_B_TEMPERATURE}  "
          f"{start}:{end}  {len(done)} done, {len(remaining)} to generate  "
          f"-> {path.name}")
    gen_secs = 0.0
    for i, prob, gold in remaining:
        prompt = build_prompt(EXEMPLARS["cot"], prob["question"])
        t0 = time.perf_counter()
        completions, n_gens, prompt_len = sample_k_chains(
            model, tokenizer, prompt, k, AXIS_B_LEN, start + i,
        )
        dt = time.perf_counter() - t0
        gen_secs += dt
        chains = []
        preds = []
        for comp, n in zip(completions, n_gens):
            pred, hashed = extract(comp)
            chains.append({"completion": comp, "pred": pred,
                           "hash_hit": int(hashed), "n_gen_tokens": n})
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
# smoke: exemplar hash, one sample per axis, per-problem determinism
# -----------------------------------------------------------------------------

def run_smoke(model, tokenizer, cells_to_run, problems, golds, start):
    import torch
    verify_exemplar_hash()
    axis_a_to_run = [c for c in cells_to_run if c.startswith("A")]
    axis_b_to_run = [c for c in cells_to_run if c.startswith("B")]
    ex0 = problems[0]
    prompt = build_prompt(EXEMPLARS["cot"], ex0["question"])
    if axis_a_to_run:
        cell = axis_a_to_run[0]
        max_new = dict(AXIS_A)[cell]
        comp, n_gen, _ = generate_greedy(model, tokenizer, prompt, max_new)
        pred, hashed = extract(comp)
        print(f"  SMOKE {cell} (greedy, len={max_new})  idx={start}  "
              f"pred={pred}  gold={golds[0]}  hash_hit={hashed}  n_gen={n_gen}")
        print(f"    completion[:200] = {comp[:200]!r}")
        a, _, _ = generate_greedy(model, tokenizer, prompt, max_new)
        b, _, _ = generate_greedy(model, tokenizer, prompt, max_new)
        if a != b:
            print(f"ERROR: greedy determinism fail on {cell}.")
            sys.exit(1)
        print(f"  GREEDY DETERMINISM OK ({cell})")
    if axis_b_to_run:
        cell = axis_b_to_run[0]
        k = dict(AXIS_B)[cell]
        comps, n_gens, _ = sample_k_chains(
            model, tokenizer, prompt, k, AXIS_B_LEN, start,
        )
        preds = [extract(c)[0] for c in comps]
        voted, voted_count = majority_vote(preds)
        print(f"  SMOKE {cell} (sampled, k={k})  idx={start}  "
              f"chain_preds={preds}  voted={voted}  gold={golds[0]}")
        a, _, _ = sample_k_chains(model, tokenizer, prompt, k, AXIS_B_LEN, start)
        b, _, _ = sample_k_chains(model, tokenizer, prompt, k, AXIS_B_LEN, start)
        if a != b:
            print(f"ERROR: sampling determinism fail on {cell} --- "
                  f"per-problem seeding is not byte-for-byte reproducible.")
            sys.exit(1)
        print(f"  SAMPLING DETERMINISM OK ({cell})")
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# FLOPs (analytical, per plan step 4)
# -----------------------------------------------------------------------------

def flops_per_problem(prompt_len, completion_len, k=1):
    """Analytical FLOPs for decoder-only inference (plan step 4):
    2 * N_active * (prompt + completion) for ff/projection, plus attention
    over growing context (quadratic). k chains cost k* a single chain."""
    ff = 2 * N_ACTIVE_PARAMS * (prompt_len + completion_len)
    attn = (N_LAYERS * N_HEADS * HEAD_DIM
            * (prompt_len * completion_len + completion_len ** 2 / 2))
    return k * (ff + 4 * attn)


def verify_model_arch(model):
    cfg = model.config
    mismatches = []
    if getattr(cfg, "num_hidden_layers", N_LAYERS) != N_LAYERS:
        mismatches.append(f"num_hidden_layers={cfg.num_hidden_layers} != {N_LAYERS}")
    if getattr(cfg, "num_attention_heads", N_HEADS) != N_HEADS:
        mismatches.append(f"num_attention_heads={cfg.num_attention_heads} != {N_HEADS}")
    if getattr(cfg, "head_dim", HEAD_DIM) != HEAD_DIM:
        mismatches.append(f"head_dim={cfg.head_dim} != {HEAD_DIM}")
    if mismatches:
        print("WARNING: model.config disagrees with FLOPs constants:")
        for m in mismatches:
            print(f"  {m}")
        print("  FLOPs estimates will be off. Update N_LAYERS / N_HEADS / "
              "HEAD_DIM in path1_length_and_sc.py.")


# -----------------------------------------------------------------------------
# plan-1 drift canary (sanity check #2)
# -----------------------------------------------------------------------------

def _load_plan1_it_cot(plan1_dir, upper):
    """Merge plan 1's IT-CoT shards into {idx: completion} for idx < upper.
    Returns (mapping, shard_paths). Empty mapping means the dir or shards
    are absent --- caller treats that as 'skip with warning'."""
    cells = Path(plan1_dir) / CELLS_SUBDIR
    if not cells.is_dir():
        return {}, []
    shards = sorted(cells.glob("it__cot__*.jsonl"))
    mapping = {}
    for s in shards:
        for line in s.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            idx = row.get("idx")
            if idx is None or idx >= upper or idx in mapping:
                continue
            mapping[idx] = row.get("completion")
    return mapping, shards


def verify_a3_reproduces_plan1(plan1_dir, results_dir,
                                shared=PLAN1_SHARED_PROBLEMS):
    """Sanity check #2: A3_len512 generations on idx [0, shared) must match
    plan 1's it:cot cell byte-for-byte. Drift here means seeds/dtype/
    transformers version shifted between plans and cross-plan comparisons
    are suspect. Returns a dict for sanity_checks in results_plan2.json."""
    plan1_map, plan1_shards = _load_plan1_it_cot(plan1_dir, shared)
    if not plan1_map:
        return {
            "skipped": True,
            "reason": (f"plan 1 IT-CoT shards not found under "
                       f"{plan1_dir}/{CELLS_SUBDIR}/it__cot__*.jsonl. "
                       f"Add them and re-run --summarize to enforce the "
                       f"drift canary."),
            "matches": 0,
            "total": 0,
            "mismatches": [],
        }
    a3_rows = load_shards(results_dir, "A3_len512", shared)
    a3_map = {r["idx"]: r.get("completion") for r in a3_rows
              if r["idx"] < shared}
    overlap = sorted(set(plan1_map) & set(a3_map))
    matches, mismatches = 0, []
    for idx in overlap:
        if plan1_map[idx] == a3_map[idx]:
            matches += 1
        else:
            mismatches.append(idx)
    result = {
        "skipped": False,
        "plan1_dir": str(plan1_dir),
        "plan1_shards": [p.name for p in plan1_shards],
        "window": shared,
        "plan1_rows_in_window": len(plan1_map),
        "plan2_rows_in_window": len(a3_map),
        "compared": len(overlap),
        "matches": matches,
        "total": len(overlap),
        "mismatches": mismatches,
    }
    return result


# -----------------------------------------------------------------------------
# summarize: merge shards, compute metrics + verdicts, write outputs
# -----------------------------------------------------------------------------

def load_shards(results_dir, cell, n_target):
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


def metrics_for_cell(rows, k=1):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(r["correct"] for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    if k == 1 and rows and "chains" not in rows[0]:
        hash_hit = sum(r.get("hash_hit", 0) for r in rows) / n
        mean_gen = sum(r.get("n_gen_tokens", 0) for r in rows) / n
        prompt_len = rows[0].get("prompt_tokens", 0)
        flops = sum(flops_per_problem(r.get("prompt_tokens", prompt_len),
                                      r.get("n_gen_tokens", 0))
                    for r in rows) / n
        mean_wall = sum(r.get("gen_secs", 0.0) for r in rows) / n
        return {
            "n": n, "correct": correct, "accuracy": acc,
            "ci95": [lo, hi], "hash_hit_rate": hash_hit,
            "mean_gen_tokens": mean_gen,
            "mean_flops_per_problem": flops,
            "mean_wallclock_sec": mean_wall,
        }
    # Axis-B row layout.
    vote_degen = sum(r.get("vote_degenerate", 0) for r in rows) / n
    chain_hash = []
    chain_gen = []
    for r in rows:
        for ch in r.get("chains", []):
            chain_hash.append(ch.get("hash_hit", 0))
            chain_gen.append(ch.get("n_gen_tokens", 0))
    hash_hit = sum(chain_hash) / len(chain_hash) if chain_hash else 0.0
    mean_gen = sum(chain_gen) / len(chain_gen) if chain_gen else 0.0
    prompt_len = rows[0].get("prompt_tokens", 0)
    flops = sum(
        flops_per_problem(r.get("prompt_tokens", prompt_len),
                          sum(ch.get("n_gen_tokens", 0)
                              for ch in r.get("chains", [])) / max(1, k),
                          k=k)
        for r in rows
    ) / n
    mean_wall = sum(r.get("gen_secs", 0.0) for r in rows) / n
    return {
        "n": n, "correct": correct, "accuracy": acc,
        "ci95": [lo, hi], "hash_hit_rate": hash_hit,
        "mean_gen_tokens": mean_gen,
        "vote_degeneracy_rate": vote_degen,
        "mean_flops_per_problem": flops,
        "mean_wallclock_sec": mean_wall,
    }


def axis_a_verdict(m):
    need = {"A1_len128", "A2_len256", "A3_len512", "A4_len1024"}
    if any(m.get(c) is None for c in need):
        return "INCOMPLETE", f"missing cells: {sorted(need - set(k for k,v in m.items() if v))}"
    a1, a2, a3, a4 = m["A1_len128"]["accuracy"], m["A2_len256"]["accuracy"], m["A3_len512"]["accuracy"], m["A4_len1024"]["accuracy"]
    a3_lo, a3_hi = m["A3_len512"]["ci95"]
    a4_lo, a4_hi = m["A4_len1024"]["ci95"]
    if a4 >= a3 + 0.05 and a4_lo > a3_hi:
        return "MONOTONIC-LIFT", (f"A4 ({a4:.3f}) >= A3 ({a3:.3f}) + 0.05 with "
                                  f"non-overlapping CIs. Length matters; "
                                  f"consider len=2048 follow-up. Default len=1024.")
    if abs(a2 - a3) < 0.02:
        return "SATURATED-EARLIER", (f"A2 ({a2:.3f}) and A3 ({a3:.3f}) within 2 "
                                      f"points. Use len=256 for head-to-head.")
    if abs(a4 - a3) < 0.03 and (a3 - a1) > 0.05 and (a3 - a2) > 0.05:
        return "KNEE-AT-512", (f"A3 ({a3:.3f}) and A4 ({a4:.3f}) within 3 points, "
                                f"A1/A2 below A3 by >5. Use len=512 for head-to-head.")
    return "AMBIGUOUS", (f"A1={a1:.3f} A2={a2:.3f} A3={a3:.3f} A4={a4:.3f}; "
                         f"no decision-rule pattern matched.")


def axis_b_verdict(m, a3_acc):
    need = {"B1_k1", "B2_k3", "B3_k5", "B4_k10"}
    if any(m.get(c) is None for c in need):
        return "INCOMPLETE", f"missing cells: {sorted(need - set(k for k,v in m.items() if v))}"
    b3, b4 = m["B3_k5"]["accuracy"], m["B4_k10"]["accuracy"]
    b3_lo, b3_hi = m["B3_k5"]["ci95"]
    b4_lo, b4_hi = m["B4_k10"]["ci95"]
    if a3_acc is None:
        return "NO-BASELINE", "A3 missing; cannot compare SC to greedy 512."
    # Use the better of B3/B4 for the win test.
    best_b_acc = max(b3, b4)
    best_b_lo = b3_lo if best_b_acc == b3 else b4_lo
    a3_lo, a3_hi = None, None
    # Re-derive A3 CI from metrics dict if present --- required for CI overlap test.
    a3_cell = m.get("A3_len512")
    if a3_cell is not None:
        a3_lo, a3_hi = a3_cell["ci95"]
    sc_wins_ci = (a3_hi is not None and best_b_lo > a3_hi)
    if best_b_acc >= a3_acc + 0.08 and sc_wins_ci:
        return "SC-WINS", (f"max(B3,B4)={best_b_acc:.3f} beats A3 ({a3_acc:.3f}) "
                            f"by >=0.08 with non-overlapping CIs. Promote best B "
                            f"cell to head-to-head.")
    if abs(b3 - a3_acc) < 0.03:
        return "SC-FLAT", (f"B3 ({b3:.3f}) within 3 points of A3 ({a3_acc:.3f}). "
                            f"Self-consistency ineffective on E2B-it; "
                            f"follow-up permitted: rerun B3 at T=1.0.")
    if b3 < a3_acc - 0.03:
        return "SC-HURTS", (f"B3 ({b3:.3f}) < A3 ({a3_acc:.3f}) by >=3 points. "
                             f"Likely temperature-too-high or sampling bug. Debug.")
    return "AMBIGUOUS", (f"B3={b3:.3f} B4={b4:.3f} vs A3={a3_acc:.3f}; "
                          f"no decision-rule pattern matched.")


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


def summarize(results_dir, n_target, plan1_dir=PLAN1_RESULTS_SUBDIR):
    root = Path(results_dir)
    mpath = root / "manifest.json"
    if not mpath.exists():
        print(f"ERROR: no manifest at {mpath}. Run at least one cell first.")
        sys.exit(1)
    man = json.loads(mpath.read_text())

    axis_a_metrics = {}
    for cell, max_new in AXIS_A:
        rows = load_shards(results_dir, cell, n_target)
        axis_a_metrics[cell] = metrics_for_cell(rows, k=1)

    axis_b_metrics = {}
    for cell, k in AXIS_B:
        rows = load_shards(results_dir, cell, n_target)
        axis_b_metrics[cell] = metrics_for_cell(rows, k=k)

    def fmt_row(name, m, extra=""):
        if m is None:
            return f"{name:12s}  (no rows)"
        status = "" if m["n"] == n_target else f"  [PARTIAL {m['n']}/{n_target}]"
        return (f"{name:12s}  acc={m['accuracy']:.3f}  "
                f"ci=({m['ci95'][0]:.3f},{m['ci95'][1]:.3f})  "
                f"hash={m['hash_hit_rate']:.2f}  "
                f"mean_tokens={m['mean_gen_tokens']:.1f}{extra}{status}")

    print()
    print(f"Axis A --- generation length (greedy, n={n_target})")
    for cell, max_new in AXIS_A:
        tag = f"  len={max_new}"
        if cell == "A3_len512":
            tag += "  <- plan 1 reference"
        print(f"  {fmt_row(cell, axis_a_metrics[cell], tag)}")

    print()
    print(f"Axis B --- self-consistency "
          f"(T={AXIS_B_TEMPERATURE}, len={AXIS_B_LEN}, n={n_target})")
    for cell, k in AXIS_B:
        extra = f"  k={k}"
        m = axis_b_metrics[cell]
        if m is not None:
            extra += f"  vote_deg={m['vote_degeneracy_rate']:.2f}"
        print(f"  {fmt_row(cell, m, extra)}")

    a_label, a_reason = axis_a_verdict(axis_a_metrics)
    a3 = axis_a_metrics.get("A3_len512")
    b_label, b_reason = axis_b_verdict(
        axis_b_metrics, a3["accuracy"] if a3 else None,
    )
    print()
    print(f"AXIS A VERDICT: {a_label} --- {a_reason}")
    print(f"AXIS B VERDICT: {b_label} --- {b_reason}")

    # Pareto frontier across all cells that have results.
    all_cells = []
    for cell, _ in AXIS_A:
        m = axis_a_metrics[cell]
        if m is not None:
            all_cells.append((cell, m["accuracy"], m["mean_flops_per_problem"]))
    for cell, _ in AXIS_B:
        m = axis_b_metrics[cell]
        if m is not None:
            all_cells.append((cell, m["accuracy"], m["mean_flops_per_problem"]))
    frontier = pareto_frontier(all_cells)
    print()
    print(f"Pareto-optimal cells (sorted by FLOPs):")
    for name, acc, flops in frontier:
        print(f"  {name:12s}  acc={acc:.3f}  flops/problem={flops:.3e}")

    # Sanity checks section of results_plan2.json (plan step 5 schema).
    a3_vs_plan1 = verify_a3_reproduces_plan1(plan1_dir, results_dir)
    vote_deg = {c: axis_b_metrics[c]["vote_degeneracy_rate"]
                for c, _ in AXIS_B if axis_b_metrics.get(c) is not None}
    sanity_checks = {
        "a3_reproduces_plan1": a3_vs_plan1,
        "b_majority_vote_degenerate_check": vote_deg,
    }
    print()
    if a3_vs_plan1["skipped"]:
        print(f"SANITY #2 A3 vs plan 1: SKIPPED --- {a3_vs_plan1['reason']}")
    else:
        tot = a3_vs_plan1["total"]
        matches = a3_vs_plan1["matches"]
        mism = a3_vs_plan1["mismatches"]
        status = "OK" if not mism and tot > 0 else "FAIL"
        print(f"SANITY #2 A3 vs plan 1: {status} --- "
              f"{matches}/{tot} completions match "
              f"(plan1 rows in window: {a3_vs_plan1['plan1_rows_in_window']}, "
              f"plan2 rows in window: {a3_vs_plan1['plan2_rows_in_window']})")
        if mism:
            print(f"  mismatched idxs: {mism[:10]}{' ...' if len(mism) > 10 else ''}")
            print(f"  harness has drifted; cross-plan comparisons are suspect.")

    out = {
        "config": {**man,
                   "n_target": n_target,
                   "plan1_dir": str(plan1_dir),
                   "flops_formula": ("ff = 2*N_active*(prompt+completion); "
                                     "attn = N_layers*N_heads*head_dim*"
                                     "(prompt*completion + completion^2/2); "
                                     "total = ff + 4*attn, times k for axis B"),
                   "flops_constants": {"N_active": N_ACTIVE_PARAMS,
                                       "N_layers": N_LAYERS,
                                       "N_heads": N_HEADS,
                                       "head_dim": HEAD_DIM}},
        "axis_a": axis_a_metrics,
        "axis_b": axis_b_metrics,
        "pareto_frontier": [{"cell": n, "accuracy": a,
                             "mean_flops_per_problem": f}
                            for n, a, f in frontier],
        "verdicts": {"axis_a": {"label": a_label, "reason": a_reason},
                     "axis_b": {"label": b_label, "reason": b_reason}},
        "sanity_checks": sanity_checks,
    }
    (root / "results_plan2.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root / 'results_plan2.json'}")

    write_pareto_csv(root, axis_a_metrics, axis_b_metrics)
    plot_path = make_pareto_plot(root, axis_a_metrics, axis_b_metrics,
                                 frontier, n_target, a_label, b_label)
    if plot_path:
        print(f"wrote {plot_path}")


def write_pareto_csv(root, axis_a_metrics, axis_b_metrics):
    path = root / "path1_pareto.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "axis", "accuracy", "ci_lo", "ci_hi",
                    "mean_flops_per_problem", "mean_wallclock_sec"])
        for cell, _ in AXIS_A:
            m = axis_a_metrics.get(cell)
            if m is None:
                continue
            w.writerow([cell, "A", f"{m['accuracy']:.6f}",
                        f"{m['ci95'][0]:.6f}", f"{m['ci95'][1]:.6f}",
                        f"{m['mean_flops_per_problem']:.6e}",
                        f"{m['mean_wallclock_sec']:.6f}"])
        for cell, _ in AXIS_B:
            m = axis_b_metrics.get(cell)
            if m is None:
                continue
            w.writerow([cell, "B", f"{m['accuracy']:.6f}",
                        f"{m['ci95'][0]:.6f}", f"{m['ci95'][1]:.6f}",
                        f"{m['mean_flops_per_problem']:.6e}",
                        f"{m['mean_wallclock_sec']:.6f}"])
    print(f"wrote {path}")


def make_pareto_plot(root, axis_a_metrics, axis_b_metrics, frontier, n_total,
                     a_label, b_label):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    frontier_names = {n for n, _, _ in frontier}

    def scatter(metrics_map, cells, marker, color, label):
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
            ax.scatter(xs, ys, marker=marker, color=color, s=70, label=label,
                       edgecolor="black", linewidth=0.6, zorder=3)
            for x, y, name in zip(xs, ys, names):
                weight = "bold" if name in frontier_names else "normal"
                ax.annotate(name, (x, y), textcoords="offset points",
                            xytext=(6, 4), fontsize=8, fontweight=weight)

    scatter(axis_a_metrics, AXIS_A, "o", "#1f6fb4", "Axis A (length, greedy)")
    scatter(axis_b_metrics, AXIS_B, "s", "#c44e4e", "Axis B (self-consistency)")

    if frontier:
        fx = [f for _, _, f in frontier]
        fy = [a for _, a, _ in frontier]
        ax.plot(fx, fy, linestyle="--", color="#333333", alpha=0.4,
                linewidth=1.2, zorder=2, label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("mean FLOPs per problem (log)")
    ax.set_ylabel("GSM8K exact-match accuracy")
    ax.set_title(f"Path 1 plan 2 --- accuracy vs compute (n={n_total})\n"
                 f"Axis A: {a_label}   |   Axis B: {b_label}", fontsize=10)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    path = root / "results_plan2.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.summarize:
        summarize(args.results_dir, args.n, plan1_dir=args.plan1_dir)
        return

    print_env()
    verify_exemplar_hash()
    start, end = parse_range(args.problems, args.n)
    check_manifest(args.results_dir, args)
    print(f"cells:     {args.cells}")
    print(f"problems:  [{start}, {end}) of n={args.n}")
    if args.n != DEFAULT_N:
        print(f"PREVIEW MODE (n={args.n} < {DEFAULT_N}): routing to "
              f"{args.results_dir}/")

    problems, golds = load_problems(start, end, args.n)

    import torch
    dtype = DTYPE_MAP[args.dtype]
    print(f"\nLoading {MODEL_ID} in {args.dtype} ...")
    tokenizer, model = load_model(MODEL_ID, dtype)
    verify_model_arch(model)

    try:
        if args.smoke:
            run_smoke(model, tokenizer, args.cells, problems, golds, start)
        for cell in args.cells:
            if cell in dict(AXIS_A):
                run_axis_a_cell(
                    model, tokenizer, cell, dict(AXIS_A)[cell],
                    problems, golds, start, end,
                    args.results_dir, args.no_resume,
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
    print(f"\nDone. Run `python path1_length_and_sc.py --summarize"
          f"{preview_suffix}` once all cells are populated.")


if __name__ == "__main__":
    main()
