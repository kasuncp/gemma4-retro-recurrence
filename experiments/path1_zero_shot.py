"""Path 1 --- plan 5: 0-shot vs 8-shot CoT --- is the 30% ceiling artificial?

Plans 1/2 fixed the prompt format at 8-shot Wei-et-al exemplars and saw a
30% plateau on GSM8K. Plan 5 probes whether that plateau is a property of
the model or of the prompt format. Two new IT-CoT cells, both at greedy
decode + 512-token cap, both scored against plan 2's deterministic first
500 GSM8K test problems:

    C1_zeroshot_simple  : chat-template, user="{q}\\n\\nLet's think step by step."
    C2_zeroshot_plain   : chat-template, user="{q}"

Both cells use ``tokenizer.apply_chat_template`` with a single user turn.
Without the chat template the IT model would underperform for reasons
unrelated to the plan question --- see sanity check #1.

The reference cell A3_len512 from plan 2 is re-used by reading its JSONL
shard directly --- no re-run. All three cells operate on the same 500
problems, model commit, and greedy decode settings; the only variable is
the prompt format.

Default --- one GPU, runs both cells sequentially:
    python path1_zero_shot.py

Quick preview (auto-routes to plan5_preview_n<N>/ to keep full-run shards
intact):
    python path1_zero_shot.py --n 20

Cell-parallel on two GPUs:
    CUDA_VISIBLE_DEVICES=0 python path1_zero_shot.py --cells C1_zeroshot_simple
    CUDA_VISIBLE_DEVICES=1 python path1_zero_shot.py --cells C2_zeroshot_plain
    python path1_zero_shot.py --summarize

Pre-registered outcome labels (A/B/C/D/E) are computed during --summarize
and stamped on the figure. See plan5.md for the full interpretation.
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

# Borrow shard IO + Wilson CI + GSM8K loader from plan 1 so the answer
# parser, problem indexing, and result-row schema are bit-identical to A3.
from path1_cot_gate import (
    GOLD_RE,
    FALLBACK_RE,
    append_jsonl,
    load_existing,
    load_problems,
    wilson_ci,
)


MODEL_KEY = "it"
MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_N = 500
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan5"
CELLS_SUBDIR = "cells"

# Cell ids --- canonical; used in JSONL filenames and results_plan5.json.
C1_CELL = "C1_zeroshot_simple"
C2_CELL = "C2_zeroshot_plain"
ALL_CELLS = [C1_CELL, C2_CELL]

# Generation params shared by both cells. Greedy + 512 tokens matches A3.
MAX_NEW = 512

# Reference cell on disk: plan 2's A3_len512 shard. The path is canonical
# and pinned --- if plan 2 ever moves, update this here. Summarize and the
# paired comparisons fall back to "skip with warning" rather than failing
# when this file is absent (e.g., on a fresh runpod that hasn't synced
# plan 2's shard down yet).
A3_REF_DIR = "results/path_1_cot_tokens/plan2"
A3_CELL = "A3_len512"

# Gemma 4 E2B architectural constants (same as plan 2). Verified at runtime.
N_ACTIVE_PARAMS = 2.3e9
HEAD_DIM = 256
N_HEADS = 8
N_LAYERS = 35

# Sanity thresholds for sanity check #2 (hash_hit_rate vs fallback). 0-shot
# completions often answer in prose ("The answer is 72.") rather than
# "#### 72", so a low hash_hit_rate is expected. As long as *something*
# extractable lands in >= 80% of completions, the accuracy is trustworthy.
# Following plan5.md's deliverables format, fallback_hit_rate is cumulative
# ("any integer extracted, primary or fallback") --- i.e. 1 - no_extract.
HASH_HIT_FLOOR = 0.30
FALLBACK_HIT_FLOOR = 0.80

# Pre-registered outcome thresholds (plan5.md "Pre-registered interpretation").
# All values are absolute percentage points on GSM8K accuracy.
OUTCOME_NEAR = 0.03   # A: |Cx - A3| <= 3pts (in points, expressed as fraction)
OUTCOME_WIN = 0.05    # B/C: >= 5pts gap
OUTCOME_PREFIX = 0.05 # D: C1 - C2 >= 5pts


# -----------------------------------------------------------------------------
# args
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Number of GSM8K test problems to evaluate (default {DEFAULT_N}). "
             f"When set below {DEFAULT_N} and --results-dir is not overridden, "
             f"results auto-route to {RESULTS_SUBDIR}_preview_n<N>/ so full-run "
             f"shards are never overwritten.",
    )
    p.add_argument(
        "--cells", nargs="+", choices=ALL_CELLS, default=ALL_CELLS,
        help=f"Cells to run. Default both: {', '.join(ALL_CELLS)}. "
             f"Drop {C2_CELL} to run only the cell that decides the plan's "
             f"main question (C1).",
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
        help="Greedy batch size. Default 1 (single-sequence, byte-deterministic). "
             "Larger values amortize GPU idle time; on a 3090 batch=8 typically "
             "lifts throughput 3-5x.",
    )
    p.add_argument(
        "--results-dir", default=RESULTS_SUBDIR,
        help="Directory for per-cell JSONL shards and final results_plan5.json.",
    )
    p.add_argument(
        "--a3-ref-dir", default=A3_REF_DIR,
        help=f"Directory holding plan 2's results (default {A3_REF_DIR}). "
             f"Summarize reads {A3_CELL}__*.jsonl from <a3-ref-dir>/{CELLS_SUBDIR}/ "
             f"for the paired McNemar comparison. Skipped with a warning if "
             f"the directory or its shards are missing.",
    )
    p.add_argument(
        "--summarize", action="store_true",
        help="Skip model loading; merge all per-cell shards in --results-dir, "
             "print the table, write results_plan5.json + results_plan5.png + "
             "path1_plan5.csv, and emit the outcome label.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Before the full run, print the chat-template'd prompt for "
             "problem 0 in both cells, generate a sample completion, and "
             "verify byte-for-byte determinism on the first 3 problems for "
             "one cell.",
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
# prompt + extraction
# -----------------------------------------------------------------------------

def cell_user_message(cell, question):
    """Return the user-turn content for ``cell`` on a raw GSM8K question."""
    if cell == C1_CELL:
        return f"{question}\n\nLet's think step by step."
    if cell == C2_CELL:
        return question
    raise ValueError(f"unknown cell: {cell!r}")


def build_chat_prompt(tokenizer, cell, question):
    """Build the IT-format prompt via tokenizer.apply_chat_template. Without
    the chat template, the user/assistant turn markers are missing and the
    IT model underperforms for reasons unrelated to the plan question
    (sanity check #1)."""
    messages = [{"role": "user", "content": cell_user_message(cell, question)}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def extract_zeroshot(text):
    """Extract a numeric answer + record which extractor fired.

    Returns (pred, hashed, fallback_used):
      - pred           : int or None
      - hashed         : bool, True iff the primary `#### N` regex matched
      - fallback_used  : bool, True iff hashed=False and the last-integer
                          fallback fired

    Same logic as path1_cot_gate.extract, but exposes the fallback flag so
    summarize() can compute fallback_hit_rate independently from
    hash_hit_rate (sanity check #2 in plan5.md).
    """
    m = GOLD_RE.search(text)
    if m:
        return int(m.group(1)), True, False
    nums = FALLBACK_RE.findall(text)
    if nums:
        return int(nums[-1]), False, True
    return None, False, False


# -----------------------------------------------------------------------------
# manifest
# -----------------------------------------------------------------------------

def manifest(args):
    return {
        "plan": "path_1_cot_tokens/plan5",
        "model_id": MODEL_ID,
        "model_key": MODEL_KEY,
        "dtype": args.dtype,
        "n_total": args.n,
        "cells": [
            {"cell": C1_CELL, "user_template": "{question}\n\nLet's think step by step.",
             "decode": "greedy", "max_new_tokens": MAX_NEW, "k": 1},
            {"cell": C2_CELL, "user_template": "{question}",
             "decode": "greedy", "max_new_tokens": MAX_NEW, "k": 1},
        ],
        "uses_chat_template": True,
        "answer_primary_regex": GOLD_RE.pattern,
        "answer_fallback_regex": FALLBACK_RE.pattern,
        "a3_ref_dir": str(args.a3_ref_dir),
        "a3_ref_cell": A3_CELL,
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
    stable = ("plan", "model_id", "model_key", "dtype", "cells",
              "uses_chat_template", "answer_primary_regex",
              "answer_fallback_regex")
    diffs = [k for k in stable if existing.get(k) != desired.get(k)]
    if diffs:
        print(f"ERROR: manifest at {p} is incompatible with this run:")
        for k in diffs:
            print(f"  {k}: existing={existing.get(k)!r}  requested={desired[k]!r}")
        print(f"  Delete {p.parent} or fix the mismatch.")
        sys.exit(1)
    existing["n_total"] = max(existing.get("n_total", 0), args.n)
    existing["a3_ref_dir"] = str(args.a3_ref_dir)
    p.write_text(json.dumps(existing, indent=2))
    return existing


# -----------------------------------------------------------------------------
# generation: greedy single + batched
# -----------------------------------------------------------------------------

def _greedy_gen_cfg(model):
    cfg = deepcopy(model.generation_config)
    cfg.do_sample = False
    cfg.top_p = None
    cfg.top_k = None
    cfg.temperature = None
    return cfg


def _count_gen_tokens(gen_ids, pad_id):
    if pad_id is None:
        return int(gen_ids.numel())
    return int((gen_ids != pad_id).sum().item())


def generate_greedy(model, tokenizer, prompt, max_new):
    import torch
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device,
    )
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
    """Batched greedy. Left-pads, attention-masks pads, returns
    (completion, n_gen, input_len) per prompt. Not bit-identical to the
    single-sequence path: batched matmul reorders FP ops."""
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


# -----------------------------------------------------------------------------
# shard paths + per-cell runner
# -----------------------------------------------------------------------------

def shard_jsonl(results_dir, cell, start, end):
    return Path(results_dir) / CELLS_SUBDIR / f"{cell}__{start:04d}_{end:04d}.jsonl"


def run_cell(model, tokenizer, cell, problems, golds, start, end,
             results_dir, no_resume, batch_size=1):
    path = shard_jsonl(results_dir, cell, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p, g) for i, (p, g) in enumerate(zip(problems, golds))
                 if (start + i) not in done]
    print(f"[{cell}] greedy len={MAX_NEW}  {start}:{end}  "
          f"{len(done)} done, {len(remaining)} to generate  -> {path.name}  "
          f"batch_size={batch_size}")
    gen_secs = 0.0
    if batch_size <= 1:
        for i, prob, gold in remaining:
            prompt = build_chat_prompt(tokenizer, cell, prob["question"])
            t0 = time.perf_counter()
            completion, n_gen, prompt_len = generate_greedy(
                model, tokenizer, prompt, MAX_NEW,
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
            prompts = [build_chat_prompt(tokenizer, cell, prob["question"])
                       for _, prob, _ in batch]
            t0 = time.perf_counter()
            per_row = generate_greedy_batch(
                model, tokenizer, prompts, MAX_NEW,
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


# -----------------------------------------------------------------------------
# smoke: chat-template print + sample completions + determinism
# -----------------------------------------------------------------------------

def run_smoke(model, tokenizer, cells_to_run, problems, golds, start):
    """Sanity checks #1 (chat-template verification), #3 (sample completion
    inspection), and #4 (greedy determinism), all in one pass."""
    import torch
    if not problems:
        print("  (no problems in range; smoke skipped)")
        return
    ex0 = problems[0]
    for cell in cells_to_run:
        prompt = build_chat_prompt(tokenizer, cell, ex0["question"])
        # Sanity #1: must contain Gemma's user/model turn markers.
        ok_user = "<start_of_turn>user" in prompt
        ok_model = "<start_of_turn>model" in prompt
        marker = "OK" if (ok_user and ok_model) else "MISSING"
        print(f"  SMOKE {cell} chat-template: {marker} "
              f"(user_marker={ok_user}, model_marker={ok_model})")
        # First 240 chars of the rendered prompt (printable, single line).
        head = prompt[:240].replace("\n", "\\n")
        tail = prompt[-160:].replace("\n", "\\n")
        print(f"    prompt head: {head!r}")
        print(f"    prompt tail: {tail!r}")
        if not (ok_user and ok_model):
            print(f"ERROR: chat template missing turn markers --- cell would "
                  f"effectively be raw-text 0-shot. Aborting before wasting "
                  f"compute.")
            sys.exit(1)
        # Sanity #3: print one completion + extracted answer.
        comp, n_gen, _ = generate_greedy(model, tokenizer, prompt, MAX_NEW)
        pred, hashed, fallback = extract_zeroshot(comp)
        print(f"    completion[:240] = {comp[:240]!r}")
        print(f"    pred={pred}  gold={golds[0]}  hash_hit={hashed}  "
              f"fallback_hit={fallback}  n_gen={n_gen}")

    # Sanity #4: byte-for-byte determinism on the first cell + first 3
    # problems. Restart-equivalence under greedy decoding.
    cell = cells_to_run[0]
    det_ok = True
    for i in range(min(3, len(problems))):
        prompt = build_chat_prompt(tokenizer, cell, problems[i]["question"])
        a, _, _ = generate_greedy(model, tokenizer, prompt, MAX_NEW)
        b, _, _ = generate_greedy(model, tokenizer, prompt, MAX_NEW)
        if a != b:
            det_ok = False
            print(f"  DETERMINISM FAIL on {cell} idx={start+i}")
    if not det_ok:
        print("ERROR: greedy decoding is not byte-for-byte deterministic. "
              "Do not trust subsequent numbers.")
        sys.exit(1)
    print(f"  GREEDY DETERMINISM OK ({cell})")
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# FLOPs (analytical, same constants as plan 2 for cross-plan comparability)
# -----------------------------------------------------------------------------

def flops_per_problem(prompt_len, completion_len, k=1):
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


# -----------------------------------------------------------------------------
# A3 reference loader (plan 2's shard --- read-only)
# -----------------------------------------------------------------------------

def load_a3_reference(a3_ref_dir):
    """Read plan 2's A3_len512 shard as a list of rows. Returns [] if the
    directory or shard is absent --- summarize() treats that as 'skip the
    paired comparison with a warning' so plan 5 can still report C1/C2
    standalone numbers on a fresh laptop or pod that hasn't synced plan 2's
    results yet."""
    cells = Path(a3_ref_dir) / CELLS_SUBDIR
    if not cells.is_dir():
        return [], None
    shards = sorted(cells.glob(f"{A3_CELL}__*.jsonl"))
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


# -----------------------------------------------------------------------------
# summarize: merge shards, compute metrics + paired McNemar + outcome
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


def metrics_for_cell(rows):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(r["correct"] for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    hash_hit = sum(r.get("hash_hit", 0) for r in rows) / n
    no_extract = sum(int(r.get("pred") is None) for r in rows) / n
    # fallback_hit_rate is cumulative per plan5.md's deliverables example
    # ("hash 0.81 fallback 0.99"): the rate at which *any* integer was
    # extracted, primary or fallback. Equivalent to 1 - no_extract_rate;
    # works whether the per-row schema includes fallback_hit (plan-5 cells)
    # or only hash_hit (plan-2 A3 reference rows).
    fallback_hit = 1.0 - no_extract
    # Strict-fallback (fallback fired BECAUSE primary missed) is a useful
    # secondary diagnostic. Compute it from per-row flags when present;
    # otherwise back it out of (pred is not None AND hash_hit == 0).
    if rows and "fallback_hit" in rows[0]:
        strict_fallback = sum(r.get("fallback_hit", 0) for r in rows) / n
    else:
        strict_fallback = sum(
            int(r.get("hash_hit", 0) == 0 and r.get("pred") is not None)
            for r in rows
        ) / n
    mean_gen = sum(r.get("n_gen_tokens", 0) for r in rows) / n
    prompt_lens = [r.get("prompt_tokens", 0) for r in rows]
    mean_prompt = sum(prompt_lens) / max(1, len(prompt_lens))
    flops = sum(flops_per_problem(r.get("prompt_tokens", 0),
                                  r.get("n_gen_tokens", 0))
                for r in rows) / n
    mean_wall = sum(r.get("gen_secs", 0.0) for r in rows) / n
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


def _binom_two_sided_p(b, n, p_null=0.5):
    """Exact two-sided binomial p-value: sum of probabilities for outcomes
    at least as extreme as observed. Used for McNemar's exact form so we
    don't pull in scipy. Mirrors scipy.stats.binom_test(b, n, 0.5).
    Numerically stable for n up to ~500 (GSM8K test slice)."""
    if n == 0:
        return 1.0
    # Compute pmf for each k in [0, n] then sum entries <= P(b).
    log_choose = [0.0] * (n + 1)
    for k in range(1, n + 1):
        log_choose[k] = log_choose[k-1] + math.log(n - k + 1) - math.log(k)
    log_p = math.log(p_null)
    log_q = math.log(1 - p_null)
    log_pmf = [log_choose[k] + k * log_p + (n - k) * log_q for k in range(n + 1)]
    target = log_pmf[b]
    # Sum probabilities for k whose log-pmf <= target (with float tol).
    s = 0.0
    for k in range(n + 1):
        if log_pmf[k] <= target + 1e-12:
            s += math.exp(log_pmf[k])
    return min(1.0, s)


def mcnemar(rows_x, rows_y, name_x, name_y):
    """Paired McNemar test on per-problem correctness. Returns dict with
    discordant counts and exact two-sided p-value. Skipped (returns None)
    if either side is empty."""
    if not rows_x or not rows_y:
        return None
    x_map = {r["idx"]: int(r["correct"]) for r in rows_x}
    y_map = {r["idx"]: int(r["correct"]) for r in rows_y}
    overlap = sorted(set(x_map) & set(y_map))
    if not overlap:
        return {"name_x": name_x, "name_y": name_y, "overlap": 0,
                "only_x": 0, "only_y": 0, "both": 0, "neither": 0,
                "p_value": None, "skipped": True,
                "reason": "no overlapping idx values"}
    only_x = sum(1 for i in overlap if x_map[i] and not y_map[i])
    only_y = sum(1 for i in overlap if y_map[i] and not x_map[i])
    both = sum(1 for i in overlap if x_map[i] and y_map[i])
    neither = sum(1 for i in overlap if not x_map[i] and not y_map[i])
    disc = only_x + only_y
    if disc == 0:
        p = 1.0
    else:
        b = min(only_x, only_y)
        p = _binom_two_sided_p(b, disc, 0.5)
    return {
        "name_x": name_x, "name_y": name_y,
        "overlap": len(overlap),
        "only_x": only_x, "only_y": only_y,
        "both": both, "neither": neither,
        "p_value": p,
        "skipped": False,
    }


def ci_overlap(a_ci, b_ci):
    a_lo, a_hi = a_ci
    b_lo, b_hi = b_ci
    return not (a_lo > b_hi or b_lo > a_hi)


def determine_outcome(c1_m, c2_m, a3_m):
    """Pre-registered interpretation from plan5.md. Labels:
        A --- 8-shot is optimal (both 0-shot cells within 3 pts of A3)
        B --- 0-shot beats 8-shot meaningfully (C1 - A3 >= 5pts, no overlap)
        C --- 0-shot underperforms substantially (A3 - C1 >= 5pts)
        D --- C1 matches A3 but C2 << C1 by >= 5pts (prefix doing real work)
        E --- C2 ~ C1 ~ A3 (IT model reasons spontaneously)
        AMBIGUOUS / INCOMPLETE --- otherwise

    Outcome A's "both within 3 pts" subsumes the simpler "C1 only" check;
    when C2 is missing we fall back to evaluating only the C1-vs-A3 axis.
    """
    if a3_m is None:
        return "INCOMPLETE", ("A3 reference shard missing; cannot decide an "
                               "outcome. Sync plan 2's results dir down or "
                               "re-run with --a3-ref-dir.")
    if c1_m is None:
        return "INCOMPLETE", ("C1_zeroshot_simple has no rows. C1 is the "
                               "cell that decides plan 5's main question.")
    a3, c1 = a3_m["accuracy"], c1_m["accuracy"]
    a3_ci, c1_ci = a3_m["ci95"], c1_m["ci95"]
    diff_c1 = c1 - a3
    overlap_c1_a3 = ci_overlap(a3_ci, c1_ci)
    c2 = c2_m["accuracy"] if c2_m is not None else None
    c2_ci = c2_m["ci95"] if c2_m is not None else None

    # Outcome B --- 0-shot beats 8-shot. Highest-precedence positive result.
    if diff_c1 >= OUTCOME_WIN and not overlap_c1_a3:
        return "B", (f"C1 ({c1:.3f}) - A3 ({a3:.3f}) = {diff_c1:+.3f} >= "
                     f"{OUTCOME_WIN:+.2f} with non-overlapping CIs. The 30% "
                     f"plateau is partly artificial; Path 1's representative "
                     f"should switch to C1's prompt format. Plan 2's "
                     f"length+SC sweep should be re-run on C1 (~8 hours).")

    # Outcome C --- 0-shot underperforms substantially.
    if -diff_c1 >= OUTCOME_WIN:
        return "C", (f"A3 ({a3:.3f}) - C1 ({c1:.3f}) = {-diff_c1:+.3f} >= "
                     f"{OUTCOME_WIN:+.2f}. 8-shot exemplars are doing real "
                     f"work; ceiling is genuine. Plan 5 is a confirming null.")

    # Outcomes A / D / E only fire when |C1 - A3| <= 3pts.
    if abs(diff_c1) <= OUTCOME_NEAR:
        if c2 is None:
            return "A", (f"|C1 - A3| = {abs(diff_c1):.3f} <= {OUTCOME_NEAR}; "
                          f"C2 not run. Path 1's plateau confirmed on the "
                          f"C1 axis. To distinguish A vs D vs E, run C2 "
                          f"(plain 0-shot) and re-summarize.")
        diff_c2_a3 = c2 - a3
        c1_minus_c2 = c1 - c2
        if abs(diff_c2_a3) <= OUTCOME_NEAR:
            return "E", (f"All three within {OUTCOME_NEAR:.2f}: "
                          f"A3={a3:.3f} C1={c1:.3f} C2={c2:.3f}. The IT "
                          f"model reasons step-by-step on its own. Path 1 "
                          f"can switch to C2 (cheapest --- no exemplars, no "
                          f"prefix) for a deployment-cost win.")
        if c1_minus_c2 >= OUTCOME_PREFIX:
            return "D", (f"C1 ({c1:.3f}) and A3 ({a3:.3f}) within "
                          f"{OUTCOME_NEAR:.2f}, but C1 - C2 = {c1_minus_c2:+.3f} "
                          f">= {OUTCOME_PREFIX:.2f}. The 'Let's think step "
                          f"by step.' prefix does real work --- a minimal-"
                          f"cost stand-in for 8-shot exemplars. Path 1's "
                          f"representative is unchanged.")
        return "A", (f"|C1 - A3| = {abs(diff_c1):.3f} <= {OUTCOME_NEAR}; "
                      f"C2 - A3 = {diff_c2_a3:+.3f} (not within "
                      f"{OUTCOME_NEAR:.2f}, but C1 - C2 = {c1_minus_c2:+.3f} "
                      f"< {OUTCOME_PREFIX:.2f}). Treat as plateau confirmed.")

    # Anything else.
    msg = (f"A3={a3:.3f} ci=({a3_ci[0]:.3f},{a3_ci[1]:.3f}), "
           f"C1={c1:.3f} ci=({c1_ci[0]:.3f},{c1_ci[1]:.3f}), "
           f"diff_C1={diff_c1:+.3f}, overlap={overlap_c1_a3}")
    if c2 is not None:
        msg += f", C2={c2:.3f} ci=({c2_ci[0]:.3f},{c2_ci[1]:.3f})"
    return "AMBIGUOUS", (msg + "; outside plan-5's pre-registered patterns.")


def warn_sanity_thresholds(metrics_by_cell):
    """Soft warnings on hash_hit / fallback_hit thresholds (sanity #2 in
    plan5.md). Non-fatal --- accuracy is still trustworthy when the
    fallback carries the load."""
    for name, m in metrics_by_cell.items():
        if m is None:
            continue
        h = m["hash_hit_rate"]
        f = m["fallback_hit_rate"]
        ne = m["no_extract_rate"]
        if h < HASH_HIT_FLOOR and f >= FALLBACK_HIT_FLOOR:
            print(f"NOTE: {name} hash_hit_rate={h:.2f} < {HASH_HIT_FLOOR} but "
                  f"fallback_hit_rate={f:.2f} >= {FALLBACK_HIT_FLOOR} --- "
                  f"fallback extractor is carrying the load (expected for "
                  f"0-shot prose answers). Accuracy is trustworthy.")
        elif h < HASH_HIT_FLOOR and f < FALLBACK_HIT_FLOOR:
            print(f"WARNING: {name} hash_hit_rate={h:.2f} AND "
                  f"fallback_hit_rate={f:.2f} both below the floors. "
                  f"Inspect a few completions; the model may not be emitting "
                  f"answers in any extractable form.")
        if ne > 0.05:
            print(f"WARNING: {name} no_extract_rate={ne:.2f} > 0.05 --- "
                  f"more than 5% of completions had no integer at all. "
                  f"Likely truncation by max_new_tokens or empty completions.")


def fmt_row(name, m, extra=""):
    if m is None:
        return f"{name:24s}  (no rows)"
    return (f"{name:24s}  acc={m['accuracy']:.3f}  "
            f"ci=({m['ci95'][0]:.3f},{m['ci95'][1]:.3f})  "
            f"hash={m['hash_hit_rate']:.2f}  "
            f"fallback={m['fallback_hit_rate']:.2f}  "
            f"mean_tok={m['mean_gen_tokens']:.1f}{extra}  n={m['n']}")


def summarize(results_dir, a3_ref_dir):
    root = Path(results_dir)
    mpath = root / "manifest.json"
    if not mpath.exists():
        print(f"ERROR: no manifest at {mpath}. Run at least one cell first.")
        sys.exit(1)
    man = json.loads(mpath.read_text())

    c1_rows = load_shards(results_dir, C1_CELL)
    c2_rows = load_shards(results_dir, C2_CELL)
    a3_rows, a3_shard_names = load_a3_reference(a3_ref_dir)

    c1_m = metrics_for_cell(c1_rows)
    c2_m = metrics_for_cell(c2_rows)
    a3_m = metrics_for_cell(a3_rows) if a3_rows else None

    n_target = man.get("n_total", DEFAULT_N)

    print()
    print(f"GSM8K (n={n_target})")
    if a3_m is None:
        print(f"  A3_len512 (reference)     (no rows --- "
              f"plan 2 shards not found at {a3_ref_dir})")
    else:
        print(f"  {fmt_row('A3_len512 (reference)', a3_m)}")
    print(f"  {fmt_row(C1_CELL, c1_m)}")
    print(f"  {fmt_row(C2_CELL, c2_m)}")

    metrics_by_cell = {C1_CELL: c1_m, C2_CELL: c2_m}
    if a3_m is not None:
        metrics_by_cell[A3_CELL] = a3_m
    warn_sanity_thresholds(metrics_by_cell)

    # Paired McNemar.
    pairs = []
    if a3_m is not None and c1_m is not None:
        pairs.append(("C1_vs_A3", mcnemar(c1_rows, a3_rows, C1_CELL, A3_CELL)))
    if a3_m is not None and c2_m is not None:
        pairs.append(("C2_vs_A3", mcnemar(c2_rows, a3_rows, C2_CELL, A3_CELL)))
    if c1_m is not None and c2_m is not None:
        pairs.append(("C1_vs_C2", mcnemar(c1_rows, c2_rows, C1_CELL, C2_CELL)))

    print()
    if not pairs:
        print("Paired comparisons: skipped (need at least one pair of cells).")
    else:
        print("Paired comparisons (McNemar exact):")
        for label, res in pairs:
            if res is None:
                print(f"  {label}: skipped (one side empty)")
                continue
            if res.get("skipped"):
                print(f"  {label}: skipped ({res.get('reason', 'unknown')})")
                continue
            p = res["p_value"]
            p_str = f"{p:.3g}" if p is not None else "n/a"
            print(f"  {label}: only_{res['name_x']}={res['only_x']}, "
                  f"only_{res['name_y']}={res['only_y']}, "
                  f"both={res['both']}, neither={res['neither']}, "
                  f"overlap={res['overlap']}, p={p_str}")

    label, reason = determine_outcome(c1_m, c2_m, a3_m)
    print()
    print(f"OUTCOME: {label} --- {reason}")

    out = {
        "config": {
            **man,
            "outcome_thresholds": {"near": OUTCOME_NEAR,
                                   "win": OUTCOME_WIN,
                                   "prefix": OUTCOME_PREFIX},
            "hash_hit_floor": HASH_HIT_FLOOR,
            "fallback_hit_floor": FALLBACK_HIT_FLOOR,
            "flops_formula": ("ff = 2*N_active*(prompt+completion); "
                              "attn = N_layers*N_heads*head_dim*"
                              "(prompt*completion + completion^2/2); "
                              "total = ff + 4*attn"),
            "flops_constants": {"N_active": N_ACTIVE_PARAMS,
                                "N_layers": N_LAYERS, "N_heads": N_HEADS,
                                "head_dim": HEAD_DIM},
            "a3_ref": {"dir": str(a3_ref_dir),
                       "cell": A3_CELL,
                       "shard_files": a3_shard_names},
        },
        "cells": {C1_CELL: c1_m, C2_CELL: c2_m, A3_CELL: a3_m},
        "paired_comparisons": {label: res for label, res in pairs},
        "outcome": {"label": label, "reason": reason},
    }
    (root / "results_plan5.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root / 'results_plan5.json'}")

    write_csv(root, c1_m, c2_m, a3_m)
    plot_path = make_plot(root, c1_m, c2_m, a3_m, n_target, label, reason)
    if plot_path is not None:
        print(f"wrote {plot_path}")


def write_csv(root, c1_m, c2_m, a3_m):
    path = root / "path1_plan5.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "accuracy", "ci_lo", "ci_hi",
                    "hash_hit_rate", "fallback_hit_rate", "no_extract_rate",
                    "mean_gen_tokens", "mean_prompt_tokens",
                    "mean_flops_per_problem", "mean_wallclock_sec", "n"])
        for cell, m in [(A3_CELL, a3_m), (C1_CELL, c1_m), (C2_CELL, c2_m)]:
            if m is None:
                continue
            w.writerow([cell, f"{m['accuracy']:.6f}",
                        f"{m['ci95'][0]:.6f}", f"{m['ci95'][1]:.6f}",
                        f"{m['hash_hit_rate']:.6f}",
                        f"{m['fallback_hit_rate']:.6f}",
                        f"{m['no_extract_rate']:.6f}",
                        f"{m['mean_gen_tokens']:.6f}",
                        f"{m['mean_prompt_tokens']:.6f}",
                        f"{m['mean_flops_per_problem']:.6e}",
                        f"{m['mean_wallclock_sec']:.6f}", m["n"]])
    print(f"wrote {path}")


def make_plot(root, c1_m, c2_m, a3_m, n_total, label, reason):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return None
    cells = [(A3_CELL, a3_m, "#1f6fb4"),
             (C1_CELL, c1_m, "#c44e4e"),
             (C2_CELL, c2_m, "#888888")]
    cells = [(name, m, color) for name, m, color in cells if m is not None]
    if not cells:
        print("  (no cells with rows; skipping plot)")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(cells)))
    accs = [m["accuracy"] for _, m, _ in cells]
    lo_err = [m["accuracy"] - m["ci95"][0] for _, m, _ in cells]
    hi_err = [m["ci95"][1] - m["accuracy"] for _, m, _ in cells]
    bar_colors = [c for _, _, c in cells]
    ax.bar(xs, accs, color=bar_colors, edgecolor="black",
           linewidth=0.6, width=0.55)
    ax.errorbar(xs, accs, yerr=[lo_err, hi_err], fmt="none",
                ecolor="black", capsize=4, linewidth=0.9)
    for x, (name, m, _) in zip(xs, cells):
        ax.text(x, m["accuracy"] + 0.015,
                f"{m['accuracy']:.3f}\n(n={m['n']}, hit={m['hash_hit_rate']:.2f})",
                ha="center", va="bottom", fontsize=8)

    # C1 - A3 lift annotation if both present.
    if a3_m is not None and c1_m is not None:
        lift = c1_m["accuracy"] - a3_m["accuracy"]
        ax.annotate(
            f"C1 - A3 = {lift:+.3f}",
            xy=(0.5,
                max(a3_m["accuracy"], c1_m["accuracy"]) + 0.10),
            ha="center", fontsize=10, color="#333333", weight="bold",
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _, _ in cells], fontsize=9)
    ax.set_ylim(0, max(0.55, max(accs) + 0.15))
    ax.set_ylabel("GSM8K exact-match accuracy")
    ax.set_title(
        f"Path 1 plan 5 --- 0-shot vs 8-shot CoT (n={n_total})\n"
        f"OUTCOME: {label}",
        fontsize=10,
    )
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.text(0.01, 0.01, reason[:240], fontsize=7,
             color="#555555", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = root / "results_plan5.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.summarize:
        summarize(args.results_dir, args.a3_ref_dir)
        return

    print_env()
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
            if cell not in ALL_CELLS:
                print(f"WARNING: unknown cell {cell!r}; skipping.")
                continue
            run_cell(
                model, tokenizer, cell, problems, golds, start, end,
                args.results_dir, args.no_resume,
                batch_size=args.batch_size,
            )
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preview_suffix = f" --n {args.n}" if args.n != DEFAULT_N else ""
    print(f"\nDone. Run `python path1_zero_shot.py --summarize"
          f"{preview_suffix}` once both cells are populated.")


if __name__ == "__main__":
    main()
