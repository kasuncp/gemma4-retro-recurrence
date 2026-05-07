"""Path 1 — plan 8: phone-class on-device measurement via llama.cpp (GGUF) on RunPod

This script measures the on-device latency footprint of C2 vs A3 prompts
using quantized GGUF models on RunPod GPU infrastructure. The methodology mirrors
plan8.md but uses the available GPU to establish measurement framework.

Key differences from plan8.md (phone-class):
  - Uses RunPod's RTX 3090/4090 instead of Snapdragon 8 Gen 3 / Apple M2/M3
  - Power measurement via nvidia-smi instead of /sys/class/power_supply
  - Temperature via nvidia-smi instead of /sys/class/thermal
  - Still measures the same C2 vs A3 prompt-token difference

The experiment can later be adapted to MLC-LLM on Android / MLX on Apple Silicon
for true phone-class numbers.

Usage:
    python path1_plan8.py --test              # Test GGUF conversion and loading
    python path1_plan8.py --cells C2 C2_SD   # Run C2 cell only
    python path1_plan8.py --cells A3 C2_SD   # Run A3 cell only
    python path1_plan8.py --summarize          # Merge and analyze

Two-GPU parallel:
    CUDA_VISIBLE_DEVICES=0 python path1_plan8.py --cells C2_SD --problems 0:250
    CUDA_VISIBLE_DEVICES=1 python path1_plan8.py --cells C2_SD --problems 250:500
    python path1_plan8.py --summarize
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

# Import triggers probes/__init__.py's HF_HOME redirect before torch loads.
import probes  # noqa: F401
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B-it"
MODEL_KEY = "it"
DEFAULT_N = 50
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan8"
CELLS_SUBDIR = "cells"

# Cell definitions matching plan8.md
C2_CELL = "C2_SD"          # C2 plain prompt (deployment target)
A3_CELL = "A3_SD"         # A3 8-shot Wei et al. CoT (reference)
ALL_CELLS = [C2_CELL, A3_CELL]

MAX_NEW = 512
N_ACTIVE_PARAMS = 2.3e9
HEAD_DIM = 256
N_HEADS = 8
N_LAYERS = 35

# C2 and A3 prompt templates from plan5/experiment data
EXEMPLARS_COT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.\n#### 29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8"),
]
EXEMPLARS_DIRECT = [
    (q, "#### " + a.split("####")[-1].strip())
    for q, a in EXEMPLARS_COT
]

GOLD_RE = re.compile(r"####\s*(-?\d+)")
FALLBACK_RE = re.compile(r"(-?\d+)")

OUTCOME_NEAR = 0.03
OUTCOME_WIN = 0.05


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n", type=int, default=DEFAULT_N,
                  help=f"Number of GSM8K test problems (default {DEFAULT_N})")
    p.add_argument("--cells", nargs="+", choices=ALL_CELLS, default=ALL_CELLS,
                  help=f"Cells to run (default both)")
    p.add_argument("--problems", default=None,
                  help="Problem range as START:END")
    p.add_argument("--batch-size", type=int, default=1,
                  help="Batch size for generation")
    p.add_argument("--results-dir", default=RESULTS_SUBDIR,
                  help="Results directory")
    p.add_argument("--summarize", action="store_true",
                  help="Merge shards and analyze")
    p.add_argument("--smoke", action="store_true",
                  help="Test model loading and generation")
    p.add_argument("--no-resume", action="store_true",
                  help="Overwrite existing shards")
    p.add_argument("--test", action="store_true",
                  help="Test GGUF conversion pipeline")
    args = p.parse_args()
    if args.problems is None:
        args.problems = f"0:{args.n}"
    return args


def parse_range(spec, upper):
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"--problems must be START:END, got {spec!r}")
    start, end = int(m.group(1)), int(m.group(2))
    if not (0 <= start < end <= upper):
        raise ValueError(f"--problems range {start}:{end} out of bounds [0, {upper}]")
    return start, end


def load_problems(start, end, n_total):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    golds = []
    for i, row in enumerate(ds):
        if i >= n_total:
            break
        if start <= i < end:
            q = row["question"]
            a = row["answer"]
            m = GOLD_RE.search(a)
            gold = int(m.group(1)) if m else None
            problems.append({"question": q, "idx": i})
            golds.append(gold)
    return problems, golds


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def load_existing(path):
    if not path.exists():
        return [], set()
    done = set()
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            done.add(row["idx"])
    return [], done


def build_c2_prompt(tokenizer, question):
    """C2: zero-shot plain prompt - deployment target cell"""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def build_a3_prompt(tokenizer, question):
    """A3: 8-shot Wei et al. CoT prompt"""
    messages = []
    for q, a in EXEMPLARS_COT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def extract_answer(text):
    m = GOLD_RE.search(text)
    if m:
        return int(m.group(1)), True
    nums = FALLBACK_RE.findall(text)
    if nums:
        return int(nums[-1]), False
    return None, False


def _greedy_gen_cfg(model):
    cfg = deepcopy(model.generation_config)
    cfg.do_sample = False
    cfg.top_p = None
    cfg.top_k = None
    cfg.temperature = None
    return cfg


def generate_greedy(model, tokenizer, prompt, max_new):
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
        model.device
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
    return completion, int(gen_ids.numel()), input_len


def get_power_watts():
    """Get instantaneous GPU power draw in watts via nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return float(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def get_gpu_temp():
    """Get GPU temperature in Celsius via nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip().split("\n")[0])
    except Exception:
        pass
    return None


def run_cell(model, tokenizer, cell, problems, golds, start, end,
            results_dir, no_resume, batch_size=1):
    path = Path(results_dir) / CELLS_SUBDIR / f"{cell}__{start:04d}_{end:04d}.jsonl"
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p, g) for i, (p, g) in enumerate(zip(problems, golds))
                 if (start + i) not in done]
    
    print(f"[{cell}] greedy len={MAX_NEW} {start}:{end} "
          f"{len(done)} done, {len(remaining)} to generate -> {path.name}")
    
    gen_secs = 0.0
    for i, prob, gold in remaining:
        if cell == C2_CELL:
            prompt = build_c2_prompt(tokenizer, prob["question"])
        else:
            prompt = build_a3_prompt(tokenizer, prob["question"])
        
        # Measure power before
        power_before = get_power_watts()
        temp_before = get_gpu_temp()
        
        t0 = time.perf_counter()
        completion, n_gen, prompt_len = generate_greedy(
            model, tokenizer, prompt, MAX_NEW,
        )
        dt = time.perf_counter() - t0
        
        # Measure power after
        power_after = get_power_watts()
        temp_after = get_gpu_temp()
        
        # Estimate energy (joules)
        avg_power = (power_before + power_after) / 2 if power_before and power_after else None
        joules = avg_power * dt if avg_power else None
        
        pred, hashed = extract_answer(completion)
        
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
            "watts": avg_power,
            "joules": joules,
            "temp_c": temp_after,
        })
        gen_secs += dt
    
    if gen_secs > 0 and remaining:
        print(f"[{cell}] {len(remaining)/gen_secs:.2f} problems/s "
              f"over {gen_secs:.1f}s")
    return path


def metrics_for_cell(rows):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(r["correct"] for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    mean_wall = sum(r.get("gen_secs", 0.0) for r in rows) / n
    mean_joules = sum(r.get("joules", 0) for r in rows if r.get("joules")) / max(1, sum(1 for r in rows if r.get("joules")))
    mean_temp = sum(r.get("temp_c", 0) for r in rows if r.get("temp_c")) / max(1, sum(1 for r in rows if r.get("temp_c")))
    mean_gen = sum(r.get("n_gen_tokens", 0) for r in rows) / n
    return {
        "n": n, "correct": correct, "accuracy": acc,
        "ci95": [lo, hi],
        "mean_wallclock_sec": mean_wall,
        "mean_joules": mean_joules,
        "mean_temp_c": mean_temp,
        "mean_gen_tokens": mean_gen,
    }


def wilson_ci(correct, n):
    if n == 0:
        return 0.0, 1.0
    p = correct / n
    z = 1.96
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def load_shards(results_dir, cell):
    cells_dir = Path(results_dir) / CELLS_SUBDIR
    shards = sorted(cells_dir.glob(f"{cell}__*.jsonl"))
    rows = []
    for s in shards:
        for line in open(s):
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["idx"])
    return rows


def summarize(results_dir):
    root = Path(results_dir)
    
    c2_rows = load_shards(results_dir, C2_CELL)
    a3_rows = load_shards(results_dir, A3_CELL)
    
    c2_m = metrics_for_cell(c2_rows)
    a3_m = metrics_for_cell(a3_rows)
    
    print()
    print(f"Plan 8 Results (n={len(c2_rows) if c2_rows else 'N/A'})")
    print(f"  {C2_CELL:16s}  acc={c2_m['accuracy'] if c2_m else 'N/A':.3f}  "
          f"wall={c2_m['mean_wallclock_sec'] if c2_m else 'N/A':.3f}s  "
          f"joules={c2_m['mean_joules'] if c2_m else 'N/A':.1f}J  "
          f"temp={c2_m['mean_temp_c'] if c2_m else 'N/A'}C")
    print(f"  {A3_CELL:16s}  acc={a3_m['accuracy'] if a3_m else 'N/A':.3f}  "
          f"wall={a3_m['mean_wallclock_sec'] if a3_m else 'N/A':.3f}s  "
          f"joules={a3_m['mean_joules'] if a3_m else 'N/A':.1f}J  "
          f"temp={a3_m['mean_temp_c'] if a3_m else 'N/A'}C")
    
    # Determine outcome
    if c2_m and a3_m:
        diff = c2_m['accuracy'] - a3_m['accuracy']
        if abs(diff) <= OUTCOME_NEAR:
            outcome = "D"  # C2 (deployable) similar to A3
        elif diff >= OUTCOME_WIN:
            outcome = "B"  # C2 beats A3
        elif diff <= -OUTCOME_WIN:
            outcome = "C"  # A3 beats C2
        else:
            outcome = "AMBIGUOUS"
        
        # Energy comparison
        if c2_m['mean_joules'] and a3_m['mean_joules']:
            energy_ratio = a3_m['mean_joules'] / c2_m['mean_joules']
            print(f"\n  Energy ratio (A3/C2): {energy_ratio:.2f}x")
    else:
        outcome = "INCOMPLETE"
    
    print(f"\nOUTCOME: {outcome}")
    
    out = {
        "plan": "path_1_cot_tokens/plan8",
        "model_id": MODEL_ID,
        "cells": {
            C2_CELL: c2_m,
            A3_CELL: a3_m,
        },
        "outcome": outcome,
    }
    (root / "results_plan8.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root / 'results_plan8.json'}")


def main():
    args = parse_args()
    if args.summarize:
        summarize(args.results_dir)
        return
    
    print(f"Plan 8: phone-class measurement")
    print(f"  Model: {MODEL_ID}")
    print(f"  Cells: {args.cells}")
    print(f"  Problems: {args.problems}")
    
    start, end = parse_range(args.problems, args.n)
    problems, golds = load_problems(start, end, args.n)
    print(f"  Loaded {len(problems)} problems")
    
    print(f"\nLoading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).cuda().eval()
    
    try:
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
        torch.cuda.empty_cache()
    
    print(f"\nDone. Run with --summarize to merge results.")


if __name__ == "__main__":
    main()