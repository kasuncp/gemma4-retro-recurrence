"""Path 1 — plan 1 gate: does CoT in-context beat no-CoT on GSM8K for
Gemma 4 E2B?

2x2 grid: {gemma-4-E2B-it, gemma-4-E2B} x {direct, cot}, 8-shot everywhere,
greedy, same 500 GSM8K test problems, reasoning present only in `cot` cells.
Decision rule keys off the `-it` CoT lift and gates whether plan2 (length
sweep + self-consistency) is worth writing.

Default --- one GPU, runs all four cells sequentially:
    python path1_cot_gate.py

Quick preview before committing to the full 500-example run (auto-routes to a
separate results dir so real-run shards are never overwritten):
    python path1_cot_gate.py --n 20
    python path1_cot_gate.py --n 20 --summarize

Two-GPU split by model:
    CUDA_VISIBLE_DEVICES=0 python path1_cot_gate.py --cells it:direct it:cot
    CUDA_VISIBLE_DEVICES=1 python path1_cot_gate.py --cells base:direct base:cot
    python path1_cot_gate.py --summarize

Three-GPU split (shard the dominant cot cells):
    CUDA_VISIBLE_DEVICES=0 python path1_cot_gate.py --cells it:direct base:direct
    CUDA_VISIBLE_DEVICES=1 python path1_cot_gate.py --cells it:cot   --problems 0:250
    CUDA_VISIBLE_DEVICES=1 python path1_cot_gate.py --cells it:cot   --problems 250:500
    CUDA_VISIBLE_DEVICES=2 python path1_cot_gate.py --cells base:cot --problems 0:250
    CUDA_VISIBLE_DEVICES=2 python path1_cot_gate.py --cells base:cot --problems 250:500
    python path1_cot_gate.py --summarize

Each (cell, shard) writes an atomic JSONL under results/path_1_cot_tokens/cells/.
--summarize merges all shards, prints the 2x2 table, writes results_gate.json,
and prints the GREEN/YELLOW/RED verdict derived from cells 1--2.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path

# Import triggers probes/__init__.py's HF_HOME redirect before torch loads.
import probes  # noqa: F401
from probes.env import DTYPE_MAP, load_model, print_env


MODELS = {
    "it":   "google/gemma-4-E2B-it",
    "base": "google/gemma-4-E2B",
}
CONDS = ("direct", "cot")
MAX_NEW = {"direct": 16, "cot": 512}
DEFAULT_N = 500
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan1"
CELLS_SUBDIR = "cells"
EXEMPLAR_SET_ID = "wei-et-al-2022-hashformat"

# Canonical Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in
# Large Language Models", Appendix G --- same 8 questions and reasoning used
# in probes/mode_round5.py (the published Gemma GSM8K 8-shot protocol), with
# the trailing "The answer is X." replaced by "#### X" so the plan's answer
# format (`(question, full reasoning ending in "#### N")`) is uniform across
# the direct and cot cells and the `####` regex lands cleanly on exemplar and
# gold alike. That substitution is the only deviation from Wei-et-al verbatim;
# it is recorded under `config.exemplar_set` in results_gate.json.
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
EXEMPLARS = {"cot": EXEMPLARS_COT, "direct": EXEMPLARS_DIRECT}

GOLD_RE = re.compile(r"####\s*(-?\d+)")
FALLBACK_RE = re.compile(r"(-?\d+)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--cells", nargs="+",
        choices=[f"{m}:{c}" for m in MODELS for c in CONDS],
        default=[f"{m}:{c}" for m in MODELS for c in CONDS],
        help="Cells to run. Default: all four. Example: --cells it:direct it:cot",
    )
    p.add_argument(
        "--problems", default=None,
        help="Problem range as START:END (half-open), over the first N GSM8K "
             "test problems (N controlled by --n). Default 0:N. Use to shard a "
             "single cell across GPUs (e.g. 0:250 and 250:500 at n=500).",
    )
    p.add_argument(
        "--n", type=int, default=DEFAULT_N,
        help=f"Number of GSM8K test problems to evaluate across all cells. "
             f"Default {DEFAULT_N}. Use a smaller value (e.g. --n 20) for a "
             f"quick direction-check before committing to the full run. When "
             f"set below {DEFAULT_N} and --results-dir is not overridden, "
             f"results auto-route to {RESULTS_SUBDIR}_preview_n<N>/ so full-run "
             f"shards are never overwritten. The verdict thresholds still "
             f"apply, but CIs will be wide --- treat preview verdicts as "
             f"directional only.",
    )
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Prompts per model.generate() call. Default 1 (sequential) to "
             "preserve backwards-compatible byte-for-byte determinism. On a "
             "single modern GPU (e.g. 4090), batch_size 4--8 typically lifts "
             "GPU utilization substantially with greedy decoding. Batched "
             "greedy should match batch=1 outputs exactly, but FP "
             "non-associativity can cause rare tied-logit flips.",
    )
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument(
        "--results-dir", default=RESULTS_SUBDIR,
        help="Directory for per-cell JSONL shards and the final results_gate.json.",
    )
    p.add_argument(
        "--summarize", action="store_true",
        help="Skip model loading; merge all per-cell shards in --results-dir, "
             "print the 2x2 table, write results_gate.json, and emit the verdict.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Before the full run, print one (prompt, completion) per cell and "
             "verify byte-for-byte determinism on the first 10 problems.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore any existing JSONL rows for the (cell, shard) pairs this "
             "invocation is going to run, and overwrite them from scratch. "
             "Rows belonging to OTHER (cell, shard) pairs are left untouched.",
    )
    args = p.parse_args()
    if args.n < 1:
        p.error(f"--n must be >= 1, got {args.n}.")
    if args.batch_size < 1:
        p.error(f"--batch-size must be >= 1, got {args.batch_size}.")
    if args.problems is None:
        args.problems = f"0:{args.n}"
    # Auto-route preview runs to a dedicated subdir so full-run shards aren't
    # overwritten. Only fires when the user didn't override --results-dir.
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


def build_prompt(exemplars, question):
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in exemplars)
    return f"{shots}\n\nQuestion: {question}\nAnswer:"


def extract(text):
    m = GOLD_RE.search(text)
    if m:
        return int(m.group(1)), True
    nums = FALLBACK_RE.findall(text)
    return (int(nums[-1]), False) if nums else (None, False)


def wilson_ci(correct, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def exemplar_hash():
    payload = json.dumps(
        {"cot": EXEMPLARS_COT, "direct": EXEMPLARS_DIRECT,
         "set_id": EXEMPLAR_SET_ID},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def manifest(args):
    return {
        "exemplar_set": EXEMPLAR_SET_ID,
        "exemplar_hash": exemplar_hash(),
        "dtype": args.dtype,
        "models": MODELS,
        "max_new": MAX_NEW,
        "n_total": args.n,
        "plan": "path_1_cot_tokens/plan1",
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


def shard_jsonl(results_dir, model_key, cond, start, end):
    return Path(results_dir) / CELLS_SUBDIR / f"{model_key}__{cond}__{start:04d}_{end:04d}.jsonl"


def load_existing(path):
    if not path.exists():
        return [], set()
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  WARNING: malformed JSONL line {i} in {path.name} ({e}); "
                  f"dropping. Will re-run that index.")
    done = {r["idx"] for r in rows if "idx" in r}
    return rows, done


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def record_commit(results_dir, model_key, model, tokenizer):
    """Opportunistically capture the HF commit SHA for a loaded model, so the
    gate JSON can pin exactly which weights produced each cell. Recorded in
    commits.json next to the manifest; --summarize merges it into results_gate.
    HF's from_pretrained populates `config._commit_hash` for cached models;
    if absent we record None rather than failing."""
    sha = getattr(model.config, "_commit_hash", None)
    tok_sha = (getattr(tokenizer, "init_kwargs", {}) or {}).get("_commit_hash")
    p = Path(results_dir) / "commits.json"
    data = {}
    if p.exists():
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            data = {}
    data[model_key] = {
        "model_id": MODELS[model_key],
        "model_commit": sha,
        "tokenizer_commit": tok_sha,
    }
    p.write_text(json.dumps(data, indent=2))


def _greedy_gen_cfg(model):
    gen_cfg = deepcopy(model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.temperature = None
    return gen_cfg


def _generate(model, tokenizer, prompt, max_new):
    import torch
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(
            **enc,
            generation_config=_greedy_gen_cfg(model),
            max_new_tokens=max_new,
            pad_token_id=pad_id,
        )
    gen_ids = out[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _generate_batch(model, tokenizer, prompts, max_new):
    """Generate completions for a batch of prompts in a single model.generate()
    call. Uses left-padding so every row's last real token is at position -1
    and `out[:, input_len:]` isolates the generated tokens. Tokenizer state is
    saved and restored so batched calls don't leak padding_side changes to
    other code paths (e.g. the per-prompt smoke check)."""
    import torch
    if tokenizer.pad_token_id is None:
        # Gemma tokenizers often lack a pad token; reuse EOS for padding.
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    finally:
        tokenizer.padding_side = prev_side
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    with torch.no_grad():
        out = model.generate(
            **enc,
            generation_config=_greedy_gen_cfg(model),
            max_new_tokens=max_new,
            pad_token_id=pad_id,
        )
    gen_ids = out[:, input_len:]
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)


def load_problems(start, end, total):
    from datasets import load_dataset
    gsm = load_dataset("gsm8k", "main", split="test")
    if total > len(gsm):
        print(f"ERROR: --n {total} exceeds GSM8K test split size {len(gsm)}.")
        sys.exit(1)
    all_problems = gsm.select(range(total))
    # Verify ground-truth parser sanity on ALL N problems we'll draw from
    # (plan sanity check #2).
    golds = []
    for row in all_problems:
        m = GOLD_RE.search(row["answer"])
        if not m:
            print(f"ERROR: GSM8K gold answer parse failed on idx=? --- "
                  f"the `####` regex missed. Split has drifted from assumption. "
                  f"Sample answer: {row['answer'][:200]!r}")
            sys.exit(1)
        golds.append(int(m.group(1)))
    return all_problems.select(range(start, end)), golds[start:end]


def run_smoke(model, tokenizer, cell_assignments, problems, golds, start):
    """Plan sanity checks #1 (print one prompt+completion per cell) and #5
    (determinism on first 10 problems of first cell in this process)."""
    import torch
    # Print one (prompt, completion) per cell on idx=start.
    ex0 = problems[0]
    for model_key, cond in cell_assignments:
        if model_key != cell_assignments[0][0]:
            continue  # smoke only covers cells for the currently loaded model
        prompt = build_prompt(EXEMPLARS[cond], ex0["question"])
        compl = _generate(model, tokenizer, prompt, MAX_NEW[cond])
        pred, hashed = extract(compl)
        print(f"  SMOKE {model_key}:{cond} idx={start}")
        print(f"    prompt[-200:] = {prompt[-200:]!r}")
        print(f"    completion    = {compl!r}")
        print(f"    pred={pred}  gold={golds[0]}  hash_hit={hashed}")

    # Determinism on first 10 problems, first cell of this process.
    model_key, cond = cell_assignments[0]
    det_ok = True
    for i in range(min(10, len(problems))):
        prompt = build_prompt(EXEMPLARS[cond], problems[i]["question"])
        a = _generate(model, tokenizer, prompt, MAX_NEW[cond])
        b = _generate(model, tokenizer, prompt, MAX_NEW[cond])
        if a != b:
            det_ok = False
            print(f"  DETERMINISM FAIL on {model_key}:{cond} idx={start+i}:\n"
                  f"    a={a!r}\n    b={b!r}")
    if not det_ok:
        print("ERROR: greedy decoding is not byte-for-byte deterministic. "
              "Do not trust subsequent numbers. Investigate before continuing.")
        sys.exit(1)
    print(f"  DETERMINISM OK (first 10 of {cell_assignments[0]})")
    torch.cuda.empty_cache()


def run_cell(model, tokenizer, model_key, cond, problems, golds, start, end,
             results_dir, no_resume, batch_size):
    path = shard_jsonl(results_dir, model_key, cond, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p, g) for i, (p, g) in enumerate(zip(problems, golds))
                 if (start + i) not in done]
    print(f"[{model_key}:{cond}] {start}:{end} --- {len(done)} done, "
          f"{len(remaining)} to generate (batch_size={batch_size}) "
          f"--> {path.name}")
    gen_secs = 0.0
    for chunk_start in range(0, len(remaining), batch_size):
        chunk = remaining[chunk_start:chunk_start + batch_size]
        t0 = time.perf_counter()
        if batch_size == 1:
            (i, p, g), = chunk
            prompt = build_prompt(EXEMPLARS[cond], p["question"])
            completions = [_generate(model, tokenizer, prompt, MAX_NEW[cond])]
        else:
            prompts = [build_prompt(EXEMPLARS[cond], p["question"])
                       for _, p, _ in chunk]
            completions = _generate_batch(model, tokenizer, prompts, MAX_NEW[cond])
        gen_secs += time.perf_counter() - t0
        for (i, p, g), compl in zip(chunk, completions):
            pred, hashed = extract(compl)
            append_jsonl(path, {
                "idx": start + i,
                "gold": g,
                "pred": pred,
                "correct": int(pred is not None and pred == g),
                "hash_hit": int(hashed),
                "completion": compl,
            })
    if gen_secs > 0 and len(remaining) > 0:
        # True throughput, not nvidia-smi util %. Compare this number across
        # batch sizes --- it should rise roughly sub-linearly with batch_size.
        print(f"[{model_key}:{cond}] {len(remaining)/gen_secs:.2f} problems/s "
              f"over {gen_secs:.1f}s of generation "
              f"(batch_size={batch_size}, max_new={MAX_NEW[cond]})")
    return path


def summarize(results_dir, args):
    root = Path(results_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no manifest at {manifest_path}. Run at least one cell "
              f"before --summarize.")
        sys.exit(1)
    m = json.loads(manifest_path.read_text())
    cells_dir = root / CELLS_SUBDIR

    per_cell = {}
    for model_key in MODELS:
        for cond in CONDS:
            shards = sorted(cells_dir.glob(f"{model_key}__{cond}__*.jsonl"))
            rows = []
            seen = set()
            for s in shards:
                for line in s.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row["idx"] in seen:
                        continue  # tolerate overlapping shards
                    seen.add(row["idx"])
                    rows.append(row)
            per_cell[f"{model_key}:{cond}"] = rows

    # Print the 2x2 table in the plan's format.
    print()
    results = {}
    n_target = m["n_total"]
    for model_key in MODELS:
        for cond in CONDS:
            key = f"{model_key}:{cond}"
            rows = per_cell[key]
            n = len(rows)
            correct = sum(r["correct"] for r in rows)
            acc = (correct / n) if n else 0.0
            ci = wilson_ci(correct, n)
            hash_hit = (sum(r["hash_hit"] for r in rows) / n) if n else 0.0
            status = "" if n == n_target else f"  [PARTIAL {n}/{n_target}]"
            print(f"{model_key:4s} {cond:6s}  acc={acc:.3f}  "
                  f"ci=({ci[0]:.3f},{ci[1]:.3f})  hash={hash_hit:.2f}{status}")
            results[key] = {
                "accuracy": acc, "ci95": list(ci), "correct": correct,
                "n": n, "hash_hit_rate": hash_hit,
            }

    verdict, verdict_reason = compute_verdict(results, n_target)
    print()
    print(f"VERDICT: {verdict} --- {verdict_reason}")

    commits_path = root / "commits.json"
    commits = {}
    if commits_path.exists():
        try:
            commits = json.loads(commits_path.read_text())
        except json.JSONDecodeError:
            commits = {}

    import torch, transformers, datasets
    gate = {
        "config": {
            **m,
            "exemplar_set_description": (
                "Wei et al. 2022 GSM8K 8-shot CoT questions and reasoning, "
                "with the terminal \"The answer is X.\" replaced by \"#### X\" "
                "to unify the answer format across direct and cot cells."
            ),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "datasets": datasets.__version__,
            },
            "commits": commits,
        },
        "cells": results,
        "verdict": {"label": verdict, "reason": verdict_reason},
    }
    out = root / "results_gate.json"
    out.write_text(json.dumps(gate, indent=2))
    print(f"wrote {out}")

    plot_path = make_plot(root, results, verdict, verdict_reason, n_target)
    if plot_path is not None:
        print(f"wrote {plot_path}")


def make_plot(root, results, verdict, verdict_reason, n_total):
    """Grouped bar chart: 4 cells, accuracy with Wilson 95% CI error bars,
    hash-hit rate annotated above each bar, verdict stamped in the corner.
    Skipped with a warning if any IT cell is missing or matplotlib isn't
    importable; everything else in summarize() still succeeds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return None

    model_keys = list(MODELS)
    conds = list(CONDS)
    if any(results.get(f"{mk}:{c}", {}).get("n", 0) == 0
           for mk in model_keys for c in conds):
        print("  (at least one cell has n=0; skipping plot)")
        return None

    width = 0.35
    x = range(len(model_keys))
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {"direct": "#888888", "cot": "#1f6fb4"}

    for j, cond in enumerate(conds):
        accs, lo_err, hi_err, hashes = [], [], [], []
        for mk in model_keys:
            r = results[f"{mk}:{cond}"]
            accs.append(r["accuracy"])
            lo_err.append(r["accuracy"] - r["ci95"][0])
            hi_err.append(r["ci95"][1] - r["accuracy"])
            hashes.append(r["hash_hit_rate"])
        xs = [xi + (j - 0.5) * width for xi in x]
        bars = ax.bar(xs, accs, width, label=cond,
                      color=colors[cond], edgecolor="black", linewidth=0.6)
        ax.errorbar(xs, accs, yerr=[lo_err, hi_err], fmt="none",
                    ecolor="black", capsize=3, linewidth=0.8)
        for xi, acc, hash_hit in zip(xs, accs, hashes):
            ax.text(xi, acc + 0.015, f"h={hash_hit:.2f}",
                    ha="center", va="bottom", fontsize=7, color="#444444")

    # IT CoT-lift annotation (the decision-rule quantity).
    d = results["it:direct"]["accuracy"]
    c = results["it:cot"]["accuracy"]
    it_x = model_keys.index("it")
    ax.annotate(
        f"lift = {c - d:+.3f}",
        xy=(it_x, max(c, d) + 0.06),
        ha="center", fontsize=9, color="#1f6fb4", weight="bold",
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"gemma-4-E2B{'-it' if mk == 'it' else ''}"
                        for mk in model_keys])
    ax.set_ylabel("GSM8K exact-match accuracy")
    ax.set_ylim(0, max(0.6, max(r["ci95"][1] for r in results.values()) + 0.1))
    ax.set_title(
        f"Path 1 plan 1 gate --- 8-shot direct vs CoT on GSM8K (n={n_total})\n"
        f"VERDICT: {verdict}",
        fontsize=10,
    )
    ax.legend(title="condition", loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.text(0.01, 0.01, verdict_reason, fontsize=7, color="#555555", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = root / "results_gate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def compute_verdict(results, n_target):
    it_direct = results.get("it:direct", {})
    it_cot = results.get("it:cot", {})
    if it_direct.get("n", 0) != n_target or it_cot.get("n", 0) != n_target:
        return ("INCOMPLETE",
                f"IT cells not at n={n_target} "
                f"(direct={it_direct.get('n',0)}, cot={it_cot.get('n',0)}).")
    d, c = it_direct["accuracy"], it_cot["accuracy"]
    d_lo, d_hi = it_direct["ci95"]
    c_lo, c_hi = it_cot["ci95"]
    lift = c - d
    ci_overlap = not (c_lo > d_hi or d_lo > c_hi)
    # RED: cot <= direct, OR (lift < 0.03 AND CIs overlap).
    if c <= d:
        return "RED", f"-it cot ({c:.3f}) <= -it direct ({d:.3f}); Path 1 wounded."
    if lift < 0.03 and ci_overlap:
        return "RED", f"lift={lift:.3f} < 0.03 and CIs overlap; Path 1 wounded."
    # GREEN: lift >= 0.10 AND CIs non-overlapping.
    if lift >= 0.10 and not ci_overlap:
        return "GREEN", f"lift={lift:.3f} >= 0.10 and CIs non-overlapping; plan2 greenlit."
    # Everything else is YELLOW: lift < 0.10, or CIs overlap, or both IT cells < 15%.
    return "YELLOW", (f"lift={lift:.3f}, ci_overlap={ci_overlap}, "
                      f"direct_acc={d:.3f}, cot_acc={c:.3f}; gate ambiguous. "
                      f"One follow-up permitted: rerun it:cot with k=5, T=0.7, "
                      f"majority vote.")


def main():
    args = parse_args()
    if args.summarize:
        summarize(args.results_dir, args)
        return

    print_env()
    start, end = parse_range(args.problems, args.n)
    check_manifest(args.results_dir, args)
    print(f"cells: {args.cells}")
    print(f"problems: [{start}, {end}) of n={args.n}   "
          f"batch_size={args.batch_size}   exemplar_hash={exemplar_hash()}")
    if args.n != DEFAULT_N:
        print(f"PREVIEW MODE (n={args.n} < {DEFAULT_N}): routing to "
              f"{args.results_dir}/")

    # Group cells by model so each model loads exactly once per process.
    by_model = {}
    for spec in args.cells:
        mk, cond = spec.split(":")
        by_model.setdefault(mk, []).append((mk, cond))

    problems, golds = load_problems(start, end, args.n)

    import torch
    dtype = DTYPE_MAP[args.dtype]

    for model_key, cell_assignments in by_model.items():
        model_id = MODELS[model_key]
        print(f"\nLoading {model_id} in {args.dtype} ...")
        tokenizer, model = load_model(model_id, dtype)
        record_commit(args.results_dir, model_key, model, tokenizer)
        try:
            if args.smoke:
                run_smoke(model, tokenizer, cell_assignments, problems, golds, start)
            for mk, cond in cell_assignments:
                run_cell(model, tokenizer, mk, cond, problems, golds,
                         start, end, args.results_dir, args.no_resume,
                         args.batch_size)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    preview_suffix = f" --n {args.n}" if args.n != DEFAULT_N else ""
    print(f"\nDone. Run `python path1_cot_gate.py --summarize{preview_suffix}` "
          f"once all cells are populated to produce results_gate.json and "
          f"the verdict.")


if __name__ == "__main__":
    main()
