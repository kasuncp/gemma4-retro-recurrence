"""Path 1 --- plan 4: ARC-Easy cross-benchmark validity for IT-CoT.

Cross-benchmark check for plan 2's GSM8K plateau finding. One axis-A cell
(A3-equivalent CoT @ len=512), one axis-B cell (k=5 self-consistency at
len=512, the most-promising plan-2 SC cell) and one optional direct-answer
cell (the plan-1 direct gate, on a second benchmark) --- all on the IT
model, all on the deterministic first ``n=500`` ARC-Easy test problems.

    A3_arc     : 8-shot CoT, greedy, max_new=512
    B3_arc     : 8-shot CoT, sampled (T=0.7, top_p=0.95), k=5, max_new=512
    direct_arc : 8-shot direct, greedy, max_new=16   (optional)

Default --- one GPU, runs all three cells sequentially:
    python path1_arc_easy.py

Quick preview (auto-routes to plan4_preview_n<N>/ to keep full-run shards
intact):
    python path1_arc_easy.py --n 20 --cells A3_arc B3_arc

Cell-parallel on two GPUs:
    CUDA_VISIBLE_DEVICES=0 python path1_arc_easy.py --cells A3_arc direct_arc
    CUDA_VISIBLE_DEVICES=1 python path1_arc_easy.py --cells B3_arc
    python path1_arc_easy.py --summarize

A3_arc is plan 4's reference cell (length-matched to plan 2's A3_len512);
B3_arc is the k=5 self-consistency cell (the most-promising plan-2 SC cell
that wasn't B4_k10's diminishing-returns 2x compute). The pre-registered
outcome interpretation (A / B / C / D) is computed during --summarize and
stamped on the figure.
"""

import argparse
import csv
import hashlib
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

# Borrowed verbatim from plan 1/plan 2: shard IO, Wilson CI, axis-B seeding
# convention. The exemplar set and prompt builder are ARC-specific and live
# in this file --- no GSM8K state is reused.
from path1_cot_gate import (
    append_jsonl,
    load_existing,
    wilson_ci,
)


MODEL_KEY = "it"
MODEL_ID = "google/gemma-4-E2B-it"
DEFAULT_N = 500
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan4"
CELLS_SUBDIR = "cells"
EXEMPLAR_SET_ID = "arc-easy-train-head8-handcrafted-cot"

# Cell definitions. Cell names are the canonical ids used in JSONL filenames
# and the results_plan4.json keys.
A3_CELL = "A3_arc"
B3_CELL = "B3_arc"
DIRECT_CELL = "direct_arc"
A3_LEN = 512
B3_LEN = 512
B3_K = 5
B3_TEMPERATURE = 0.7
B3_TOP_P = 0.95
DIRECT_LEN = 16
B3_BASE_SEED = 0

# Gemma 4 E2B architectural constants used in the analytical FLOPs formula
# (same as plan 2). Verified at runtime against model.config --- mismatches
# produce a warning, not a failure.
N_ACTIVE_PARAMS = 2.3e9
HEAD_DIM = 256
N_HEADS = 8
N_LAYERS = 35

# Plan-4 sanity-check thresholds (see "Sanity checks before trusting numbers"
# in plan4.md). Soft warnings only --- the cells still write their JSONLs.
ANSWER_HIT_FLOOR = 0.80                 # below this on A3 = prompt format wrong
A3_PLAUSIBLE_LO, A3_PLAUSIBLE_HI = 0.65, 0.92  # warn outside this band
DIRECT_PLAUSIBLE_LO = 0.40              # well above 25% chance baseline
SATURATION_THRESHOLD = 0.90             # outcome C (benchmark saturated)

# Hand-crafted 8-shot CoT exemplars drawn from the deterministic first 8
# rows of allenai/ai2_arc ARC-Easy *train* split, with reasoning written by
# the plan author. Each entry pins the original train-split id, the question
# text exactly as published, the four (or fewer) choice texts, the original
# labels (which are A/B/C/D for 7 of 8 and 1/2/3/4 for one --- see #4), the
# normalized A/B/C/D labels actually shown in the prompt, the normalized
# gold letter, and the reasoning string.
#
# Why hand-crafted reasoning: ARC-Easy train rows ship with no reasoning
# text. The Wei-et-al GSM8K plan-1 exemplars are the only standard CoT prompt
# for that benchmark; ARC-Easy CoT exemplars are not standardized in the
# literature, so we write our own and pin the exemplar_hash so they cannot
# silently drift.
#
# Why normalize labels: row #4 (NYSEDREGENTS_2006_8_10) uses 1/2/3/4 labels
# instead of A/B/C/D --- the plan-4 spec specifically calls this out
# (sanity check #2). Normalizing every prompt to A/B/C/D keeps the answer-
# extraction regex uniform across exemplars and test problems.
EXEMPLARS_COT = [
    {
        "id": "Mercury_7220990",
        "question": ("Which factor will most likely cause a person to "
                     "develop a fever?"),
        "labels": ["A", "B", "C", "D"],
        "texts": [
            "a leg muscle relaxing after exercise",
            "a bacterial population in the bloodstream",
            "several viral particles on the skin",
            "carbohydrates being digested in the stomach",
        ],
        "orig_labels": ["A", "B", "C", "D"],
        "orig_gold": "B",
        "gold": "B",
        "reasoning": ("Fever is the body's response to an internal "
                      "infection. Muscle relaxation (A) and digestion (D) "
                      "are normal processes that don't trigger fever. "
                      "Viral particles only on the skin (C) haven't yet "
                      "entered the body. A bacterial population already "
                      "in the bloodstream (B) provokes the immune "
                      "response that produces fever."),
    },
    {
        "id": "MCAS_2007_8_5189",
        "question": ("Lichens are symbiotic organisms made of green "
                     "algae and fungi. What do the green algae supply "
                     "to the fungi in this symbiotic relationship?"),
        "labels": ["A", "B", "C", "D"],
        "texts": ["carbon dioxide", "food", "protection", "water"],
        "orig_labels": ["A", "B", "C", "D"],
        "orig_gold": "B",
        "gold": "B",
        "reasoning": ("In a lichen, the green algae photosynthesize and "
                      "produce sugars; the fungus provides structure and "
                      "absorbs water. So the algae supply food (sugars) "
                      "to the fungi."),
    },
    {
        "id": "Mercury_SC_401169",
        "question": "When a switch is used in an electrical circuit, the switch can",
        "labels": ["A", "B", "C", "D"],
        "texts": [
            "cause the charge to build.",
            "increase and decrease the voltage.",
            "cause the current to change direction.",
            "stop and start the flow of current.",
        ],
        "orig_labels": ["A", "B", "C", "D"],
        "orig_gold": "D",
        "gold": "D",
        "reasoning": ("A switch is a two-state device. It opens or closes "
                      "the circuit, which stops or starts the flow of "
                      "current. It does not change voltage or reverse "
                      "current direction on its own."),
    },
    {
        "id": "MCAS_2004_8_27",
        "question": "Which of the following is an example of an assistive device?",
        "labels": ["A", "B", "C", "D"],
        "texts": ["contact lens", "motorcycle", "raincoat", "coffee pot"],
        "orig_labels": ["A", "B", "C", "D"],
        "orig_gold": "A",
        "gold": "A",
        "reasoning": ("An assistive device helps a person overcome a "
                      "physical limitation. A contact lens corrects "
                      "vision impairment. A motorcycle, raincoat, and "
                      "coffee pot are general-purpose items, not "
                      "assistive devices."),
    },
    {
        "id": "NYSEDREGENTS_2006_8_10",
        "question": ("Rocks are classified as igneous, metamorphic, or "
                     "sedimentary according to"),
        "labels": ["A", "B", "C", "D"],
        "texts": [
            "their color",
            "their shape",
            "how they formed",
            "the minerals they contain",
        ],
        "orig_labels": ["1", "2", "3", "4"],
        "orig_gold": "3",
        "gold": "C",
        "reasoning": ("Igneous, metamorphic, and sedimentary are "
                      "categories defined by formation process: igneous "
                      "from cooled magma, sedimentary from deposited "
                      "sediment, metamorphic from heat and pressure. So "
                      "rocks are classified by how they formed."),
    },
    {
        "id": "Mercury_7013388",
        "question": ("A chewable calcium carbonate tablet is a common "
                     "treatment for stomach discomfort. Calcium carbonate "
                     "is most likely used as this type of medicine "
                     "because calcium carbonate"),
        "labels": ["A", "B", "C", "D"],
        "texts": [
            "has a pleasant flavor.",
            "is inexpensive to produce.",
            "neutralizes digestive acid.",
            "occurs naturally in the body.",
        ],
        "orig_labels": ["A", "B", "C", "D"],
        "orig_gold": "C",
        "gold": "C",
        "reasoning": ("Stomach discomfort is often caused by excess "
                      "stomach acid. Calcium carbonate is a base that "
                      "reacts with acid; this neutralizes the acid and "
                      "relieves the discomfort. The other options are "
                      "side properties, not the medical mechanism."),
    },
    {
        "id": "Mercury_7179953",
        "question": "Which two body systems are directly involved in movement?",
        "labels": ["A", "B", "C", "D"],
        "texts": [
            "muscular and skeletal",
            "digestive and muscular",
            "skeletal and respiratory",
            "respiratory and digestive",
        ],
        "orig_labels": ["A", "B", "C", "D"],
        "orig_gold": "A",
        "gold": "A",
        "reasoning": ("Movement requires both a structural framework and "
                      "a force generator. Bones (skeletal system) provide "
                      "the framework, and muscles (muscular system) "
                      "contract to pull on the bones. So muscular and "
                      "skeletal are the two systems directly involved."),
    },
    {
        "id": "Mercury_7205118",
        "question": ("Which change in the state of water particles "
                     "causes the particles to become arranged in a "
                     "fixed position?"),
        "labels": ["A", "B", "C", "D"],
        "texts": ["boiling", "melting", "freezing", "evaporating"],
        "orig_labels": ["A", "B", "C", "D"],
        "orig_gold": "C",
        "gold": "C",
        "reasoning": ("When liquid water freezes into ice, the water "
                      "particles arrange into a rigid lattice in fixed "
                      "positions. Boiling, melting, and evaporating all "
                      "increase particle motion rather than fixing it."),
    },
]

# Direct-answer exemplars: same questions and choices, no reasoning, just
# "The answer is <letter>." The plan-1 direct gate's analog on ARC.
EXEMPLARS_DIRECT = [
    {**ex, "reasoning": ""}  # reasoning omitted; build_prompt handles this.
    for ex in EXEMPLARS_COT
]
EXEMPLARS = {"cot": EXEMPLARS_COT, "direct": EXEMPLARS_DIRECT}

# Answer-extraction regexes. Primary: "The answer is X" (the format taught
# by the exemplars). Fallback: last single uppercase letter A-E mentioned
# anywhere in the completion --- correct often enough to keep accuracy from
# being dragged down by trailing-text variation.
PRIMARY_ANSWER_RE = re.compile(r"[Tt]he answer is\s*\(?([A-E])\)?")
FALLBACK_ANSWER_RE = re.compile(r"\b([A-E])\b")

# Default cell list; --cells overrides.
ALL_CELLS = [A3_CELL, B3_CELL, DIRECT_CELL]


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
        help=f"Number of ARC-Easy test problems to evaluate (default {DEFAULT_N}). "
             f"When set below {DEFAULT_N} and --results-dir is not overridden, "
             f"results auto-route to {RESULTS_SUBDIR}_preview_n<N>/ so full-run "
             f"shards are never overwritten.",
    )
    p.add_argument(
        "--cells", nargs="+", choices=ALL_CELLS, default=ALL_CELLS,
        help=f"Specific cell names to run. Default all three: {', '.join(ALL_CELLS)}. "
             f"Drop {DIRECT_CELL} to skip the optional direct gate.",
    )
    p.add_argument(
        "--problems", default=None,
        help="Problem range as START:END (half-open) over the first --n ARC-Easy "
             "test problems. Default 0:N. Used to shard a single cell across "
             "GPUs (e.g. 0:250 and 250:500 at n=500).",
    )
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Greedy batch size for A3_arc and direct_arc. Default 1 (single-"
             "sequence) is byte-deterministic. Larger values amortize GPU idle "
             "time; B3_arc sampling ignores this flag (it already batches k "
             "chains via num_return_sequences=k).",
    )
    p.add_argument(
        "--results-dir", default=RESULTS_SUBDIR,
        help="Directory for per-cell JSONL shards and final results_plan4.json.",
    )
    p.add_argument(
        "--summarize", action="store_true",
        help="Skip model loading; merge all per-cell shards in --results-dir, "
             "print the table, write results_plan4.json + results_plan4.png + "
             "path1_plan4.csv, and emit the outcome label.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Before the full run, verify exemplar hash, print one "
             "(prompt, completion) per cell, and assert per-problem "
             "determinism on the first 3 problems for one cell.",
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

def normalize_problem(row, idx):
    """Convert a raw HF ARC-Easy row into a problem dict with normalized
    A/B/C/D labels in the prompt and a normalized gold letter, regardless
    of whether the source labels were A-D or 1-4. Returns None if the gold
    answer is missing from the choice list (data corruption); the caller
    drops such rows and surfaces the count in the manifest."""
    orig_labels = list(row["choices"]["label"])
    texts = list(row["choices"]["text"])
    n = len(orig_labels)
    if n == 0 or n > 26:
        return None
    if row["answerKey"] not in orig_labels:
        return None
    gold_pos = orig_labels.index(row["answerKey"])
    norm_labels = [chr(ord("A") + j) for j in range(n)]
    return {
        "idx": idx,
        "id": row["id"],
        "question": row["question"],
        "labels": norm_labels,
        "texts": texts,
        "orig_labels": orig_labels,
        "orig_gold": row["answerKey"],
        "gold": norm_labels[gold_pos],
        "n_choices": n,
        "remapped": orig_labels != norm_labels,
    }


def build_prompt(exemplars, problem, condition):
    """Build the 8-shot prompt for a problem. ``condition`` is 'cot' or
    'direct'. CoT exemplars include the reasoning text; direct exemplars
    skip straight to "The answer is X."""
    shots = []
    for ex in exemplars:
        choice_lines = "\n".join(f"{lbl}) {txt}"
                                 for lbl, txt in zip(ex["labels"], ex["texts"]))
        if condition == "cot" and ex.get("reasoning"):
            ans = f"{ex['reasoning']} The answer is {ex['gold']}."
        else:
            ans = f"The answer is {ex['gold']}."
        shots.append(f"Question: {ex['question']}\n{choice_lines}\nAnswer: {ans}")
    test_lines = "\n".join(f"{lbl}) {txt}"
                            for lbl, txt in zip(problem["labels"],
                                                problem["texts"]))
    return ("\n\n".join(shots)
            + f"\n\nQuestion: {problem['question']}\n{test_lines}\nAnswer:")


def extract(text, valid_labels):
    """Extract answer letter. Returns (pred, hashed) where hashed=True iff
    the primary 'The answer is X' regex matched. Pred restricted to letters
    in ``valid_labels`` (the per-problem normalized label set). Falls back
    to the last in-set letter mention if the primary pattern doesn't fire.
    Returns (None, False) if neither fires."""
    valid = set(valid_labels)
    m = PRIMARY_ANSWER_RE.search(text)
    if m and m.group(1) in valid:
        return m.group(1), True
    matches = [c for c in FALLBACK_ANSWER_RE.findall(text) if c in valid]
    if matches:
        return matches[-1], False
    return None, False


def exemplar_hash():
    """Hash the canonical exemplar set so any in-place edit to the embedded
    list above causes a verbatim mismatch with results recorded by an
    earlier run --- preventing a silent drift in the prompt."""
    payload = json.dumps(
        {"set_id": EXEMPLAR_SET_ID, "cot": EXEMPLARS_COT,
         "direct": EXEMPLARS_DIRECT},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


# -----------------------------------------------------------------------------
# manifest
# -----------------------------------------------------------------------------

def manifest(args, gold_stats=None):
    return {
        "plan": "path_1_cot_tokens/plan4",
        "model_id": MODEL_ID,
        "exemplar_set": EXEMPLAR_SET_ID,
        "exemplar_hash": exemplar_hash(),
        "exemplar_ids": [ex["id"] for ex in EXEMPLARS_COT],
        "dtype": args.dtype,
        "n_total": args.n,
        "cells": [
            {"cell": A3_CELL, "condition": "cot", "decode": "greedy",
             "max_new_tokens": A3_LEN, "k": 1},
            {"cell": B3_CELL, "condition": "cot", "decode": "sampled",
             "max_new_tokens": B3_LEN, "k": B3_K,
             "temperature": B3_TEMPERATURE, "top_p": B3_TOP_P},
            {"cell": DIRECT_CELL, "condition": "direct", "decode": "greedy",
             "max_new_tokens": DIRECT_LEN, "k": 1},
        ],
        "axis_b_base_seed": B3_BASE_SEED,
        "primary_answer_regex": PRIMARY_ANSWER_RE.pattern,
        "fallback_answer_regex": FALLBACK_ANSWER_RE.pattern,
        "gold_stats": gold_stats or {},
    }


def check_manifest(results_dir, args, gold_stats):
    p = Path(results_dir) / "manifest.json"
    desired = manifest(args, gold_stats=gold_stats)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(desired, indent=2))
        return desired
    existing = json.loads(p.read_text())
    # Compare on stable fields only --- gold_stats is informational and
    # depends on --n, so a different --n shouldn't fail manifest check.
    stable = ("plan", "model_id", "exemplar_set", "exemplar_hash",
              "exemplar_ids", "dtype", "cells", "axis_b_base_seed",
              "primary_answer_regex", "fallback_answer_regex")
    diffs = [k for k in stable if existing.get(k) != desired.get(k)]
    if diffs:
        print(f"ERROR: manifest at {p} is incompatible with this run:")
        for k in diffs:
            print(f"  {k}: existing={existing.get(k)!r}  requested={desired[k]!r}")
        print(f"  Delete {p.parent} or fix the mismatch.")
        sys.exit(1)
    # Refresh gold_stats / n_total without failing if they widened.
    existing["gold_stats"] = gold_stats
    existing["n_total"] = max(existing.get("n_total", 0), args.n)
    p.write_text(json.dumps(existing, indent=2))
    return existing


# -----------------------------------------------------------------------------
# data loading
# -----------------------------------------------------------------------------

def load_problems(start, end, total):
    """Load the deterministic first ``total`` rows of ARC-Easy test, validate
    each gold answer, normalize labels, and slice [start, end). Returns
    (problems, gold_stats). gold_stats reports the count of remapped
    (numeric) golds and any rows dropped for missing/bad gold."""
    from datasets import load_dataset
    print(f"Loading ARC-Easy (allenai/ai2_arc, ARC-Easy, split=test) ...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    if total > len(ds):
        print(f"ERROR: --n {total} exceeds ARC-Easy test split size {len(ds)}.")
        sys.exit(1)

    problems = []
    n_remapped = 0
    n_3_choice = 0
    n_dropped = 0
    n_choices_hist = {}
    for i, row in enumerate(ds):
        if i >= total:
            break
        prob = normalize_problem(row, i)
        if prob is None:
            n_dropped += 1
            print(f"  WARNING: row idx={i} id={row['id']!r} dropped "
                  f"(gold {row['answerKey']!r} not in labels "
                  f"{row['choices']['label']!r}).")
            continue
        if prob["remapped"]:
            n_remapped += 1
        if prob["n_choices"] != 4:
            if prob["n_choices"] == 3:
                n_3_choice += 1
        n_choices_hist[prob["n_choices"]] = (
            n_choices_hist.get(prob["n_choices"], 0) + 1)
        problems.append(prob)

    print(f"  loaded {len(problems)} ARC-Easy test problems "
          f"(remapped numeric gold: {n_remapped}, 3-choice: {n_3_choice}, "
          f"dropped: {n_dropped}).")
    gold_stats = {
        "n_loaded": len(problems),
        "n_remapped_numeric": n_remapped,
        "n_3_choice": n_3_choice,
        "n_dropped_bad_gold": n_dropped,
        "n_choices_histogram": n_choices_hist,
    }
    # Slice after counting to keep gold_stats informative across the full
    # n window.
    return problems[start:end], gold_stats


# -----------------------------------------------------------------------------
# generation: greedy and sampled-k-chains
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
    cfg.temperature = B3_TEMPERATURE
    cfg.top_p = B3_TOP_P
    cfg.top_k = None
    return cfg


def _count_gen_tokens(gen_ids, pad_id):
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


def generate_greedy_batch(model, tokenizer, prompts, max_new):
    """Batched greedy. Left-pads, attention-masks the pad positions, and
    returns (completion, n_gen, input_len) per prompt in order. Not
    bit-identical to generate_greedy --- batched matmul reorders ops."""
    import torch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    was_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(list(prompts), return_tensors="pt",
                        padding=True).to(model.device)
    finally:
        tokenizer.padding_side = was_padding_side
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
    """Sample k chains in a single generate() call. Per-problem seeded for
    resume-stable determinism (same convention as plan-2 axis B)."""
    import torch
    torch.manual_seed(B3_BASE_SEED + idx)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(B3_BASE_SEED + idx)
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
    """Plurality of non-None letter predictions. Ties broken by first-chain-
    wins, deterministic given the per-problem seed."""
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
# shard paths
# -----------------------------------------------------------------------------

def shard_jsonl(results_dir, cell, start, end):
    return Path(results_dir) / CELLS_SUBDIR / f"{cell}__{start:04d}_{end:04d}.jsonl"


# -----------------------------------------------------------------------------
# per-cell runners
# -----------------------------------------------------------------------------

def run_greedy_cell(model, tokenizer, cell, condition, max_new, problems,
                    start, end, results_dir, no_resume, batch_size=1):
    path = shard_jsonl(results_dir, cell, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p) for i, p in enumerate(problems)
                 if (start + i) not in done]
    print(f"[{cell}] cond={condition} max_new={max_new}  {start}:{end}  "
          f"{len(done)} done, {len(remaining)} to generate  -> {path.name}  "
          f"batch_size={batch_size}")
    gen_secs = 0.0
    if batch_size <= 1:
        for i, prob in remaining:
            prompt = build_prompt(EXEMPLARS[condition], prob, condition)
            t0 = time.perf_counter()
            completion, n_gen, prompt_len = generate_greedy(
                model, tokenizer, prompt, max_new,
            )
            dt = time.perf_counter() - t0
            gen_secs += dt
            pred, hashed = extract(completion, prob["labels"])
            append_jsonl(path, {
                "idx": start + i,
                "id": prob["id"],
                "gold": prob["gold"],
                "orig_gold": prob["orig_gold"],
                "remapped": prob["remapped"],
                "n_choices": prob["n_choices"],
                "pred": pred,
                "correct": int(pred is not None and pred == prob["gold"]),
                "answer_hit": int(hashed),
                "completion": completion,
                "n_gen_tokens": n_gen,
                "prompt_tokens": prompt_len,
                "gen_secs": dt,
            })
    else:
        for b in range(0, len(remaining), batch_size):
            batch = remaining[b:b + batch_size]
            prompts = [build_prompt(EXEMPLARS[condition], prob, condition)
                       for _, prob in batch]
            t0 = time.perf_counter()
            per_row = generate_greedy_batch(
                model, tokenizer, prompts, max_new,
            )
            dt = time.perf_counter() - t0
            gen_secs += dt
            per_row_dt = dt / len(batch)
            for (i, prob), (completion, n_gen, prompt_len) in zip(batch, per_row):
                pred, hashed = extract(completion, prob["labels"])
                append_jsonl(path, {
                    "idx": start + i,
                    "id": prob["id"],
                    "gold": prob["gold"],
                    "orig_gold": prob["orig_gold"],
                    "remapped": prob["remapped"],
                    "n_choices": prob["n_choices"],
                    "pred": pred,
                    "correct": int(pred is not None and pred == prob["gold"]),
                    "answer_hit": int(hashed),
                    "completion": completion,
                    "n_gen_tokens": n_gen,
                    "prompt_tokens": prompt_len,
                    "gen_secs": per_row_dt,
                })
    if gen_secs > 0 and remaining:
        print(f"[{cell}] {len(remaining)/gen_secs:.2f} problems/s  "
              f"over {gen_secs:.1f}s")
    return path


def run_b3_cell(model, tokenizer, cell, problems, start, end, results_dir,
                no_resume):
    path = shard_jsonl(results_dir, cell, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p) for i, p in enumerate(problems)
                 if (start + i) not in done]
    print(f"[{cell}] k={B3_K} len={B3_LEN} T={B3_TEMPERATURE}  {start}:{end}  "
          f"{len(done)} done, {len(remaining)} to generate  -> {path.name}")
    gen_secs = 0.0
    for i, prob in remaining:
        prompt = build_prompt(EXEMPLARS["cot"], prob, "cot")
        t0 = time.perf_counter()
        completions, n_gens, prompt_len = sample_k_chains(
            model, tokenizer, prompt, B3_K, B3_LEN, start + i,
        )
        dt = time.perf_counter() - t0
        gen_secs += dt
        chains = []
        preds = []
        for comp, n in zip(completions, n_gens):
            pred, hashed = extract(comp, prob["labels"])
            chains.append({"completion": comp, "pred": pred,
                           "answer_hit": int(hashed), "n_gen_tokens": n})
            preds.append(pred)
        voted, voted_count = majority_vote(preds)
        append_jsonl(path, {
            "idx": start + i,
            "id": prob["id"],
            "gold": prob["gold"],
            "orig_gold": prob["orig_gold"],
            "remapped": prob["remapped"],
            "n_choices": prob["n_choices"],
            "k": B3_K,
            "chains": chains,
            "voted_pred": voted,
            "voted_count": voted_count,
            "vote_degenerate": int(voted_count == B3_K),
            "correct": int(voted is not None and voted == prob["gold"]),
            "prompt_tokens": prompt_len,
            "gen_secs": dt,
        })
    if gen_secs > 0 and remaining:
        print(f"[{cell}] {len(remaining)/gen_secs:.2f} problems/s  "
              f"over {gen_secs:.1f}s")
    return path


# -----------------------------------------------------------------------------
# smoke
# -----------------------------------------------------------------------------

def verify_exemplar_hash():
    """The exemplar_hash printed here gets baked into manifest.json on the
    first cell run. If you edit EXEMPLARS_COT and re-run, the cells running
    after the edit will fail manifest_check; older shards are still on
    disk under the old hash."""
    h = exemplar_hash()
    print(f"exemplar_hash = {h}")


def run_smoke(model, tokenizer, cells_to_run, problems, start):
    import torch
    verify_exemplar_hash()
    if not problems:
        print("  (no problems in range; smoke skipped)")
        return
    ex0 = problems[0]
    for cell in cells_to_run:
        condition = "cot" if cell != DIRECT_CELL else "direct"
        max_new = (A3_LEN if cell == A3_CELL else
                   B3_LEN if cell == B3_CELL else DIRECT_LEN)
        prompt = build_prompt(EXEMPLARS[condition], ex0, condition)
        if cell == B3_CELL:
            comps, n_gens, _ = sample_k_chains(
                model, tokenizer, prompt, B3_K, B3_LEN, start,
            )
            preds = [extract(c, ex0["labels"])[0] for c in comps]
            voted, _ = majority_vote(preds)
            print(f"  SMOKE {cell} (sampled, k={B3_K})  idx={start}  "
                  f"chain_preds={preds}  voted={voted}  gold={ex0['gold']}")
            a, _, _ = sample_k_chains(model, tokenizer, prompt, B3_K, B3_LEN, start)
            b, _, _ = sample_k_chains(model, tokenizer, prompt, B3_K, B3_LEN, start)
            if a != b:
                print(f"ERROR: sampling determinism fail on {cell}.")
                sys.exit(1)
            print(f"  SAMPLING DETERMINISM OK ({cell})")
        else:
            comp, n_gen, _ = generate_greedy(model, tokenizer, prompt, max_new)
            pred, hashed = extract(comp, ex0["labels"])
            print(f"  SMOKE {cell} (greedy, len={max_new})  idx={start}  "
                  f"pred={pred}  gold={ex0['gold']}  answer_hit={hashed}  "
                  f"n_gen={n_gen}")
            print(f"    completion[:200] = {comp[:200]!r}")
            a, _, _ = generate_greedy(model, tokenizer, prompt, max_new)
            b, _, _ = generate_greedy(model, tokenizer, prompt, max_new)
            if a != b:
                print(f"ERROR: greedy determinism fail on {cell}.")
                sys.exit(1)
            print(f"  GREEDY DETERMINISM OK ({cell})")
    torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# FLOPs (analytical, same constants as plan 2)
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
# summarize: merge shards, compute metrics, outcome interpretation
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


def metrics_greedy(rows):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(r["correct"] for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    answer_hit = sum(r.get("answer_hit", 0) for r in rows) / n
    mean_gen = sum(r.get("n_gen_tokens", 0) for r in rows) / n
    prompt_len = rows[0].get("prompt_tokens", 0)
    flops = sum(flops_per_problem(r.get("prompt_tokens", prompt_len),
                                  r.get("n_gen_tokens", 0))
                for r in rows) / n
    mean_wall = sum(r.get("gen_secs", 0.0) for r in rows) / n
    return {
        "n": n, "correct": correct, "accuracy": acc,
        "ci95": [lo, hi], "answer_hit_rate": answer_hit,
        "mean_gen_tokens": mean_gen,
        "mean_flops_per_problem": flops,
        "mean_wallclock_sec": mean_wall,
    }


def metrics_voted(rows, k):
    n = len(rows)
    if n == 0:
        return None
    correct = sum(r["correct"] for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    vote_degen = sum(r.get("vote_degenerate", 0) for r in rows) / n
    chain_hit, chain_gen = [], []
    for r in rows:
        for ch in r.get("chains", []):
            chain_hit.append(ch.get("answer_hit", 0))
            chain_gen.append(ch.get("n_gen_tokens", 0))
    answer_hit = sum(chain_hit) / len(chain_hit) if chain_hit else 0.0
    mean_gen = sum(chain_gen) / len(chain_gen) if chain_gen else 0.0
    prompt_len = rows[0].get("prompt_tokens", 0)
    flops = sum(
        flops_per_problem(
            r.get("prompt_tokens", prompt_len),
            (sum(ch.get("n_gen_tokens", 0) for ch in r.get("chains", []))
             / max(1, k)),
            k=k,
        )
        for r in rows
    ) / n
    mean_wall = sum(r.get("gen_secs", 0.0) for r in rows) / n
    return {
        "n": n, "correct": correct, "accuracy": acc,
        "ci95": [lo, hi], "answer_hit_rate": answer_hit,
        "mean_gen_tokens": mean_gen,
        "vote_degeneracy_rate": vote_degen,
        "mean_flops_per_problem": flops,
        "mean_wallclock_sec": mean_wall,
    }


def ci_overlap(a_ci, b_ci):
    a_lo, a_hi = a_ci
    b_lo, b_hi = b_ci
    return not (a_lo > b_hi or b_lo > a_hi)


def determine_outcome(a3_m, b3_m, direct_m):
    """Pre-registered interpretation from plan4.md ("Pre-registered
    interpretation"). Returns (label, reason). The labels:
        A --- plateau confirmed
        B --- ARC-Easy shows SC response GSM8K hides
        C --- ARC-Easy saturates above 90% on both cells
        D --- (secondary) CoT doesn't beat direct on ARC-Easy
        AMBIGUOUS --- no decision-rule pattern matched
        INCOMPLETE --- A3 or B3 missing
    """
    if a3_m is None or b3_m is None:
        return "INCOMPLETE", (f"A3_arc or B3_arc missing "
                              f"(A3 n={a3_m['n'] if a3_m else 0}, "
                              f"B3 n={b3_m['n'] if b3_m else 0}).")
    a, b = a3_m["accuracy"], b3_m["accuracy"]
    a_ci, b_ci = a3_m["ci95"], b3_m["ci95"]
    diff = b - a
    overlap = ci_overlap(a_ci, b_ci)

    # Outcome C --- saturation (precedence first; the cell distinction
    # stops being meaningful when both clip the ceiling).
    if a >= SATURATION_THRESHOLD and b >= SATURATION_THRESHOLD:
        return "C", (f"A3_arc ({a:.3f}) and B3_arc ({b:.3f}) both >= "
                     f"{SATURATION_THRESHOLD}. Benchmark ceiling is too "
                     f"close for plan-4's plateau-vs-SC contrast to "
                     f"differentiate the cells. The four-paths writeup "
                     f"should flag this and consider adding ARC-Challenge "
                     f"to the shared rig.")
    # Outcome B --- ARC-Easy SC response GSM8K hides.
    if diff >= 0.05 and not overlap:
        return "B", (f"B3_arc ({b:.3f}) - A3_arc ({a:.3f}) = {diff:+.3f} "
                     f">= 0.05 with non-overlapping CIs. Self-consistency "
                     f"lifts ARC-Easy where it didn't lift GSM8K. Two "
                     f"mechanisms could explain this --- see plan4.md "
                     f"outcome B for the follow-up.")
    # Outcome A --- plateau confirmed.
    if abs(diff) <= 0.03 and overlap:
        return "A", (f"|B3_arc - A3_arc| = {abs(diff):.3f} <= 0.03 with "
                     f"overlapping CIs. Path 1's plateau holds on both "
                     f"GSM8K and ARC-Easy. The four-paths writeup can "
                     f"state inference-time compute doesn't help Path 1 "
                     f"on either benchmark.")
    return "AMBIGUOUS", (f"A3={a:.3f} ci=({a_ci[0]:.3f},{a_ci[1]:.3f}), "
                         f"B3={b:.3f} ci=({b_ci[0]:.3f},{b_ci[1]:.3f}), "
                         f"diff={diff:+.3f}, overlap={overlap}; outside "
                         f"plan-4's pre-registered patterns A/B/C.")


def determine_secondary_d(a3_m, direct_m):
    """Outcome D is independent of A/B/C and only meaningful when the
    direct cell ran. Returns None or a (label, reason) tuple."""
    if a3_m is None or direct_m is None:
        return None
    a = a3_m["accuracy"]
    d = direct_m["accuracy"]
    lift = a - d
    if lift < 0.10:
        return ("D", f"A3_arc ({a:.3f}) - direct_arc ({d:.3f}) = {lift:+.3f} "
                f"< 0.10. CoT does not give a 10-point lift over direct on "
                f"ARC-Easy, even though it does on GSM8K (~30 pts). Most "
                f"likely cause: 4-way MC + 8-shot priming makes direct-"
                f"answer trivially good. Less likely: CoT is GSM8K-specific. "
                f"Investigate before trusting; consider rerunning direct_arc "
                f"with max_new_tokens=4 to tighten format.")
    return None


def warn_sanity_thresholds(a3_m, direct_m):
    """Soft warnings on plan-4 sanity thresholds (sanity checks #3, #4, #5
    in plan4.md). All non-fatal --- they print to stdout and the script
    still completes."""
    if a3_m is not None:
        if a3_m["answer_hit_rate"] < ANSWER_HIT_FLOOR:
            print(f"WARNING: A3_arc answer_hit_rate "
                  f"{a3_m['answer_hit_rate']:.3f} < {ANSWER_HIT_FLOOR}. "
                  f"The model is not following the 'The answer is X' "
                  f"format reliably. Inspect a few completions before "
                  f"trusting accuracy.")
        a = a3_m["accuracy"]
        if a < A3_PLAUSIBLE_LO:
            print(f"WARNING: A3_arc accuracy {a:.3f} < {A3_PLAUSIBLE_LO} "
                  f"--- below plan-4's plausible range. Likely the prompt "
                  f"format is wrong. Check (prompt, completion) pairs.")
        elif a > A3_PLAUSIBLE_HI:
            print(f"NOTE: A3_arc accuracy {a:.3f} > {A3_PLAUSIBLE_HI} "
                  f"--- benchmark may be saturated. Outcome C will fire "
                  f"if B3_arc is also above {SATURATION_THRESHOLD}.")
    if direct_m is not None:
        d = direct_m["accuracy"]
        if d < DIRECT_PLAUSIBLE_LO:
            print(f"NOTE: direct_arc accuracy {d:.3f} < "
                  f"{DIRECT_PLAUSIBLE_LO}. Lower than expected for "
                  f"4-way MC + 8-shot priming; verify direct exemplars.")


def fmt_row(name, m, extra=""):
    if m is None:
        return f"{name:12s}  (no rows)"
    return (f"{name:12s}  acc={m['accuracy']:.3f}  "
            f"ci=({m['ci95'][0]:.3f},{m['ci95'][1]:.3f})  "
            f"answer_hit={m['answer_hit_rate']:.2f}  "
            f"mean_tokens={m['mean_gen_tokens']:.1f}{extra}  "
            f"n={m['n']}")


def summarize(results_dir):
    root = Path(results_dir)
    mpath = root / "manifest.json"
    if not mpath.exists():
        print(f"ERROR: no manifest at {mpath}. Run at least one cell first.")
        sys.exit(1)
    man = json.loads(mpath.read_text())

    a3_rows = load_shards(results_dir, A3_CELL)
    b3_rows = load_shards(results_dir, B3_CELL)
    direct_rows = load_shards(results_dir, DIRECT_CELL)
    a3_m = metrics_greedy(a3_rows)
    b3_m = metrics_voted(b3_rows, B3_K)
    direct_m = metrics_greedy(direct_rows)

    n_target = man.get("n_total", DEFAULT_N)
    print()
    print(f"ARC-Easy (n={n_target})")
    print(f"  {fmt_row(A3_CELL, a3_m, f'  len={A3_LEN}')}")
    b3_extra = f"  k={B3_K} len={B3_LEN}"
    if b3_m is not None:
        b3_extra += f"  vote_deg={b3_m['vote_degeneracy_rate']:.2f}"
    print(f"  {fmt_row(B3_CELL, b3_m, b3_extra)}")
    cells_in_manifest = [c["cell"] for c in man.get("cells", [])]
    if direct_m is not None or DIRECT_CELL in cells_in_manifest:
        print(f"  {fmt_row(DIRECT_CELL, direct_m, f'  len={DIRECT_LEN}')}")

    warn_sanity_thresholds(a3_m, direct_m)

    primary_label, primary_reason = determine_outcome(a3_m, b3_m, direct_m)
    secondary = determine_secondary_d(a3_m, direct_m)

    print()
    print(f"OUTCOME (primary): {primary_label} --- {primary_reason}")
    if secondary is not None:
        sl, sr = secondary
        print(f"OUTCOME (secondary): {sl} --- {sr}")

    out = {
        "config": {
            **man,
            "a3_plausible_range": [A3_PLAUSIBLE_LO, A3_PLAUSIBLE_HI],
            "answer_hit_floor": ANSWER_HIT_FLOOR,
            "saturation_threshold": SATURATION_THRESHOLD,
            "flops_formula": ("ff = 2*N_active*(prompt+completion); "
                              "attn = N_layers*N_heads*head_dim*"
                              "(prompt*completion + completion^2/2); "
                              "total = ff + 4*attn, times k for B3"),
            "flops_constants": {"N_active": N_ACTIVE_PARAMS,
                                "N_layers": N_LAYERS, "N_heads": N_HEADS,
                                "head_dim": HEAD_DIM},
        },
        "cells": {A3_CELL: a3_m, B3_CELL: b3_m, DIRECT_CELL: direct_m},
        "outcome": {"primary": {"label": primary_label,
                                 "reason": primary_reason},
                    "secondary": (None if secondary is None
                                  else {"label": secondary[0],
                                        "reason": secondary[1]})},
    }
    (root / "results_plan4.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root / 'results_plan4.json'}")

    write_csv(root, a3_m, b3_m, direct_m)
    plot_path = make_plot(root, a3_m, b3_m, direct_m, n_target,
                          primary_label, primary_reason, secondary)
    if plot_path is not None:
        print(f"wrote {plot_path}")


def write_csv(root, a3_m, b3_m, direct_m):
    path = root / "path1_plan4.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "accuracy", "ci_lo", "ci_hi",
                    "answer_hit_rate", "mean_gen_tokens",
                    "mean_flops_per_problem", "mean_wallclock_sec", "n"])
        for cell, m in [(A3_CELL, a3_m), (B3_CELL, b3_m),
                        (DIRECT_CELL, direct_m)]:
            if m is None:
                continue
            w.writerow([cell, f"{m['accuracy']:.6f}",
                        f"{m['ci95'][0]:.6f}", f"{m['ci95'][1]:.6f}",
                        f"{m['answer_hit_rate']:.6f}",
                        f"{m['mean_gen_tokens']:.6f}",
                        f"{m['mean_flops_per_problem']:.6e}",
                        f"{m['mean_wallclock_sec']:.6f}", m["n"]])
    print(f"wrote {path}")


def make_plot(root, a3_m, b3_m, direct_m, n_total, primary_label,
              primary_reason, secondary):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed; skipping plot)")
        return None
    cells = [(A3_CELL, a3_m, "#1f6fb4"),
             (B3_CELL, b3_m, "#c44e4e"),
             (DIRECT_CELL, direct_m, "#888888")]
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
                f"{m['accuracy']:.3f}\n(n={m['n']}, "
                f"hit={m['answer_hit_rate']:.2f})",
                ha="center", va="bottom", fontsize=8)

    # B3 - A3 lift annotation if both present.
    if a3_m is not None and b3_m is not None:
        lift = b3_m["accuracy"] - a3_m["accuracy"]
        ax.annotate(
            f"B3 - A3 = {lift:+.3f}",
            xy=(0.5 if len(cells) >= 2 else 0,
                max(a3_m["accuracy"], b3_m["accuracy"]) + 0.10),
            ha="center", fontsize=10, color="#333333", weight="bold",
        )

    # Reference bands: 25% chance line and 90% saturation line.
    ax.axhline(0.25, color="#999999", linestyle=":", linewidth=0.8)
    ax.text(len(cells) - 0.4, 0.255, "chance (25%)", fontsize=7,
            color="#777777")
    ax.axhline(SATURATION_THRESHOLD, color="#5fa55a", linestyle="--",
               linewidth=0.8, alpha=0.7)
    ax.text(len(cells) - 0.4, SATURATION_THRESHOLD + 0.005,
            f"saturation ({SATURATION_THRESHOLD:.0%})", fontsize=7,
            color="#3a7536")

    ax.set_xticks(xs)
    ax.set_xticklabels([name for name, _, _ in cells])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("ARC-Easy accuracy")
    title_extra = (f"  [secondary: {secondary[0]}]" if secondary else "")
    title = (f"Path 1 plan 4 --- ARC-Easy cross-benchmark validity "
             f"(n={n_total})\n"
             f"OUTCOME: {primary_label}{title_extra}")
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.text(0.01, 0.01, primary_reason[:240], fontsize=7,
             color="#555555", wrap=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = root / "results_plan4.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.summarize:
        summarize(args.results_dir)
        return

    print_env()
    verify_exemplar_hash()
    start, end = parse_range(args.problems, args.n)
    print(f"cells:     {args.cells}")
    print(f"problems:  [{start}, {end}) of n={args.n}")
    if args.n != DEFAULT_N:
        print(f"PREVIEW MODE (n={args.n} < {DEFAULT_N}): routing to "
              f"{args.results_dir}/")

    problems, gold_stats = load_problems(start, end, args.n)
    check_manifest(args.results_dir, args, gold_stats)

    import torch
    dtype = DTYPE_MAP[args.dtype]
    print(f"\nLoading {MODEL_ID} in {args.dtype} ...")
    tokenizer, model = load_model(MODEL_ID, dtype)
    verify_model_arch(model)

    try:
        if args.smoke:
            run_smoke(model, tokenizer, args.cells, problems, start)
        for cell in args.cells:
            if cell == A3_CELL:
                run_greedy_cell(
                    model, tokenizer, A3_CELL, "cot", A3_LEN,
                    problems, start, end,
                    args.results_dir, args.no_resume,
                    batch_size=args.batch_size,
                )
            elif cell == B3_CELL:
                run_b3_cell(
                    model, tokenizer, B3_CELL,
                    problems, start, end,
                    args.results_dir, args.no_resume,
                )
            elif cell == DIRECT_CELL:
                run_greedy_cell(
                    model, tokenizer, DIRECT_CELL, "direct", DIRECT_LEN,
                    problems, start, end,
                    args.results_dir, args.no_resume,
                    batch_size=args.batch_size,
                )
            else:
                print(f"WARNING: unknown cell {cell!r}; skipping.")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preview_suffix = f" --n {args.n}" if args.n != DEFAULT_N else ""
    print(f"\nDone. Run `python path1_arc_easy.py --summarize"
          f"{preview_suffix}` once all cells are populated.")


if __name__ == "__main__":
    main()
