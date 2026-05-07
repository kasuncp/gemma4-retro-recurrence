"""Path 1 --- plan 7: cross-benchmark validity on harder reasoning.

Plan 5 revealed C2 (zero-shot plain via chat template) beats A3 by 41.6 pp on
GSM8K. Plan 6 is running now. Plan 7 tests whether C2's prompt-format advantage
and the test-time-compute plateau hold on harder reasoning benchmarks where the
model is NOT already saturated:

    - ARC-Challenge (n=500, deterministic first test problems)
    - MATH (n=200, deterministic first test problems)
    - BBH-lite (n=500, curated subset of Big-Bench Hard)

Three cells per benchmark:

    C2_greedy : zero-shot plain, chat template, max_new_tokens=512, greedy
    A3_style  : 8-shot CoT in the same Wei et al. format as plan 2/4
    Direct    : 8-shot direct, max_new_tokens=16

Default --- one GPU, runs all 9 cells sequentially:
    python path1_harder_benchmarks.py

Quick preview:
    python path1_harder_benchmarks.py --n 20 --benchmark arc

Parallel on multiple GPUs (shard by --problems):
    CUDA_VISIBLE_DEVICES=0 python path1_harder_benchmarks.py --benchmark arc --problems 0:250
    CUDA_VISIBLE_DEVICES=1 python path1_harder_benchmarks.py --benchmark arc --problems 250:500
    # ... repeat for MATH and BBH-lite
    python path1_harder_benchmarks.py --summarize

Pre-registered outcome labels (A/B/C/D/E) are computed during --summarize.
See plan7.md for the full interpretation.
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

import probes
from probes.env import DTYPE_MAP, load_model, print_env
from path1_cot_gate import (
    GOLD_RE,
    FALLBACK_RE,
    append_jsonl,
    load_existing,
    wilson_ci,
)
from path1_zero_shot import (
    C2_CELL,
    build_chat_prompt as build_c2_prompt,
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
RESULTS_SUBDIR = "results/path_1_cot_tokens/plan7"
CELLS_SUBDIR = "cells"

BENCHMARKS = {
    "arc": {"n": 500, "name": "ARC-Challenge"},
    "math": {"n": 200, "name": "MATH"},
    "bbh": {"n": 500, "name": "BBH-lite"},
}

C2_CELL_NAME = "C2_greedy"
A3_STYLE_CELL = "A3_style"
DIRECT_CELL = "direct"

ALL_CELLS = [C2_CELL_NAME, A3_STYLE_CELL, DIRECT_CELL]

MAX_NEW_C2 = 512
MAX_NEW_DIRECT = 16

HASH_HIT_FLOOR = 0.30
FALLBACK_HIT_FLOOR = 0.85

OUTCOME_NEAR = 0.03
OUTCOME_WIN = 0.10
OUTCOME_BIG = 0.20


def parse_args():
    all_benchmarks = list(BENCHMARKS.keys())
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--benchmark", choices=all_benchmarks, default=None,
        help=f"Benchmark to run. Default all three. Use to parallelize.",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Override n for the selected benchmark. Auto-defaults per benchmark.",
    )
    p.add_argument(
        "--cells", nargs="+", choices=ALL_CELLS, default=ALL_CELLS,
        help=f"Cells to run. Default all three: {', '.join(ALL_CELLS)}.",
    )
    p.add_argument(
        "--problems", default=None,
        help="Problem range as START:END (half-open). Default 0:N. Used to shard.",
    )
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument(
        "--batch-size", type=int, default=1,
        help="Greedy batch size. Default 1 (byte-deterministic).",
    )
    p.add_argument(
        "--results-dir", default=RESULTS_SUBDIR,
        help="Directory for per-cell JSONL shards and final results_plan7.json.",
    )
    p.add_argument(
        "--summarize", action="store_true",
        help="Skip model loading; merge shards, compute metrics, write outputs.",
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Before full run, print one prompt/completion and verify determinism.",
    )
    p.add_argument(
        "--no-resume", action="store_true",
        help="Ignore existing JSONL rows, overwrite from scratch.",
    )
    args = p.parse_args()

    if args.benchmark is None:
        args.benchmarks_to_run = all_benchmarks
    else:
        args.benchmarks_to_run = [args.benchmark]

    for bm in args.benchmarks_to_run:
        n_default = BENCHMARKS[bm]["n"]
        if args.n is not None and args.benchmark is not None:
            args.n_per_benchmark = {bm: args.n}
        elif args.n is not None:
            p.error("--n requires --benchmark to be set")
        else:
            args.n_per_benchmark = {bm: BENCHMARKS[bm]["n"] for bm in args.benchmarks_to_run}

    if args.problems is None:
        args.problems = {bm: f"0:{args.n_per_benchmark[bm]}" for bm in args.benchmarks_to_run}

    return args


def parse_range(spec, upper):
    m = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", spec)
    if not m:
        raise ValueError(f"--problems must be START:END, got {spec!r}")
    start, end = int(m.group(1)), int(m.group(2))
    if not (0 <= start < end <= upper):
        raise ValueError(f"--problems range {start}:{end} out of bounds [0, {upper}]")
    return start, end


def load_arc_challenge(problems_dir=None):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rows = []
    for i, row in enumerate(ds):
        if i >= 500:
            break
        question = row["question"]
        choices_dict = row["choices"]
        choice_texts = choices_dict["text"]
        choice_labels = choices_dict["label"]
        options_text = "\n".join(f"{l}. {t}" for l, t in zip(choice_labels, choice_texts))
        full_q = f"{question}\n{options_text}"
        gold = row["answerKey"]
        rows.append({"idx": i, "question": full_q, "gold": gold})
    return rows


def load_math(problems_dir=None):
    from datasets import load_dataset
    ds = load_dataset("nlile/hendrycks-MATH-benchmark", split="test")
    rows = []
    for i, row in enumerate(ds):
        if i >= 200:
            break
        question = row["problem"]
        rows.append({"idx": i, "question": question, "gold": row["solution"]})
    return rows


def load_bbh_lite(problems_dir=None):
    from datasets import load_dataset
    ds = load_dataset("EleutherAI/bbh", split="test")
    tasks = ["boolean_expressions", "causal_judgment", "city", "colored_nodes",
             "formal_fallacies", "geometric_shapes", "logical_deduction",
             "movie_recommendation", "multistep_arithmetic_two", "navigate",
             "object_counting", "penguins", "reasoning_about_colored_objects",
             "ruin_words", "salient_translation_error_detection",
             "snarks", "sports_understanding", "temporal_sequences",
             "tracking_shuffled_objects", "web_of_lies"]
    rows = []
    idx = 0
    for task in tasks:
        task_ds = ds.filter(lambda x: x["task"] == task)
        for row in task_ds:
            if idx >= 500:
                break
            question = row["question"]
            rows.append({"idx": idx, "task": task, "question": question, "gold": row["answer"]})
            idx += 1
        if idx >= 500:
            break
    return rows[:500]


LOADERS = {
    "arc": load_arc_challenge,
    "math": load_math,
    "bbh": load_bbh_lite,
}


def build_c2_prompt_arc(tokenizer, question):
    return build_c2_prompt(tokenizer, C2_CELL, question)


def build_c2_prompt_math(tokenizer, question):
    return build_c2_prompt(tokenizer, C2_CELL, question)


def build_c2_prompt_bbh(tokenizer, question, task=None):
    return build_c2_prompt(tokenizer, C2_CELL, question)


EXEMPLARS_ARC = [
    {
        "id": "exemplar_1",
        "question": "Which factor will most likely cause a person to develop a fever?",
        "labels": ["A", "B", "C", "D"],
        "texts": [
            "a leg muscle relaxing after exercise",
            "a bacterial population in the bloodstream",
            "several viral particles on the skin",
            "carbohydrates being digested in the stomach",
        ],
        "gold": "B",
        "reasoning": ("Fever is the body's response to an internal infection. "
                      "Muscle relaxation (A) and digestion (D) are normal processes "
                      "that don't trigger fever. Viral particles on the skin (C) "
                      "haven't entered the body. A bacterial population in the "
                      "bloodstream (B) provokes the immune response that produces fever."),
    },
    {
        "id": "exemplar_2",
        "question": "Lichens are symbiotic organisms made of green algae and fungi. "
                    "What do the green algae supply to the fungi in this symbiotic relationship?",
        "labels": ["A", "B", "C", "D"],
        "texts": ["carbon dioxide", "food", "protection", "water"],
        "gold": "B",
        "reasoning": ("In a lichen, the green algae photosynthesize and produce sugars; "
                      "the fungus provides structure and absorbs water. So the algae "
                      "supply food (sugars) to the fungi."),
    },
    {
        "id": "exemplar_3",
        "question": "When a switch is used in an electrical circuit, the switch can",
        "labels": ["A", "B", "C", "D"],
        "texts": [
            "cause the charge to build",
            "increase and decrease the voltage",
            "cause the current to change direction",
            "stop and start the flow of current",
        ],
        "gold": "D",
        "reasoning": ("A switch is a two-state device. It opens or closes the circuit, "
                      "which stops or starts the flow of current."),
    },
    {
        "id": "exemplar_4",
        "question": "Which of the following is an example of an assistive device?",
        "labels": ["A", "B", "C", "D"],
        "texts": ["contact lens", "motorcycle", "raincoat", "coffee pot"],
        "gold": "A",
        "reasoning": ("An assistive device helps a person overcome a physical limitation. "
                      "A contact lens corrects vision impairment."),
    },
    {
        "id": "exemplar_5",
        "question": "Rocks are classified as igneous, metamorphic, or sedimentary according to",
        "labels": ["A", "B", "C", "D"],
        "texts": ["their color", "their shape", "how they formed", "the minerals they contain"],
        "gold": "C",
        "reasoning": ("Igneous, metamorphic, and sedimentary are categories defined by "
                      "formation process."),
    },
    {
        "id": "exemplar_6",
        "question": "A chewable calcium carbonate tablet is a common treatment for stomach "
                    "discomfort. Calcium carbonate is most likely used as this type of "
                    "medicine because calcium carbonate",
        "labels": ["A", "B", "C", "D"],
        "texts": ["has a pleasant flavor", "is inexpensive to produce",
                  "neutralizes digestive acid", "occurs naturally in the body"],
        "gold": "C",
        "reasoning": ("Calcium carbonate is a base that reacts with acid; this neutralizes "
                      "the acid and relieves the discomfort."),
    },
    {
        "id": "exemplar_7",
        "question": "Which two body systems are directly involved in movement?",
        "labels": ["A", "B", "C", "D"],
        "texts": ["muscular and skeletal", "digestive and muscular",
                  "skeletal and respiratory", "respiratory and digestive"],
        "gold": "A",
        "reasoning": ("Bones provide the framework, and muscles contract to pull on the bones."),
    },
    {
        "id": "exemplar_8",
        "question": "Which change in the state of water particles causes the particles "
                    "to become arranged in a fixed position?",
        "labels": ["A", "B", "C", "D"],
        "texts": ["boiling", "melting", "freezing", "evaporating"],
        "gold": "C",
        "reasoning": ("When liquid water freezes into ice, the water particles arrange "
                      "into a rigid lattice in fixed positions."),
    },
]


def build_a3_style_prompt_arc(tokenizer, question):
    prompt = "Solve the following ARC challenge problem step by step.\n\n"
    for ex in EXEMPLARS_ARC:
        opts = "\n".join(f"{l}. {t}" for l, t in zip(ex["labels"], ex["texts"]))
        prompt += f"Problem:\n{ex['question']}\n{opts}\n"
        prompt += f"Chain of Thought: {ex['reasoning']}\n"
        prompt += f"The answer is {ex['gold']}.\n\n"
    opts = "\n".join(f"{l}. {t}" for l, t in zip(["A", "B", "C", "D"], ["?", "?", "?", "?"]))
    prompt += f"Problem:\n{question}\n{opts}\n"
    prompt += "Chain of Thought:"
    return prompt


def build_direct_prompt_arc(tokenizer, question):
    prompt = "Solve the following ARC challenge problem.\n\n"
    for ex in EXEMPLARS_ARC:
        opts = "\n".join(f"{l}. {t}" for l, t in zip(ex["labels"], ex["texts"]))
        prompt += f"Problem:\n{ex['question']}\n{opts}\n"
        prompt += f"The answer is {ex['gold']}.\n\n"
    opts = "\n".join(f"{l}. {t}" for l, t in zip(["A", "B", "C", "D"], ["?", "?", "?", "?"]))
    prompt += f"Problem:\n{question}\n{opts}\n"
    prompt += "The answer is"
    return prompt


EXEMPLARS_MATH = [
    {
        "id": "math_ex_1",
        "question": "If $3x + 7 = 22$, what is $x$?",
        "gold": "5",
        "reasoning": ("Subtract 7 from both sides: 3x = 15. Divide by 3: x = 5."),
    },
    {
        "id": "math_ex_2",
        "question": "What is the area of a rectangle with length 8 and width 5?",
        "gold": "40",
        "reasoning": ("Area = length × width = 8 × 5 = 40."),
    },
    {
        "id": "math_ex_3",
        "question": "Simplify: $2^3 \\times 2^2$",
        "gold": "32",
        "reasoning": ("2^3 × 2^2 = 2^(3+2) = 2^5 = 32."),
    },
    {
        "id": "math_ex_4",
        "question": "Find the square root of 144.",
        "gold": "12",
        "reasoning": ("12 × 12 = 144, so sqrt(144) = 12."),
    },
    {
        "id": "math_ex_5",
        "question": "What is 25% of 80?",
        "gold": "20",
        "reasoning": ("25% = 1/4, so 80/4 = 20."),
    },
    {
        "id": "math_ex_6",
        "question": "If a car travels 240 miles in 4 hours, what is its average speed?",
        "gold": "60",
        "reasoning": ("Speed = distance/time = 240/4 = 60 mph."),
    },
    {
        "id": "math_ex_7",
        "question": "What is the perimeter of a square with side length 9?",
        "gold": "36",
        "reasoning": ("Perimeter = 4 × side = 4 × 9 = 36."),
    },
    {
        "id": "math_ex_8",
        "question": "Solve for y: 2y + 3 = 11",
        "gold": "4",
        "reasoning": ("Subtract 3: 2y = 8. Divide by 2: y = 4."),
    },
]


def build_a3_style_prompt_math(tokenizer, question):
    prompt = "Solve the following math problem step by step. Put your final answer in \\boxed{} format.\n\n"
    for ex in EXEMPLARS_MATH:
        prompt += f"Problem: {ex['question']}\n"
        prompt += f"Solution: {ex['reasoning']} Therefore, \\boxed{{{ex['gold']}}}\n\n"
    prompt += f"Problem: {question}\n"
    prompt += "Solution:"
    return prompt


def build_direct_prompt_math(tokenizer, question):
    prompt = "Solve the following math problem. Put your final answer in \\boxed{} format.\n\n"
    for ex in EXEMPLARS_MATH:
        prompt += f"Problem: {ex['question']}\nAnswer: \\boxed{{{ex['gold']}}}\n\n"
    prompt += f"Problem: {question}\nAnswer:"
    return prompt


EXEMPLARS_BBH = [
    {
        "id": "bbh_ex_1",
        "task": "boolean_expressions",
        "question": "not (True and False)",
        "gold": "True",
        "reasoning": ("First evaluate True and False = False. Then not False = True."),
    },
    {
        "id": "bbh_ex_2",
        "task": "causal_judgment",
        "question": "A person deliberately floods their neighbor's house. Was this caused by nature or human action?",
        "gold": "human action",
        "reasoning": ("Deliberate action by a person is human action, not natural causation."),
    },
    {
        "id": "bbh_ex_3",
        "task": "navigate",
        "question": "If facing north, turn left, then turn right, then turn left. What direction are you facing now?",
        "gold": "west",
        "reasoning": ("North -> turn left = West -> turn right = North -> turn left = West."),
    },
]


def build_a3_style_prompt_bbh(tokenizer, question, task=None):
    prompt = "Solve the following task step by step.\n\n"
    for ex in EXEMPLARS_BBH:
        prompt += f"Task ({ex['task']}): {ex['question']}\n"
        prompt += f"Chain of Thought: {ex['reasoning']}\n"
        prompt += f"Answer: {ex['gold']}\n\n"
    if task:
        prompt += f"Task ({task}): {question}\n"
    else:
        prompt += f"Task: {question}\n"
    prompt += "Chain of Thought:"
    return prompt


def build_direct_prompt_bbh(tokenizer, question, task=None):
    prompt = "Solve the following task.\n\n"
    for ex in EXEMPLARS_BBH:
        prompt += f"Task ({ex['task']}): {ex['question']}\n"
        prompt += f"Answer: {ex['gold']}\n\n"
    if task:
        prompt += f"Task ({task}): {question}\n"
    else:
        prompt += f"Task: {question}\n"
    prompt += "Answer:"
    return prompt


PROMPT_BUILDERS = {
    "arc": {
        C2_CELL_NAME: build_c2_prompt_arc,
        A3_STYLE_CELL: build_a3_style_prompt_arc,
        DIRECT_CELL: build_direct_prompt_arc,
    },
    "math": {
        C2_CELL_NAME: build_c2_prompt_math,
        A3_STYLE_CELL: build_a3_style_prompt_math,
        DIRECT_CELL: build_direct_prompt_math,
    },
    "bbh": {
        C2_CELL_NAME: build_c2_prompt_bbh,
        A3_STYLE_CELL: build_a3_style_prompt_bbh,
        DIRECT_CELL: build_direct_prompt_bbh,
    },
}


def extract_arc(text):
    m = re.search(r"[Tt]he answer is\s*\(?([A-D])\)?", text)
    if m:
        return m.group(1), True, False
    m = re.search(r"\b([A-D])\b", text)
    if m:
        return m.group(1), False, True
    return None, False, False


def extract_math(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        return m.group(1).strip(), True, False
    nums = re.findall(r"-?\d+\.?\d*", text)
    if nums:
        return nums[-1], False, True
    return None, False, False


def extract_bbh(text):
    text_clean = text.strip()
    for ans in ["True", "False", "true", "false"]:
        if text_clean.endswith(ans):
            return ans, True, False
    m = re.search(r"\b(True|False)\b", text, re.IGNORECASE)
    if m:
        return m.group(1), False, True
    return None, False, False


EXTRACTORS = {
    "arc": extract_arc,
    "math": extract_math,
    "bbh": extract_bbh,
}


def manifest(args):
    cells = []
    for bm in args.benchmarks_to_run:
        n = args.n_per_benchmark[bm]
        for cell in ALL_CELLS:
            max_new = MAX_NEW_C2 if cell == C2_CELL_NAME else MAX_NEW_DIRECT
            cells.append({
                "benchmark": bm,
                "cell": f"{bm}_{cell}",
                "n": n,
                "max_new_tokens": max_new,
                "decode": "greedy",
            })
    return {
        "plan": "path_1_cot_tokens/plan7",
        "model_id": MODEL_ID,
        "model_key": MODEL_KEY,
        "dtype": args.dtype,
        "cells": cells,
        "uses_chat_template": (C2_CELL_NAME in ALL_CELLS),
    }


def check_manifest(results_dir, args):
    p = Path(results_dir) / "manifest.json"
    desired = manifest(args)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(desired, indent=2))
        return desired
    existing = json.loads(p.read_text())
    existing["n_total"] = max(existing.get("n_total", 0), max(args.n_per_benchmark.values()))
    p.write_text(json.dumps(existing, indent=2))
    return existing


def shard_jsonl(results_dir, benchmark, cell, start, end):
    return Path(results_dir) / CELLS_SUBDIR / f"{benchmark}_{cell}__{start:04d}_{end:04d}.jsonl"


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
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
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
    import torch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
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


def run_cell(model, tokenizer, benchmark, cell, problems, start, end,
             results_dir, no_resume, batch_size, max_new):
    path = shard_jsonl(results_dir, benchmark, cell, start, end)
    if no_resume and path.exists():
        path.unlink()
    _, done = load_existing(path)
    remaining = [(i, p) for i, p in enumerate(problems) if (start + i) not in done]

    print(f"[{benchmark}_{cell}] max_new={max_new}  {start}:{end}  "
          f"{len(done)} done, {len(remaining)} to generate  -> {path.name}")

    builder = PROMPT_BUILDERS[benchmark][cell]
    extractor = EXTRACTORS[benchmark]

    gen_secs = 0.0
    if batch_size <= 1:
        for i, prob in remaining:
            if benchmark == "bbh":
                prompt = builder(tokenizer, prob["question"], prob.get("task"))
            else:
                prompt = builder(tokenizer, prob["question"])
            t0 = time.perf_counter()
            completion, n_gen, prompt_len = generate_greedy(model, tokenizer, prompt, max_new)
            dt = time.perf_counter() - t0
            gen_secs += dt

            pred, hashed, fallback = extractor(completion)
            gold = prob["gold"]
            correct = int(pred is not None and pred == gold)

            append_jsonl(path, {
                "idx": start + i,
                "gold": gold,
                "pred": pred,
                "correct": correct,
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
            if benchmark == "bbh":
                prompts = [builder(tokenizer, p["question"], p.get("task")) for _, p in batch]
            else:
                prompts = [builder(tokenizer, p["question"]) for _, p in batch]
            t0 = time.perf_counter()
            per_row = generate_greedy_batch(model, tokenizer, prompts, max_new)
            dt = time.perf_counter() - t0
            gen_secs += dt
            per_row_dt = dt / len(batch)

            for (i, prob), (completion, n_gen, prompt_len) in zip(batch, per_row):
                pred, hashed, fallback = extractor(completion)
                gold = prob["gold"]
                correct = int(pred is not None and pred == gold)

                append_jsonl(path, {
                    "idx": start + i,
                    "gold": gold,
                    "pred": pred,
                    "correct": correct,
                    "hash_hit": int(hashed),
                    "fallback_hit": int(fallback),
                    "completion": completion,
                    "n_gen_tokens": n_gen,
                    "prompt_tokens": prompt_len,
                    "gen_secs": per_row_dt,
                })

    if gen_secs > 0 and remaining:
        print(f"[{benchmark}_{cell}] {len(remaining)/gen_secs:.2f} problems/s over {gen_secs:.1f}s")
    return path


def run_smoke(model, tokenizer, benchmark, cell, problems, start):
    import torch
    if not problems:
        print("  (no problems in range; smoke skipped)")
        return
    builder = PROMPT_BUILDERS[benchmark][cell]
    if benchmark == "bbh":
        prompt = builder(tokenizer, problems[0]["question"], problems[0].get("task"))
    else:
        prompt = builder(tokenizer, problems[0]["question"])
    head = prompt[:200].replace("\n", "\\n")
    print(f"  SMOKE {benchmark}_{cell} prompt head: {head!r}")

    max_new = MAX_NEW_C2 if cell == C2_CELL_NAME else MAX_NEW_DIRECT
    comp, n_gen, _ = generate_greedy(model, tokenizer, prompt, max_new)
    print(f"  SMOKE {benchmark}_{cell} completion[:200] = {comp[:200]!r}")

    extractor = EXTRACTORS[benchmark]
    pred, _, _ = extractor(comp)
    print(f"  SMOKE {benchmark}_{cell} pred={pred} gold={problems[0]['gold']}")


def load_shards(results_dir, benchmark, cell):
    cells_dir = Path(results_dir) / CELLS_SUBDIR
    shards = sorted(cells_dir.glob(f"{benchmark}_{cell}__*.jsonl"))
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
    correct = sum(int(r.get("correct", 0)) for r in rows)
    acc = correct / n
    lo, hi = wilson_ci(correct, n)
    hash_hit = sum(int(r.get("hash_hit", 0)) for r in rows) / n
    no_extract = sum(int(r.get("pred") is None) for r in rows) / n
    fallback_hit = 1.0 - no_extract
    mean_gen = sum(int(r.get("n_gen_tokens", 0)) for r in rows) / n
    prompt_lens = [int(r.get("prompt_tokens", 0)) for r in rows]
    mean_prompt = sum(prompt_lens) / max(1, len(prompt_lens))
    flops = sum(flops_per_problem(int(r.get("prompt_tokens", 0)), int(r.get("n_gen_tokens", 0)))
                for r in rows) / n
    mean_wall = sum(float(r.get("gen_secs", 0.0)) for r in rows) / n
    return {
        "n": n, "correct": correct, "accuracy": acc,
        "ci95": [lo, hi],
        "hash_hit_rate": hash_hit,
        "fallback_hit_rate": fallback_hit,
        "no_extract_rate": no_extract,
        "mean_gen_tokens": mean_gen,
        "mean_prompt_tokens": mean_prompt,
        "mean_flops_per_problem": flops,
        "mean_wallclock_sec": mean_wall,
    }


def ci_overlap(a_ci, b_ci):
    a_lo, a_hi = a_ci
    b_lo, b_hi = b_ci
    return not (a_lo > b_hi or b_lo > a_hi)


def mcnemar(c2_correct, a3_correct, n):
    b_and_a = sum(1 for i in range(n) if c2_correct[i] and not a3_correct[i])
    a_and_b = sum(1 for i in range(n) if not c2_correct[i] and a3_correct[i])
    if b_and_a + a_and_b == 0:
        return 0.0
    stat = (abs(b_and_a - a_and_b) - 1) ** 2 / (b_and_a + a_and_b)
    return stat


def determine_outcome(benchmark_metrics):
    outcomes = {}
    for bm in ["arc", "math", "bbh"]:
        m = benchmark_metrics.get(bm)
        if m is None:
            outcomes[bm] = ("INCOMPLETE", "No metrics computed")
            continue

        c2 = m.get(f"{bm}_{C2_CELL_NAME}")
        a3 = m.get(f"{bm}_{A3_STYLE_CELL}")
        direct = m.get(f"{bm}_{DIRECT_CELL}")

        if c2 is None or a3 is None:
            outcomes[bm] = ("INCOMPLETE", "Missing C2 or A3 cell")
            continue

        c2_acc = c2["accuracy"]
        a3_acc = a3["accuracy"]
        diff = c2_acc - a3_acc

        if diff >= OUTCOME_WIN and not ci_overlap(c2["ci95"], a3["ci95"]):
            label = "A" if diff >= OUTCOME_BIG else "B"
            outcomes[bm] = (label, f"C2 ({c2_acc:.3f}) >= A3 ({a3_acc:.3f}) + {OUTCOME_WIN:.2f}")
        elif diff <= -OUTCOME_WIN and not ci_overlap(c2["ci95"], a3["ci95"]):
            outcomes[bm] = ("E", f"C2 ({c2_acc:.3f}) < A3 ({a3_acc:.3f}) - {OUTCOME_WIN:.2f}")
        elif abs(diff) <= OUTCOME_NEAR:
            outcomes[bm] = ("B", f"C2 ≈ A3 within {OUTCOME_NEAR:.2f}")
        else:
            outcomes[bm] = ("AMBIGUOUS", f"C2-A3 diff = {diff:+.3f}")

        if direct and a3:
            if direct["accuracy"] >= a3["accuracy"] - 0.05:
                outcomes[bm] = (outcomes[bm][0] + "_SATURATED", outcomes[bm][1] + f" | Direct≈CoT")

    return outcomes


def summarize(results_dir):
    root = Path(results_dir)
    mpath = root / "manifest.json"
    if not mpath.exists():
        print(f"ERROR: no manifest at {mpath}. Run at least one cell first.")
        sys.exit(1)

    benchmark_metrics = {}
    for bm in ["arc", "math", "bbh"]:
        benchmark_metrics[bm] = {}
        for cell in ALL_CELLS:
            rows = load_shards(results_dir, bm, cell)
            benchmark_metrics[bm][f"{bm}_{cell}"] = metrics_for_cell(rows)

    print("\n=== Per-benchmark results ===")
    for bm, metrics in benchmark_metrics.items():
        print(f"\n{bm.upper()} ({BENCHMARKS[bm]['name']})")
        for cell in ALL_CELLS:
            m = metrics.get(f"{bm}_{cell}")
            if m is None:
                print(f"  {cell}: (no rows)")
            else:
                print(f"  {cell}: acc={m['accuracy']:.3f} ci=({m['ci95'][0]:.3f},{m['ci95'][1]:.3f}) n={m['n']}")

    outcomes = determine_outcome(benchmark_metrics)
    print("\n=== Outcomes ===")
    for bm, (label, reason) in outcomes.items():
        print(f"{bm}: {label} - {reason}")

    out = {
        "config": json.loads(mpath.read_text()),
        "benchmark_metrics": benchmark_metrics,
        "outcomes": outcomes,
    }
    (root / "results_plan7.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {root / 'results_plan7.json'}")

    write_csv(root, benchmark_metrics)


def write_csv(root, benchmark_metrics):
    path = root / "path1_cross_benchmark.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["benchmark", "cell", "accuracy", "ci_lo", "ci_hi",
                    "mean_flops_per_problem", "mean_wallclock_sec",
                    "hash_hit_rate", "fallback_hit_rate", "n"])
        for bm, metrics in benchmark_metrics.items():
            for cell in ALL_CELLS:
                m = metrics.get(f"{bm}_{cell}")
                if m is None:
                    continue
                w.writerow([bm, f"{bm}_{cell}", f"{m['accuracy']:.6f}",
                            f"{m['ci95'][0]:.6f}", f"{m['ci95'][1]:.6f}",
                            f"{m['mean_flops_per_problem']:.6e}",
                            f"{m['mean_wallclock_sec']:.6f}",
                            f"{m['hash_hit_rate']:.6f}",
                            f"{m['fallback_hit_rate']:.6f}",
                            m["n"]])
    print(f"wrote {path}")


def main():
    args = parse_args()
    print_env()

    if args.summarize:
        summarize(args.results_dir)
        return

    check_manifest(args.results_dir, args)

    tokenizer, model = load_model(MODEL_ID, args.dtype)
    verify_model_arch(model)

    for benchmark in args.benchmarks_to_run:
        n = args.n_per_benchmark[benchmark]
        spec = args.problems[benchmark] if isinstance(args.problems, dict) else args.problems
        start, end = parse_range(spec, n)
        print(f"\n=== Loading {benchmark} problems ({n} total, running {start}:{end}) ===")
        problems = LOADERS[benchmark]()

        if args.smoke:
            for cell in args.cells:
                run_smoke(model, tokenizer, benchmark, cell, problems[start:end], start)

        for cell in args.cells:
            max_new = MAX_NEW_C2 if cell == C2_CELL_NAME else MAX_NEW_DIRECT
            run_cell(model, tokenizer, benchmark, cell, problems, start, end,
                     args.results_dir, args.no_resume, args.batch_size, max_new)

    print("\nDone. Run with --summarize to compute metrics.")


if __name__ == "__main__":
    main()