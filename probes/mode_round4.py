"""Round 4 --- zero-shot reasoning benchmark check.

Rounds 1-3c characterised the perplexity landscape of looping pretrained
Gemma 4 E2B. Perplexity is a language-modelling metric; the target use
case is mobile reasoning. This round probes whether the architecturally
stable recurrent configurations (blocks D and G) retain reasoning ability
zero-shot, *before* any retrofit training --- so we can pick a bucket
(plans/plan4.md) without burning training compute.

Scope:
  * 7 configurations: baseline, D-r4, D-r8, G-r4, G-r8, A-r8, D-r1.
  * GSM8K main test (first 250 problems) --- primary signal.
  * ARC-Easy test (first 200 problems) --- optional secondary axis.
  * Greedy decoding, ``use_cache=False`` globally (the loop hooks
    re-enter decoder layers, which breaks incremental KV caches on
    sliding-attention layers --- same reason perplexity runs disable
    the cache).
  * No few-shot, no chain-of-thought prompt ("Question: ...\nAnswer:").
  * Per-problem raw outputs are saved so we can recompute agreement
    matrices without re-running the sweep.

Sanity check: D-r1 installs the block-loop hook at r=1, which does zero
extra iterations and must produce bitwise-identical generations to the
unhooked baseline. If not, stop and debug the hook before trusting the
other configurations (plan4 Bucket 5).
"""

import itertools
import re
import sys

import torch

from .env import _results_path, _write_results_json
from .hooks import install_block_loop_hooks
from .introspect import _inspect_and_require_strategy_a


# Plan4 canonical configuration set. "block" is None for the baseline
# (no hook installed); for every other config the block-loop hook is
# installed over the named layer range at the given r.
CONFIGS = [
    {"name": "baseline", "block": None,     "r": 1},
    {"name": "D-r4",     "block": (15, 22), "r": 4},
    {"name": "D-r8",     "block": (15, 22), "r": 8},
    {"name": "G-r4",     "block": (15, 24), "r": 4},
    {"name": "G-r8",     "block": (15, 24), "r": 8},
    {"name": "A-r8",     "block": (15, 19), "r": 8},
    {"name": "D-r1",     "block": (15, 22), "r": 1},
]


# Flexible-extract regex for GSM8K numeric answers. Matches optional
# leading minus, then digits possibly separated by commas (thousands),
# optionally followed by a decimal fraction. Mirrors lm-eval-harness'
# "flexible-extract" regex closely enough for zero-shot greedy outputs.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _parse_gsm8k_number(text):
    """Return the last number in ``text`` as float, or None if absent."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    raw = matches[-1].replace(",", "").rstrip(".")
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_gsm8k_gold(reference_solution):
    """Extract the gold numeric answer from a GSM8K reference solution
    (delimited by '####'). Returns float; raises on malformed rows."""
    tail = reference_solution.split("####")[-1].strip().replace(",", "")
    return float(tail)


def _parse_mc_answer(text, valid_labels):
    """Return the first character in ``text`` that is a valid label,
    or None. Handles arbitrary label alphabets (ARC has a mix of
    A/B/C/D, A/B/C/D/E, and 1/2/3/4)."""
    valid_set = set(valid_labels)
    for ch in text:
        if ch in valid_set:
            return ch
    return None


def _load_gsm8k(n):
    from datasets import load_dataset

    print(f"Loading GSM8K (openai/gsm8k, config='main', split='test') ...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        problems.append({
            "idx": i,
            "question": row["question"],
            "reference": row["answer"],
            "gold": _parse_gsm8k_gold(row["answer"]),
        })
    print(f"  loaded {len(problems)} GSM8K problems.")
    return problems


def _load_arc_easy(n):
    from datasets import load_dataset

    print(f"Loading ARC-Easy (allenai/ai2_arc, config='ARC-Easy', split='test') ...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    problems = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        problems.append({
            "idx": i,
            "question": row["question"],
            "choices": {
                "label": list(row["choices"]["label"]),
                "text": list(row["choices"]["text"]),
            },
            "gold": row["answerKey"],
        })
    print(f"  loaded {len(problems)} ARC-Easy problems.")
    return problems


def _format_gsm8k_prompt(problem):
    return f"Question: {problem['question']}\nAnswer:"


def _format_arc_prompt(problem):
    lines = [f"Question: {problem['question']}", "Options:"]
    for label, text in zip(problem["choices"]["label"], problem["choices"]["text"]):
        lines.append(f"{label}. {text}")
    lines.append("Answer:")
    return "\n".join(lines)


def _generate(model, tokenizer, prompt, max_new_tokens):
    """Greedy decode, no sampling, no KV cache. Returns the generated text
    only (not including the prompt).

    Gemma 4's saved generation_config carries sampling defaults (top_p,
    top_k, temperature). With do_sample=False those are silently ignored
    but transformers prints a noisy "flags not valid" warning per call.
    We clone the config and null those fields so the warning never fires;
    the model's own config on disk is left untouched.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    from copy import deepcopy
    gen_cfg = deepcopy(model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.temperature = None

    with torch.no_grad():
        out = model.generate(
            **enc,
            generation_config=gen_cfg,
            max_new_tokens=max_new_tokens,
            use_cache=False,
            pad_token_id=pad_id,
        )
    gen_ids = out[0, input_len:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def _install_config_hook(decoder_layers, cfg):
    """Install the block-loop hook for ``cfg`` or return a no-op uninstall
    for the unhooked baseline."""
    block = cfg["block"]
    if block is None:
        return lambda: None
    start, end = block
    return install_block_loop_hooks(decoder_layers, start, end, r=cfg["r"])


def _run_gsm8k(cfg, model, tokenizer, decoder_layers, problems, max_new_tokens):
    uninstall = _install_config_hook(decoder_layers, cfg)
    results = []
    n_correct = 0
    try:
        for p in problems:
            prompt = _format_gsm8k_prompt(p)
            generated = _generate(model, tokenizer, prompt, max_new_tokens)
            predicted = _parse_gsm8k_number(generated)
            correct = (
                predicted is not None
                and p["gold"] is not None
                and abs(predicted - p["gold"]) < 1e-4
            )
            if correct:
                n_correct += 1
            results.append({
                "idx": p["idx"],
                "question": p["question"],
                "gold": p["gold"],
                "generated": generated,
                "parsed_answer": predicted,
                "correct": bool(correct),
            })
            if (len(results) % 25) == 0 or len(results) == len(problems):
                print(
                    f"    [{cfg['name']}] gsm8k {len(results)}/{len(problems)}  "
                    f"running accuracy={n_correct / len(results):.3f}"
                )
    finally:
        uninstall()
    accuracy = n_correct / len(problems) if problems else 0.0
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(problems),
            "per_problem": results}


def _run_arc_easy(cfg, model, tokenizer, decoder_layers, problems, max_new_tokens):
    uninstall = _install_config_hook(decoder_layers, cfg)
    results = []
    n_correct = 0
    try:
        for p in problems:
            prompt = _format_arc_prompt(p)
            generated = _generate(model, tokenizer, prompt, max_new_tokens)
            predicted = _parse_mc_answer(generated, p["choices"]["label"])
            correct = predicted is not None and predicted == p["gold"]
            if correct:
                n_correct += 1
            results.append({
                "idx": p["idx"],
                "question": p["question"],
                "gold": p["gold"],
                "generated": generated,
                "parsed_answer": predicted,
                "correct": bool(correct),
            })
            if (len(results) % 25) == 0 or len(results) == len(problems):
                print(
                    f"    [{cfg['name']}] arc    {len(results)}/{len(problems)}  "
                    f"running accuracy={n_correct / len(results):.3f}"
                )
    finally:
        uninstall()
    accuracy = n_correct / len(problems) if problems else 0.0
    return {"accuracy": accuracy, "n_correct": n_correct, "n_total": len(problems),
            "per_problem": results}


def _agreement_matrix(per_config_results):
    """Build the pairwise agreement matrix described in plan4:
      { (cfg_A, cfg_B): {both_correct, only_A_correct, only_B_correct, both_wrong} }
    Assumes per_config_results preserves problem order (it does --- we iterate
    the same ``problems`` list per config).
    """
    names = list(per_config_results.keys())
    matrix = {}
    for a, b in itertools.combinations(names, 2):
        ra = [row["correct"] for row in per_config_results[a]]
        rb = [row["correct"] for row in per_config_results[b]]
        if len(ra) != len(rb):
            # Should not happen --- every config runs the same problems ---
            # but guard against silent truncation.
            raise RuntimeError(
                f"agreement matrix: {a} has {len(ra)} problems, {b} has {len(rb)}"
            )
        both_c = sum(1 for x, y in zip(ra, rb) if x and y)
        only_a = sum(1 for x, y in zip(ra, rb) if x and not y)
        only_b = sum(1 for x, y in zip(ra, rb) if not x and y)
        both_w = sum(1 for x, y in zip(ra, rb) if not x and not y)
        matrix[f"{a}__vs__{b}"] = {
            "a": a, "b": b,
            "both_correct": both_c,
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "both_wrong": both_w,
        }
    return matrix


def _unique_correct_vs_baseline(config_name, per_config_results):
    """Count problems ``config_name`` got right that baseline got wrong."""
    if "baseline" not in per_config_results or config_name == "baseline":
        return None
    base = per_config_results["baseline"]
    mine = per_config_results[config_name]
    return sum(
        1 for m, b in zip(mine, base) if m["correct"] and not b["correct"]
    )


def _d_r1_bitwise_match(per_config_results):
    """Compare D-r1 generated strings against baseline. Hook must be a
    bitwise no-op at r=1; anything else means plan4 Bucket 5 (hook bug).

    Returns None if either config is missing from the run.
    """
    if "baseline" not in per_config_results or "D-r1" not in per_config_results:
        return None
    base = per_config_results["baseline"]
    dr1 = per_config_results["D-r1"]
    matches = sum(1 for a, b in zip(base, dr1) if a["generated"] == b["generated"])
    return {"matches": matches, "total": len(base)}


def _summary_table(per_config_results, label):
    print(f"\n=== {label} zero-shot results ===")
    print(
        f"{'config':>10}  {'accuracy':>9}  {'vs baseline':>11}  "
        f"{'unique_vs_baseline':>19}"
    )
    base_acc = None
    if "baseline" in per_config_results:
        base_correct = sum(1 for r in per_config_results["baseline"] if r["correct"])
        base_total = len(per_config_results["baseline"])
        base_acc = base_correct / base_total if base_total else 0.0

    rows = []
    for name, results in per_config_results.items():
        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        acc = correct / total if total else 0.0
        if base_acc is None or name == "baseline":
            delta_str = "   ---"
            delta = None
        else:
            delta = acc - base_acc
            delta_str = f"{delta:+.3f}"
        uniq = _unique_correct_vs_baseline(name, per_config_results)
        uniq_str = "---" if uniq is None else str(uniq)
        marker = ""
        if name == "D-r1":
            marker = "  <- must match baseline"
        print(
            f"{name:>10}  {acc:>8.3f}  {delta_str:>11}  {uniq_str:>19}{marker}"
        )
        rows.append({
            "config": name,
            "n_correct": correct,
            "n_total": total,
            "accuracy": acc,
            "vs_baseline": delta,
            "unique_correct_vs_baseline": uniq,
        })
    return {"rows": rows, "baseline_accuracy": base_acc}


def _select_configs(all_configs, wanted):
    """Filter ``all_configs`` in the order given by ``wanted``. If the
    user passes names that aren't known, we fail loudly so a typo doesn't
    silently drop a config from the sweep."""
    by_name = {c["name"]: c for c in all_configs}
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        valid = ", ".join(c["name"] for c in all_configs)
        print(f"ERROR: unknown config names {unknown}. Valid: {valid}")
        sys.exit(1)
    return [by_name[n] for n in wanted]


def run_reasoning_eval_mode(args, model, tokenizer, decoder_layers):
    """Plan4 reasoning-eval runner.

    Loads GSM8K (and optionally ARC-Easy) once, then runs each selected
    configuration end-to-end --- install hook, generate answers, parse,
    record, uninstall --- before moving to the next. Configs run
    sequentially to keep memory usage bounded and to isolate each
    sweep's hooks from the next. After the sweep completes, we print
    the plan4 summary table and agreement matrix, and save the full
    per-problem outputs so downstream analysis doesn't require a
    re-run.

    CUDA is already required upstream by ``print_env`` / ``load_model``.
    Hook strategy A is asserted against a reference layer from the D
    block (layer 15) if at least one looping config is selected --- this
    mirrors the preamble of mode_round3b/3c and guards against
    transformers upgrades silently moving the PLE kwarg.
    """
    output_json = args.output_json or _results_path("results_round4_reasoning.json")
    max_new_tokens = args.max_gen_tokens
    n_gsm8k = args.num_gsm8k_problems
    n_arc = args.num_arc_problems
    run_arc = args.run_arc

    if args.configs:
        configs = _select_configs(CONFIGS, args.configs)
    else:
        configs = [dict(c) for c in CONFIGS]

    # Validate every looping block lives inside the model.
    n_layers = len(decoder_layers)
    for cfg in configs:
        if cfg["block"] is None:
            continue
        s, e = cfg["block"]
        if s < 0 or e >= n_layers or s > e:
            print(
                f"ERROR: config {cfg['name']} block [{s}..{e}] out of range "
                f"for {n_layers} layers."
            )
            sys.exit(1)

    any_looping = any(c["block"] is not None for c in configs)
    if any_looping:
        # Inspect Strategy A at the first looping block's start layer. The
        # hook doesn't strictly need kwarg-style PLE (it captures whatever
        # is passed), but a Strategy change is a sign the model's forward
        # signature moved, which would invalidate other assumptions too.
        first_loop = next(c for c in configs if c["block"] is not None)
        inspection = _inspect_and_require_strategy_a(
            decoder_layers[first_loop["block"][0]]
        )
        ple_kwarg = inspection["ple_kwarg"]
        print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")
    else:
        inspection = None
        ple_kwarg = None

    gsm8k_problems = _load_gsm8k(n_gsm8k) if n_gsm8k > 0 else []
    arc_problems = _load_arc_easy(n_arc) if (run_arc and n_arc > 0) else []

    gsm8k_per_config = {}
    arc_per_config = {}

    for i, cfg in enumerate(configs, 1):
        print(
            f"\n--- [{i}/{len(configs)}] config={cfg['name']}  "
            f"block={cfg['block']}  r={cfg['r']} ---"
        )
        if gsm8k_problems:
            gsm_out = _run_gsm8k(
                cfg, model, tokenizer, decoder_layers, gsm8k_problems, max_new_tokens
            )
            gsm8k_per_config[cfg["name"]] = gsm_out["per_problem"]
            print(
                f"  GSM8K {cfg['name']}: {gsm_out['n_correct']}/{gsm_out['n_total']} "
                f"= {gsm_out['accuracy']:.3f}"
            )
        if arc_problems:
            arc_out = _run_arc_easy(
                cfg, model, tokenizer, decoder_layers, arc_problems, max_new_tokens
            )
            arc_per_config[cfg["name"]] = arc_out["per_problem"]
            print(
                f"  ARC-Easy {cfg['name']}: {arc_out['n_correct']}/{arc_out['n_total']} "
                f"= {arc_out['accuracy']:.3f}"
            )

    # Summary tables + agreement matrices.
    analysis = {}
    if gsm8k_per_config:
        gsm_summary = _summary_table(gsm8k_per_config, "GSM8K")
        gsm_matrix = _agreement_matrix(gsm8k_per_config)
        analysis["gsm8k"] = {
            "summary": gsm_summary,
            "agreement_matrix": gsm_matrix,
        }
    if arc_per_config:
        arc_summary = _summary_table(arc_per_config, "ARC-Easy")
        arc_matrix = _agreement_matrix(arc_per_config)
        analysis["arc_easy"] = {
            "summary": arc_summary,
            "agreement_matrix": arc_matrix,
        }

    # D-r1 bitwise sanity on GSM8K (if both configs ran).
    sanity = {}
    if gsm8k_per_config:
        match = _d_r1_bitwise_match(gsm8k_per_config)
        if match is not None:
            print(
                f"\n=== D-r1 vs baseline sanity (GSM8K generated-text match) ===\n"
                f"  {match['matches']}/{match['total']} generated strings match "
                f"bitwise."
            )
            if match["matches"] != match["total"]:
                print(
                    "  WARNING: D-r1 should be a bitwise no-op w.r.t. baseline. "
                    "Plan4 Bucket 5 --- hook bug. Inspect before interpreting "
                    "other configurations."
                )
            sanity["gsm8k_d_r1_vs_baseline"] = match

    output = {
        "config": {
            "mode": "reasoning-eval",
            "model_id": args.model_id,
            "configs": [
                {"name": c["name"], "block": c["block"], "r": c["r"]}
                for c in configs
            ],
            "num_gsm8k_problems": n_gsm8k,
            "num_arc_problems": n_arc if run_arc else 0,
            "run_arc": run_arc,
            "max_gen_tokens": max_new_tokens,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
        },
        "inspection": (
            {
                "strategy": inspection["strategy"],
                "layer_class": inspection["layer_class"],
                "source_file": inspection["source_file"],
                "start_lineno": inspection["start_lineno"],
                "signature": inspection["signature"],
                "ple_kwarg": inspection["ple_kwarg"],
            }
            if inspection is not None
            else None
        ),
        "gsm8k": {
            "per_config": gsm8k_per_config,
            "summary": analysis.get("gsm8k", {}).get("summary"),
            "agreement_matrix": analysis.get("gsm8k", {}).get("agreement_matrix"),
        } if gsm8k_per_config else None,
        "arc_easy": {
            "per_config": arc_per_config,
            "summary": analysis.get("arc_easy", {}).get("summary"),
            "agreement_matrix": analysis.get("arc_easy", {}).get("agreement_matrix"),
        } if arc_per_config else None,
        "sanity_checks": sanity,
    }
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")
