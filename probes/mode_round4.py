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

Crash-tolerance: every problem's result is appended (with fsync) to a
JSONL file under ``results/round4_partial/`` *as soon as it finishes*.
Re-running the same command resumes from wherever the previous run was
killed --- already-done problem indices are skipped per (dataset,
config). A manifest in the same directory pins the model_id, dtype, and
max_gen_tokens so that a config change (which would invalidate prior
generations) aborts loudly instead of silently mixing data. Pass
``--no-resume`` to wipe the checkpoint dir and start fresh.

Live inspection: while a run is in progress, the JSONL files are
plain-text-readable (``wc -l results/round4_partial/gsm8k__G-r8.jsonl``
for progress, ``jq`` for filtering). For the plan4 summary table
without re-loading the model, run
``--mode reasoning-eval --summarize-only``.

Sanity check: D-r1 installs the block-loop hook at r=1, which does zero
extra iterations and must produce bitwise-identical generations to the
unhooked baseline. If not, stop and debug the hook before trusting the
other configurations (plan4 Bucket 5).
"""

import itertools
import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch

from .env import _results_path, _write_results_json
from .hooks import install_block_loop_hooks
from .introspect import _inspect_and_require_strategy_a


# Subdirectory of results/ where per-problem JSONL files and the run
# manifest live. Hard-coded so the --summarize-only path can find them
# without the user having to remember a flag.
CHECKPOINT_DIRNAME = "round4_partial"


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


# ----------------------------------------------------------------------
# Checkpoint / resume infrastructure
# ----------------------------------------------------------------------

def _checkpoint_dir(args):
    """Where per-problem JSONLs and the manifest live for this run."""
    override = getattr(args, "checkpoint_dir", None)
    return Path(override) if override else Path(_results_path(CHECKPOINT_DIRNAME))


def _manifest_path(ckpt_dir):
    return ckpt_dir / "manifest.json"


def _jsonl_path(ckpt_dir, dataset, config_name):
    """Two-underscore separator avoids collision with config names like
    ``D-r1`` (the dash is fine as a literal here, just being defensive)."""
    return ckpt_dir / f"{dataset}__{config_name}.jsonl"


def _load_jsonl(p):
    """Read a JSONL file and return the list of parsed rows. Malformed
    lines (e.g. partial last write after a hard crash) are skipped with
    a warning rather than aborting --- the next run will simply re-do
    that single problem."""
    if not p.exists():
        return []
    rows = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(
                f"  WARNING: malformed JSONL line {i} in {p.name} "
                f"({e}); skipping --- the missing problem will be re-run."
            )
    return rows


def _append_jsonl(p, row):
    """Append one row + newline + flush + fsync. fsync makes the row
    durable even if the host loses power immediately afterwards. Cost is
    a few ms per problem against generation latency measured in seconds,
    so the tradeoff is favourable."""
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _check_or_write_manifest(ckpt_dir, args):
    """Detect incompatible reruns. The fields tracked here all influence
    generated text, so a mismatch means the existing JSONLs were produced
    under different conditions and must not be merged with new rows.

    On first run we just write the manifest. On rerun, mismatch aborts
    with a hint pointing at --no-resume.
    """
    new_manifest = {
        "model_id": args.model_id,
        "dtype": args.dtype,
        "max_gen_tokens": args.max_gen_tokens,
    }
    p = _manifest_path(ckpt_dir)
    if not p.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(new_manifest, indent=2))
        return new_manifest, True  # fresh
    try:
        existing = json.loads(p.read_text())
    except json.JSONDecodeError:
        print(f"ERROR: checkpoint manifest at {p} is corrupt. "
              f"Delete {ckpt_dir} or pass --no-resume.")
        sys.exit(1)

    diffs = [k for k in new_manifest if existing.get(k) != new_manifest[k]]
    if diffs:
        print(f"\nERROR: checkpoint at {ckpt_dir} is incompatible with this run:")
        for k in diffs:
            print(f"  {k}: existing={existing.get(k)!r}  requested={new_manifest[k]!r}")
        print(f"  Delete {ckpt_dir} or pass --no-resume to start fresh.\n")
        sys.exit(1)
    return existing, False  # resumed


def _wipe_checkpoint_dir(ckpt_dir):
    if ckpt_dir.exists():
        print(f"--no-resume: wiping {ckpt_dir}")
        shutil.rmtree(ckpt_dir)


def _existing_progress(ckpt_dir, datasets, config_names):
    """Build a {dataset: {config: count}} table of per-config progress
    found on disk. Used for the startup banner so the user can see at a
    glance where the previous run was killed."""
    progress = {}
    for dataset in datasets:
        progress[dataset] = {}
        for name in config_names:
            p = _jsonl_path(ckpt_dir, dataset, name)
            progress[dataset][name] = len(_load_jsonl(p)) if p.exists() else 0
    return progress


def _print_progress_banner(progress, problem_counts):
    """Print one line per (dataset, config) showing how much is already
    on disk vs how many problems we plan to run. Done configs are tagged
    so the user knows to expect zero generation work for them."""
    print("\n=== Resume status ===")
    for dataset, per_cfg in progress.items():
        n = problem_counts.get(dataset, 0)
        if n == 0:
            continue
        print(f"  {dataset}:")
        for name, done in per_cfg.items():
            tag = " [done]" if done >= n else ""
            print(f"    {name:>10}: {done}/{n}{tag}")


def _resume_existing(jsonl_path, problems):
    """Load any prior rows for this (dataset, config), drop ones whose
    idx is no longer in scope (user reduced N between runs), and return
    (existing_rows_filtered, set_of_done_idx)."""
    valid_idx = {p["idx"] for p in problems}
    existing = [r for r in _load_jsonl(jsonl_path) if r.get("idx") in valid_idx]
    done = {r["idx"] for r in existing}
    return existing, done


def _run_dataset(
    cfg, model, tokenizer, decoder_layers, problems, max_new_tokens,
    jsonl_path,
    *,
    label,                  # "gsm8k" / "arc"  --- log tag
    format_prompt,          # problem -> prompt string
    score,                  # (problem, generated) -> (parsed_answer, correct)
):
    """Generic per-config eval loop with JSONL checkpoint + resume.

    Both GSM8K and ARC follow the same skeleton: prompt -> generate ->
    parse -> score -> append. The only differences are the prompt
    formatter and the scoring function, both injected via callbacks.
    The function only installs the loop hook if there is *new* work to
    do --- a fully-resumed config skips hook installation entirely so
    we don't pay the (small) per-call overhead for nothing.
    """
    existing, done_idx = _resume_existing(jsonl_path, problems)
    remaining = [p for p in problems if p["idx"] not in done_idx]
    n_correct = sum(1 for r in existing if r["correct"])

    if not remaining:
        print(
            f"    [{cfg['name']}] {label} already complete "
            f"({len(existing)}/{len(problems)})"
        )
    else:
        if existing:
            print(
                f"    [{cfg['name']}] {label} resuming: {len(existing)}/"
                f"{len(problems)} on disk, generating {len(remaining)} more"
            )
        uninstall = _install_config_hook(decoder_layers, cfg)
        try:
            for p in remaining:
                prompt = format_prompt(p)
                generated = _generate(model, tokenizer, prompt, max_new_tokens)
                parsed, correct = score(p, generated)
                row = {
                    "idx": p["idx"],
                    "question": p["question"],
                    "gold": p["gold"],
                    "generated": generated,
                    "parsed_answer": parsed,
                    "correct": bool(correct),
                }
                _append_jsonl(jsonl_path, row)
                existing.append(row)
                if correct:
                    n_correct += 1
                if (len(existing) % 25) == 0 or len(existing) == len(problems):
                    print(
                        f"    [{cfg['name']}] {label} {len(existing)}/{len(problems)}  "
                        f"running accuracy={n_correct / len(existing):.3f}"
                    )
        finally:
            uninstall()

    # Canonical order for downstream analysis regardless of resume order.
    existing.sort(key=lambda r: r["idx"])
    accuracy = n_correct / len(problems) if problems else 0.0
    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(problems),
        "per_problem": existing,
    }


def _score_gsm8k(problem, generated):
    parsed = _parse_gsm8k_number(generated)
    correct = (
        parsed is not None
        and problem["gold"] is not None
        and abs(parsed - problem["gold"]) < 1e-4
    )
    return parsed, correct


def _score_arc(problem, generated):
    parsed = _parse_mc_answer(generated, problem["choices"]["label"])
    correct = parsed is not None and parsed == problem["gold"]
    return parsed, correct


def _run_gsm8k(cfg, model, tokenizer, decoder_layers, problems, max_new_tokens, jsonl_path):
    return _run_dataset(
        cfg, model, tokenizer, decoder_layers, problems, max_new_tokens, jsonl_path,
        label="gsm8k",
        format_prompt=_format_gsm8k_prompt,
        score=_score_gsm8k,
    )


def _run_arc_easy(cfg, model, tokenizer, decoder_layers, problems, max_new_tokens, jsonl_path):
    return _run_dataset(
        cfg, model, tokenizer, decoder_layers, problems, max_new_tokens, jsonl_path,
        label="arc  ",
        format_prompt=_format_arc_prompt,
        score=_score_arc,
    )


def _by_idx(rows):
    """Index rows by problem idx so callers can intersect cleanly even if
    two configs are at different points in their run (resume-aware)."""
    return {r["idx"]: r for r in rows if "idx" in r}


def _agreement_matrix(per_config_results):
    """Build the pairwise agreement matrix described in plan4. Rows are
    matched by problem ``idx`` (not list position) so the matrix stays
    correct under partial-inspection or resume-in-progress conditions:
    if config A has finished 200 problems and config B only 150, only
    the 150 they share are scored. The intersection size is recorded.
    """
    names = list(per_config_results.keys())
    matrix = {}
    for a, b in itertools.combinations(names, 2):
        ra = _by_idx(per_config_results[a])
        rb = _by_idx(per_config_results[b])
        common = sorted(set(ra) & set(rb))
        both_c = only_a = only_b = both_w = 0
        for i in common:
            ca, cb = ra[i]["correct"], rb[i]["correct"]
            if ca and cb:
                both_c += 1
            elif ca and not cb:
                only_a += 1
            elif cb and not ca:
                only_b += 1
            else:
                both_w += 1
        matrix[f"{a}__vs__{b}"] = {
            "a": a, "b": b,
            "n_common": len(common),
            "both_correct": both_c,
            "only_a_correct": only_a,
            "only_b_correct": only_b,
            "both_wrong": both_w,
        }
    return matrix


def _unique_correct_vs_baseline(config_name, per_config_results):
    """Count problems ``config_name`` got right that baseline got wrong,
    over the intersection of the two configs' completed indices."""
    if "baseline" not in per_config_results or config_name == "baseline":
        return None
    base = _by_idx(per_config_results["baseline"])
    mine = _by_idx(per_config_results[config_name])
    common = set(base) & set(mine)
    return sum(1 for i in common if mine[i]["correct"] and not base[i]["correct"])


def _d_r1_bitwise_match(per_config_results):
    """Compare D-r1 generated strings against baseline by ``idx``. Hook
    must be a bitwise no-op at r=1; any mismatch is plan4 Bucket 5.

    Returns None if either config is missing. ``total`` reflects the
    intersection so partial runs report meaningful numbers.
    """
    if "baseline" not in per_config_results or "D-r1" not in per_config_results:
        return None
    base = _by_idx(per_config_results["baseline"])
    dr1 = _by_idx(per_config_results["D-r1"])
    common = sorted(set(base) & set(dr1))
    matches = sum(1 for i in common if base[i]["generated"] == dr1[i]["generated"])
    return {"matches": matches, "total": len(common)}


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
    """Plan4 reasoning-eval runner with JSONL checkpoint + resume.

    Loads GSM8K (and optionally ARC-Easy) once, then runs each selected
    configuration end-to-end --- install hook, generate answers, parse,
    record, uninstall --- before moving to the next. Configs run
    sequentially to keep memory usage bounded and to isolate each
    sweep's hooks from the next. After the sweep completes, we print
    the plan4 summary table and agreement matrix, and save the full
    per-problem outputs so downstream analysis doesn't require a
    re-run.

    Each problem's result is appended to a per-(dataset, config) JSONL
    file under ``results/round4_partial/`` as soon as it finishes. A
    re-invocation of the same command resumes wherever the previous
    run was killed; no work is lost. ``--no-resume`` wipes the
    checkpoint dir and starts fresh.

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

    # ---- Checkpoint setup ----
    ckpt_dir = _checkpoint_dir(args)
    if getattr(args, "no_resume", False):
        _wipe_checkpoint_dir(ckpt_dir)
    manifest, fresh = _check_or_write_manifest(ckpt_dir, args)
    print(
        f"Checkpoint dir: {ckpt_dir}  "
        f"({'fresh start' if fresh else 'resume mode'})"
    )

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

    # Resume banner: tells the user what's already on disk before we
    # touch the model. Cheap (just file reads) and helps debug if a
    # prior run wrote to an unexpected directory.
    datasets_in_play = []
    if gsm8k_problems:
        datasets_in_play.append("gsm8k")
    if arc_problems:
        datasets_in_play.append("arc_easy")
    progress = _existing_progress(
        ckpt_dir, datasets_in_play, [c["name"] for c in configs]
    )
    _print_progress_banner(
        progress,
        {"gsm8k": len(gsm8k_problems), "arc_easy": len(arc_problems)},
    )

    gsm8k_per_config = {}
    arc_per_config = {}

    for i, cfg in enumerate(configs, 1):
        print(
            f"\n--- [{i}/{len(configs)}] config={cfg['name']}  "
            f"block={cfg['block']}  r={cfg['r']} ---"
        )
        if gsm8k_problems:
            gsm_jsonl = _jsonl_path(ckpt_dir, "gsm8k", cfg["name"])
            gsm_out = _run_gsm8k(
                cfg, model, tokenizer, decoder_layers, gsm8k_problems,
                max_new_tokens, gsm_jsonl,
            )
            gsm8k_per_config[cfg["name"]] = gsm_out["per_problem"]
            print(
                f"  GSM8K {cfg['name']}: {gsm_out['n_correct']}/{gsm_out['n_total']} "
                f"= {gsm_out['accuracy']:.3f}"
            )
        if arc_problems:
            arc_jsonl = _jsonl_path(ckpt_dir, "arc_easy", cfg["name"])
            arc_out = _run_arc_easy(
                cfg, model, tokenizer, decoder_layers, arc_problems,
                max_new_tokens, arc_jsonl,
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
            "checkpoint_dir": str(ckpt_dir),
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


def summarize_checkpoints(args):
    """Read JSONLs from the checkpoint dir and print the plan4 summary
    table without loading the model. Useful for inspecting an
    in-progress run from a separate shell:

        python ple_sanity_check.py --mode reasoning-eval --summarize-only

    Iterates every JSONL file in the checkpoint dir (so this works even
    if --configs differs between the running job and this inspector).
    Datasets and configs are inferred from filenames using the
    ``{dataset}__{config}.jsonl`` convention. The agreement matrix and
    D-r1 sanity check are computed over each config-pair's intersection
    of completed indices, so partial data is tolerated.
    """
    ckpt_dir = _checkpoint_dir(args)
    if not ckpt_dir.exists():
        print(f"No checkpoint dir at {ckpt_dir} --- nothing to summarize.")
        return

    manifest_p = _manifest_path(ckpt_dir)
    if manifest_p.exists():
        print(f"Manifest ({manifest_p}):")
        print("  " + manifest_p.read_text().replace("\n", "\n  ").rstrip())
        print()

    by_dataset = {}
    for p in sorted(ckpt_dir.glob("*.jsonl")):
        if "__" not in p.stem:
            continue
        dataset, config = p.stem.split("__", 1)
        rows = _load_jsonl(p)
        if not rows:
            continue
        by_dataset.setdefault(dataset, {})[config] = rows

    if not by_dataset:
        print(f"No JSONL data in {ckpt_dir}.")
        return

    for dataset, per_config in by_dataset.items():
        # Show problem-count distribution so the user sees who's behind.
        print(f"=== {dataset} progress ===")
        max_n = max(len(rows) for rows in per_config.values())
        for name in sorted(per_config):
            n = len(per_config[name])
            tag = "" if n == max_n else f"  ({max_n - n} behind leader)"
            print(f"  {name:>10}: {n} problems{tag}")
        _summary_table(per_config, dataset)
        if dataset == "gsm8k":
            match = _d_r1_bitwise_match(per_config)
            if match is not None and match["total"] > 0:
                print(
                    f"\n=== D-r1 vs baseline sanity (intersection={match['total']}) ==="
                )
                print(
                    f"  {match['matches']}/{match['total']} generated strings match "
                    f"bitwise."
                )
                if match["matches"] != match["total"]:
                    print(
                        "  WARNING: hook should be a no-op at r=1. Plan4 Bucket 5."
                    )
        # Pairwise agreement only when 2+ configs share data.
        if len(per_config) >= 2:
            mat = _agreement_matrix(per_config)
            print(f"\n=== {dataset} pairwise agreement ===")
            for key, cell in mat.items():
                print(
                    f"  {cell['a']:>10} vs {cell['b']:<10}  "
                    f"both_correct={cell['both_correct']:>3}  "
                    f"only_a={cell['only_a_correct']:>3}  "
                    f"only_b={cell['only_b_correct']:>3}  "
                    f"both_wrong={cell['both_wrong']:>3}  "
                    f"(n={cell['n_common']})"
                )
        print()
