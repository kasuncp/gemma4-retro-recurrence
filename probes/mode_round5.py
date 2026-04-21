"""Round 5 --- reasoning eval v2 (fixed GSM8K baseline + width sweep).

Round 4 produced three actionable takeaways:
  1. GSM8K baseline of 4.8% was a harness bug, not a model result ---
     155/238 wrong baselines truncated mid-arithmetic at max_gen_tokens=256,
     and the remaining generations produced "Question: ...\nAnswer: ..."
     textbook-continuation loops instead of answering.
  2. On ARC-Easy, narrow blocks (A=[15,19]) degraded less than wide ones,
     suggesting block width is a meaningful axis.
  3. PLE is re-injected on every loop iteration. Semantically dubious
     for a "per-layer" embedding; worth ablating.

Plan5 has three parts:

Part 1 --- fix GSM8K baseline:
  * Raise max_gen_tokens 256 -> 512 (handled by CLI default).
  * Switch GSM8K to google/gemma-4-E2B-it + tokenizer.apply_chat_template
    so the instruction-tuned turn format is used.
  * 8-shot Wei et al. CoT exemplars (the standard GSM8K protocol).
  * MANDATORY validation gate: baseline-only on 50 problems, accuracy
    must land in [40%, 55%]. Invoke with
    ``--mode reasoning-eval-r5 --configs baseline --num-gsm8k-problems 50
      --num-arc-problems 0``.
    If the gate fails, do NOT run the full sweep.

Part 2 --- width sweep on ARC-Easy (primary experiment):
  * ARC-Easy stays on base E2B with raw prompt to preserve the round-4
    cross-round anchor (A-r8 = 40.0%).
  * Fixed r=8, fixed start=15, widths 2/3/4/5.
  * W5-r1 sanity: the hook is a no-op at r=1 and must produce
    bitwise-identical generations to the baseline on every problem.
  * W5-r8 must land within +/-2% of round-4 A-r8 accuracy.

Part 3 --- start-position sweep (secondary):
  * Fixed width=3, r=8, starts 10 / 15 / 20. (S15-W3 is the same run
    as W3-r8 from Part 2; one physical run covers both.)

Part 4 --- PLE ablation (strategy pilot):
  * Same block as W5-r8 but PLE is fed only on iteration 1; iterations
    2..r receive a zero tensor of the same shape in the per_layer_input
    slot. Implementation: ``install_block_loop_hooks(..., ple_strategy=
    "iter1-only")``.

Two-pass model loading:
  Pass 1: GSM8K on google/gemma-4-E2B-it (9 configs).
  Pass 2: ARC-Easy on google/gemma-4-E2B (9 configs).
  Unloaded between passes to keep VRAM headroom sane.

Crash-tolerance: per-problem JSONL append + fsync after every generation
(same as round 4). Manifest in the checkpoint dir pins both model IDs,
dtype, and max_gen_tokens so a config change aborts the resume loudly
rather than silently mixing incompatible data. ``--no-resume`` wipes and
starts fresh.

New fields in each per-row JSONL record vs round 4:
  * ``truncated`` --- True if generation hit max_new_tokens (no EOS /
    stop-string). Combined with ``parsed_answer is None`` this yields
    the per-config ``truncation_rate`` summary field that would have
    caught the round-4 baseline bug at a glance.
"""

import gc
import itertools
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import torch

from .env import (
    DTYPE_MAP,
    EXPECTED_NUM_LAYERS,
    _results_path,
    _write_results_json,
    load_model,
)
from .hooks import install_block_loop_hooks
from .introspect import _inspect_and_require_strategy_a, find_decoder_layers
from .mode_round4 import (
    _agreement_matrix,
    _by_idx,
    _format_arc_prompt,
    _load_arc_easy,
    _load_gsm8k,
    _parse_gsm8k_number,
    _parse_mc_answer,
    _score_arc,
    _score_gsm8k,
    _unique_correct_vs_baseline,
)


CHECKPOINT_DIRNAME = "round5_partial"


# Plan5 canonical configuration set. Each entry is:
#   name           --- config key used in checkpoints and output JSON
#   block          --- (L_start, L_end) inclusive range, or None for baseline
#   r              --- loop count (1 = hook is a no-op)
#   ple_strategy   --- passed to install_block_loop_hooks; "every-iter"
#                      matches round 4, "iter1-only" is the plan5 Part 4
#                      ablation where PLE is zero on replay iterations.
CONFIGS = [
    {"name": "baseline",    "block": None,     "r": 1, "ple_strategy": "every-iter"},
    {"name": "W2-r8",       "block": (15, 16), "r": 8, "ple_strategy": "every-iter"},
    {"name": "W3-r8",       "block": (15, 17), "r": 8, "ple_strategy": "every-iter"},
    {"name": "W4-r8",       "block": (15, 18), "r": 8, "ple_strategy": "every-iter"},
    {"name": "W5-r8",       "block": (15, 19), "r": 8, "ple_strategy": "every-iter"},
    {"name": "W5-r1",       "block": (15, 19), "r": 1, "ple_strategy": "every-iter"},
    {"name": "S10-W3",      "block": (10, 12), "r": 8, "ple_strategy": "every-iter"},
    {"name": "S20-W3",      "block": (20, 22), "r": 8, "ple_strategy": "every-iter"},
    {"name": "W5-r8-noPLE", "block": (15, 19), "r": 8, "ple_strategy": "iter1-only"},
]

# Note: S15-W3 (start=15, width=3) is identical to W3-r8 above. The
# start-position sweep (Part 3) uses S10-W3, W3-r8, S20-W3 as its three
# points; we don't add a duplicate config.


# Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in
# Large Language Models", Appendix G. These are the canonical 8 GSM8K
# few-shot exemplars used by every published Gemma GSM8K number; the
# "The answer is X." suffix lines up with _parse_gsm8k_number (last
# number in generated text = final answer).
WEI_COT_EXEMPLARS = [
    {
        "q": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "a": "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6. The answer is 6.",
    },
    {
        "q": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "a": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    },
    {
        "q": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "a": "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39.",
    },
    {
        "q": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
        "a": "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8. The answer is 8.",
    },
    {
        "q": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
        "a": "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. The answer is 9.",
    },
    {
        "q": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
        "a": "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.",
    },
    {
        "q": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
        "a": "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    },
    {
        "q": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "a": "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    },
]


# Extra stop-string we pass to HF generate for GSM8K. Prevents the model
# from continuing past its own answer into a new "Q:" exemplar turn when
# the chat template's EOS isn't asserted cleanly.
GSM8K_STOP_STRINGS = ["\nQ:", "\nQuestion:"]


def _format_gsm8k_prompt_chat(problem, tokenizer):
    """Format a GSM8K problem as a chat-template prompt with 8-shot CoT.

    Each exemplar becomes a user/assistant turn pair, followed by the
    target question as the final user turn. ``add_generation_prompt=True``
    ensures the template leaves the assistant turn open so the model
    starts generating the answer directly.
    """
    msgs = []
    for ex in WEI_COT_EXEMPLARS:
        msgs.append({"role": "user", "content": f"Q: {ex['q']}"})
        msgs.append({"role": "assistant", "content": f"A: {ex['a']}"})
    msgs.append({"role": "user", "content": f"Q: {problem['question']}"})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )


def _generate_r5(model, tokenizer, prompt, max_new_tokens, stop_strings=None):
    """Greedy decode with KV cache disabled. Returns (text, truncated,
    n_gen_tokens).

    ``truncated`` is True iff generation hit ``max_new_tokens`` without
    emitting EOS or any stop string. That's the signal plan5 asks us to
    record per problem so ``truncation_rate`` summarises harness health.

    Sampling flags on generation_config are nulled (same shim as round 4)
    to suppress the "flags not valid" warning under do_sample=False.
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    gen_cfg = deepcopy(model.generation_config)
    gen_cfg.do_sample = False
    gen_cfg.top_p = None
    gen_cfg.top_k = None
    gen_cfg.temperature = None

    gen_kwargs = dict(
        generation_config=gen_cfg,
        max_new_tokens=max_new_tokens,
        use_cache=False,
        pad_token_id=pad_id,
    )
    if stop_strings:
        # HF's generate accepts stop_strings + tokenizer directly.
        gen_kwargs["stop_strings"] = list(stop_strings)
        gen_kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)
    gen_ids = out[0, input_len:]
    n_gen = int(gen_ids.shape[0])

    # Truncation = reached the length cap. If the model emitted EOS the
    # generation ends earlier. If it emitted a stop-string, HF strips the
    # stop sequence itself but the total length will be < max_new_tokens.
    truncated = (n_gen >= max_new_tokens)
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text, truncated, n_gen


def _install_config_hook(decoder_layers, cfg):
    """Install the block-loop hook for ``cfg`` or return a no-op
    uninstall for the unhooked baseline."""
    block = cfg["block"]
    if block is None:
        return lambda: None
    start, end = block
    return install_block_loop_hooks(
        decoder_layers, start, end, r=cfg["r"],
        ple_strategy=cfg.get("ple_strategy", "every-iter"),
    )


# ----------------------------------------------------------------------
# Checkpoint / resume infrastructure. Same layout as round 4 but with a
# round5-specific dir name and a manifest that tracks two model IDs.
# ----------------------------------------------------------------------

def _checkpoint_dir(args):
    override = getattr(args, "checkpoint_dir", None)
    return Path(override) if override else Path(_results_path(CHECKPOINT_DIRNAME))


def _manifest_path(ckpt_dir):
    return ckpt_dir / "manifest.json"


def _jsonl_path(ckpt_dir, dataset, config_name):
    return ckpt_dir / f"{dataset}__{config_name}.jsonl"


def _load_jsonl(p):
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
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def _check_or_write_manifest(ckpt_dir, args):
    """Detect incompatible reruns. All tracked fields influence
    generated text, so a mismatch invalidates the JSONLs.

    Round 5 tracks two model IDs (GSM8K + ARC). Each config changes the
    sampling / tokenisation pathway, so we pin them together.
    """
    new_manifest = {
        "gsm8k_model_id": args.gsm8k_model_id,
        "arc_model_id": args.arc_model_id,
        "dtype": args.dtype,
        "max_gen_tokens": args.max_gen_tokens,
        "plan": "round5",
    }
    p = _manifest_path(ckpt_dir)
    if not p.exists():
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(new_manifest, indent=2))
        return new_manifest, True
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
    return existing, False


def _wipe_checkpoint_dir(ckpt_dir):
    if ckpt_dir.exists():
        print(f"--no-resume: wiping {ckpt_dir}")
        shutil.rmtree(ckpt_dir)


def _resume_existing(jsonl_path, problems):
    valid_idx = {p["idx"] for p in problems}
    existing = [r for r in _load_jsonl(jsonl_path) if r.get("idx") in valid_idx]
    done = {r["idx"] for r in existing}
    return existing, done


def _run_dataset(
    cfg, model, tokenizer, decoder_layers, problems, max_new_tokens,
    jsonl_path,
    *,
    label,
    format_prompt,
    score,
    stop_strings=None,
):
    """Round-5 per-config eval loop. Same skeleton as round 4 but each
    row also records ``truncated`` and ``n_gen_tokens`` so downstream
    analysis can compute truncation rate without re-running.
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
                generated, truncated, n_gen = _generate_r5(
                    model, tokenizer, prompt, max_new_tokens,
                    stop_strings=stop_strings,
                )
                parsed, correct = score(p, generated)
                row = {
                    "idx": p["idx"],
                    "question": p["question"],
                    "gold": p["gold"],
                    "generated": generated,
                    "parsed_answer": parsed,
                    "correct": bool(correct),
                    "truncated": bool(truncated),
                    "n_gen_tokens": n_gen,
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

    existing.sort(key=lambda r: r["idx"])
    accuracy = n_correct / len(problems) if problems else 0.0
    return {
        "accuracy": accuracy,
        "n_correct": n_correct,
        "n_total": len(problems),
        "per_problem": existing,
    }


def _truncation_rate(rows):
    """Fraction of rows that hit max_gen_tokens without producing a
    parseable answer. This is the exact metric plan5 asks for --- the
    one that would have caught round 4's baseline bug in seconds."""
    if not rows:
        return 0.0
    bad = sum(1 for r in rows if r.get("truncated") and r.get("parsed_answer") is None)
    return bad / len(rows)


def _w5_r1_bitwise_match(per_config_results):
    """Compare W5-r1 generated strings against baseline by idx. The
    block-loop hook must be a no-op at r=1, so every generation must be
    bitwise-identical to the unhooked baseline. ``total`` reflects the
    intersection of completed idx so partial runs still report.
    """
    if "baseline" not in per_config_results or "W5-r1" not in per_config_results:
        return None
    base = _by_idx(per_config_results["baseline"])
    w5 = _by_idx(per_config_results["W5-r1"])
    common = sorted(set(base) & set(w5))
    matches = sum(1 for i in common if base[i]["generated"] == w5[i]["generated"])
    return {"matches": matches, "total": len(common)}


def _summary_table(per_config_results, label):
    """Round 4's summary table plus a truncation_rate column. Returns a
    dict suitable for serialisation alongside the raw per-problem data.
    """
    print(f"\n=== {label} zero-shot results ===")
    print(
        f"{'config':>12}  {'accuracy':>9}  {'vs baseline':>11}  "
        f"{'unique_vs_baseline':>19}  {'truncation':>10}"
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
        trunc = _truncation_rate(results)
        marker = ""
        if name == "W5-r1":
            marker = "  <- must match baseline bitwise"
        print(
            f"{name:>12}  {acc:>8.3f}  {delta_str:>11}  "
            f"{uniq_str:>19}  {trunc:>10.3f}{marker}"
        )
        rows.append({
            "config": name,
            "n_correct": correct,
            "n_total": total,
            "accuracy": acc,
            "vs_baseline": delta,
            "unique_correct_vs_baseline": uniq,
            "truncation_rate": trunc,
        })
    return {"rows": rows, "baseline_accuracy": base_acc}


def _load_round4_a_r8_arc_accuracy(round4_path):
    """Read round 4's ARC-Easy A-r8 accuracy for the cross-round delta
    check. Returns None if the file is missing or malformed --- the
    sanity check is then reported as unavailable rather than aborting.
    """
    p = Path(round4_path)
    if not p.exists():
        print(f"  (round 4 results not found at {p}; skipping cross-round check.)")
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"  WARNING: could not parse round 4 JSON at {p}: {e}")
        return None
    arc = data.get("arc_easy") or {}
    summary = arc.get("summary") or {}
    for row in summary.get("rows", []):
        if row.get("config") == "A-r8":
            return row.get("accuracy")
    return None


def _free_model(model, tokenizer):
    """Best-effort GPU cleanup between the two model passes. Without
    this the second load can OOM on smaller cards even though E2B is
    well under a 16GB ceiling on paper --- cached allocators don't give
    memory back until forced."""
    try:
        del model
    except Exception:
        pass
    try:
        del tokenizer
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_pass(
    *, args, configs, model_id, dtype, dataset_label, problems,
    ckpt_dir, format_prompt_factory, score, stop_strings, max_new_tokens,
):
    """Load ``model_id``, run ``problems`` on every config in ``configs``,
    then unload. Returns ``{config_name: per_problem_rows}``.

    ``format_prompt_factory`` takes ``tokenizer`` and returns a
    ``problem -> prompt`` callable. This lets the GSM8K path bind the
    chat template to its tokenizer while ARC uses a pure function.
    """
    if not problems:
        return {}, None

    print(f"\n================ Loading {dataset_label} model: {model_id} ================")
    tokenizer, model = load_model(model_id, dtype)
    decoder_layers = find_decoder_layers(model)
    if len(decoder_layers) != EXPECTED_NUM_LAYERS:
        print(
            f"WARNING: expected {EXPECTED_NUM_LAYERS} decoder layers, "
            f"found {len(decoder_layers)}. Continuing anyway."
        )

    # Range-check every block against the actual layer count of THIS
    # model. Same architecture family across E2B / E2B-it, but guard
    # against surprises.
    n_layers = len(decoder_layers)
    for cfg in configs:
        if cfg["block"] is None:
            continue
        s, e = cfg["block"]
        if s < 0 or e >= n_layers or s > e:
            print(
                f"ERROR: config {cfg['name']} block [{s}..{e}] out of range "
                f"for {n_layers} layers in {model_id}."
            )
            sys.exit(1)

    # Strategy-A assertion at the first looping block's start layer.
    # A Strategy change would invalidate the hook, so catch it before
    # burning compute.
    any_looping = any(c["block"] is not None for c in configs)
    inspection = None
    if any_looping:
        first_loop = next(c for c in configs if c["block"] is not None)
        inspection = _inspect_and_require_strategy_a(
            decoder_layers[first_loop["block"][0]]
        )
        print(f"Strategy A confirmed. PLE kwarg = {inspection['ple_kwarg']!r}.\n")

    format_prompt = format_prompt_factory(tokenizer)

    per_config = {}
    for i, cfg in enumerate(configs, 1):
        print(
            f"\n--- [{dataset_label}][{i}/{len(configs)}] config={cfg['name']}  "
            f"block={cfg['block']}  r={cfg['r']}  ple={cfg['ple_strategy']} ---"
        )
        jsonl = _jsonl_path(ckpt_dir, dataset_label, cfg["name"])
        out = _run_dataset(
            cfg, model, tokenizer, decoder_layers, problems,
            max_new_tokens, jsonl,
            label=dataset_label,
            format_prompt=format_prompt,
            score=score,
            stop_strings=stop_strings,
        )
        per_config[cfg["name"]] = out["per_problem"]
        print(
            f"  {dataset_label} {cfg['name']}: {out['n_correct']}/{out['n_total']} "
            f"= {out['accuracy']:.3f}  truncation_rate={_truncation_rate(out['per_problem']):.3f}"
        )

    _free_model(model, tokenizer)
    return per_config, inspection


def run_reasoning_eval_r5_mode(args):
    """Plan5 reasoning-eval-v2 runner.

    Two-pass execution: GSM8K on the instruction-tuned model (chat
    template + 8-shot CoT), then ARC-Easy on the base model (raw prompt
    matching round 4 for cross-round comparability). Each pass runs all
    requested configs end-to-end and checkpoints per-problem to JSONL.

    Argparse fields consumed:
      * args.gsm8k_model_id / args.arc_model_id
      * args.dtype
      * args.max_gen_tokens
      * args.num_gsm8k_problems / args.num_arc_problems / args.run_arc
      * args.configs (subset names), args.output_json, args.checkpoint_dir,
        args.no_resume
      * args.round4_json (path to round-4 results for cross-round check)
    """
    output_json = args.output_json or _results_path("results_round5_reasoning.json")
    max_new_tokens = args.max_gen_tokens
    n_gsm8k = args.num_gsm8k_problems
    n_arc = args.num_arc_problems
    run_arc = args.run_arc

    if args.configs:
        by_name = {c["name"]: c for c in CONFIGS}
        unknown = [n for n in args.configs if n not in by_name]
        if unknown:
            valid = ", ".join(c["name"] for c in CONFIGS)
            print(f"ERROR: unknown config names {unknown}. Valid: {valid}")
            sys.exit(1)
        configs = [dict(by_name[n]) for n in args.configs]
    else:
        configs = [dict(c) for c in CONFIGS]

    # Checkpoint setup and manifest check.
    ckpt_dir = _checkpoint_dir(args)
    if getattr(args, "no_resume", False):
        _wipe_checkpoint_dir(ckpt_dir)
    manifest, fresh = _check_or_write_manifest(ckpt_dir, args)
    print(
        f"Checkpoint dir: {ckpt_dir}  "
        f"({'fresh start' if fresh else 'resume mode'})"
    )
    print(f"  GSM8K model: {args.gsm8k_model_id}")
    print(f"  ARC   model: {args.arc_model_id}")
    print(f"  max_gen_tokens: {max_new_tokens}")

    dtype = DTYPE_MAP[args.dtype]

    gsm8k_problems = _load_gsm8k(n_gsm8k) if n_gsm8k > 0 else []
    arc_problems = _load_arc_easy(n_arc) if (run_arc and n_arc > 0) else []

    # --- Pass 1: GSM8K on the IT model with chat template + CoT ---
    def make_gsm8k_formatter(tokenizer):
        def fmt(problem):
            return _format_gsm8k_prompt_chat(problem, tokenizer)
        return fmt

    gsm8k_per_config = {}
    gsm8k_inspection = None
    if gsm8k_problems:
        gsm8k_per_config, gsm8k_inspection = _run_pass(
            args=args,
            configs=configs,
            model_id=args.gsm8k_model_id,
            dtype=dtype,
            dataset_label="gsm8k",
            problems=gsm8k_problems,
            ckpt_dir=ckpt_dir,
            format_prompt_factory=make_gsm8k_formatter,
            score=_score_gsm8k,
            stop_strings=GSM8K_STOP_STRINGS,
            max_new_tokens=max_new_tokens,
        )

    # --- Pass 2: ARC-Easy on the base model with raw prompt (round-4 parity) ---
    def make_arc_formatter(tokenizer):
        # ARC prompt is tokenizer-independent; factory just binds.
        return _format_arc_prompt

    arc_per_config = {}
    arc_inspection = None
    if arc_problems:
        arc_per_config, arc_inspection = _run_pass(
            args=args,
            configs=configs,
            model_id=args.arc_model_id,
            dtype=dtype,
            dataset_label="arc_easy",
            problems=arc_problems,
            ckpt_dir=ckpt_dir,
            format_prompt_factory=make_arc_formatter,
            score=_score_arc,
            stop_strings=None,
            max_new_tokens=max_new_tokens,
        )

    # --- Summaries, matrices, sanity checks ---
    analysis = {}
    if gsm8k_per_config:
        analysis["gsm8k"] = {
            "summary": _summary_table(gsm8k_per_config, "GSM8K"),
            "agreement_matrix": _agreement_matrix(gsm8k_per_config),
        }
    if arc_per_config:
        analysis["arc_easy"] = {
            "summary": _summary_table(arc_per_config, "ARC-Easy"),
            "agreement_matrix": _agreement_matrix(arc_per_config),
        }

    sanity = {}

    # W5-r1 bitwise check on both datasets: the hook is a no-op at r=1
    # and must produce identical generations to the unhooked baseline.
    # Plan5 explicitly requires this on both benchmarks.
    for ds_key, per_cfg, total_wanted in (
        ("arc",   arc_per_config,   len(arc_problems)),
        ("gsm8k", gsm8k_per_config, len(gsm8k_problems)),
    ):
        match = _w5_r1_bitwise_match(per_cfg)
        if match is not None:
            print(
                f"\n=== W5-r1 vs baseline sanity ({ds_key}, "
                f"intersection={match['total']}) ===\n"
                f"  {match['matches']}/{match['total']} generated strings match "
                f"bitwise."
            )
            if match["matches"] != match["total"]:
                print(
                    "  WARNING: W5-r1 must be a bitwise no-op w.r.t. baseline. "
                    "Hook / PLE / cache plumbing has regressed; the sweep is "
                    "not interpretable until this is fixed."
                )
            sanity[f"w5_r1_vs_baseline_{ds_key}"] = match

    # Cross-round W5-r8 vs round-4 A-r8 on ARC-Easy. Plan5 wants a delta
    # with +/-2% tolerance. Reproducing round-4 numbers proves the
    # harness / weights / generation config didn't drift between rounds.
    if arc_per_config and "W5-r8" in arc_per_config:
        w5_rows = arc_per_config["W5-r8"]
        if w5_rows:
            w5_acc = sum(1 for r in w5_rows if r["correct"]) / len(w5_rows)
            a_r8 = _load_round4_a_r8_arc_accuracy(
                getattr(args, "round4_json", None)
                or _results_path("results_round4_reasoning.json")
            )
            if a_r8 is not None:
                delta = w5_acc - a_r8
                within = abs(delta) <= 0.02
                sanity["w5_r8_vs_round4_a_r8"] = {
                    "round4_accuracy": a_r8,
                    "round5_accuracy": w5_acc,
                    "delta": delta,
                    "within_tolerance": within,
                }
                tag = "OK" if within else "OUT OF TOLERANCE"
                print(
                    f"\n=== W5-r8 vs round-4 A-r8 (ARC-Easy) ===\n"
                    f"  round 4 A-r8 accuracy: {a_r8:.3f}\n"
                    f"  round 5 W5-r8 accuracy: {w5_acc:.3f}\n"
                    f"  delta: {delta:+.3f}  [{tag}]"
                )

    output = {
        "config": {
            "mode": "reasoning-eval-r5",
            "gsm8k_model_id": args.gsm8k_model_id,
            "arc_model_id": args.arc_model_id,
            "configs": [
                {
                    "name": c["name"],
                    "block": c["block"],
                    "r": c["r"],
                    "ple_strategy": c["ple_strategy"],
                }
                for c in configs
            ],
            "num_gsm8k_problems": n_gsm8k,
            "num_arc_problems": n_arc if run_arc else 0,
            "run_arc": run_arc,
            "max_gen_tokens": max_new_tokens,
            "dtype": args.dtype,
            "gsm8k_stop_strings": GSM8K_STOP_STRINGS,
            "checkpoint_dir": str(ckpt_dir),
            "round4_comparison_path": getattr(args, "round4_json", None)
                or _results_path("results_round4_reasoning.json"),
        },
        "inspections": {
            "gsm8k_model": _inspection_summary(gsm8k_inspection),
            "arc_model":   _inspection_summary(arc_inspection),
        },
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


def _inspection_summary(inspection):
    if inspection is None:
        return None
    return {
        "strategy": inspection["strategy"],
        "layer_class": inspection["layer_class"],
        "source_file": inspection["source_file"],
        "start_lineno": inspection["start_lineno"],
        "signature": inspection["signature"],
        "ple_kwarg": inspection["ple_kwarg"],
    }


def summarize_checkpoints_r5(args):
    """CPU-only inspector: read JSONLs from the round-5 checkpoint dir
    and print the plan5 summary table. Useful mid-run from a second
    shell while the GPU keeps generating.

    Iterates every ``{dataset}__{config}.jsonl`` in the checkpoint dir
    so the summary works even if --configs differs between the running
    job and this inspector.
    """
    ckpt_dir = _checkpoint_dir(args)
    if not ckpt_dir.exists():
        print(f"No checkpoint dir at {ckpt_dir} --- nothing to summarize.")
        return

    mp = _manifest_path(ckpt_dir)
    if mp.exists():
        print(f"Manifest ({mp}):")
        print("  " + mp.read_text().replace("\n", "\n  ").rstrip())
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
        print(f"=== {dataset} progress ===")
        max_n = max(len(rows) for rows in per_config.values())
        for name in sorted(per_config):
            n = len(per_config[name])
            tag = "" if n == max_n else f"  ({max_n - n} behind leader)"
            print(f"  {name:>12}: {n} problems{tag}")
        _summary_table(per_config, dataset)
        match = _w5_r1_bitwise_match(per_config)
        if match is not None and match["total"] > 0:
            print(
                f"\n=== W5-r1 vs baseline sanity ({dataset}, "
                f"intersection={match['total']}) ==="
            )
            print(
                f"  {match['matches']}/{match['total']} generated strings "
                f"match bitwise."
            )
            if match["matches"] != match["total"]:
                print("  WARNING: W5-r1 should be bitwise-identical to baseline.")
        if len(per_config) >= 2:
            mat = _agreement_matrix(per_config)
            print(f"\n=== {dataset} pairwise agreement ===")
            for key, cell in mat.items():
                print(
                    f"  {cell['a']:>12} vs {cell['b']:<12}  "
                    f"both_correct={cell['both_correct']:>3}  "
                    f"only_a={cell['only_a_correct']:>3}  "
                    f"only_b={cell['only_b_correct']:>3}  "
                    f"both_wrong={cell['both_wrong']:>3}  "
                    f"(n={cell['n_common']})"
                )
        print()
