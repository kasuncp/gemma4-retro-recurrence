"""Path 2 v2 unified eval entry point (Phase 0 -> Phase 4).

Modes:
  --summarize-only   : aggregate JSONL cells in --output-dir and print
                       the compact table; no model load. Runs on a CPU
                       laptop. This is the only mode that should run
                       on Phase 0 acceptance.
  (default)          : load model + tokenizer, fetch benchmark, run
                       generation per problem with a block-loop hook
                       wired according to --config, append to JSONL.
                       Requires GPU; Phase 1+ only.

Per-cell output:
  <output-dir>/<benchmark>__<config>.jsonl       per-problem rows
  <output-dir>/<benchmark>__<config>.summary.json summary (lazy)

Resume: re-running the same (config, benchmark, output-dir) appends
problems whose ``idx`` isn't already in the JSONL. Pass --no-resume
to wipe the cell first.

Examples:

  # CPU laptop, Phase 0 acceptance (no GPU needed):
  python experiments/path2_v2_eval.py --summarize-only \\
      --output-dir tests/fixtures/path2_v2_smoke/

  # Phase 1 sanity gate:
  python experiments/path2_v2_eval.py --phase sanity --config baseline-C2 \\
      --benchmark gsm8k --n 50 \\
      --output-dir results/path_2_depth_recurrence_v2/sanity/

  # Phase 1 r=1 token-match check:
  python experiments/path2_v2_eval.py --phase sanity --config W5-r1 \\
      --benchmark gsm8k --n 20 \\
      --output-dir results/path_2_depth_recurrence_v2/sanity/

The model + hook layer is intentionally guarded behind the
``--summarize-only`` short-circuit: pure-Python summarisation must
work on a laptop with no torch in the environment.
"""

import argparse
import sys
import time
from pathlib import Path

# Importing ``probes`` triggers HF_HOME redirect; safe even on CPU.
import probes  # noqa: F401
from probes import eval_v3
from probes.eval_v3 import (
    CONFIGS, append_row, cell_jsonl_path, cell_summary_path,
    existing_idxs, get_config, print_summary_table, read_jsonl,
    summarise,
)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--phase",
        choices=["sanity", "layer-map", "block-sweep", "r-sweep"],
        default="sanity",
        help="Phase identifier; recorded in the summary JSON for "
             "downstream filtering. Does not change harness behaviour.",
    )
    p.add_argument(
        "--config",
        default="baseline-C2",
        help=(
            f"Config name from probes.eval_v3.CONFIGS. "
            f"Registered: {sorted(CONFIGS)}."
        ),
    )
    p.add_argument(
        "--benchmark",
        choices=["gsm8k", "arc-c", "bbh-lite"],
        default="gsm8k",
    )
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--model-id", default="google/gemma-4-E2B-it")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument(
        "--output-dir",
        default="results/path_2_depth_recurrence_v2/sanity",
    )
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help="Skip model load; just aggregate JSONL in --output-dir "
             "and print the table. Safe on a CPU laptop.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Wipe the cell's JSONL before running.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# CPU-only path (summarize-only)
# ---------------------------------------------------------------------------

def _summarize_only(args) -> int:
    print_summary_table(Path(args.output_dir), benchmark=None)
    return 0


# ---------------------------------------------------------------------------
# Generation path (Phase 1+, requires GPU)
# ---------------------------------------------------------------------------

def _import_torch_and_friends():
    """Lazy-import the GPU stack so --summarize-only never hits torch.

    Returns ``(torch, find_decoder_layers, install_block_loop_hooks,
    inspect_strategy_a, load_model, print_env, DTYPE_MAP)``.
    """
    import torch  # noqa: F401
    from probes.env import DTYPE_MAP, load_model, print_env
    from probes.hooks import install_block_loop_hooks
    from probes.introspect import (
        _inspect_and_require_strategy_a,
        find_decoder_layers,
    )
    return (
        torch, find_decoder_layers, install_block_loop_hooks,
        _inspect_and_require_strategy_a, load_model, print_env, DTYPE_MAP,
    )


def _build_prompt(tokenizer, *, benchmark: str, prompt_name: str, row: dict) -> str:
    from probes.prompts import build_prompt
    return build_prompt(
        tokenizer, benchmark=benchmark, prompt_name=prompt_name, row=row,
    )


def _score_row(*, benchmark: str, row: dict, completion: str,
               n_gen_tokens: int, t_gen_seconds: float, max_new_tokens: int) -> dict:
    """Dispatch to the right per-row scorer in probes.extractors."""
    from probes import extractors
    if benchmark == "gsm8k":
        out = extractors.score_gsm8k_row(
            idx=row["idx"], question=row["question"], gold=row["gold"],
            completion=completion, n_gen_tokens=n_gen_tokens,
            max_new_tokens=max_new_tokens,
        )
    elif benchmark == "arc-c":
        out = extractors.score_arc_row(
            idx=row["idx"], question=row["question"],
            choices=row["choices"], gold=row["gold"],
            gold_letter=row["gold_letter"],
            completion=completion, n_gen_tokens=n_gen_tokens,
            max_new_tokens=max_new_tokens,
        )
    elif benchmark == "bbh-lite":
        out = extractors.score_bbh_row(
            idx=row["idx"], task=row.get("task"),
            question=row["question"], gold=row["gold"],
            completion=completion, n_gen_tokens=n_gen_tokens,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"unknown benchmark {benchmark!r}")
    out["t_gen_seconds"] = t_gen_seconds
    return out


def _run_one_cell(args, cfg: dict) -> int:
    (
        torch, find_decoder_layers, install_block_loop_hooks,
        inspect_strategy_a, load_model, print_env, DTYPE_MAP,
    ) = _import_torch_and_friends()

    print_env()
    output_dir = Path(args.output_dir)
    jsonl = cell_jsonl_path(
        output_dir=output_dir,
        benchmark=args.benchmark,
        config_name=args.config,
    )
    if args.no_resume and jsonl.is_file():
        jsonl.unlink()
        print(f"--no-resume: wiped {jsonl}")

    done = existing_idxs(jsonl)
    if done:
        print(f"resuming: {len(done)} rows already in {jsonl}")

    # Load benchmark.
    from probes.datasets_v2 import load_benchmark
    rows = load_benchmark(args.benchmark, n=args.n, seed=args.seed)
    rows_to_run = [r for r in rows if r["idx"] not in done]
    print(
        f"benchmark={args.benchmark} n_total={len(rows)} "
        f"n_remaining={len(rows_to_run)} cell={args.config}"
    )
    if not rows_to_run:
        print("nothing to do; emitting summary.")
        _emit_summary(args, jsonl)
        return 0

    # Load model + tokenizer.
    dtype = DTYPE_MAP[args.dtype]
    print(f"Loading {args.model_id} in {args.dtype} ...")
    tokenizer, model = load_model(args.model_id, dtype)
    decoder_layers = find_decoder_layers(model)

    # Wire the block-loop hook for the duration of the cell.
    block = cfg.get("block")
    r = cfg.get("r", 1)
    ple_strategy = cfg.get("ple_strategy", "every-iter")
    uninstall = None
    if block is not None:
        # Strategy-A guard so a transformers update doesn't silently
        # break PLE plumbing.
        inspect_strategy_a(decoder_layers[block[0]])
        uninstall = install_block_loop_hooks(
            decoder_layers, block[0], block[1],
            r=r, ple_strategy=ple_strategy,
        )
        print(
            f"block-loop hook installed: layers [{block[0]}, {block[1]}], "
            f"r={r}, ple_strategy={ple_strategy}"
        )

    try:
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=False,  # structural; required by the loop hook
            pad_token_id=tokenizer.eos_token_id,
        )
        for i, row in enumerate(rows_to_run, 1):
            prompt = _build_prompt(
                tokenizer,
                benchmark=args.benchmark,
                prompt_name=cfg["prompt"],
                row=row,
            )
            inputs = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False,
            ).to("cuda")
            t0 = time.time()
            with torch.no_grad():
                out_ids = model.generate(**inputs, **gen_kwargs)
            t1 = time.time()
            gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
            completion = tokenizer.decode(
                gen_ids, skip_special_tokens=True,
            )
            n_gen_tokens = int(gen_ids.shape[0])
            scored = _score_row(
                benchmark=args.benchmark,
                row=row,
                completion=completion,
                n_gen_tokens=n_gen_tokens,
                t_gen_seconds=t1 - t0,
                max_new_tokens=args.max_new_tokens,
            )
            append_row(jsonl, scored)
            if i % 10 == 0 or i == len(rows_to_run):
                print(
                    f"  [{i}/{len(rows_to_run)}] idx={row['idx']} "
                    f"n_tok={n_gen_tokens} t={t1 - t0:.1f}s"
                )
    finally:
        if uninstall is not None:
            uninstall()

    _emit_summary(args, jsonl)
    return 0


def _emit_summary(args, jsonl: Path) -> None:
    rows = read_jsonl(jsonl)
    summary = summarise(rows, args.benchmark)
    out = {
        "phase": args.phase,
        "config": args.config,
        "config_dict": get_config(args.config),
        "benchmark": args.benchmark,
        "n_requested": args.n,
        "model_id": args.model_id,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "summary": summary,
    }
    p = cell_summary_path(
        output_dir=Path(args.output_dir),
        benchmark=args.benchmark,
        config_name=args.config,
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    import json
    p.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {p}")
    print_summary_table(Path(args.output_dir), benchmark=args.benchmark)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if args.summarize_only:
        return _summarize_only(args)

    cfg = get_config(args.config)
    return _run_one_cell(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
