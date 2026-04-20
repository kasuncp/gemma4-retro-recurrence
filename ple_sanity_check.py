"""
PLE x Loop Recurrence Probes for Gemma 4 E2B.

Round 1 (mode=original): naively loop a single middle decoder layer `r` times
and confirm graceful perplexity degradation.

Round 2a (mode=ple-variants): compare three PLE injection policies under the
same loop --- vanilla / scaled / once --- across r in {1, 4, 8}.

Round 2b (mode=ple-importance-scan): vanilla vs zero at r=1 across all 35
layers, to quantify per-layer PLE contribution to perplexity.

Round 2b (mode=layer-location): {vanilla, once} x {r=4, r=8} across three
layer locations (default: 5, 17, 28), to test whether "once beats vanilla"
at layer 17 generalises across depths.

Round 2c (mode=full-looping-map): 35 layers x {vanilla, once} x {r=2, r=4,
r=8} = 210 cells, plus per-layer r=1 regression checks, plus correlation
analysis against PLE importance / attention type / KV role / layer depth.

Round 3a (mode=pair-looping-map): 34 pair starting positions (L, L+1) x
{r=2, r=4, r=8} = 102 cells, vanilla PLE only. Each cell loops the pair
as a unit: inside one forward pass, run layer L then L+1, then feed that
output back as the next input to layer L, repeating r times. Directly
comparable to round 2c single-layer vanilla results; tests the "cascade
hypothesis" for block looping.

Round 3b (mode=block-looping): 6 hand-picked contiguous blocks (widths
4-10, anchored in/around the round-2c valley) x {r=2, r=4, r=8} = 18
cells, vanilla PLE only. Each cell loops the entire block as a unit.
Discriminates between "valley is a hard ceiling" (only a tight 4-5 layer
block is loopable) and "valley is a floor" (a wider valley-anchored
block works because within-block re-processing dominates at scale).

Round 3c (mode=block-looping-3c): 6 blocks x {r=2, r=4, r=8} = 18
cells, vanilla PLE only. Re-runs A (15-19), B (15-18), D (15-22) as
references from round 3b, plus three new KV-consumer probes: G (15-24,
10 layers), H (15-23, 9 layers), I (20-27, 8 layers shifted late). Pins
down the maximum viable block width (G vs H vs D) and tests whether
block position matters within the consumer zone (I vs D).

Round 4 (mode=reasoning-eval): 7 configurations (baseline, D-r4, D-r8,
G-r4, G-r8, A-r8, D-r1) x zero-shot GSM8K (and optionally ARC-Easy).
Greedy decoding, KV cache disabled during generation (the loop hooks
re-enter decoder layers so per-step caching is unsafe). Produces per-
configuration accuracy, a pairwise agreement matrix, and a D-r1 vs
baseline bitwise check. Answers plan4's question: does recurrence
preserve, help, or degrade reasoning zero-shot?

Cloud-server friendly: argparse flags, HF cache redirect to persistent
volume, env-info dump, JSON result save.

Recommended RunPod usage:
    # PyTorch 2.x template, single GPU >= 16 GB VRAM
    pip install -U transformers datasets accelerate
    export HF_TOKEN=hf_xxx                       # accept Gemma license first

    # Round 1 (original sanity check, default --- backward compatible):
    python ple_sanity_check.py

    # Round 2a (PLE variants probe):
    python ple_sanity_check.py --mode ple-variants

    # Round 2b (per-layer PLE importance scan):
    python ple_sanity_check.py --mode ple-importance-scan

    # Round 2b (layer-location x PLE policy sweep):
    python ple_sanity_check.py --mode layer-location

    # Round 2c (full 210-cell looping tolerance map):
    python ple_sanity_check.py --mode full-looping-map

    # Round 3a (pair-of-layers looping probe, 102 cells):
    python ple_sanity_check.py --mode pair-looping-map

    # Round 3b (block-looping probe, 18 cells):
    python ple_sanity_check.py --mode block-looping

    # Round 3c (extended-blocks probe, 18 cells = A/B/D refs + G/H/I):
    python ple_sanity_check.py --mode block-looping-3c

    # Round 4 (zero-shot reasoning eval: 7 configs x GSM8K +/- ARC):
    python ple_sanity_check.py --mode reasoning-eval
"""

import argparse
import sys

# Importing `probes` runs the HF cache redirect BEFORE any submodule loads
# torch/transformers. Keep this at the top.
import probes  # noqa: F401
from probes.env import (
    DTYPE_MAP,
    EXPECTED_NUM_LAYERS,
    MODEL_ID,
    _results_path,
    load_model,
    print_env,
)
from probes.data import prepare_inputs
from probes.introspect import find_decoder_layers
from probes.mode_round1 import run_original_mode
from probes.mode_round2a import run_ple_variants_mode, run_zero_diagnostic
from probes.mode_round2b import run_importance_scan_mode, run_layer_location_mode
from probes.mode_round2c import run_full_looping_map
from probes.mode_round3a import run_pair_looping_map
from probes.mode_round3b import run_block_looping_map
from probes.mode_round3c import run_block_looping_3c_map
from probes.mode_round4 import run_reasoning_eval_mode


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--mode",
        choices=[
            "original",
            "ple-variants",
            "ple-importance-scan",
            "layer-location",
            "full-looping-map",
            "pair-looping-map",
            "block-looping",
            "block-looping-3c",
            "reasoning-eval",
        ],
        default="original",
        help=(
            "original = round-1 sanity check; "
            "ple-variants = round-2a probe; "
            "ple-importance-scan = round-2b per-layer vanilla-vs-zero at r=1; "
            "layer-location = round-2b {vanilla, once} x {r=4,8} across layers; "
            "full-looping-map = round-2c 35 x {vanilla,once} x {r=2,4,8} sweep; "
            "pair-looping-map = round-3a 34 pairs x {r=2,4,8}, vanilla only; "
            "block-looping = round-3b 6 blocks x {r=2,4,8}, vanilla only; "
            "block-looping-3c = round-3c A/B/D refs + G/H/I extension probes "
            "x {r=2,4,8}, vanilla only; "
            "reasoning-eval = round-4 zero-shot GSM8K (+/- ARC) over 7 "
            "configs (baseline, D-r4, D-r8, G-r4, G-r8, A-r8, D-r1)"
        ),
    )
    p.add_argument("--target-layer", type=int, default=17)
    p.add_argument(
        "--r-values",
        type=int,
        nargs="+",
        default=None,
        help=(
            "r values to sweep. "
            "Default: [1,2,4,8] for original, [1,4,8] for ple-variants, "
            "[1] for ple-importance-scan, [4,8] for layer-location, "
            "[2,4,8] for full-looping-map, pair-looping-map, "
            "block-looping, and block-looping-3c (r=1 is used as a "
            "regression check only)."
        ),
    )
    p.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Space-separated layer indices. "
            "ple-importance-scan default: all 0..34. "
            "layer-location default: 5 17 28. "
            "full-looping-map default: all 0..34. "
            "pair-looping-map: pair starting indices L (pairs are (L, L+1)); "
            "default 0..33."
        ),
    )
    p.add_argument(
        "--round2c-json",
        default=_results_path("results_round2c_full_map.json"),
        help=(
            "Path to round-2c full-looping-map JSON. Used by --mode "
            "pair-looping-map to merge single-layer vanilla ppl into the "
            "comparison table (pair vs min/geom-mean of single-layer), "
            "and by --mode block-looping to merge single-layer vanilla "
            "ppl for the worst/best-single-in-block comparison columns. "
            "Default: results/results_round2c_full_map.json."
        ),
    )
    p.add_argument(
        "--blocks",
        nargs="+",
        default=None,
        help=(
            "Block specs as space-separated START-END pairs (e.g. "
            "--blocks 15-19 12-19 13-22). Used under --mode=block-looping "
            "and --mode=block-looping-3c. Defaults: the 6 plan3b candidates "
            "(A 15-19, B 15-18, C 12-19, D 15-22, E 13-22, F 25-32) for "
            "block-looping; the 6 plan3c candidates (A 15-19, B 15-18, "
            "D 15-22, G 15-24, H 15-23, I 20-27) for block-looping-3c."
        ),
    )
    p.add_argument(
        "--importance-json",
        default=_results_path("results_round2b_importance.json"),
        help=(
            "Path to round-2b PLE importance scan JSON. Used by --mode "
            "full-looping-map to merge per-layer PLE importance into the "
            "output and compute correlations against it. Default: "
            "results/results_round2b_importance.json."
        ),
    )
    p.add_argument("--num-sequences", type=int, default=50)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument(
        "--output-json",
        default=None,
        help=(
            "Output JSON path. If relative, it is interpreted as given (no "
            "implicit results/ prefix). Defaults are routed under results/: "
            "results/results.json (original), results/results_round2a.json "
            "(ple-variants), results/results_round2b_importance.json "
            "(ple-importance-scan), results/results_round2b_location.json "
            "(layer-location), results/results_round2c_full_map.json "
            "(full-looping-map), results/results_round3a_pair_looping.json "
            "(pair-looping-map), results/results_round3b_blocks.json "
            "(block-looping), results/results_round3c_extended_blocks.json "
            "(block-looping-3c)."
        ),
    )
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument(
        "--only-diagnostic",
        action="store_true",
        help="plan2a-add1: under --mode ple-variants, run only the zero-PLE "
             "diagnostic (vanilla r=1 vs zero r=1) and skip the full grid.",
    )

    # Round 4 (reasoning-eval) specific flags.
    p.add_argument(
        "--num-gsm8k-problems",
        type=int,
        default=250,
        help="reasoning-eval: number of GSM8K problems to run (first N of "
             "test split). Set to 0 to skip GSM8K. Default 250.",
    )
    p.add_argument(
        "--num-arc-problems",
        type=int,
        default=200,
        help="reasoning-eval: number of ARC-Easy problems to run (first N of "
             "test split). Ignored unless --run-arc is passed. Default 200.",
    )
    p.add_argument(
        "--max-gen-tokens",
        type=int,
        default=256,
        help="reasoning-eval: max new tokens per generation. Default 256.",
    )
    p.add_argument(
        "--run-arc",
        action="store_true",
        help="reasoning-eval: also run ARC-Easy as a secondary axis. GSM8K "
             "runs unconditionally (unless --num-gsm8k-problems 0). ARC is "
             "opt-in because it roughly doubles eval time.",
    )
    p.add_argument(
        "--configs",
        nargs="+",
        default=None,
        help="reasoning-eval: restrict to a subset of configuration names "
             "(baseline, D-r4, D-r8, G-r4, G-r8, A-r8, D-r1). Preserves "
             "the order given. Default: all 7.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    print_env()

    dtype = DTYPE_MAP[args.dtype]
    print(f"Loading {args.model_id} in {args.dtype} ...")
    tokenizer, model = load_model(args.model_id, dtype)

    decoder_layers = find_decoder_layers(model)
    if len(decoder_layers) != EXPECTED_NUM_LAYERS:
        print(
            f"WARNING: expected {EXPECTED_NUM_LAYERS} decoder layers, "
            f"found {len(decoder_layers)}. Continuing anyway."
        )
    if args.target_layer >= len(decoder_layers):
        print(f"ERROR: target-layer={args.target_layer} out of range.")
        sys.exit(1)

    # Wikitext perplexity inputs are unused by reasoning-eval. Skip the
    # dataset download + tokenisation to keep that mode snappy.
    if args.mode != "reasoning-eval":
        inputs = prepare_inputs(tokenizer, args.num_sequences, args.max_length)
    else:
        inputs = None

    if args.mode == "original":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_original_mode(args, model, decoder_layers, inputs)
    elif args.mode == "ple-variants":
        if args.only_diagnostic:
            run_zero_diagnostic(args, model, decoder_layers, inputs)
        else:
            run_ple_variants_mode(args, model, decoder_layers, inputs)
    elif args.mode == "ple-importance-scan":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_importance_scan_mode(args, model, decoder_layers, inputs)
    elif args.mode == "layer-location":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_layer_location_mode(args, model, decoder_layers, inputs)
    elif args.mode == "full-looping-map":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_full_looping_map(args, model, decoder_layers, inputs)
    elif args.mode == "pair-looping-map":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_pair_looping_map(args, model, decoder_layers, inputs)
    elif args.mode == "block-looping":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_block_looping_map(args, model, decoder_layers, inputs)
    elif args.mode == "block-looping-3c":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_block_looping_3c_map(args, model, decoder_layers, inputs)
    elif args.mode == "reasoning-eval":
        if args.only_diagnostic:
            print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
            sys.exit(1)
        run_reasoning_eval_mode(args, model, tokenizer, decoder_layers)
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
