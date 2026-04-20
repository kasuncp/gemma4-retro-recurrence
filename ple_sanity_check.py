"""
PLE x Loop Recurrence Probes for Gemma 4 E2B.

Round 1 (mode=original): naively loop a single middle decoder layer `r` times
and confirm graceful perplexity degradation.

Round 2a (mode=ple-variants): compare three PLE injection policies under the
same loop --- vanilla / scaled / once --- across r in {1, 4, 8}.

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
"""

import argparse
import inspect
import json
import math
import os
import sys
from pathlib import Path

# --- Bootstrap: redirect HF cache to persistent volume if present ----------
# RunPod (and most cloud providers) mount persistent storage at /workspace.
# Putting the HF cache there means the ~5 GB model download survives pod
# restarts.
_PERSISTENT = Path("/workspace")
if _PERSISTENT.is_dir() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_PERSISTENT / ".cache" / "huggingface")

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

MODEL_ID = "google/gemma-4-E2B"
EXPECTED_NUM_LAYERS = 35
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Round 1's anchor for regression check 1.
ROUND1_UNMODIFIED_PPL = 12.53165455614523


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--mode",
        choices=["original", "ple-variants"],
        default="original",
        help="original = round-1 sanity check; ple-variants = round-2a probe",
    )
    p.add_argument("--target-layer", type=int, default=17)
    p.add_argument(
        "--r-values",
        type=int,
        nargs="+",
        default=None,
        help="r values to sweep. Default: [1,2,4,8] for original, [1,4,8] for ple-variants",
    )
    p.add_argument("--num-sequences", type=int, default=50)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument("--output-json", default=None,
                   help="default: results.json (original) / results_round2a.json (ple-variants)")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument(
        "--only-diagnostic",
        action="store_true",
        help="plan2a-add1: under --mode ple-variants, run only the zero-PLE "
             "diagnostic (vanilla r=1 vs zero r=1) and skip the full grid.",
    )
    return p.parse_args()


def print_env():
    print("=" * 60)
    print(f"torch:        {torch.__version__}")
    import transformers
    print(f"transformers: {transformers.__version__}")
    print(f"CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"GPU:          {props.name} ({props.total_memory / 1e9:.1f} GB)")
        print(f"capability:   sm_{props.major}{props.minor}")
    print(f"HF_HOME:      {os.environ.get('HF_HOME', '(default)')}")
    has_token = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    print(f"HF token set: {has_token}")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    if not has_token:
        print(
            "WARNING: no HF_TOKEN / HUGGING_FACE_HUB_TOKEN set. Gemma is a "
            "gated model --- load will fail unless you've cached the weights "
            "or logged in via `huggingface-cli login`."
        )


def load_model(model_id, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="cuda",
        )
    except (ValueError, KeyError) as e:
        print(f"AutoModelForCausalLM failed ({e}); trying multimodal loader.")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map="cuda",
        )
    model.train(False)  # inference mode
    return tokenizer, model


def find_decoder_layers(model):
    """Locate the ModuleList of decoder layers, handling multimodal wrappers."""
    candidates = [
        ("model.model.layers", lambda m: m.model.layers),
        ("model.language_model.model.layers", lambda m: m.language_model.model.layers),
        ("model.model.language_model.layers", lambda m: m.model.language_model.layers),
        ("model.language_model.layers", lambda m: m.language_model.layers),
    ]
    for path, getter in candidates:
        try:
            layers = getter(model)
            if hasattr(layers, "__len__") and len(layers) > 0:
                print(f"Found decoder layers at: {path}  (count={len(layers)})")
                return layers
        except AttributeError:
            continue
    raise RuntimeError(
        "Could not locate decoder layers. Print `model` to find the path."
    )


def prepare_inputs(tokenizer, num_sequences, max_length):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 500][:num_sequences]
    inputs = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
        for t in texts
    ]
    print(f"Prepared {len(inputs)} eval sequences (max_length={max_length}).")
    return inputs


def make_looped_forward(orig_forward, r):
    """Round-1 hook: naive loop, no PLE control."""
    def looped(hidden_states, *args, **kwargs):
        out = None
        for _ in range(r):
            out = orig_forward(hidden_states, *args, **kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped


def compute_perplexity(model, inputs):
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for inp in inputs:
            out = model(**inp, labels=inp["input_ids"])
            n_tokens = inp["input_ids"].numel() - 1
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens
    mean_nll = total_nll / total_tokens
    return mean_nll, math.exp(mean_nll)


# ============================================================================
# Round 2a: PLE variants
# ============================================================================

# Candidate names for the PLE kwarg on a Gemma decoder layer's forward.
# Different transformers versions have used slightly different names for the
# per-layer-embedding tensor; we'll accept any of these.
PLE_KWARG_CANDIDATES = (
    "per_layer_input",
    "per_layer_inputs",
    "per_layer_embedding",
    "ple",
    "ple_input",
)


def inspect_ple_strategy(decoder_layer):
    """
    Step 0 of plan2a: locate where PLE is plumbed in the decoder layer.

    Returns dict with:
      - strategy: "A" | "B" | "C"
      - ple_kwarg: name of the PLE kwarg (Strategy A) or None
      - source_file: path to modeling file
      - source: layer.forward source text (for diagnostic dump)
      - signature: str of the forward signature
      - layer_class: class name of the decoder layer
    """
    layer_cls = type(decoder_layer)
    fwd = layer_cls.forward
    sig = inspect.signature(fwd)
    try:
        source_file = inspect.getfile(layer_cls)
    except TypeError:
        source_file = "<unknown>"
    try:
        src_lines, start_lineno = inspect.getsourcelines(fwd)
        source = "".join(src_lines)
    except OSError:
        start_lineno = -1
        source = "<source unavailable>"

    ple_kwarg = next(
        (name for name in PLE_KWARG_CANDIDATES if name in sig.parameters), None,
    )

    if ple_kwarg is not None:
        strategy = "A"
    else:
        # No PLE kwarg --- check if PLE is referenced inside forward by name
        # (computed there or fused in), or applied entirely outside.
        if any(name in source for name in PLE_KWARG_CANDIDATES) or "per_layer" in source:
            strategy = "C"  # PLE referenced internally; complex/unknown wiring
        else:
            strategy = "B"  # No PLE reference in layer at all --- applied outside

    return {
        "strategy": strategy,
        "ple_kwarg": ple_kwarg,
        "source_file": source_file,
        "start_lineno": start_lineno,
        "signature": str(sig),
        "source": source,
        "layer_class": layer_cls.__name__,
    }


def print_ple_inspection(report):
    print("=" * 60)
    print("PLE wiring inspection (plan2a Step 0)")
    print(f"  layer class:   {report['layer_class']}")
    print(f"  source file:   {report['source_file']}:{report['start_lineno']}")
    print(f"  signature:     {report['signature']}")
    print(f"  ple kwarg:     {report['ple_kwarg']}")
    print(f"  strategy:      {report['strategy']}")
    print("=" * 60)


def make_looped_forward_ple(orig_forward, r, ple_mode, ple_kwarg):
    """
    Round-2a hook: loop the decoder layer `r` times with controlled PLE
    re-injection.

    ple_mode in {"vanilla", "scaled", "once", "zero"}.
      - "zero" (plan2a-add1): always pass scale=0.0, including at i=0.

    Approach: leave the decoder layer's forward untouched; intercept the PLE
    kwarg and multiply by the chosen scale before delegating. This avoids any
    monkey-patching of model source code.

    For r=1 with mode in {scaled, once}, ple_scale == 1.0 by construction,
    and we pass the original tensor unchanged --- so plan2a's regression
    checks 2 and 3 are bitwise-identical to vanilla r=1.
    """
    def looped(hidden_states, *args, **kwargs):
        original_ple = kwargs.get(ple_kwarg, None)
        out = None
        for i in range(r):
            if ple_mode == "vanilla":
                scale = 1.0
            elif ple_mode == "scaled":
                scale = 1.0 / r
            elif ple_mode == "once":
                scale = 1.0 if i == 0 else 0.0
            elif ple_mode == "zero":
                scale = 0.0  # plan2a-add1: always zero, from iteration 0
            else:
                raise ValueError(f"unknown ple_mode={ple_mode!r}")

            call_kwargs = dict(kwargs)
            if original_ple is not None and scale != 1.0:
                # scale=0.0 zeros out PLE; fractional scales dilute it.
                call_kwargs[ple_kwarg] = original_ple * scale
            # else: scale==1.0 --- pass original tensor unchanged for bitwise identity.

            out = orig_forward(hidden_states, *args, **call_kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped


def run_original_mode(args, model, decoder_layers, inputs):
    """Round-1 sanity check, preserving exact prior behaviour."""
    r_values = args.r_values if args.r_values is not None else [1, 2, 4, 8]
    output_json = args.output_json or "results.json"

    print("\nRunning unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.2f}")

    orig_forward = decoder_layers[args.target_layer].forward
    results = {}
    try:
        for r in r_values:
            decoder_layers[args.target_layer].forward = make_looped_forward(orig_forward, r)
            mean_nll, ppl = compute_perplexity(model, inputs)
            results[r] = {"mean_nll": mean_nll, "ppl": ppl}
            print(f"r={r:2d}:  mean NLL = {mean_nll:.4f}   perplexity = {ppl:.2f}")
    finally:
        decoder_layers[args.target_layer].forward = orig_forward

    print("\n=== Summary ===")
    baseline_ppl = results[r_values[0]]["ppl"]
    summary = []
    for r, res in results.items():
        ratio = res["ppl"] / baseline_ppl
        line = f"r={r:2d}:  ppl={res['ppl']:7.2f}   (x{ratio:.2f} vs r={r_values[0]})"
        print(line)
        summary.append({"r": r, "ppl": res["ppl"], "ratio": ratio})

    drift = abs(results[1]["ppl"] - unmod_ppl) / unmod_ppl if 1 in results else None
    if drift is not None:
        print(
            f"\nHook sanity: r=1 ppl={results[1]['ppl']:.4f} vs unmodified "
            f"ppl={unmod_ppl:.4f} (relative drift={drift:.2e})"
        )
        if drift > 1e-3:
            print("WARNING: r=1 hooked perplexity differs from unmodified - hook may be buggy.")

    output = {
        "config": {
            "mode": "original",
            "model_id": args.model_id,
            "target_layer": args.target_layer,
            "r_values": r_values,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
        },
        "unmodified": {"mean_nll": unmod_nll, "ppl": unmod_ppl},
        "results": {str(r): v for r, v in results.items()},
        "summary": summary,
        "hook_drift": drift,
    }
    Path(output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {output_json}")


def run_zero_diagnostic(args, model, decoder_layers, inputs):
    """plan2a-add1: distinguish 'three-way tie is real' from 'patch broken'.

    Round 2a produced bitwise-identical perplexities for vanilla / scaled / once
    at every r. That is ambiguous: either PLE at TARGET_LAYER genuinely doesn't
    move the needle under our loop, or the patch never intercepted PLE at all.

    This function runs the minimal cell that distinguishes the two: vanilla r=1
    and zero r=1 at the same TARGET_LAYER, in the same execution (so evaluation
    state is identical). If they differ, the patch is working.
    """
    output_json = args.output_json or "results_round2a_addendum.json"

    target_layer = decoder_layers[args.target_layer]
    inspection = inspect_ple_strategy(target_layer)
    print_ple_inspection(inspection)

    if inspection["strategy"] != "A":
        print(
            f"ERROR: PLE wiring is Strategy {inspection['strategy']}, not A. "
            "Cannot run zero-PLE diagnostic via kwarg interception."
        )
        sys.exit(2)

    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    orig_forward = target_layer.forward
    try:
        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
        )
        nll_v1, ppl_v1 = compute_perplexity(model, inputs)

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="zero", ple_kwarg=ple_kwarg,
        )
        nll_z1, ppl_z1 = compute_perplexity(model, inputs)
    finally:
        target_layer.forward = orig_forward

    nll_diff = abs(nll_z1 - nll_v1)
    status = "WORKING" if nll_diff > 1e-6 else "BROKEN"

    print("=== Zero-PLE diagnostic ===")
    print(f"vanilla r=1:  ppl = {ppl_v1}")
    print(f"zero    r=1:  ppl = {ppl_z1}")
    print()
    print(f"Difference: {nll_diff:.6f}  (absolute NLL difference)")
    print(f"Patch status: {status}")

    output = {
        "config": {
            "mode": "ple-variants",
            "diagnostic": "zero-ple",
            "model_id": args.model_id,
            "target_layer": args.target_layer,
            "r": 1,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
        },
        "inspection": {
            "strategy": inspection["strategy"],
            "layer_class": inspection["layer_class"],
            "source_file": inspection["source_file"],
            "start_lineno": inspection["start_lineno"],
            "signature": inspection["signature"],
            "ple_kwarg": inspection["ple_kwarg"],
        },
        "cells": [
            {"ple_mode": "vanilla", "r": 1, "mean_nll": nll_v1, "ppl": ppl_v1},
            {"ple_mode": "zero", "r": 1, "mean_nll": nll_z1, "ppl": ppl_z1},
        ],
        "nll_difference": nll_diff,
        "patch_status": status,
    }
    Path(output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {output_json}")


def run_ple_variants_mode(args, model, decoder_layers, inputs):
    """Round-2a: vanilla / scaled / once x r in {1,4,8} at TARGET_LAYER."""
    r_values = args.r_values if args.r_values is not None else [1, 4, 8]
    output_json = args.output_json or "results_round2a.json"

    target_layer = decoder_layers[args.target_layer]
    inspection = inspect_ple_strategy(target_layer)
    print_ple_inspection(inspection)

    if inspection["strategy"] != "A":
        msg = (
            f"\nPLE wiring is Strategy {inspection['strategy']}, not the expected "
            f"Strategy A (PLE passed as kwarg into the decoder layer's forward).\n"
        )
        if inspection["strategy"] == "B":
            msg += (
                "Strategy B means PLE is applied in the outer model forward, "
                "OUTSIDE the per-layer forward. Round 1's hook fundamentally "
                "could not control PLE: round-1 'vanilla' was actually 'once' "
                "in disguise. Stop and report this finding (see plan2a).\n"
            )
        else:  # "C"
            msg += (
                "Strategy C means PLE is referenced inside the layer in a way "
                "that is not a simple kwarg (e.g. computed from input_ids, "
                "fused into a projection, ...). Source dump above. Stop and "
                "report (see plan2a).\n"
            )
        print("--- Decoder layer forward source ---")
        print(inspection["source"])
        print("--- end source ---")
        print(msg)
        out_payload = {
            "mode": "ple-variants",
            "aborted": True,
            "inspection": {k: v for k, v in inspection.items() if k != "source"},
            "inspection_source": inspection["source"],
            "reason": msg.strip(),
        }
        Path(output_json).write_text(json.dumps(out_payload, indent=2))
        print(f"Wrote diagnostic-only {output_json}")
        sys.exit(2)

    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    print("Running unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.4f}")

    orig_forward = target_layer.forward
    cells = []
    cell_index = {}  # (mode, r) -> ppl, for table rendering

    try:
        # ---- Step 4: regression checks BEFORE collecting experiment data ----
        print("\n=== Regression checks (plan2a Step 4) ===")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
        )
        nll_v1, ppl_v1 = compute_perplexity(model, inputs)
        print(f"  vanilla r=1: ppl = {ppl_v1:.6f}")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="scaled", ple_kwarg=ple_kwarg,
        )
        nll_s1, ppl_s1 = compute_perplexity(model, inputs)
        print(f"  scaled  r=1: ppl = {ppl_s1:.6f}")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="once", ple_kwarg=ple_kwarg,
        )
        nll_o1, ppl_o1 = compute_perplexity(model, inputs)
        print(f"  once    r=1: ppl = {ppl_o1:.6f}")

        # Tolerances: vanilla-vs-round1 we allow numerical drift across HW;
        # scaled/once vs vanilla at r=1 must be bitwise equal (scale=1.0).
        check1 = math.isclose(ppl_v1, ROUND1_UNMODIFIED_PPL, rel_tol=1e-3)
        check2 = ppl_s1 == ppl_v1
        check3 = ppl_o1 == ppl_v1
        print(f"  check 1 (vanilla r=1 ~= round1 unmodified={ROUND1_UNMODIFIED_PPL:.4f}): {check1}")
        print(f"  check 2 (scaled r=1 == vanilla r=1):  {check2}")
        print(f"  check 3 (once   r=1 == vanilla r=1):  {check3}")
        if not (check1 and check2 and check3):
            target_layer.forward = orig_forward
            print("\nERROR: regression checks failed --- aborting before main run.")
            sys.exit(3)

        # Reuse r=1 measurements (identical across modes by construction).
        for ple_mode, (nll1, ppl1) in [
            ("vanilla", (nll_v1, ppl_v1)),
            ("scaled", (nll_s1, ppl_s1)),
            ("once", (nll_o1, ppl_o1)),
        ]:
            cell_index[(ple_mode, 1)] = ppl1
            cells.append({"ple_mode": ple_mode, "r": 1, "mean_nll": nll1, "ppl": ppl1})

        # ---- Step 5: 9-cell experiment (excluding r=1 already done) ----
        print("\n=== Main sweep (plan2a Step 5) ===")
        for ple_mode in ("vanilla", "scaled", "once"):
            for r in r_values:
                if r == 1:
                    continue
                target_layer.forward = make_looped_forward_ple(
                    orig_forward, r=r, ple_mode=ple_mode, ple_kwarg=ple_kwarg,
                )
                mean_nll, ppl = compute_perplexity(model, inputs)
                cell_index[(ple_mode, r)] = ppl
                cells.append({"ple_mode": ple_mode, "r": r, "mean_nll": mean_nll, "ppl": ppl})
                print(f"  {ple_mode:7s} r={r:2d}: mean NLL = {mean_nll:.4f}   perplexity = {ppl:.4f}")
    finally:
        target_layer.forward = orig_forward

    # ---- Step 6: printed table ----
    print("\n=== PLE variants (layer {}) ===".format(args.target_layer))
    header = "              " + "    ".join(f"r={r:<5d}" for r in r_values)
    print(header)
    for ple_mode in ("vanilla", "scaled", "once"):
        row_vals = "  ".join(f"{cell_index[(ple_mode, r)]:8.2f}" for r in r_values)
        print(f"  {ple_mode:8s}  {row_vals}")

    output = {
        "config": {
            "mode": "ple-variants",
            "model_id": args.model_id,
            "target_layer": args.target_layer,
            "r_values": r_values,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
        },
        "round1_baseline": {"unmodified_ppl": ROUND1_UNMODIFIED_PPL},
        "unmodified": {"mean_nll": unmod_nll, "ppl": unmod_ppl},
        "inspection": {
            "strategy": inspection["strategy"],
            "layer_class": inspection["layer_class"],
            "source_file": inspection["source_file"],
            "start_lineno": inspection["start_lineno"],
            "signature": inspection["signature"],
            "ple_kwarg": inspection["ple_kwarg"],
        },
        "regression_checks": {
            "vanilla_r1_matches_round1": check1,
            "scaled_r1_equals_vanilla_r1": check2,
            "once_r1_equals_vanilla_r1": check3,
            "vanilla_r1_ppl": ppl_v1,
            "round1_unmodified_ppl": ROUND1_UNMODIFIED_PPL,
        },
        "cells": cells,
    }
    Path(output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {output_json}")


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

    inputs = prepare_inputs(tokenizer, args.num_sequences, args.max_length)

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
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
