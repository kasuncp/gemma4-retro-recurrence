"""
PLE x Loop Recurrence Sanity Check for Gemma 4 E2B.

Probes whether naively looping a single middle decoder layer `r` times
(with PLE re-injected on every iteration) breaks the pretrained model.

Cloud-server friendly: argparse flags, HF cache redirect to persistent
volume, env-info dump, JSON result save.

Recommended RunPod usage:
    # PyTorch 2.x template, single GPU >= 16 GB VRAM
    pip install -U transformers datasets accelerate
    export HF_TOKEN=hf_xxx                # accept Gemma license first
    python ple_sanity_check.py            # bf16 default, results.json at end
"""

import argparse
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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target-layer", type=int, default=17)
    p.add_argument("--r-values", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--num-sequences", type=int, default=50)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--dtype", choices=list(DTYPE_MAP), default="bf16")
    p.add_argument("--output-json", default="results.json")
    p.add_argument("--model-id", default=MODEL_ID)
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
            "gated model — load will fail unless you've cached the weights "
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
    model.train(False)  # inference mode (equivalent to .eval())
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

    print("\nRunning unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.2f}")

    orig_forward = decoder_layers[args.target_layer].forward
    results = {}
    try:
        for r in args.r_values:
            decoder_layers[args.target_layer].forward = make_looped_forward(orig_forward, r)
            mean_nll, ppl = compute_perplexity(model, inputs)
            results[r] = {"mean_nll": mean_nll, "ppl": ppl}
            print(f"r={r:2d}:  mean NLL = {mean_nll:.4f}   perplexity = {ppl:.2f}")
    finally:
        decoder_layers[args.target_layer].forward = orig_forward

    print("\n=== Summary ===")
    baseline_ppl = results[args.r_values[0]]["ppl"]
    summary = []
    for r, res in results.items():
        ratio = res["ppl"] / baseline_ppl
        line = f"r={r:2d}:  ppl={res['ppl']:7.2f}   (x{ratio:.2f} vs r={args.r_values[0]})"
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
            "model_id": args.model_id,
            "target_layer": args.target_layer,
            "r_values": args.r_values,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
        },
        "unmodified": {"mean_nll": unmod_nll, "ppl": unmod_ppl},
        "results": {str(r): v for r, v in results.items()},
        "summary": summary,
        "hook_drift": drift,
    }
    Path(args.output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
