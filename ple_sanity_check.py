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

# Round 2b default layer choices for the layer-location sweep.
# 5  = early, close to embedding
# 17 = middle, anchor from round 2a
# 28 = late, near the KV-consumer region
DEFAULT_LOCATION_LAYERS = [5, 17, 28]


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
        ],
        default="original",
        help=(
            "original = round-1 sanity check; "
            "ple-variants = round-2a probe; "
            "ple-importance-scan = round-2b per-layer vanilla-vs-zero at r=1; "
            "layer-location = round-2b {vanilla, once} x {r=4,8} across layers; "
            "full-looping-map = round-2c 35 x {vanilla,once} x {r=2,4,8} sweep"
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
            "[2,4,8] for full-looping-map (r=1 is used as a regression check only)."
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
            "full-looping-map default: all 0..34."
        ),
    )
    p.add_argument(
        "--importance-json",
        default="results_round2b_importance.json",
        help=(
            "Path to round-2b PLE importance scan JSON. Used by --mode "
            "full-looping-map to merge per-layer PLE importance into the "
            "output and compute correlations against it."
        ),
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
    # use_cache=False: we are computing forward-pass perplexity, not generating.
    # When the loop hooks re-enter a decoder layer with caching on, the layer's
    # self_attn appends new K/V to past_key_values on each iteration, making
    # the K-length grow beyond the attention mask's shape (crash observed as
    # "expanded size of tensor (1023) must match existing size (512)" on
    # sliding-attention layers). Disabling cache makes the measurement pure.
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for inp in inputs:
            out = model(**inp, labels=inp["input_ids"], use_cache=False)
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


def make_looped_forward_ple(orig_forward, r, ple_mode, ple_kwarg, location_recorder=None):
    """
    Round-2a hook (post-addendum2): loop the decoder layer `r` times with
    controlled PLE re-injection, handling per_layer_input passed either
    positionally or as a kwarg.

    Background (plan2a-add2): the Gemma 4 decoder layer signature is
        forward(self, hidden_states, per_layer_input=None, shared_kv_states=None, ...)
    and Gemma4Model.forward passes per_layer_input positionally on the
    transformers version we tested. The pre-fix version of this hook only
    looked in kwargs, so it never actually intercepted PLE --- every loop
    iteration got the original tensor back via *args, making zero-mode
    bitwise-identical to vanilla. Now we detect both paths.

    location_recorder: optional dict; on the first hook call, its
    "ple_location" key is set to "positional" | "kwarg" | "missing". Lets
    callers observe which code path the hook actually took.

    ple_mode in {"vanilla", "scaled", "once", "zero"}.
      - "zero": always pass scale=0.0, including at i=0.

    For r=1 with mode in {vanilla, scaled, once}, scale == 1.0 by
    construction and we pass the original args/kwargs through unchanged ---
    so plan2a's regression checks 2 and 3 stay bitwise-identical to vanilla
    r=1.
    """
    first_call = [True]

    def looped(hidden_states, *args, **kwargs):
        if ple_kwarg in kwargs:
            original_ple = kwargs[ple_kwarg]
            ple_location = "kwarg"
        elif len(args) >= 1:
            # *args captures everything after hidden_states; per_layer_input
            # is the first positional after hidden_states, so args[0].
            original_ple = args[0]
            ple_location = "positional"
        else:
            original_ple = None
            ple_location = "missing"

        if first_call[0]:
            print(f"  [hook] PLE arrives via {ple_location} (mode={ple_mode}, r={r})")
            if location_recorder is not None:
                location_recorder.setdefault("ple_location", ple_location)
            first_call[0] = False

        out = None
        for i in range(r):
            if ple_mode == "vanilla":
                scale = 1.0
            elif ple_mode == "scaled":
                scale = 1.0 / r
            elif ple_mode == "once":
                scale = 1.0 if i == 0 else 0.0
            elif ple_mode == "zero":
                scale = 0.0
            else:
                raise ValueError(f"unknown ple_mode={ple_mode!r}")

            if original_ple is None or scale == 1.0:
                # Pass through unchanged --- preserves bitwise identity at
                # scale=1.0, and there's nothing to scale when PLE is None.
                call_args = args
                call_kwargs = kwargs
            else:
                scaled_ple = original_ple * scale
                if ple_location == "kwarg":
                    call_args = args
                    call_kwargs = dict(kwargs)
                    call_kwargs[ple_kwarg] = scaled_ple
                else:  # "positional"
                    call_args = (scaled_ple,) + args[1:]
                    call_kwargs = kwargs

            out = orig_forward(hidden_states, *call_args, **call_kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out
    return looped


def get_layer_attention_info(model, layer_idx):
    """Best-effort extraction of attention-type / KV-sharing info for a layer.

    Round 2b bonus: the plan says to log attention type (local vs global) and
    whether a layer is a KV consumer *if* that is cheap. We inspect the model
    config for the common Gemma attributes and fall back to "unknown" otherwise
    --- no source diving.
    """
    info = {"attention_type": "unknown", "is_kv_consumer": "unknown"}
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)

    for source in (text_cfg, cfg):
        if source is None:
            continue
        layer_types = getattr(source, "layer_types", None)
        if layer_types is not None and 0 <= layer_idx < len(layer_types):
            info["attention_type"] = layer_types[layer_idx]
            break

    # KV-sharing heuristic: Gemma variants expose either a boundary index
    # (first consumer layer) or a count (how many tail layers reuse KV).
    for source in (text_cfg, cfg):
        if source is None:
            continue
        num_layers = getattr(source, "num_hidden_layers", None)
        first_consumer = getattr(source, "first_kv_shared_layer_idx", None)
        if first_consumer is None:
            first_consumer = getattr(source, "kv_shared_layer_idx", None)
        shared_count = getattr(source, "num_kv_shared_layers", None)
        if first_consumer is not None:
            info["is_kv_consumer"] = layer_idx >= first_consumer
            break
        if num_layers is not None and shared_count is not None:
            info["is_kv_consumer"] = layer_idx >= (num_layers - shared_count)
            break
    return info


def _inspect_and_require_strategy_a(decoder_layer):
    """Shared preamble for round-2b modes: inspect wiring and assert Strategy A."""
    inspection = inspect_ple_strategy(decoder_layer)
    print_ple_inspection(inspection)
    if inspection["strategy"] != "A":
        print(
            f"ERROR: PLE wiring is Strategy {inspection['strategy']}, not A. "
            "Round-2b modes require kwarg-style PLE interception."
        )
        sys.exit(2)
    return inspection


def run_importance_scan_mode(args, model, decoder_layers, inputs):
    """Round 2b: per-layer vanilla-vs-zero at r=1, across all layers.

    For each layer l in the requested set, install the hook at r=1 first with
    ple_mode="vanilla" and then with ple_mode="zero", measure perplexity, and
    restore the original forward before moving on. nll_diff = nll_zero -
    nll_vanilla is positive when PLE at that layer is useful, near-zero when
    PLE at that layer is approximately inert.
    """
    r_values = args.r_values if args.r_values is not None else [1]
    if r_values != [1]:
        print(
            "ERROR: --mode ple-importance-scan requires r=1 (it measures PLE "
            f"importance, not recurrence tolerance). Got r_values={r_values}."
        )
        sys.exit(1)
    output_json = args.output_json or "results_round2b_importance.json"
    layer_indices = (
        args.layers if args.layers is not None else list(range(len(decoder_layers)))
    )
    for l in layer_indices:
        if l < 0 or l >= len(decoder_layers):
            print(f"ERROR: layer index {l} out of range [0, {len(decoder_layers)}).")
            sys.exit(1)

    # Inspect wiring once on a representative layer; all decoder layers share
    # the same class, so the kwarg name is identical everywhere.
    inspection = _inspect_and_require_strategy_a(decoder_layers[layer_indices[0]])
    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    print("Running unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.4f}\n")

    per_layer = []
    location_recorder = {}
    total = len(layer_indices)
    for i, l in enumerate(layer_indices):
        layer = decoder_layers[l]
        attn_info = get_layer_attention_info(model, l)
        orig_forward = layer.forward
        try:
            layer.forward = make_looped_forward_ple(
                orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
                location_recorder=location_recorder,
            )
            nll_v, ppl_v = compute_perplexity(model, inputs)

            layer.forward = make_looped_forward_ple(
                orig_forward, r=1, ple_mode="zero", ple_kwarg=ple_kwarg,
                location_recorder=location_recorder,
            )
            nll_z, ppl_z = compute_perplexity(model, inputs)
        finally:
            layer.forward = orig_forward

        nll_diff = nll_z - nll_v
        per_layer.append({
            "layer": l,
            "attention_type": attn_info["attention_type"],
            "is_kv_consumer": attn_info["is_kv_consumer"],
            "ppl_vanilla": ppl_v,
            "ppl_zero": ppl_z,
            "nll_vanilla": nll_v,
            "nll_zero": nll_z,
            "nll_diff": nll_diff,
        })
        print(
            f"  [{i+1:2d}/{total}] layer {l:2d}  "
            f"ppl_vanilla={ppl_v:8.4f}  ppl_zero={ppl_z:9.4f}  "
            f"nll_diff={nll_diff:+.4f}  "
            f"(attn={attn_info['attention_type']}, "
            f"kv_consumer={attn_info['is_kv_consumer']})"
        )

    ple_location = location_recorder.get("ple_location", "unknown")
    print(f"\nPLE location detected: {ple_location}")

    # ---- Printed table ----
    print("\n=== PLE importance scan (r=1) ===")
    print(f"{'layer':>5}  {'ppl_vanilla':>12}  {'ppl_zero':>12}  {'nll_diff':>10}  "
          f"{'attn':>10}  {'kv_consumer':>11}")
    for row in per_layer:
        print(
            f"{row['layer']:>5}  "
            f"{row['ppl_vanilla']:>12.4f}  "
            f"{row['ppl_zero']:>12.4f}  "
            f"{row['nll_diff']:>+10.4f}  "
            f"{str(row['attention_type']):>10}  "
            f"{str(row['is_kv_consumer']):>11}"
        )

    top_high = sorted(per_layer, key=lambda r: r["nll_diff"], reverse=True)[:5]
    top_low = sorted(per_layer, key=lambda r: r["nll_diff"])[:5]

    print("\nTop 5 most PLE-important layers (highest nll_diff):")
    for row in top_high:
        print(f"  layer {row['layer']:>2}: {row['nll_diff']:+.4f}")

    print("\nTop 5 least PLE-important layers (lowest nll_diff):")
    for row in top_low:
        print(f"  layer {row['layer']:>2}: {row['nll_diff']:+.4f}")

    output = {
        "config": {
            "mode": "ple-importance-scan",
            "model_id": args.model_id,
            "layers": layer_indices,
            "r": 1,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
        },
        "unmodified": {"mean_nll": unmod_nll, "ppl": unmod_ppl},
        "inspection": {
            "strategy": inspection["strategy"],
            "layer_class": inspection["layer_class"],
            "source_file": inspection["source_file"],
            "start_lineno": inspection["start_lineno"],
            "signature": inspection["signature"],
            "ple_kwarg": inspection["ple_kwarg"],
        },
        "ple_location": ple_location,
        "per_layer": per_layer,
        "top_high": [{"layer": r["layer"], "nll_diff": r["nll_diff"]} for r in top_high],
        "top_low": [{"layer": r["layer"], "nll_diff": r["nll_diff"]} for r in top_low],
    }
    Path(output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {output_json}")


def run_layer_location_mode(args, model, decoder_layers, inputs):
    """Round 2b: does 'once beats vanilla' at layer 17 generalise across depths?

    Grid: layer in {5, 17, 28} x ple_mode in {vanilla, once} x r in {4, 8}.
    Plus per-layer regression checks at r=1 (vanilla must match round-1
    unmodified baseline; once must bitwise-equal vanilla by construction of
    the scale=1.0 passthrough).
    """
    r_values = args.r_values if args.r_values is not None else [4, 8]
    output_json = args.output_json or "results_round2b_location.json"
    layer_indices = args.layers if args.layers is not None else list(DEFAULT_LOCATION_LAYERS)
    for l in layer_indices:
        if l < 0 or l >= len(decoder_layers):
            print(f"ERROR: layer index {l} out of range [0, {len(decoder_layers)}).")
            sys.exit(1)

    inspection = _inspect_and_require_strategy_a(decoder_layers[layer_indices[0]])
    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    print("Running unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.4f}\n")

    regressions = []
    cells = []
    cell_index = {}  # (layer, mode, r) -> ppl
    location_recorder = {}

    for l in layer_indices:
        layer = decoder_layers[l]
        attn_info = get_layer_attention_info(model, l)
        orig_forward = layer.forward
        print(f"\n--- Layer {l} "
              f"(attn={attn_info['attention_type']}, "
              f"kv_consumer={attn_info['is_kv_consumer']}) ---")
        try:
            # ---- Regression checks ----
            layer.forward = make_looped_forward_ple(
                orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
                location_recorder=location_recorder,
            )
            nll_v1, ppl_v1 = compute_perplexity(model, inputs)
            print(f"  [regression] vanilla r=1: ppl = {ppl_v1:.6f}")

            layer.forward = make_looped_forward_ple(
                orig_forward, r=1, ple_mode="once", ple_kwarg=ple_kwarg,
                location_recorder=location_recorder,
            )
            nll_o1, ppl_o1 = compute_perplexity(model, inputs)
            print(f"  [regression] once    r=1: ppl = {ppl_o1:.6f}")

            check_baseline = math.isclose(ppl_v1, ROUND1_UNMODIFIED_PPL, rel_tol=1e-3)
            check_once_eq_vanilla = ppl_o1 == ppl_v1
            print(
                f"  check baseline  (vanilla r=1 ~= round1 "
                f"{ROUND1_UNMODIFIED_PPL:.4f}): {check_baseline}"
            )
            print(
                f"  check once==van (once r=1 == vanilla r=1 bitwise): "
                f"{check_once_eq_vanilla}"
            )
            regressions.append({
                "layer": l,
                "ppl_vanilla_r1": ppl_v1,
                "ppl_once_r1": ppl_o1,
                "round1_baseline_matches": check_baseline,
                "once_eq_vanilla_r1": check_once_eq_vanilla,
            })
            if not (check_baseline and check_once_eq_vanilla):
                layer.forward = orig_forward
                print(f"\nERROR: regression checks failed at layer {l} --- aborting.")
                sys.exit(3)

            # Reuse the r=1 vanilla measurement as a recorded cell so downstream
            # analysis has the r=1 point.
            cell_index[(l, "vanilla", 1)] = ppl_v1
            cell_index[(l, "once", 1)] = ppl_o1
            cells.append({
                "layer": l, "ple_mode": "vanilla", "r": 1,
                "mean_nll": nll_v1, "ppl": ppl_v1,
            })
            cells.append({
                "layer": l, "ple_mode": "once", "r": 1,
                "mean_nll": nll_o1, "ppl": ppl_o1,
            })

            # ---- Main grid ----
            for r in r_values:
                if r == 1:
                    continue
                for ple_mode in ("vanilla", "once"):
                    layer.forward = make_looped_forward_ple(
                        orig_forward, r=r, ple_mode=ple_mode, ple_kwarg=ple_kwarg,
                        location_recorder=location_recorder,
                    )
                    mean_nll, ppl = compute_perplexity(model, inputs)
                    cell_index[(l, ple_mode, r)] = ppl
                    cells.append({
                        "layer": l, "ple_mode": ple_mode, "r": r,
                        "mean_nll": mean_nll, "ppl": ppl,
                    })
                    print(
                        f"  {ple_mode:7s} r={r:2d}: mean NLL = {mean_nll:.4f}   "
                        f"perplexity = {ppl:.4f}"
                    )
        finally:
            layer.forward = orig_forward

    ple_location = location_recorder.get("ple_location", "unknown")
    print(f"\nPLE location detected: {ple_location}")

    # ---- Printed table ----
    non_trivial_r = [r for r in r_values if r != 1]
    print("\n=== Layer-location x PLE policy sweep ===")
    print(f"{'layer':>5}  {'r':>3}  {'vanilla_ppl':>12}  {'once_ppl':>12}  "
          f"{'once/vanilla':>13}  {'nll_improve':>12}")
    summary_rows = []
    for l in layer_indices:
        for r in non_trivial_r:
            van = cell_index[(l, "vanilla", r)]
            onc = cell_index[(l, "once", r)]
            ratio = onc / van if van else float("nan")
            # absolute NLL improvement: vanilla - once (positive = once better)
            nll_van = math.log(van) if van > 0 else float("nan")
            nll_onc = math.log(onc) if onc > 0 else float("nan")
            nll_improve = nll_van - nll_onc
            summary_rows.append({
                "layer": l, "r": r,
                "vanilla_ppl": van, "once_ppl": onc,
                "once_over_vanilla": ratio,
                "nll_improvement": nll_improve,
            })
            print(
                f"{l:>5}  {r:>3}  {van:>12.4f}  {onc:>12.4f}  "
                f"{ratio:>13.4f}  {nll_improve:>+12.4f}"
            )

    # Layer info summary (attention / kv consumer) for each target layer.
    layer_info = {
        l: get_layer_attention_info(model, l) for l in layer_indices
    }

    output = {
        "config": {
            "mode": "layer-location",
            "model_id": args.model_id,
            "layers": layer_indices,
            "r_values": r_values,
            "ple_modes": ["vanilla", "once"],
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
        "ple_location": ple_location,
        "layer_info": layer_info,
        "regressions": regressions,
        "cells": cells,
        "summary": summary_rows,
    }
    Path(output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {output_json}")


# ============================================================================
# Round 2c: full 35-layer looping-tolerance map
# ============================================================================


def _pearson(xs, ys):
    """Pearson correlation coefficient. NaN if insufficient variance."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    denom = (sx2 * sy2) ** 0.5
    if denom == 0.0:
        return float("nan")
    return sxy / denom


def _average_ranks(values):
    """Return average ranks (1-indexed, mid-ranks for ties), same order as input."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mid-rank, 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    """Spearman rank correlation, tie-safe."""
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _analyze_full_map(layer_metadata, cell_index, layer_indices, r_values):
    """Print tables + correlations + top-5s; return structured analysis dict.

    Uses the highest r in ``r_values`` as the "stress" reference for top-5
    tables, once-vs-vanilla table, and correlation target.
    """
    meta_by_layer = {m["layer"]: m for m in layer_metadata}
    top_r = max(r_values)
    first_r = min(r_values)

    # ---- Table 1: vanilla looping across all layers ----
    print("\n=== Vanilla looping (all layers) ===")
    r_cols = "  ".join(f"{'r=' + str(r):>10}" for r in r_values)
    ratio_label = f"r{top_r}/r{first_r}"
    print(
        f"{'layer':>5}  {'attn':>6}  {'kv_cons':>7}  {'PLE_imp':>8}  "
        f"{r_cols}  {ratio_label:>10}"
    )
    vanilla_summary = []
    for l in layer_indices:
        m = meta_by_layer[l]
        row_ppls = [cell_index.get((l, "vanilla", r), float("nan")) for r in r_values]
        base = row_ppls[0]
        ratio = (
            row_ppls[-1] / base
            if math.isfinite(base) and base > 0 and math.isfinite(row_ppls[-1])
            else float("nan")
        )
        attn = m["attention_type"]
        attn_short = (
            "slide" if attn == "sliding_attention"
            else "full" if attn == "full_attention"
            else str(attn)[:6]
        )
        kv = m["is_kv_consumer"]
        ple = m["ple_importance"]
        ple_str = f"{ple:8.4f}" if isinstance(ple, float) and math.isfinite(ple) else f"{'n/a':>8}"
        ppl_cells = "  ".join(f"{p:>10.2f}" for p in row_ppls)
        print(
            f"{l:>5}  {attn_short:>6}  {str(kv):>7}  {ple_str}  "
            f"{ppl_cells}  {ratio:>10.3f}"
        )
        row_dict = {
            "layer": l,
            "attention_type": attn,
            "is_kv_consumer": kv,
            "ple_importance": ple,
        }
        for r, p in zip(r_values, row_ppls):
            row_dict[f"vanilla_ppl_r{r}"] = p
        row_dict[f"ratio_r{top_r}_over_r{first_r}"] = ratio
        vanilla_summary.append(row_dict)

    # ---- Table 2: once vs vanilla at top_r ----
    print(f"\n=== Once vs vanilla (r={top_r}) ===")
    print(f"{'layer':>5}  {'vanilla_ppl':>12}  {'once_ppl':>12}  {'delta_%':>9}  {'helps?':>6}")
    once_vs_vanilla = []
    for l in layer_indices:
        van = cell_index.get((l, "vanilla", top_r), float("nan"))
        onc = cell_index.get((l, "once", top_r), float("nan"))
        if math.isfinite(van) and van > 0 and math.isfinite(onc):
            delta_pct = (onc - van) / van * 100.0
            helps = onc < van
        else:
            delta_pct = float("nan")
            helps = False
        helps_str = "yes" if helps else "no"
        print(
            f"{l:>5}  {van:>12.4f}  {onc:>12.4f}  "
            f"{delta_pct:>+8.2f}%  {helps_str:>6}"
        )
        once_vs_vanilla.append({
            "layer": l,
            "vanilla_ppl": van,
            "once_ppl": onc,
            "delta_pct": delta_pct,
            "once_helps": helps,
        })

    # ---- Correlations against ppl(r=top_r, vanilla) ----
    feat_ple = []
    feat_attn = []
    feat_kv = []
    feat_idx = []
    targets = []
    excluded = 0
    for l in layer_indices:
        y = cell_index.get((l, "vanilla", top_r), float("nan"))
        m = meta_by_layer[l]
        ple = m["ple_importance"]
        kv = m["is_kv_consumer"]
        attn = m["attention_type"]
        ok = (
            math.isfinite(y)
            and isinstance(ple, float) and math.isfinite(ple)
            and isinstance(kv, bool)
            and attn in ("sliding_attention", "full_attention")
        )
        if not ok:
            excluded += 1
            continue
        feat_ple.append(ple)
        feat_attn.append(1.0 if attn == "full_attention" else 0.0)
        feat_kv.append(1.0 if kv else 0.0)
        feat_idx.append(float(l))
        targets.append(y)

    print(
        f"\n=== Correlations with ppl(r={top_r}, vanilla) "
        f"(n={len(targets)}, excluded={excluded}) ==="
    )
    correlations = {}
    for name, xs in (
        ("ple_importance", feat_ple),
        ("attention_type", feat_attn),
        ("is_kv_consumer", feat_kv),
        ("layer_index", feat_idx),
    ):
        rp = _pearson(xs, targets)
        rs = _spearman(xs, targets)
        correlations[name] = {"pearson": rp, "spearman": rs}
        print(f"  {name:>18}  pearson={rp:+.4f}   spearman={rs:+.4f}")

    # ---- Top-5 tables (finite values only) ----
    vanilla_finite = [
        (l, cell_index.get((l, "vanilla", top_r), float("nan")))
        for l in layer_indices
    ]
    vanilla_finite = [(l, p) for l, p in vanilla_finite if math.isfinite(p)]
    top_loopable = sorted(vanilla_finite, key=lambda t: t[1])[:5]
    top_fragile = sorted(vanilla_finite, key=lambda t: t[1], reverse=True)[:5]

    print(f"\n=== Top 5 most loopable (lowest vanilla ppl at r={top_r}) ===")
    for l, p in top_loopable:
        print(f"  layer {l:>2}: ppl={p:.4f}")
    print(f"\n=== Top 5 least loopable (highest vanilla ppl at r={top_r}) ===")
    for l, p in top_fragile:
        print(f"  layer {l:>2}: ppl={p:.4f}")

    once_deltas = []
    for l in layer_indices:
        van = cell_index.get((l, "vanilla", top_r), float("nan"))
        onc = cell_index.get((l, "once", top_r), float("nan"))
        if math.isfinite(van) and math.isfinite(onc):
            once_deltas.append((l, van - onc, van, onc))  # positive = once helps
    sorted_helps = sorted(once_deltas, key=lambda t: t[1], reverse=True)
    once_helps_most = sorted_helps[:5]
    once_hurts_most = list(reversed(sorted_helps[-5:]))

    print(f"\n=== Top 5 where `once` most helps (r={top_r}) ===")
    for l, d, van, onc in once_helps_most:
        print(f"  layer {l:>2}: vanilla={van:.4f}  once={onc:.4f}  delta={d:+.4f}")
    print(f"\n=== Top 5 where `once` most hurts (r={top_r}) ===")
    for l, d, van, onc in once_hurts_most:
        print(f"  layer {l:>2}: vanilla={van:.4f}  once={onc:.4f}  delta={d:+.4f}")

    return {
        "target_r": top_r,
        "baseline_r": first_r,
        "n_layers_in_correlation": len(targets),
        "n_layers_excluded_from_correlation": excluded,
        f"correlations_r{top_r}_vanilla": correlations,
        "vanilla_summary": vanilla_summary,
        f"once_vs_vanilla_r{top_r}": once_vs_vanilla,
        "top_loopable": [{"layer": l, "ppl": p} for l, p in top_loopable],
        "top_fragile": [{"layer": l, "ppl": p} for l, p in top_fragile],
        "once_helps_most": [
            {"layer": l, "delta": d, "vanilla_ppl": van, "once_ppl": onc}
            for l, d, van, onc in once_helps_most
        ],
        "once_hurts_most": [
            {"layer": l, "delta": d, "vanilla_ppl": van, "once_ppl": onc}
            for l, d, van, onc in once_hurts_most
        ],
    }


def run_full_looping_map(args, model, decoder_layers, inputs):
    """Round 2c: 35 x {vanilla, once} x {r=2,4,8} = 210 cells.

    Before the sweep, run a per-layer r=1 vanilla regression check -- at r=1
    with ple_mode=vanilla, ``make_looped_forward_ple`` passes every argument
    through unchanged (scale=1.0), so perplexity must match the unmodified
    baseline to numerical tolerance. A failure means the hook installation
    itself perturbs the model and invalidates the sweep before it starts.

    After the sweep, run the analysis pass (tables + correlations + top-5s)
    and write a single JSON with everything.
    """
    n_layers = len(decoder_layers)
    r_values = args.r_values if args.r_values is not None else [2, 4, 8]
    ple_modes = ["vanilla", "once"]
    output_json = args.output_json or "results_round2c_full_map.json"

    if 1 in r_values:
        print(
            "ERROR: --mode full-looping-map uses r=1 only as a regression "
            "check. Do not include 1 in --r-values."
        )
        sys.exit(1)

    layer_indices = (
        args.layers if args.layers is not None else list(range(n_layers))
    )
    for l in layer_indices:
        if l < 0 or l >= n_layers:
            print(f"ERROR: layer index {l} out of range [0, {n_layers}).")
            sys.exit(1)

    # ---- Load PLE importance metadata from round 2b ----
    importance_path = Path(args.importance_json)
    if not importance_path.is_file():
        print(
            f"ERROR: importance JSON not found at {importance_path}. "
            "Run --mode ple-importance-scan first, or pass "
            "--importance-json to point at an existing file."
        )
        sys.exit(1)
    try:
        importance_data = json.loads(importance_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: failed to parse {importance_path}: {e}")
        sys.exit(1)
    importance_by_layer = {
        row["layer"]: row["nll_diff"]
        for row in importance_data.get("per_layer", [])
    }
    missing_importance = [l for l in layer_indices if l not in importance_by_layer]
    if missing_importance:
        print(
            f"WARNING: PLE importance missing for layers {missing_importance}; "
            "those layers will be excluded from correlation analysis."
        )

    inspection = _inspect_and_require_strategy_a(decoder_layers[layer_indices[0]])
    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    # ---- Unmodified baseline ----
    print("Running unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.4f}\n")

    # ---- Per-layer r=1 regression check ----
    # We compare against *this run's* unmodified ppl rather than the round-1
    # constant. Across GPUs / dtypes the unmodified number drifts by a few
    # parts in 1e-5; the invariant we actually care about is that installing
    # the hook at r=1 vanilla does not move the number from its same-run
    # reference.
    print("=== Per-layer r=1 regression ===")
    regression_rows = []
    max_drift = 0.0
    all_match = True
    location_recorder = {}
    for l in layer_indices:
        layer = decoder_layers[l]
        orig_forward = layer.forward
        try:
            layer.forward = make_looped_forward_ple(
                orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
                location_recorder=location_recorder,
            )
            nll_v1, ppl_v1 = compute_perplexity(model, inputs)
        finally:
            layer.forward = orig_forward

        drift = abs(ppl_v1 - unmod_ppl) / unmod_ppl if unmod_ppl else float("inf")
        match = drift < 1e-3
        max_drift = max(max_drift, drift)
        regression_rows.append({
            "layer": l,
            "ppl_vanilla_r1": ppl_v1,
            "rel_drift": drift,
            "matches_baseline": match,
        })
        marker = "OK" if match else "FAIL"
        print(
            f"  layer {l:2d}  vanilla r=1: ppl={ppl_v1:.6f}  "
            f"drift={drift:.2e}  [{marker}]"
        )
        if not match:
            all_match = False
            print(
                f"\nERROR: layer {l} r=1 regression failed "
                f"(ppl={ppl_v1:.6f} vs baseline={unmod_ppl:.6f}, "
                f"drift={drift:.2e} > 1e-3). Aborting before main sweep."
            )
            sys.exit(3)

    print(f"\nAll {len(layer_indices)} layers passed r=1 regression "
          f"(max drift = {max_drift:.2e}).\n")

    # ---- Collect per-layer metadata up front ----
    layer_metadata = []
    for l in layer_indices:
        info = get_layer_attention_info(model, l)
        layer_metadata.append({
            "layer": l,
            "attention_type": info["attention_type"],
            "is_kv_consumer": info["is_kv_consumer"],
            "ple_importance": importance_by_layer.get(l, float("nan")),
        })

    # ---- Main 210-cell sweep ----
    total_cells = len(layer_indices) * len(ple_modes) * len(r_values)
    print(f"=== Main sweep ({total_cells} cells) ===")
    cells = []
    cell_index = {}
    done = 0
    for l in layer_indices:
        layer = decoder_layers[l]
        orig_forward = layer.forward
        try:
            for ple_mode in ple_modes:
                for r in r_values:
                    layer.forward = make_looped_forward_ple(
                        orig_forward, r=r, ple_mode=ple_mode, ple_kwarg=ple_kwarg,
                        location_recorder=location_recorder,
                    )
                    mean_nll, ppl = compute_perplexity(model, inputs)
                    cell_index[(l, ple_mode, r)] = ppl
                    cells.append({
                        "layer": l,
                        "ple_mode": ple_mode,
                        "r": r,
                        "mean_nll": mean_nll,
                        "ppl": ppl,
                    })
                    done += 1
                    print(
                        f"  [{done:3d}/{total_cells}] layer {l:2d}  "
                        f"{ple_mode:7s} r={r}: ppl={ppl:.4f}"
                    )
        finally:
            layer.forward = orig_forward

    ple_location = location_recorder.get("ple_location", "unknown")
    print(f"\nPLE location detected: {ple_location}")

    # ---- Analysis ----
    analysis = _analyze_full_map(
        layer_metadata, cell_index, layer_indices, r_values,
    )

    output = {
        "config": {
            "mode": "full-looping-map",
            "model_id": args.model_id,
            "layers": layer_indices,
            "r_values": r_values,
            "ple_modes": ple_modes,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
            "importance_json": str(importance_path),
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
        "ple_location": ple_location,
        "layer_metadata": layer_metadata,
        "regression_checks": {
            "all_r1_match_baseline": all_match,
            "max_rel_drift": max_drift,
            "per_layer": regression_rows,
        },
        "cells": cells,
        "analysis": analysis,
    }
    Path(output_json).write_text(json.dumps(output, indent=2))
    print(f"\nWrote {output_json}")


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
    location_recorder = {}
    try:
        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_v1, ppl_v1 = compute_perplexity(model, inputs)

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="zero", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_z1, ppl_z1 = compute_perplexity(model, inputs)
    finally:
        target_layer.forward = orig_forward

    nll_diff = abs(nll_z1 - nll_v1)
    status = "WORKING" if nll_diff > 1e-6 else "BROKEN"
    ple_location = location_recorder.get("ple_location", "unknown")

    print("=== Zero-PLE diagnostic ===")
    print(f"vanilla r=1:  ppl = {ppl_v1}")
    print(f"zero    r=1:  ppl = {ppl_z1}")
    print()
    print(f"Difference: {nll_diff:.6f}  (absolute NLL difference)")
    print(f"Patch status: {status}")
    print(f"PLE location: {ple_location}")

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
        "ple_location": ple_location,
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
    location_recorder = {}  # plan2a-add2: capture detected PLE arrival path

    try:
        # ---- Step 4: regression checks BEFORE collecting experiment data ----
        print("\n=== Regression checks (plan2a Step 4 + add2) ===")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_v1, ppl_v1 = compute_perplexity(model, inputs)
        print(f"  vanilla r=1: ppl = {ppl_v1:.6f}")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="scaled", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_s1, ppl_s1 = compute_perplexity(model, inputs)
        print(f"  scaled  r=1: ppl = {ppl_s1:.6f}")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="once", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_o1, ppl_o1 = compute_perplexity(model, inputs)
        print(f"  once    r=1: ppl = {ppl_o1:.6f}")

        # plan2a-add2 check 4: inline zero-PLE diagnostic. The original bug
        # made every mode behave like vanilla, so zero r=1 was bitwise-equal
        # to vanilla r=1. After the fix, zero must produce a different ppl.
        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="zero", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_z1, ppl_z1 = compute_perplexity(model, inputs)
        print(f"  zero    r=1: ppl = {ppl_z1:.6f}")

        # Tolerances: vanilla-vs-round1 we allow numerical drift across HW;
        # scaled/once vs vanilla at r=1 must be bitwise equal (scale=1.0).
        check1 = math.isclose(ppl_v1, ROUND1_UNMODIFIED_PPL, rel_tol=1e-3)
        check2 = ppl_s1 == ppl_v1
        check3 = ppl_o1 == ppl_v1
        check4 = abs(nll_z1 - nll_v1) > 1e-6  # zero must differ from vanilla
        print(f"  check 1 (vanilla r=1 ~= round1 unmodified={ROUND1_UNMODIFIED_PPL:.4f}): {check1}")
        print(f"  check 2 (scaled r=1 == vanilla r=1):  {check2}")
        print(f"  check 3 (once   r=1 == vanilla r=1):  {check3}")
        print(f"  check 4 (zero   r=1 != vanilla r=1):  {check4}   "
              f"[NLL diff = {abs(nll_z1 - nll_v1):.6f}]")
        if not (check1 and check2 and check3 and check4):
            target_layer.forward = orig_forward
            print("\nERROR: regression checks failed --- aborting before main run.")
            if not check4:
                print("  check 4 failure means the PLE patch is BROKEN: zero-mode "
                      "produced bitwise-identical perplexity to vanilla. The hook "
                      "is not actually intercepting per_layer_input.")
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
                    location_recorder=location_recorder,
                )
                mean_nll, ppl = compute_perplexity(model, inputs)
                cell_index[(ple_mode, r)] = ppl
                cells.append({"ple_mode": ple_mode, "r": r, "mean_nll": mean_nll, "ppl": ppl})
                print(f"  {ple_mode:7s} r={r:2d}: mean NLL = {mean_nll:.4f}   perplexity = {ppl:.4f}")
    finally:
        target_layer.forward = orig_forward

    ple_location = location_recorder.get("ple_location", "unknown")
    print(f"\nPLE location detected: {ple_location}")

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
            "zero_r1_differs_from_vanilla_r1": check4,
            "vanilla_r1_ppl": ppl_v1,
            "zero_r1_ppl": ppl_z1,
            "zero_vs_vanilla_nll_diff": abs(nll_z1 - nll_v1),
            "round1_unmodified_ppl": ROUND1_UNMODIFIED_PPL,
        },
        "addendum2": {
            "ple_location": ple_location,
            "zero_diagnostic_passed": check4,
            "fix_applied": "handle_positional_ple_args",
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
    else:
        raise ValueError(f"unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
