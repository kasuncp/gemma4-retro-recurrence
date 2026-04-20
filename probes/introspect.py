"""Decoder-layer discovery, PLE wiring inspection, per-layer metadata.

PLE wiring strategies (round 2a, plan2a Step 0):
  A --- PLE arrives as a keyword argument in the layer's ``forward``.
        This is what the current Gemma 4 E2B transformers release uses.
  B --- PLE is applied by the outer model, not inside the decoder layer.
  C --- PLE is referenced inside the layer in a non-kwarg way (computed
        from input_ids, fused into a projection, ...).

Rounds 2a+ require Strategy A; the modes that require it call
``_inspect_and_require_strategy_a`` to assert early.
"""

import inspect
import sys


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
