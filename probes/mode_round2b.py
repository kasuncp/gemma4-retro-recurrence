"""Round 2b --- per-layer PLE importance scan and layer-location sweep.

Two experiments sharing the same infrastructure:

  run_importance_scan_mode: for every layer l, measure vanilla-vs-zero PLE
    at r=1. nll_diff = nll_zero - nll_vanilla quantifies PLE importance at
    layer l.

  run_layer_location_mode: {vanilla, once} x {r=4, r=8} across a small set
    of anchor layers, to test whether "once beats vanilla at layer 17"
    generalises across depths.
"""

import math
import sys

from .data import compute_perplexity
from .env import ROUND1_UNMODIFIED_PPL, _results_path, _write_results_json
from .hooks import make_looped_forward_ple
from .introspect import _inspect_and_require_strategy_a, get_layer_attention_info


# Round 2b default layer choices for the layer-location sweep.
# 5  = early, close to embedding
# 17 = middle, anchor from round 2a
# 28 = late, near the KV-consumer region
DEFAULT_LOCATION_LAYERS = [5, 17, 28]


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
    output_json = args.output_json or _results_path("results_round2b_importance.json")
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
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")


def run_layer_location_mode(args, model, decoder_layers, inputs):
    """Round 2b: does 'once beats vanilla' at layer 17 generalise across depths?

    Grid: layer in {5, 17, 28} x ple_mode in {vanilla, once} x r in {4, 8}.
    Plus per-layer regression checks at r=1 (vanilla must match round-1
    unmodified baseline; once must bitwise-equal vanilla by construction of
    the scale=1.0 passthrough).
    """
    r_values = args.r_values if args.r_values is not None else [4, 8]
    output_json = args.output_json or _results_path("results_round2b_location.json")
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
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")
