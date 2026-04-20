"""Round 2c --- full 35-layer looping-tolerance map.

35 layers x {vanilla, once} x {r=2, r=4, r=8} = 210 cells, plus per-layer
r=1 regression checks, plus correlation analysis against PLE importance,
attention type, KV-sharing role, and layer depth.
"""

import json
import math
import sys
from pathlib import Path

from .data import compute_perplexity
from .env import ROUND1_UNMODIFIED_PPL, _results_path, _write_results_json
from .hooks import make_looped_forward_ple
from .introspect import _inspect_and_require_strategy_a, get_layer_attention_info
from .stats import _pearson, _spearman


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
    output_json = args.output_json or _results_path("results_round2c_full_map.json")

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
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")
