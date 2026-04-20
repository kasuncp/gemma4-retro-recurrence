"""Round 3a --- pair-of-layers looping probe.

34 pair starting positions (L, L+1) x {r=2, r=4, r=8} = 102 cells, vanilla
PLE only. Each cell loops the pair as a unit. Directly compared against
round-2c single-layer vanilla results to test the cascade hypothesis.
"""

import json
import math
import sys
from pathlib import Path

from .data import compute_perplexity
from .env import _results_path, _write_results_json
from .hooks import install_pair_loop_hooks
from .introspect import _inspect_and_require_strategy_a, get_layer_attention_info
from .stats import _pearson, _spearman


def _load_round2c_vanilla_r8(path):
    """Load round-2c single-layer vanilla results, indexed by (layer, r).

    Returns {(layer, r): ppl} covering every cell found in the file. Empty
    dict if the file is missing or malformed --- callers should warn but
    proceed (the sweep still runs; comparison columns just come out n/a).
    """
    p = Path(path)
    if not p.is_file():
        print(f"WARNING: round-2c json not found at {p}; comparison columns will be n/a.")
        return {}
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"WARNING: failed to parse {p}: {e}; comparison columns will be n/a.")
        return {}
    out = {}
    for cell in d.get("cells", []):
        if cell.get("ple_mode") != "vanilla":
            continue
        out[(cell["layer"], cell["r"])] = cell["ppl"]
    if not out:
        print(f"WARNING: no vanilla cells in {p}; comparison columns will be n/a.")
    return out


def _analyze_pair_map(
    pair_starts, r_values, pair_ppl, single_ppl, unmod_ppl, meta_by_layer,
):
    """Print the pair-vs-single comparison and build the analysis dict.

    pair_ppl: {(L, r): ppl}. single_ppl: {(layer, r): ppl} from round 2c.
    Returns dict with summary counts, per-pair comparison rows, correlations.
    """
    top_r = max(r_values)

    # ---- Main comparison table at top_r ----
    print(f"\n=== Pair-looping vs single-layer (r={top_r}, vanilla) ===")
    print(
        f"{'pair':>7}  {'pair_ppl':>10}  {'single_L':>10}  {'single_L+1':>10}  "
        f"{'min_single':>10}  {'geom_mean':>10}  {'improve':>8}"
    )
    comparison = []
    for L in pair_starts:
        pp = pair_ppl.get((L, top_r), float("nan"))
        sL = single_ppl.get((L, top_r), float("nan"))
        sL1 = single_ppl.get((L + 1, top_r), float("nan"))
        if math.isfinite(sL) and math.isfinite(sL1) and sL > 0 and sL1 > 0:
            min_s = min(sL, sL1)
            geom_s = math.sqrt(sL * sL1)
        else:
            min_s = float("nan")
            geom_s = float("nan")
        if math.isfinite(min_s) and math.isfinite(pp) and pp > 0:
            improve = min_s / pp  # >1 means pair beats better-of-the-two.
        else:
            improve = float("nan")
        comparison.append({
            "L": L,
            "L1": L + 1,
            "pair_ppl": pp,
            "single_L_ppl": sL,
            "single_L1_ppl": sL1,
            "min_single_ppl": min_s,
            "geom_mean_single_ppl": geom_s,
            "improvement": improve,
        })
        print(
            f"({L:>2},{L + 1:<2})  {pp:>10.2f}  {sL:>10.2f}  {sL1:>10.2f}  "
            f"{min_s:>10.2f}  {geom_s:>10.2f}  {improve:>8.3f}"
        )

    # ---- Counts within multiples of baseline, at top_r ----
    thresholds = (5.0, 10.0, 50.0, 100.0)
    pair_ppls_top = [pair_ppl.get((L, top_r), float("nan")) for L in pair_starts]
    finite_pair_ppls = [p for p in pair_ppls_top if math.isfinite(p)]
    counts = {}
    for t in thresholds:
        counts[f"within_{int(t)}x"] = sum(
            1 for p in finite_pair_ppls if p <= t * unmod_ppl
        )
    print(f"\n=== Pairs within Nx baseline (r={top_r}, baseline ppl={unmod_ppl:.2f}) ===")
    for t in thresholds:
        k = f"within_{int(t)}x"
        print(f"  within {int(t):>3d}x baseline: {counts[k]:>3d} / {len(finite_pair_ppls)}")

    # Comparison with round-2c single-layer valley (ppl within 10x baseline).
    single_within_10x = 0
    single_layers_counted = 0
    for l in meta_by_layer:
        s = single_ppl.get((l, top_r), float("nan"))
        if math.isfinite(s):
            single_layers_counted += 1
            if s <= 10.0 * unmod_ppl:
                single_within_10x += 1

    print(
        f"\nRound-2c single-layer within 10x baseline: "
        f"{single_within_10x} / {single_layers_counted}"
    )
    print(
        f"Round-3a pair within 10x baseline:           "
        f"{counts.get('within_10x', 0)} / {len(finite_pair_ppls)}"
    )

    # ---- Correlations at top_r ----
    feat_ple_sum = []
    feat_attn_cross = []  # pair crosses global/local boundary?
    feat_kv_cross = []    # pair straddles the KV-consumer boundary?
    feat_midpoint = []
    targets = []
    for L in pair_starts:
        y = pair_ppl.get((L, top_r), float("nan"))
        mL = meta_by_layer.get(L)
        mL1 = meta_by_layer.get(L + 1)
        if mL is None or mL1 is None or not math.isfinite(y):
            continue
        ple_L = mL["ple_importance"]
        ple_L1 = mL1["ple_importance"]
        if not (isinstance(ple_L, float) and math.isfinite(ple_L)
                and isinstance(ple_L1, float) and math.isfinite(ple_L1)):
            continue
        feat_ple_sum.append(ple_L + ple_L1)
        feat_attn_cross.append(
            1.0 if mL["attention_type"] != mL1["attention_type"] else 0.0
        )
        kv_L = mL["is_kv_consumer"]
        kv_L1 = mL1["is_kv_consumer"]
        feat_kv_cross.append(
            1.0 if (isinstance(kv_L, bool) and isinstance(kv_L1, bool)
                    and kv_L != kv_L1) else 0.0
        )
        feat_midpoint.append(L + 0.5)
        targets.append(y)

    print(
        f"\n=== Correlations with pair_ppl(r={top_r}, vanilla) "
        f"(n={len(targets)}) ==="
    )
    correlations = {}
    for name, xs in (
        ("ple_importance_sum", feat_ple_sum),
        ("attention_boundary_cross", feat_attn_cross),
        ("kv_boundary_cross", feat_kv_cross),
        ("pair_midpoint", feat_midpoint),
    ):
        rp = _pearson(xs, targets)
        rs = _spearman(xs, targets)
        correlations[name] = {"pearson": rp, "spearman": rs}
        print(f"  {name:>28}  pearson={rp:+.4f}   spearman={rs:+.4f}")

    return {
        "target_r": top_r,
        "unmodified_ppl": unmod_ppl,
        "counts_within_baseline_multiples": counts,
        "n_pairs_scored": len(finite_pair_ppls),
        "single_layer_within_10x": single_within_10x,
        "n_single_layers_scored": single_layers_counted,
        "comparison": comparison,
        f"correlations_r{top_r}": correlations,
    }


def run_pair_looping_map(args, model, decoder_layers, inputs):
    """Round 3a: 34 pair starts x {r=2, r=4, r=8} = 102 cells, vanilla only.

    Regression (before the main sweep): r=1 pair-loop at three
    representative pairs --- (5,6), (17,18), (28,29) --- must match the
    unmodified baseline. At r=1 the forward_hook does zero extra
    iterations and returns the original output unmodified, so drift
    indicates the hook installation itself perturbs the model.
    """
    n_layers = len(decoder_layers)
    r_values = args.r_values if args.r_values is not None else [2, 4, 8]
    output_json = args.output_json or _results_path("results_round3a_pair_looping.json")

    if 1 in r_values:
        print(
            "ERROR: --mode pair-looping-map uses r=1 only as a regression "
            "check. Do not include 1 in --r-values."
        )
        sys.exit(1)

    pair_starts = (
        args.layers if args.layers is not None else list(range(n_layers - 1))
    )
    for L in pair_starts:
        if L < 0 or L + 1 >= n_layers:
            print(
                f"ERROR: pair start L={L} out of range for {n_layers} "
                f"layers (need L+1 < {n_layers})."
            )
            sys.exit(1)

    # ---- Wiring inspection (shared across the whole sweep) ----
    inspection = _inspect_and_require_strategy_a(decoder_layers[pair_starts[0]])
    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    # ---- Round-2c single-layer comparison data ----
    single_ppl = _load_round2c_vanilla_r8(args.round2c_json)

    # ---- Unmodified baseline ----
    print("Running unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.4f}\n")

    # ---- Regression check at r=1 on representative pairs ----
    regression_pairs_requested = [(5, 6), (17, 18), (28, 29)]
    regression_pairs = [
        (L, L1) for (L, L1) in regression_pairs_requested
        if L1 < n_layers and L in pair_starts
    ]
    if not regression_pairs:
        # Fallback: use the first requested pair_start.
        L = pair_starts[0]
        regression_pairs = [(L, L + 1)]
        print(
            f"WARNING: no default regression pairs in {pair_starts}; "
            f"using fallback ({L},{L + 1})."
        )

    print("=== Regression check: r=1 pair-loop must match unmodified ===")
    regression_rows = []
    all_match = True
    max_drift = 0.0
    for L, L1 in regression_pairs:
        uninstall = install_pair_loop_hooks(decoder_layers, L, r=1)
        try:
            nll_r1, ppl_r1 = compute_perplexity(model, inputs)
        finally:
            uninstall()
        drift = abs(ppl_r1 - unmod_ppl) / unmod_ppl if unmod_ppl else float("inf")
        match = drift < 1e-3
        max_drift = max(max_drift, drift)
        marker = "OK" if match else "FAIL"
        print(
            f"  pair ({L:>2},{L1:<2})  r=1: ppl={ppl_r1:.6f}  "
            f"drift={drift:.2e}  [{marker}]"
        )
        regression_rows.append({
            "L": L, "L1": L1,
            "ppl_pair_r1": ppl_r1,
            "rel_drift": drift,
            "matches_baseline": match,
        })
        if not match:
            all_match = False

    if not all_match:
        print(
            "\nERROR: pair-loop regression at r=1 failed. The hook is "
            "perturbing the model even when it should be a no-op. Aborting "
            "before the main sweep."
        )
        out_payload = {
            "config": {
                "mode": "pair-looping-map",
                "model_id": args.model_id,
                "pair_starts": pair_starts,
                "r_values": r_values,
                "ple_mode": "vanilla",
                "num_sequences": args.num_sequences,
                "max_length": args.max_length,
                "dtype": args.dtype,
                "ple_kwarg": ple_kwarg,
                "round2c_json": args.round2c_json,
            },
            "aborted": True,
            "reason": "regression check failed",
            "unmodified": {"mean_nll": unmod_nll, "ppl": unmod_ppl},
            "regression_checks": {
                "pairs_tested": [
                    {"L": L, "L1": L1} for L, L1 in regression_pairs
                ],
                "max_rel_drift": max_drift,
                "all_pass": all_match,
                "per_pair": regression_rows,
            },
        }
        _write_results_json(output_json, out_payload)
        sys.exit(3)

    print(
        f"\nAll {len(regression_pairs)} regression pairs passed "
        f"(max drift = {max_drift:.2e}).\n"
    )

    # ---- Per-layer metadata (for correlation analysis) ----
    importance_by_layer = {}
    importance_path = Path(args.importance_json)
    if importance_path.is_file():
        try:
            importance_data = json.loads(importance_path.read_text())
            importance_by_layer = {
                row["layer"]: row["nll_diff"]
                for row in importance_data.get("per_layer", [])
            }
        except json.JSONDecodeError as e:
            print(f"WARNING: failed to parse {importance_path}: {e}")
    else:
        print(f"WARNING: importance json not found at {importance_path}")

    meta_by_layer = {}
    for l in range(n_layers):
        info = get_layer_attention_info(model, l)
        meta_by_layer[l] = {
            "layer": l,
            "attention_type": info["attention_type"],
            "is_kv_consumer": info["is_kv_consumer"],
            "ple_importance": importance_by_layer.get(l, float("nan")),
        }

    # ---- Main sweep ----
    total_cells = len(pair_starts) * len(r_values)
    print(f"=== Main sweep ({total_cells} cells) ===")
    pair_cells = []
    pair_ppl = {}
    done = 0
    for L in pair_starts:
        for r in r_values:
            uninstall = install_pair_loop_hooks(decoder_layers, L, r=r)
            try:
                mean_nll, ppl = compute_perplexity(model, inputs)
            finally:
                uninstall()
            pair_ppl[(L, r)] = ppl
            pair_cells.append({
                "L": L, "L1": L + 1, "r": r,
                "mean_nll": mean_nll, "ppl": ppl,
            })
            done += 1
            print(
                f"  [{done:3d}/{total_cells}] pair ({L:>2},{L + 1:<2})  "
                f"r={r}: ppl={ppl:.4f}"
            )

    # ---- Analysis ----
    analysis = _analyze_pair_map(
        pair_starts, r_values, pair_ppl, single_ppl, unmod_ppl, meta_by_layer,
    )

    # ---- Interpretation bucket hint ----
    n_pairs_within_10x = analysis["counts_within_baseline_multiples"].get(
        "within_10x", 0
    )
    single_within_10x = analysis["single_layer_within_10x"]
    hint = _interpret_pair_map(
        n_pairs_within_10x,
        analysis["n_pairs_scored"],
        single_within_10x,
        analysis["n_single_layers_scored"],
    )
    print(f"\n=== Interpretation hint ===\n{hint}")

    output = {
        "config": {
            "mode": "pair-looping-map",
            "model_id": args.model_id,
            "pair_starts": pair_starts,
            "r_values": r_values,
            "ple_mode": "vanilla",
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
            "round2c_json": args.round2c_json,
            "importance_json": str(importance_path),
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
        "layer_metadata": [meta_by_layer[l] for l in range(n_layers)],
        "regression_checks": {
            "pairs_tested": [
                {"L": L, "L1": L1} for L, L1 in regression_pairs
            ],
            "max_rel_drift": max_drift,
            "all_pass": all_match,
            "per_pair": regression_rows,
        },
        "pair_cells": pair_cells,
        "analysis": analysis,
        "interpretation_hint": hint,
    }
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")


def _interpret_pair_map(n_pairs_within_10x, n_pairs, single_within_10x, n_singles):
    """Rough heuristic mapping onto plan3a's four interpretation buckets.

    The final interpretation paragraph is written by the analyst looking at
    the full table; this function just gives a first-pass label so the
    console output contains an explicit pointer to which bucket the counts
    suggest.
    """
    if n_pairs == 0 or n_singles == 0:
        return (
            "Bucket UNKNOWN --- insufficient data. Check regression results "
            "and comparison table manually."
        )
    pair_rate = n_pairs_within_10x / n_pairs
    single_rate = single_within_10x / n_singles
    # "Dramatic" improvement: pair rate >= 2x single rate AND >= 50% of pairs
    # within 10x.
    if pair_rate >= max(0.5, 2.0 * single_rate):
        return (
            f"Bucket 1 (cascade hypothesis confirmed) --- {n_pairs_within_10x}"
            f"/{n_pairs} pairs within 10x baseline vs {single_within_10x}"
            f"/{n_singles} single layers. Pair-looping dramatically improves "
            "over single-layer looping in most of the model. Round 3b (full "
            "block looping) is a green light."
        )
    if pair_rate > single_rate + 0.05:
        return (
            f"Bucket 2 (partial / regional) --- {n_pairs_within_10x}/{n_pairs}"
            f" pairs vs {single_within_10x}/{n_singles} single layers. Some "
            "regions improve, others don't. Inspect the comparison table to "
            "identify which pair ranges are loopable."
        )
    if pair_rate < single_rate - 0.05:
        return (
            f"Bucket 4 (WORSE --- investigate) --- {n_pairs_within_10x}/"
            f"{n_pairs} pairs vs {single_within_10x}/{n_singles} single "
            "layers. Pair-looping is systematically worse; this is unexpected"
            " and likely a hook bug. Stop and investigate."
        )
    return (
        f"Bucket 3 (identical / fragile) --- {n_pairs_within_10x}/{n_pairs} "
        f"pairs vs {single_within_10x}/{n_singles} single layers. Pair-"
        "looping is essentially the same as single-layer looping; the "
        "cascade hypothesis is not supported. Retrofit design needs a "
        "rethink before Round 3b."
    )
