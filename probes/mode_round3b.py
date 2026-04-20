"""Round 3b --- block-looping probe.

6 candidate blocks x {r=2, r=4, r=8} = 18 cells, vanilla PLE only. Each
cell loops a contiguous range of decoder layers as a unit r times. Tests
whether the round-2c "valley" (layers ~15-19) is a hard ceiling (scenario
X: only a tight block is loopable) or a floor (scenario Y: a wider
valley-anchored block behaves much better than single/pair probes
suggested because within-block re-processing dominates at that scale).

Hook strategy mirrors round 3a (see ``probes/hooks.install_block_loop_hooks``):
per-layer forward_pre_hooks capture whatever the outer model passes
(including the per-layer PLE slice, which arrives positionally), and a
single forward_hook on the last layer re-runs the block (r-1) additional
times. No outer-model patching required.
"""

import json
import math
import sys
from pathlib import Path

from .data import compute_perplexity
from .env import _results_path, _write_results_json
from .hooks import install_block_loop_hooks
from .introspect import _inspect_and_require_strategy_a, get_layer_attention_info


# Plan3b candidate blocks. Hand-picked to discriminate between "valley is
# a hard ceiling" (scenario X) and "valley is a floor" (scenario Y). See
# plans/plan3b.md for rationale.
DEFAULT_BLOCKS = [
    {"name": "A", "label": "valley-core",        "start": 15, "end": 19},
    {"name": "B", "label": "valley-narrow",      "start": 15, "end": 18},
    {"name": "C", "label": "valley-extend-down", "start": 12, "end": 19},
    {"name": "D", "label": "valley-extend-up",   "start": 15, "end": 22},
    {"name": "E", "label": "valley-centered",    "start": 13, "end": 22},
    {"name": "F", "label": "late-block",         "start": 25, "end": 32},
]


def _parse_block_specs(raw):
    """Parse --blocks specs like '15-19' into dicts with start/end.

    Custom blocks have no named label; the label string echoes the range.
    Caller is responsible for validating start/end against layer count.
    """
    out = []
    for i, s in enumerate(raw):
        try:
            lo, hi = s.split("-", 1)
            start, end = int(lo), int(hi)
        except (ValueError, AttributeError):
            raise ValueError(f"bad block spec {s!r}: expected 'START-END'")
        if start > end:
            raise ValueError(f"block spec {s!r} has start > end")
        out.append({
            "name": f"custom{i}",
            "label": f"{start}-{end}",
            "start": start,
            "end": end,
        })
    return out


def _load_round2c_vanilla(path):
    """Load round-2c single-layer vanilla ppl, indexed by (layer, r).

    Empty dict (with a warning) if the file is missing or unparseable.
    Comparison columns come out n/a in that case but the sweep still runs.
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


def _fmt_layers(b):
    return f"{b['start']}-{b['end']}"


def _analyze_block_map(blocks, r_values, block_ppl, single_ppl, unmod_ppl):
    """Print the plan3b comparison table at top_r, build the analysis dict.

    block_ppl: {(name, r): ppl}. single_ppl: {(layer, r): ppl} from round 2c.
    Columns: block_ppl, worst_single_in_block, best_single_in_block,
    improvement = worst_single / block_ppl (>1 means the block rescues its
    weakest layer).
    """
    top_r = max(r_values)

    print(f"\n=== Block-looping results at r={top_r} ===")
    print(
        f"{'name':>4}  {'layers':>7}  {'width':>5}  {'block_ppl':>11}  "
        f"{'worst_single':>13}  {'best_single':>12}  {'ratio':>9}"
    )
    rows = []
    for b in blocks:
        name, start, end = b["name"], b["start"], b["end"]
        width = end - start + 1
        pp = block_ppl.get((name, top_r), float("nan"))
        singles = [single_ppl.get((l, top_r), float("nan"))
                   for l in range(start, end + 1)]
        finite_singles = [s for s in singles if math.isfinite(s)]
        worst = max(finite_singles) if finite_singles else float("nan")
        best = min(finite_singles) if finite_singles else float("nan")
        if math.isfinite(worst) and math.isfinite(pp) and pp > 0:
            ratio = worst / pp
        else:
            ratio = float("nan")
        rows.append({
            "name": name,
            "label": b.get("label"),
            "start": start,
            "end": end,
            "width": width,
            "block_ppl": pp,
            "worst_single_in_block_ppl": worst,
            "best_single_in_block_ppl": best,
            "improvement_over_worst_single": ratio,
        })
        print(
            f"{name:>4}  {_fmt_layers(b):>7}  {width:>5}  "
            f"{pp:>11.2f}  {worst:>13.2f}  {best:>12.2f}  {ratio:>9.3f}"
        )

    thresholds = (5.0, 10.0, 50.0, 100.0)
    ppls_top = [block_ppl.get((b["name"], top_r), float("nan")) for b in blocks]
    finite = [p for p in ppls_top if math.isfinite(p)]
    counts = {
        f"within_{int(t)}x": sum(1 for p in finite if p <= t * unmod_ppl)
        for t in thresholds
    }
    print(
        f"\n=== Blocks within Nx baseline (r={top_r}, "
        f"baseline ppl={unmod_ppl:.2f}) ==="
    )
    for t in thresholds:
        k = f"within_{int(t)}x"
        print(f"  within {int(t):>3d}x baseline: {counts[k]:>3d} / {len(finite)}")

    return {
        "target_r": top_r,
        "unmodified_ppl": unmod_ppl,
        "comparison": rows,
        "counts_within_baseline_multiples": counts,
        "n_blocks_scored": len(finite),
    }


def _interpret_block_map(blocks, r_values, block_ppl, unmod_ppl):
    """First-pass label for which of plan3b's 5 interpretation buckets the
    numbers suggest. Heuristic only --- the final call is made by the analyst
    looking at the full table.
    """
    top_r = max(r_values)
    by_name = {b["name"]: b for b in blocks}
    pp = {n: block_ppl.get((n, top_r), float("nan")) for n in by_name}

    default_names = {"A", "B", "C", "D", "E", "F"}
    if not default_names.issubset(by_name.keys()):
        return (
            "Bucket UNKNOWN --- custom blocks were used, so plan3b's named "
            "bucket heuristic does not apply. Inspect the table manually."
        )

    def within(name, mult):
        p = pp.get(name, float("nan"))
        return math.isfinite(p) and p <= mult * unmod_ppl

    valley_ok = within("A", 5.0) and within("B", 5.0)
    ext_ok = all(within(n, 10.0) for n in ("C", "D", "E"))
    far_ok = within("F", 10.0)
    # Bucket 4 = BOTH valley anchors catastrophic. If only one is bad, that's
    # an unexpected pattern (Bucket 5), not the clean "hook bug" signal.
    valley_both_catastrophic = (
        not within("A", 50.0) and not within("B", 50.0)
    )

    if valley_both_catastrophic:
        return (
            "Bucket 4 (hook bug suspected) --- both valley anchors A and B "
            "degrade >50x at r=8. This contradicts prior single/pair-loop "
            "results in the valley. Stop and debug the block-loop hook."
        )
    if valley_ok and ext_ok and far_ok:
        return (
            "Bucket 3 (cascade hypothesis fully confirmed) --- all 6 blocks "
            "within 10x baseline, including F (25-32). Within-block re-"
            "processing dominates regardless of location; retrofit has "
            "maximum flexibility for block placement."
        )
    if valley_ok and ext_ok and not far_ok:
        return (
            "Bucket 2 (valley-anchored cascade works, distant block does "
            "not) --- A, B, C, D, E within tolerance; F catastrophic. "
            "Retrofit should use the widest valley-anchored block that "
            "still holds (likely E: 13-22)."
        )
    # Strict Bucket 1: ALL extensions broken (>50x). Mixed patterns (e.g., C
    # ok but D broken) fall through to Bucket 5 --- plan3b lists that exact
    # kind of non-monotone result as its archetypal Bucket-5 signal.
    extensions_all_broken = all(
        not within(n, 50.0) for n in ("C", "D", "E")
    )
    if valley_ok and extensions_all_broken:
        return (
            "Bucket 1 (valley-only hard ceiling) --- only A/B hold at r=8; "
            "extensions (C, D, E) all degrade 50x+. Retrofit should use a "
            "tight 4-5 layer recurrent block."
        )
    return (
        "Bucket 5 (unexpected pattern) --- the results do not fit scenarios "
        "X or Y cleanly (e.g., non-monotone in width, or one valley anchor "
        "catastrophic but not the other). Pause, paste the comparison table, "
        "and discuss before further plans."
    )


def run_block_looping_map(args, model, decoder_layers, inputs):
    """Round 3b: 6 candidate blocks x {r=2, r=4, r=8} = 18 cells, vanilla only.

    Thin wrapper around ``_run_block_looping_sweep`` that pins the
    round-3b default block set, output filename, mode name, and
    bucket interpreter. Round 3c reuses the same helper with its own
    defaults (see ``probes/mode_round3c.py``).
    """
    _run_block_looping_sweep(
        args, model, decoder_layers, inputs,
        default_blocks=DEFAULT_BLOCKS,
        default_output_filename="results_round3b_blocks.json",
        mode_name="block-looping",
        interpreter_fn=_interpret_block_map,
    )


def _run_block_looping_sweep(
    args, model, decoder_layers, inputs,
    *,
    default_blocks,
    default_output_filename,
    mode_name,
    interpreter_fn,
    extra_analysis_fn=None,
):
    """Shared block-looping runner for rounds 3b and 3c.

    Parameters
    ----------
    default_blocks : list of dict
        Blocks to sweep when the user did not pass ``--blocks``. Each
        dict has keys ``name``, ``label``, ``start``, ``end``.
    default_output_filename : str
        Filename (not full path) to write under ``results/`` when
        ``--output-json`` is not set.
    mode_name : str
        String recorded in the output JSON's ``config.mode`` field.
        Lets downstream consumers tell rounds apart in merged datasets.
    interpreter_fn : callable(blocks, r_values, block_ppl, unmod_ppl) -> str
        Round-specific bucket classifier printed after the main sweep.
    extra_analysis_fn : callable(blocks, r_values, block_ppl, unmod_ppl) -> dict, optional
        Optional round-specific extra analysis; return dict is merged into
        the output JSON under ``analysis_extra``.

    Regression (before the main sweep): r=1 block-loop on at least two
    blocks (smallest and largest by width) must match the unmodified
    baseline. At r=1 the forward_hook does zero extra iterations and
    returns the original output unchanged --- drift indicates the hook
    installation itself perturbs the model.
    """
    n_layers = len(decoder_layers)
    r_values = args.r_values if args.r_values is not None else [2, 4, 8]
    output_json = args.output_json or _results_path(default_output_filename)

    if 1 in r_values:
        print(
            f"ERROR: --mode {mode_name} uses r=1 only as a regression "
            "check. Do not include 1 in --r-values."
        )
        sys.exit(1)

    if args.blocks:
        try:
            blocks = _parse_block_specs(args.blocks)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        blocks = [dict(b) for b in default_blocks]

    for b in blocks:
        if b["start"] < 0 or b["end"] >= n_layers or b["start"] > b["end"]:
            print(
                f"ERROR: block {b['name']} [{b['start']}..{b['end']}] out of "
                f"range for {n_layers} layers."
            )
            sys.exit(1)

    # ---- Wiring inspection (shared across the whole sweep) ----
    inspection = _inspect_and_require_strategy_a(decoder_layers[blocks[0]["start"]])
    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    # ---- Round-2c single-layer comparison data ----
    single_ppl = _load_round2c_vanilla(args.round2c_json)

    # ---- Unmodified baseline ----
    print("Running unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.4f}\n")

    # ---- Regression check at r=1 on smallest + largest block ----
    sorted_by_width = sorted(blocks, key=lambda b: b["end"] - b["start"])
    regression_blocks = []
    seen = set()
    for b in (sorted_by_width[0], sorted_by_width[-1]):
        key = (b["start"], b["end"])
        if key not in seen:
            regression_blocks.append(b)
            seen.add(key)

    print("=== Regression check: r=1 block-loop must match unmodified ===")
    regression_rows = []
    all_match = True
    max_drift = 0.0
    for b in regression_blocks:
        uninstall = install_block_loop_hooks(
            decoder_layers, b["start"], b["end"], r=1,
        )
        try:
            nll_r1, ppl_r1 = compute_perplexity(model, inputs)
        finally:
            uninstall()
        drift = abs(ppl_r1 - unmod_ppl) / unmod_ppl if unmod_ppl else float("inf")
        match = drift < 1e-3
        max_drift = max(max_drift, drift)
        marker = "OK" if match else "FAIL"
        width = b["end"] - b["start"] + 1
        print(
            f"  block {b['name']} ({b['start']:>2}-{b['end']:<2}, width "
            f"{width:>2})  r=1: ppl={ppl_r1:.6f}  drift={drift:.2e}  [{marker}]"
        )
        regression_rows.append({
            "name": b["name"],
            "start": b["start"],
            "end": b["end"],
            "width": width,
            "ppl_block_r1": ppl_r1,
            "rel_drift": drift,
            "matches_baseline": match,
        })
        if not match:
            all_match = False

    if not all_match:
        print(
            "\nERROR: block-loop regression at r=1 failed. The hook is "
            "perturbing the model even when it should be a no-op. Aborting "
            "before the main sweep."
        )
        out_payload = {
            "config": {
                "mode": mode_name,
                "model_id": args.model_id,
                "blocks": blocks,
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
                "blocks_tested": [
                    {"name": b["name"], "start": b["start"], "end": b["end"]}
                    for b in regression_blocks
                ],
                "max_rel_drift": max_drift,
                "all_pass": all_match,
                "per_block": regression_rows,
            },
        }
        _write_results_json(output_json, out_payload)
        sys.exit(3)

    print(
        f"\nAll {len(regression_blocks)} regression blocks passed "
        f"(max drift = {max_drift:.2e}).\n"
    )

    # ---- Per-layer metadata (attention type, KV role) for context ----
    layer_metadata = []
    for l in range(n_layers):
        info = get_layer_attention_info(model, l)
        layer_metadata.append({
            "layer": l,
            "attention_type": info["attention_type"],
            "is_kv_consumer": info["is_kv_consumer"],
        })

    # ---- Main sweep ----
    total_cells = len(blocks) * len(r_values)
    print(f"=== Main sweep ({total_cells} cells) ===")
    block_cells = []
    block_ppl = {}
    done = 0
    for b in blocks:
        width = b["end"] - b["start"] + 1
        for r in r_values:
            uninstall = install_block_loop_hooks(
                decoder_layers, b["start"], b["end"], r=r,
            )
            try:
                mean_nll, ppl = compute_perplexity(model, inputs)
            finally:
                uninstall()
            block_ppl[(b["name"], r)] = ppl
            block_cells.append({
                "name": b["name"],
                "label": b.get("label"),
                "start": b["start"],
                "end": b["end"],
                "width": width,
                "r": r,
                "mean_nll": mean_nll,
                "ppl": ppl,
            })
            done += 1
            print(
                f"  [{done:3d}/{total_cells}] block {b['name']} "
                f"({b['start']:>2}-{b['end']:<2}, width {width:>2})  "
                f"r={r}: ppl={ppl:.4f}"
            )

    # ---- Analysis ----
    analysis = _analyze_block_map(
        blocks, r_values, block_ppl, single_ppl, unmod_ppl,
    )
    hint = interpreter_fn(blocks, r_values, block_ppl, unmod_ppl)
    print(f"\n=== Interpretation hint ===\n{hint}")

    analysis_extra = None
    if extra_analysis_fn is not None:
        analysis_extra = extra_analysis_fn(
            blocks, r_values, block_ppl, unmod_ppl,
        )

    output = {
        "config": {
            "mode": mode_name,
            "model_id": args.model_id,
            "blocks": blocks,
            "r_values": r_values,
            "ple_mode": "vanilla",
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
            "round2c_json": args.round2c_json,
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
        "layer_metadata": layer_metadata,
        "regression_checks": {
            "blocks_tested": [
                {"name": b["name"], "start": b["start"], "end": b["end"]}
                for b in regression_blocks
            ],
            "max_rel_drift": max_drift,
            "all_pass": all_match,
            "per_block": regression_rows,
        },
        "block_cells": block_cells,
        "analysis": analysis,
        "interpretation_hint": hint,
    }
    if analysis_extra is not None:
        output["analysis_extra"] = analysis_extra
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")
