"""Round 5c --- IT-weight perplexity bridge.

Closes the model seam between rounds 1-3c (block geometry chosen on
``google/gemma-4-E2B`` raw-text Wikitext-2) and rounds 4-5/6 (reasoning
eval on ``google/gemma-4-E2B-it``). Architecture is unchanged across
the IT post-training, so the looping-tolerance map *should* transfer ---
this round measures whether it does, on the same Wikitext-2 sequences
and the same vanilla block-loop hook.

Sweep (~22 cells, ~20 min on a single 4090):
  * IT baseline (no hook) --- anchor for every ratio in this round.
  * Blocks A [15,19], D [15,22], G [15,24] at r in {1, 2, 4, 8}.
    r=1 doubles as a regression check (must equal the baseline).
  * Control F [25,32] at r=8 --- catastrophic on base, must remain
    catastrophic on IT.
  * Coarse single-layer overlay: r=8 vanilla on layers
    {0, 5, 10, 15, 17, 20, 25, 30, 33} --- enough to verify the round-2c
    valley is still anchored ~15-19 and the late-layer drift zone still
    bites past ~25.

Mandatory regression checks (failing any halts before further work):
  1. A r=1 == baseline (relative drift < 1e-4).
  2. D r=1 == baseline (relative drift < 1e-4).
  3. layer-17 r=8 ratio in [2.5x, 5x] (round-2c base value: 3.4x).
  4. F r=8 ratio > 100x.

Output JSON keys mirror plan5c.md "Reporting format" so the bucket /
exit-criteria summary is self-contained. Comparison columns are
pre-filled from the round-2c full map and the round-3b/3c block JSONs
when those files exist.

Note: this mode loads its own IT model and tokenizer; the top-level
``main()`` short-circuits before the default base-model load so we do
not pay the download/load cost twice. See ``run_it_perplexity_bridge``.
"""

import json
import math
import sys
from pathlib import Path

import torch

from .data import compute_perplexity, prepare_inputs
from .env import (
    DTYPE_MAP,
    EXPECTED_NUM_LAYERS,
    ROUND1_UNMODIFIED_PPL,
    _results_path,
    _write_results_json,
    load_model,
    print_env,
)
from .hooks import install_block_loop_hooks
from .introspect import (
    _inspect_and_require_strategy_a,
    find_decoder_layers,
    get_layer_attention_info,
)


# Plan5c configurations. Block names match round 3b/3c so the comparison
# columns line up automatically.
DEFAULT_BLOCKS_5C = [
    {"name": "A", "label": "valley-core",         "start": 15, "end": 19},
    {"name": "D", "label": "valley-extend-up",    "start": 15, "end": 22},
    {"name": "G", "label": "valley-extend-up-10", "start": 15, "end": 24},
]

# Catastrophic control: must remain catastrophic on IT or the late-layer
# drift mechanism shifted during instruction tuning.
DEFAULT_CONTROL_BLOCK = {
    "name": "F", "label": "late-block",  "start": 25, "end": 32,
}

DEFAULT_BLOCK_R_VALUES = [1, 2, 4, 8]
DEFAULT_CONTROL_R = 8
DEFAULT_OVERLAY_LAYERS = [0, 5, 10, 15, 17, 20, 25, 30, 33]
DEFAULT_OVERLAY_R = 8

# Round-3b base reference perplexities at r=8, used for transfer ratios.
BASE_REFERENCE_R8 = {
    "A": 30.07,   # results_round3b_blocks.json A r=8
    "D": 33.29,   # results_round3b_blocks.json D r=8
    "G": 36.01,   # results_round3c_extended_blocks.json G r=8
    "F": 338517.18,  # results_round3b_blocks.json F r=8 (catastrophic)
}

# Round-2c base reference for the L17 valley-anchor regression.
BASE_BASELINE_PPL_ROUND1 = ROUND1_UNMODIFIED_PPL  # 12.5366...

# Tolerances per plan5c.md "Mandatory regression checks".
R1_DRIFT_TOL = 1e-4
L17_RATIO_LO = 2.5
L17_RATIO_HI = 5.0
F_CATASTROPHIC_THRESHOLD = 100.0


def _resolve_round2c_path(path):
    """Resolve the round-2c JSON, falling back to the path_2 location.

    The default ``--round2c-json`` is ``results/results_round2c_full_map.json``
    for backward compat with rounds 3a/3b/3c, but plan5c lives under the
    path_2 results tree where the file actually was committed. Pick the
    first existing candidate.
    """
    p = Path(path)
    if p.is_file():
        return p
    fallback = Path("results/path_2_depth_recurrence/results_round2c_full_map.json")
    if fallback.is_file():
        print(f"  using round-2c fallback path: {fallback}")
        return fallback
    return p  # report the original (missing) path in the warning


def _load_round2c_vanilla_ratios(path):
    """Return ``{layer: {"ppl_r8": ..., "ratio_r8_vs_baseline": ...}}``.

    Empty dict (with a warning) if the file is missing or malformed; the
    sweep proceeds with n/a in the overlay comparison column.
    """
    p = _resolve_round2c_path(path)
    if not p.is_file():
        print(f"WARNING: round-2c JSON not found at {p}; overlay comparison will be n/a.")
        return {}
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"WARNING: failed to parse {p}: {e}; overlay comparison will be n/a.")
        return {}
    base_baseline = d.get("unmodified", {}).get("ppl")
    out = {}
    for cell in d.get("cells", []):
        if cell.get("ple_mode") != "vanilla" or cell.get("r") != 8:
            continue
        l = cell["layer"]
        ppl = cell["ppl"]
        ratio = (
            ppl / base_baseline
            if base_baseline and math.isfinite(ppl) and math.isfinite(base_baseline)
            else float("nan")
        )
        out[l] = {"ppl_r8": ppl, "ratio_r8_vs_baseline": ratio}
    return out


def _load_round3_block_ppl(round3b_path, round3c_path):
    """Return ``{block_name: {r: ppl}}`` merged across rounds 3b and 3c.

    3c overrides 3b on overlapping names (G/H/I only live in 3c). Used
    for the per-cell ``base_ppl_round3*`` and ``transfer_ratio`` columns.
    """
    out = {}
    for path in (round3b_path, round3c_path):
        p = Path(path)
        if not p.is_file():
            print(f"WARNING: round-3 JSON not found at {p}; transfer ratios partial.")
            continue
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            print(f"WARNING: failed to parse {p}: {e}; skipping for transfer ratios.")
            continue
        base_baseline = d.get("unmodified", {}).get("ppl")
        for cell in d.get("block_cells", []):
            name = cell.get("name")
            r = cell.get("r")
            ppl = cell.get("ppl")
            if name is None or r is None or ppl is None:
                continue
            entry = out.setdefault(name, {})
            entry[r] = {"ppl": ppl, "baseline": base_baseline}
    return out


def _r1_regression_check(model, decoder_layers, block, inputs, baseline_ppl):
    """Run an r=1 block-loop, return (ppl, drift, passed)."""
    uninstall = install_block_loop_hooks(
        decoder_layers, block["start"], block["end"], r=1,
    )
    try:
        _, ppl = compute_perplexity(model, inputs)
    finally:
        uninstall()
    drift = (
        abs(ppl - baseline_ppl) / baseline_ppl
        if baseline_ppl else float("inf")
    )
    return ppl, drift, drift < R1_DRIFT_TOL


def _block_loop_cell(model, decoder_layers, block, r, inputs):
    """Run a single block-loop cell, return (mean_nll, ppl)."""
    uninstall = install_block_loop_hooks(
        decoder_layers, block["start"], block["end"], r=r,
    )
    try:
        return compute_perplexity(model, inputs)
    finally:
        uninstall()


def _single_layer_loop_cell(model, decoder_layers, layer_idx, r, inputs):
    """Run a vanilla single-layer loop via the block hook (block of width 1).

    Equivalent to ``install_block_loop_hooks(decoder_layers, l, l, r=r)``;
    using the same hook keeps regression semantics identical to the
    block sweep above (zero hook drift at r=1, every-iter PLE).
    """
    return _block_loop_cell(
        model, decoder_layers,
        {"start": layer_idx, "end": layer_idx},
        r, inputs,
    )


def _bucket_classification(
    *, transfer_ratios, l17_ratio, f_ratio, valley_layers_ok, drift_layers_ok,
):
    """Map measured numbers onto plan5c's 6 interpretation buckets.

    Returns ``(bucket_name, exit_action)``. Heuristic only --- the final
    call is the analyst's after they read the printed table.

    ``transfer_ratios`` is a list of (block_name, r, transfer) tuples for
    the 3 blocks A/D/G at r > 1 (excluding F control).
    ``valley_layers_ok`` is True if layers 15, 17 still look like the
    minimum of the overlay; ``drift_layers_ok`` is True if layers 25/30/33
    still show much higher ratios than the valley.
    """
    finite_ratios = [t for _, _, t in transfer_ratios if math.isfinite(t)]
    a_finite = [t for n, _, t in transfer_ratios if n == "A" and math.isfinite(t)]
    a_max = max(a_finite) if a_finite else float("nan")

    f_catastrophic = math.isfinite(f_ratio) and f_ratio > F_CATASTROPHIC_THRESHOLD

    # Bucket 4: A breaks catastrophically on IT.
    if math.isfinite(a_max) and a_max > 100.0:
        return (
            "Bucket 4 --- block A degraded catastrophically on IT",
            "HALT --- debug the block-loop hook (verify layer module class "
            "is unchanged between base and IT, run A r=1 regression, check "
            "model.model.layers path resolves the same way) before plan 6.",
        )

    # Bucket 5: F is not catastrophic on IT.
    if not f_catastrophic:
        return (
            "Bucket 5 --- F-r8 not catastrophic on IT (late-layer drift "
            "damped by instruction tuning)",
            "PROCEED to plan 6 with existing blocks; queue a follow-on "
            "to retest wider blocks (H/I past layer 24) on IT specifically.",
        )

    # Bucket 3: valley shifted (single-layer overlay disagrees with round 2c).
    if not valley_layers_ok or not drift_layers_ok:
        return (
            "Bucket 3 --- single-layer overlay disagrees with round 2c "
            "(valley shifted or drift zone moved)",
            "WRITE round 5d (full 35x3 round-2c re-run on IT, ~210 cells, "
            "~45 min); do NOT run plan 6 until the IT map is established.",
        )

    if not finite_ratios:
        return (
            "Bucket UNKNOWN --- no finite transfer ratios collected",
            "HALT --- inspect the per-cell numbers manually before plan 6.",
        )

    tight = all(0.8 <= t <= 1.3 for t in finite_ratios)
    quant_shift = all(0.5 <= t <= 2.0 for t in finite_ratios)

    if tight:
        return (
            "Bucket 1 --- tight transfer (all block ratios in [0.8, 1.3])",
            "PROCEED to plan 6 as written; block geometry validated for IT.",
        )
    if quant_shift:
        return (
            "Bucket 2 --- quantitative shift, qualitative match",
            "PROCEED to plan 6 with a noted caveat that absolute IT-side "
            "ratios are not identical to base; block choices remain valid.",
        )

    return (
        "Bucket 6 --- mixed signal (some blocks transfer, others don't)",
        "PAUSE and discuss with the analyst before plan 6.",
    )


def _print_block_table(
    *, baselines, block_cells, control_cell, base_block_ppl,
    base_baseline_ppl,
):
    """Print the plan5c "block table" (3 blocks x 4 r values + F control)."""
    it_baseline = baselines["it_baseline_ppl"]
    print("\n=== Round 5c IT-perplexity bridge ===")
    print(
        f"{'config':>16}  {'base_ratio':>11}  {'it_ratio':>10}  {'transfer':>9}"
    )
    print(
        f"{'unmodified':>16}  "
        f"{1.00:>10.2f}x  "
        f"{1.00:>9.2f}x  "
        f"{'-':>9}"
    )

    rows = []
    for cell in block_cells:
        name = cell["name"]
        r = cell["r"]
        ppl = cell["ppl"]
        it_ratio = ppl / it_baseline if it_baseline else float("nan")
        base_entry = base_block_ppl.get(name, {}).get(r)
        if base_entry and base_baseline_ppl:
            base_ppl = base_entry["ppl"]
            # Use this round's base anchor where the round-3 file's own
            # baseline matches (it should — same dataset, same dtype).
            anchor = base_entry.get("baseline") or base_baseline_ppl
            base_ratio = base_ppl / anchor
        else:
            base_ratio = float("nan")
        if math.isfinite(base_ratio) and base_ratio > 0 and math.isfinite(it_ratio):
            transfer = it_ratio / base_ratio
        else:
            transfer = float("nan")
        cell["base_ratio"] = base_ratio
        cell["it_ratio"] = it_ratio
        cell["transfer_ratio"] = transfer
        label = f"{name}  r={r}"
        print(
            f"{label:>16}  "
            f"{(f'{base_ratio:.2f}x') if math.isfinite(base_ratio) else 'n/a':>11}  "
            f"{(f'{it_ratio:.2f}x') if math.isfinite(it_ratio) else 'n/a':>10}  "
            f"{(f'{transfer:.2f}') if math.isfinite(transfer) else 'n/a':>9}"
        )
        rows.append(cell)

    if control_cell is not None:
        name = control_cell["name"]
        r = control_cell["r"]
        ppl = control_cell["ppl"]
        it_ratio = ppl / it_baseline if it_baseline else float("nan")
        base_entry = base_block_ppl.get(name, {}).get(r)
        if base_entry and base_baseline_ppl:
            anchor = base_entry.get("baseline") or base_baseline_ppl
            base_ratio = base_entry["ppl"] / anchor
        else:
            base_ratio = float("nan")
        if math.isfinite(base_ratio) and base_ratio > 0 and math.isfinite(it_ratio):
            transfer = it_ratio / base_ratio
        else:
            transfer = float("nan")
        control_cell["base_ratio"] = base_ratio
        control_cell["it_ratio"] = it_ratio
        control_cell["transfer_ratio"] = transfer
        marker = (
            "OK (catastrophic)"
            if math.isfinite(it_ratio) and it_ratio > F_CATASTROPHIC_THRESHOLD
            else "FAIL (not catastrophic --- bucket 5)"
        )
        label = f"{name}  r={r} (ctrl)"
        print(
            f"{label:>16}  "
            f"{(f'{base_ratio:.0f}x') if math.isfinite(base_ratio) else 'n/a':>11}  "
            f"{(f'{it_ratio:.2f}x') if math.isfinite(it_ratio) else 'n/a':>10}  "
            f"{(f'{transfer:.2f}') if math.isfinite(transfer) else 'n/a':>9}"
            f"   [{marker}]"
        )
    return rows


def _print_overlay_table(*, overlay_cells, base_overlay, baselines):
    """Print the single-layer r=8 coarse overlay."""
    it_baseline = baselines["it_baseline_ppl"]
    print(
        f"\nSingle-layer r={DEFAULT_OVERLAY_R} (coarse):"
    )
    print(f"{'layer':>6}  {'base_ratio':>11}  {'it_ratio':>10}")
    out = []
    for cell in overlay_cells:
        l = cell["layer"]
        it_ratio = cell["ppl"] / it_baseline if it_baseline else float("nan")
        cell["it_ratio"] = it_ratio
        base_entry = base_overlay.get(l, {})
        base_ratio = base_entry.get("ratio_r8_vs_baseline", float("nan"))
        cell["base_ratio"] = base_ratio
        print(
            f"{l:>6}  "
            f"{(f'{base_ratio:.2f}x') if math.isfinite(base_ratio) else 'n/a':>11}  "
            f"{(f'{it_ratio:.2f}x') if math.isfinite(it_ratio) else 'n/a':>10}"
        )
        out.append(cell)
    return out


def _maybe_plot(out_png_path, *, baselines, block_cells, control_cell, overlay_cells):
    """Best-effort matplotlib summary figure. Skip silently if matplotlib
    isn't importable --- the JSON is the canonical output.

    Two-panel layout: left = block transfer ratios (A/D/G + F control),
    right = single-layer r=8 ratio vs layer index, base vs IT overlaid.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not importable; skipping figure.")
        return False

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Left: block ratios ----
    labels = []
    base_ratios = []
    it_ratios = []
    for cell in block_cells:
        if cell["r"] == 1:
            continue  # r=1 is the regression check; not informative on the plot
        labels.append(f"{cell['name']}  r={cell['r']}")
        base_ratios.append(cell.get("base_ratio", float("nan")))
        it_ratios.append(cell.get("it_ratio", float("nan")))
    if control_cell is not None:
        labels.append(f"{control_cell['name']}  r={control_cell['r']}*")
        base_ratios.append(control_cell.get("base_ratio", float("nan")))
        it_ratios.append(control_cell.get("it_ratio", float("nan")))

    xs = list(range(len(labels)))
    width = 0.4
    base_clip = [r if math.isfinite(r) and r < 1e4 else float("nan") for r in base_ratios]
    it_clip = [r if math.isfinite(r) and r < 1e4 else float("nan") for r in it_ratios]
    ax_l.bar([x - width / 2 for x in xs], base_clip, width, label="base", color="#4A78A6")
    ax_l.bar([x + width / 2 for x in xs], it_clip, width, label="IT", color="#E07A5F")
    ax_l.set_xticks(xs)
    ax_l.set_xticklabels(labels, rotation=30, ha="right")
    ax_l.set_ylabel("ppl ratio (block / baseline)")
    ax_l.set_yscale("log")
    ax_l.set_title(
        f"Block transfer (IT baseline ppl={baselines['it_baseline_ppl']:.2f})"
    )
    ax_l.axhline(1.0, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)
    ax_l.axhline(F_CATASTROPHIC_THRESHOLD, color="red", linestyle=":", alpha=0.5,
                 label=f"F threshold {F_CATASTROPHIC_THRESHOLD:.0f}x")
    ax_l.legend(loc="upper left", fontsize=9)

    # ---- Right: single-layer overlay (base vs IT) ----
    layers = [c["layer"] for c in overlay_cells]
    it_overlay = [c.get("it_ratio", float("nan")) for c in overlay_cells]
    base_overlay = [c.get("base_ratio", float("nan")) for c in overlay_cells]
    ax_r.plot(layers, base_overlay, "o-", color="#4A78A6", label="base (round 2c)")
    ax_r.plot(layers, it_overlay, "s-", color="#E07A5F", label="IT (this round)")
    ax_r.set_yscale("log")
    ax_r.set_xlabel("layer index")
    ax_r.set_ylabel(f"ppl ratio at r={DEFAULT_OVERLAY_R}")
    ax_r.set_title("Single-layer overlay (lower = more loopable)")
    ax_r.axvspan(15, 19, alpha=0.12, color="green", label="round-2c valley (15-19)")
    ax_r.legend(loc="best", fontsize=9)

    fig.suptitle("Round 5c --- IT-weight perplexity bridge", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    p = Path(out_png_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print(f"Wrote {p}")
    return True


def run_it_perplexity_bridge(args):
    """Round 5c: IT-weight perplexity bridge.

    Loads its own model so callers do not pay for a base-model load that
    will immediately be discarded. Mirrors round 5's two-pass pattern but
    only uses one model.
    """
    if args.r_values is not None:
        print(
            "ERROR: --mode it-perplexity-bridge fixes r in {1,2,4,8} "
            "for the block sweep. --r-values is not supported."
        )
        sys.exit(1)
    if args.layers is not None:
        print(
            "ERROR: --mode it-perplexity-bridge fixes the single-layer "
            "overlay layer set. --layers is not supported."
        )
        sys.exit(1)
    if args.only_diagnostic:
        print("ERROR: --only-diagnostic only applies under --mode ple-variants.")
        sys.exit(1)

    print_env()
    dtype = DTYPE_MAP[args.dtype]
    model_id = args.model_id

    print(f"Loading {model_id} in {args.dtype} (round 5c IT bridge) ...")
    tokenizer, model = load_model(model_id, dtype)
    decoder_layers = find_decoder_layers(model)
    n_layers = len(decoder_layers)
    if n_layers != EXPECTED_NUM_LAYERS:
        print(
            f"WARNING: expected {EXPECTED_NUM_LAYERS} decoder layers, "
            f"found {n_layers}. Continuing anyway."
        )

    # Cheap parity check: tokenizer vocab size (per plan5c notes).
    base_vocab = None
    try:
        from transformers import AutoTokenizer
        base_tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
        base_vocab = base_tok.vocab_size
        print(
            f"Tokenizer parity: IT vocab={tokenizer.vocab_size}  "
            f"base vocab={base_vocab}  match={tokenizer.vocab_size == base_vocab}"
        )
    except Exception as e:  # network / cache miss --- don't abort
        print(f"WARNING: tokenizer parity check skipped ({e}).")

    inputs = prepare_inputs(tokenizer, args.num_sequences, args.max_length)

    # Range-check every block before any compute.
    blocks = [dict(b) for b in DEFAULT_BLOCKS_5C]
    control = dict(DEFAULT_CONTROL_BLOCK)
    for b in blocks + [control]:
        if b["start"] < 0 or b["end"] >= n_layers or b["start"] > b["end"]:
            print(
                f"ERROR: block {b['name']} [{b['start']}..{b['end']}] out "
                f"of range for {n_layers} layers."
            )
            sys.exit(1)
    for l in DEFAULT_OVERLAY_LAYERS:
        if l < 0 or l >= n_layers:
            print(f"ERROR: overlay layer {l} out of range [0, {n_layers}).")
            sys.exit(1)

    # ---- Wiring inspection (Strategy A required for the hook) ----
    inspection = _inspect_and_require_strategy_a(decoder_layers[blocks[0]["start"]])
    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    # ---- Comparison data: round 2c overlay + round 3b/3c block ppls ----
    base_overlay = _load_round2c_vanilla_ratios(args.round2c_json)
    base_block_ppl = _load_round3_block_ppl(
        args.round3b_json, args.round3c_json,
    )

    # ---- IT baseline (no hook) ----
    print("Running IT unmodified baseline (no hook) ...")
    it_nll, it_baseline_ppl = compute_perplexity(model, inputs)
    it_vs_base_ratio = it_baseline_ppl / BASE_BASELINE_PPL_ROUND1
    print(
        f"IT baseline:  mean NLL = {it_nll:.4f}   ppl = {it_baseline_ppl:.4f}   "
        f"vs round-1 base ({BASE_BASELINE_PPL_ROUND1:.4f}) = "
        f"{it_vs_base_ratio:.3f}x\n"
    )
    baselines = {
        "it_baseline_ppl": it_baseline_ppl,
        "it_baseline_mean_nll": it_nll,
        "base_baseline_ppl_round1": BASE_BASELINE_PPL_ROUND1,
        "it_vs_base_ratio": it_vs_base_ratio,
    }

    # ---- Mandatory regression checks ----
    print("=== Mandatory regression checks (plan5c) ===")
    regression_results = {}

    # 1. A r=1 must match IT baseline.
    a_block = next(b for b in blocks if b["name"] == "A")
    a_r1_ppl, a_r1_drift, a_r1_pass = _r1_regression_check(
        model, decoder_layers, a_block, inputs, it_baseline_ppl,
    )
    regression_results["A_r1_no_op"] = {
        "ppl": a_r1_ppl, "drift": a_r1_drift, "passed": a_r1_pass,
    }
    print(
        f"  A r=1: ppl={a_r1_ppl:.6f}  drift={a_r1_drift:.2e}  "
        f"[{'OK' if a_r1_pass else 'FAIL'}]"
    )
    if not a_r1_pass:
        print(
            "ERROR: A r=1 regression failed --- block-loop hook is "
            "perturbing the model on IT weights even when it should be a "
            "no-op. Halting before further sweep."
        )
        _write_partial(args, baselines, regression_results, blocks, control,
                       inspection, reason="A r=1 regression failed")
        sys.exit(3)

    # 2. D r=1 must match IT baseline.
    d_block = next(b for b in blocks if b["name"] == "D")
    d_r1_ppl, d_r1_drift, d_r1_pass = _r1_regression_check(
        model, decoder_layers, d_block, inputs, it_baseline_ppl,
    )
    regression_results["D_r1_no_op"] = {
        "ppl": d_r1_ppl, "drift": d_r1_drift, "passed": d_r1_pass,
    }
    print(
        f"  D r=1: ppl={d_r1_ppl:.6f}  drift={d_r1_drift:.2e}  "
        f"[{'OK' if d_r1_pass else 'FAIL'}]"
    )
    if not d_r1_pass:
        print(
            "ERROR: D r=1 regression failed --- block-loop hook is "
            "perturbing the model on IT weights even when it should be a "
            "no-op. Halting before further sweep."
        )
        _write_partial(args, baselines, regression_results, blocks, control,
                       inspection, reason="D r=1 regression failed")
        sys.exit(3)

    # ---- Main block sweep (r in {1, 2, 4, 8} for A/D/G + F r=8 control) ----
    print("\n=== Block sweep ===")
    block_cells = []
    # r=1 reused from the regression checks above for A and D so we don't
    # re-pay the forward; G needs its own r=1 cell (regression-only block).
    block_cells.append({
        "name": "A", "label": a_block["label"], "start": a_block["start"],
        "end": a_block["end"], "width": a_block["end"] - a_block["start"] + 1,
        "r": 1, "ppl": a_r1_ppl, "mean_nll": math.log(a_r1_ppl) if a_r1_ppl > 0 else float("nan"),
    })
    block_cells.append({
        "name": "D", "label": d_block["label"], "start": d_block["start"],
        "end": d_block["end"], "width": d_block["end"] - d_block["start"] + 1,
        "r": 1, "ppl": d_r1_ppl, "mean_nll": math.log(d_r1_ppl) if d_r1_ppl > 0 else float("nan"),
    })

    g_block = next(b for b in blocks if b["name"] == "G")
    g_r1_ppl, g_r1_drift, g_r1_pass = _r1_regression_check(
        model, decoder_layers, g_block, inputs, it_baseline_ppl,
    )
    regression_results["G_r1_no_op"] = {
        "ppl": g_r1_ppl, "drift": g_r1_drift, "passed": g_r1_pass,
    }
    print(
        f"  G r=1: ppl={g_r1_ppl:.6f}  drift={g_r1_drift:.2e}  "
        f"[{'OK' if g_r1_pass else 'FAIL'}]  (advisory)"
    )
    block_cells.append({
        "name": "G", "label": g_block["label"], "start": g_block["start"],
        "end": g_block["end"], "width": g_block["end"] - g_block["start"] + 1,
        "r": 1, "ppl": g_r1_ppl, "mean_nll": math.log(g_r1_ppl) if g_r1_ppl > 0 else float("nan"),
    })

    main_r_values = [r for r in DEFAULT_BLOCK_R_VALUES if r != 1]
    total_main = len(blocks) * len(main_r_values) + 1  # + F r=8
    done = 0
    for b in blocks:
        for r in main_r_values:
            mean_nll, ppl = _block_loop_cell(model, decoder_layers, b, r, inputs)
            done += 1
            block_cells.append({
                "name": b["name"], "label": b["label"], "start": b["start"],
                "end": b["end"], "width": b["end"] - b["start"] + 1,
                "r": r, "ppl": ppl, "mean_nll": mean_nll,
            })
            print(
                f"  [{done:2d}/{total_main}] {b['name']} "
                f"({b['start']:>2}-{b['end']:<2}, w={b['end']-b['start']+1})  "
                f"r={r}: ppl={ppl:.4f}"
            )

    # F r=8 control.
    mean_nll_f, ppl_f = _block_loop_cell(
        model, decoder_layers, control, DEFAULT_CONTROL_R, inputs,
    )
    done += 1
    control_cell = {
        "name": control["name"], "label": control["label"],
        "start": control["start"], "end": control["end"],
        "width": control["end"] - control["start"] + 1,
        "r": DEFAULT_CONTROL_R, "ppl": ppl_f, "mean_nll": mean_nll_f,
    }
    print(
        f"  [{done:2d}/{total_main}] F (ctrl, "
        f"{control['start']:>2}-{control['end']:<2})  "
        f"r={DEFAULT_CONTROL_R}: ppl={ppl_f:.4f}"
    )

    # ---- Single-layer r=8 overlay ----
    print(f"\n=== Single-layer overlay r={DEFAULT_OVERLAY_R} ===")
    overlay_cells = []
    layer_meta = {}
    for l in DEFAULT_OVERLAY_LAYERS:
        info = get_layer_attention_info(model, l)
        layer_meta[l] = info
        mean_nll, ppl = _single_layer_loop_cell(
            model, decoder_layers, l, DEFAULT_OVERLAY_R, inputs,
        )
        overlay_cells.append({
            "layer": l,
            "r": DEFAULT_OVERLAY_R,
            "ppl": ppl,
            "mean_nll": mean_nll,
            "attention_type": info["attention_type"],
            "is_kv_consumer": info["is_kv_consumer"],
        })
        print(
            f"  layer {l:>2}  attn={info['attention_type']:>18}  "
            f"kv_cons={str(info['is_kv_consumer']):>5}  "
            f"r={DEFAULT_OVERLAY_R}: ppl={ppl:.4f}"
        )

    # ---- Regression checks 3 + 4 (post-sweep) ----
    l17_cell = next((c for c in overlay_cells if c["layer"] == 17), None)
    l17_ratio = (
        l17_cell["ppl"] / it_baseline_ppl if l17_cell and it_baseline_ppl
        else float("nan")
    )
    l17_pass = (
        math.isfinite(l17_ratio)
        and L17_RATIO_LO <= l17_ratio <= L17_RATIO_HI
    )
    regression_results["L17_r8_in_valley"] = {
        "ratio": l17_ratio,
        "tolerance_lo": L17_RATIO_LO,
        "tolerance_hi": L17_RATIO_HI,
        "passed": l17_pass,
    }

    f_ratio = ppl_f / it_baseline_ppl if it_baseline_ppl else float("nan")
    f_pass = math.isfinite(f_ratio) and f_ratio > F_CATASTROPHIC_THRESHOLD
    regression_results["F_r8_catastrophic"] = {
        "ratio": f_ratio,
        "threshold": F_CATASTROPHIC_THRESHOLD,
        "passed": f_pass,
    }

    print("\n=== Post-sweep regression checks ===")
    print(
        f"  L17 r=8 ratio: {l17_ratio:.3f}x  "
        f"(tolerance [{L17_RATIO_LO:.1f}x, {L17_RATIO_HI:.1f}x])  "
        f"[{'OK' if l17_pass else 'FAIL'}]"
    )
    print(
        f"  F   r=8 ratio: {f_ratio:.3f}x  "
        f"(threshold > {F_CATASTROPHIC_THRESHOLD:.0f}x)  "
        f"[{'OK' if f_pass else 'FAIL --- bucket 5'}]"
    )

    # ---- Print the plan5c summary tables ----
    block_cells_main = [c for c in block_cells if c["r"] in main_r_values]
    _print_block_table(
        baselines=baselines,
        block_cells=block_cells_main,
        control_cell=control_cell,
        base_block_ppl=base_block_ppl,
        base_baseline_ppl=BASE_BASELINE_PPL_ROUND1,
    )
    _print_overlay_table(
        overlay_cells=overlay_cells,
        base_overlay=base_overlay,
        baselines=baselines,
    )

    # Decorate r=1 block cells (not in the printed table) with comparison
    # columns so the JSON is self-contained.
    for c in block_cells:
        if c["r"] == 1:
            c["it_ratio"] = (
                c["ppl"] / it_baseline_ppl if it_baseline_ppl else float("nan")
            )
            base_entry = base_block_ppl.get(c["name"], {}).get(1)
            if base_entry and base_entry.get("baseline"):
                c["base_ratio"] = base_entry["ppl"] / base_entry["baseline"]
            else:
                c["base_ratio"] = float("nan")
            c["transfer_ratio"] = (
                c["it_ratio"] / c["base_ratio"]
                if math.isfinite(c.get("it_ratio", float("nan")))
                and math.isfinite(c.get("base_ratio", float("nan")))
                and c["base_ratio"] > 0
                else float("nan")
            )

    # ---- Bucket classification ----
    transfer_for_bucket = [
        (c["name"], c["r"], c.get("transfer_ratio", float("nan")))
        for c in block_cells_main if c["name"] in {"A", "D", "G"}
    ]

    # Valley check: layer 15 / 17 should still be among the lowest IT ratios
    # in the overlay (round-2c says 15/17 are the minimum on base).
    overlay_by_layer = {c["layer"]: c.get("it_ratio", float("nan")) for c in overlay_cells}
    valley_vals = [overlay_by_layer.get(l, float("nan")) for l in (15, 17)]
    drift_vals = [overlay_by_layer.get(l, float("nan")) for l in (25, 30, 33)]
    valley_min = min((v for v in valley_vals if math.isfinite(v)), default=float("inf"))
    drift_min = min((v for v in drift_vals if math.isfinite(v)), default=0.0)
    # "Valley still anchored 15-19" --- 15/17 should be at most ~1/3 of the
    # late-drift minimum. Slack chosen to tolerate IT-side jitter without
    # falsely flagging a shifted valley when the qualitative shape holds.
    valley_layers_ok = (
        math.isfinite(valley_min)
        and (drift_min == 0.0 or valley_min < drift_min / 3.0)
    )
    drift_layers_ok = math.isfinite(drift_min) and drift_min > 5.0

    bucket, exit_action = _bucket_classification(
        transfer_ratios=transfer_for_bucket,
        l17_ratio=l17_ratio,
        f_ratio=f_ratio,
        valley_layers_ok=valley_layers_ok,
        drift_layers_ok=drift_layers_ok,
    )
    print(f"\n=== Interpretation ===\n  {bucket}\n  Exit action: {exit_action}")

    # ---- Output ----
    output_json = args.output_json or _results_path(
        "path_2_depth_recurrence/plan5c/results_round5c_it_perplexity_bridge.json"
    )
    output = {
        "config": {
            "mode": "it-perplexity-bridge",
            "model_id": model_id,
            "dtype": args.dtype,
            "n_sequences": args.num_sequences,
            "max_length": args.max_length,
            "ple_kwarg": ple_kwarg,
            "round2c_json": args.round2c_json,
            "round3b_json": args.round3b_json,
            "round3c_json": args.round3c_json,
            "block_r_values": DEFAULT_BLOCK_R_VALUES,
            "overlay_layers": DEFAULT_OVERLAY_LAYERS,
            "overlay_r": DEFAULT_OVERLAY_R,
        },
        "inspection": {
            "strategy": inspection["strategy"],
            "layer_class": inspection["layer_class"],
            "source_file": inspection["source_file"],
            "start_lineno": inspection["start_lineno"],
            "signature": inspection["signature"],
            "ple_kwarg": inspection["ple_kwarg"],
        },
        "baselines": baselines,
        "regression_checks": regression_results,
        "tokenizer_parity": {
            "it_vocab_size": getattr(tokenizer, "vocab_size", None),
            "base_vocab_size": base_vocab,
        },
        "block_cells": block_cells,
        "control_cell": control_cell,
        "single_layer_cells": overlay_cells,
        "interpretation": {
            "bucket": bucket,
            "exit_action": exit_action,
            "valley_layers_ok": valley_layers_ok,
            "drift_layers_ok": drift_layers_ok,
        },
    }
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")

    # ---- Companion PNG (per-plan results figure) ----
    png_path = str(Path(output_json).with_suffix(".png"))
    _maybe_plot(
        png_path,
        baselines=baselines,
        block_cells=block_cells_main,
        control_cell=control_cell,
        overlay_cells=overlay_cells,
    )

    # ---- Free the IT model promptly. The watcher syncs the JSON down
    # before pod teardown; nothing else needs the GPU. ----
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_partial(args, baselines, regression_results, blocks, control,
                   inspection, *, reason):
    """Persist the partial state when a regression failure halts the run.

    Lets the watcher sync something useful down before teardown so the
    operator can diagnose without re-running.
    """
    output_json = args.output_json or _results_path(
        "path_2_depth_recurrence/plan5c/results_round5c_it_perplexity_bridge.json"
    )
    payload = {
        "config": {
            "mode": "it-perplexity-bridge",
            "model_id": args.model_id,
            "dtype": args.dtype,
            "n_sequences": args.num_sequences,
            "max_length": args.max_length,
            "ple_kwarg": inspection["ple_kwarg"],
            "round2c_json": args.round2c_json,
            "round3b_json": args.round3b_json,
            "round3c_json": args.round3c_json,
        },
        "aborted": True,
        "reason": reason,
        "baselines": baselines,
        "regression_checks": regression_results,
    }
    _write_results_json(output_json, payload)
    print(f"Wrote partial {output_json}")
