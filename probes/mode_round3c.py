"""Round 3c --- pinning down the stable block region.

Round 3b established that (a) blocks crossing the KV producer/consumer
boundary at layer 14/15 break catastrophically, and (b) inside the
consumer region (layers 15-34) there is a stable subregion and a
slow-drift subregion, with the boundary somewhere between layers 22
(D, stable) and 25 (F, drift).

Round 3c probes the exact boundary with three new blocks, all
KV-consumer:

    G  15-24  (10 layers, widest extension up)
    H  15-23  ( 9 layers, intermediate width)
    I  20-27  ( 8 layers, same width as D but shifted +5 --- does
              position matter, or just width?)

The run also re-executes A (15-19), B (15-18), and D (15-22) from
round 3b as in-sweep references. That makes the comparison table
self-contained and doubles as a sanity check that round-3b numbers
reproduce on the current host.

All heavy lifting (hooks, regression checks, JSON output, single-layer
merge) is shared with round 3b via ``_run_block_looping_sweep``. This
module provides the 3c-specific default block set, the multi-r
summary table, and the bucket interpreter defined in plans/plan3c.md.
"""

import math

from .env import _results_path
from .mode_round3b import _fmt_layers, _run_block_looping_sweep


# Plan3c block set: three round-3b references (A, B, D) plus three new
# probes (G, H, I). See plans/plan3c.md for rationale.
DEFAULT_BLOCKS_3C = [
    {"name": "A", "label": "valley-core",        "start": 15, "end": 19},
    {"name": "B", "label": "valley-narrow",      "start": 15, "end": 18},
    {"name": "D", "label": "valley-extend-up",   "start": 15, "end": 22},
    {"name": "G", "label": "valley-extend-up-10", "start": 15, "end": 24},
    {"name": "H", "label": "valley-extend-up-9",  "start": 15, "end": 23},
    {"name": "I", "label": "valley-shifted-late", "start": 20, "end": 27},
]


def _multi_r_summary(blocks, r_values, block_ppl, unmod_ppl):
    """Print and return the plan3c-format summary: one row per block
    with ppl at every r, plus a ``vs_block_D_r_top`` ratio column.

    If block D is absent from the run (custom block set), the ratio
    column is emitted as n/a and the header reflects that.
    """
    top_r = max(r_values)
    by_name = {b["name"]: b for b in blocks}
    d_top = block_ppl.get(("D", top_r), float("nan")) if "D" in by_name else float("nan")
    have_d = "D" in by_name and math.isfinite(d_top) and d_top > 0

    r_header = "  ".join(f"r={r:<4}" for r in r_values)
    ratio_header = f"vs_D_r{top_r}" if have_d else "vs_D(n/a)"
    print(f"\n=== Round 3c summary (baseline ppl={unmod_ppl:.4f}) ===")
    print(
        f"{'block':>5}  {'layers':>7}  {'width':>5}  {r_header}  {ratio_header:>10}"
    )

    rows = []
    for b in blocks:
        name = b["name"]
        width = b["end"] - b["start"] + 1
        r_ppls = [block_ppl.get((name, r), float("nan")) for r in r_values]
        top_ppl = block_ppl.get((name, top_r), float("nan"))
        if have_d and math.isfinite(top_ppl):
            ratio = top_ppl / d_top
            ratio_str = f"{ratio:.2f}x"
        else:
            ratio = float("nan")
            ratio_str = "n/a"

        r_ppls_str = "  ".join(f"{p:>5.1f}" for p in r_ppls)
        print(
            f"{name:>5}  {_fmt_layers(b):>7}  {width:>5}  {r_ppls_str}  {ratio_str:>10}"
        )

        rows.append({
            "name": name,
            "label": b.get("label"),
            "start": b["start"],
            "end": b["end"],
            "width": width,
            "ppl_by_r": {str(r): p for r, p in zip(r_values, r_ppls)},
            "vs_block_D_top_r": ratio,
        })

    # Also emit the plan3c stability ratio (r_top / r_min) for each block.
    print(f"\n=== Stability (r={max(r_values)} / r={min(r_values)} ratio) ===")
    r_lo = min(r_values)
    r_hi = max(r_values)
    for b in blocks:
        name = b["name"]
        p_lo = block_ppl.get((name, r_lo), float("nan"))
        p_hi = block_ppl.get((name, r_hi), float("nan"))
        if math.isfinite(p_lo) and math.isfinite(p_hi) and p_lo > 0:
            stability = p_hi / p_lo
            if stability < 1.2:
                tag = "stable"
            elif stability < 2.0:
                tag = "slight drift"
            elif stability < 10.0:
                tag = "compounding"
            else:
                tag = "catastrophic"
            print(f"  {name:>5}: r{r_hi}/r{r_lo} = {stability:.3f}  [{tag}]")
        else:
            print(f"  {name:>5}: n/a")

    return {
        "baseline_ppl": unmod_ppl,
        "top_r": top_r,
        "d_reference_ppl_top_r": d_top if have_d else None,
        "rows": rows,
    }


def _interpret_block_map_3c(blocks, r_values, block_ppl, unmod_ppl):
    """Implements the plan3c bucket logic (see plans/plan3c.md lines 74-91).

    Two orthogonal axes are tested:

      - **Width axis** (G, H vs D): how far can the stable region extend
        upward from layer 22?
      - **Position axis** (I vs D): must the block anchor at layer 15,
        or does any 8-layer all-consumer block work?

    The plan defines each axis's buckets with a flat ppl < 50 threshold
    at r=top. Returns a multi-paragraph hint covering both axes plus a
    consistency check that flags Bucket 6 patterns.
    """
    top_r = max(r_values)
    by_name = {b["name"]: b for b in blocks}

    required = {"D", "G", "H", "I"}
    missing = sorted(required - by_name.keys())
    if missing:
        return (
            f"Bucket UNKNOWN --- required blocks {missing} not in this run; "
            "plan3c bucket heuristic does not apply. Inspect the table manually."
        )

    def ppl_of(name):
        return block_ppl.get((name, top_r), float("nan"))

    def works(name):
        p = ppl_of(name)
        return math.isfinite(p) and p < 50.0

    g_ok = works("G")
    h_ok = works("H")
    i_ok = works("I")
    d_ok = works("D")

    # ----- Width axis (buckets 1-3) -----
    if g_ok:
        width_verdict = (
            "Bucket 1 (G works) --- stable region extends to >=layer 24. "
            "Use block G (15-24) as the retrofit recurrent block: 10 layers, "
            "25% more effective compute than D. Worth pushing further to "
            "layer 25+ (follow-up: block 15-25)."
        )
    elif h_ok:
        width_verdict = (
            "Bucket 2 (G breaks, H works) --- stable region ends at layer 23. "
            "Use block H (15-23): 9 layers, 12% more than D. The layer 23->24 "
            "boundary matters but probably not worth investigating further; "
            "proceed with H."
        )
    elif d_ok:
        width_verdict = (
            "Bucket 3 (G and H both break; D remains the maximum) --- stable "
            "region ends at layer 22 exactly. Use block D (15-22). Single-layer "
            "probes at 23 (ppl 592) and 24 (ppl 4391) should have been rescued "
            "by block-looping but weren't --- an unseen constraint at layer 23+."
        )
    else:
        width_verdict = (
            "Bucket WIDTH-UNEXPECTED --- D itself broke (ppl>=50 at r=8). "
            "Round 3b had D viable at ppl=33.3. Suspect compute drift or a "
            "hook regression; stop and debug before drawing block conclusions."
        )

    # ----- Position axis (buckets 4-5) -----
    if i_ok:
        position_verdict = (
            "Bucket 4 (I works) --- position within the consumer zone is "
            "flexible. Any 8-layer all-consumer block is viable. The stable "
            "subregion is broader than 15-22; follow-up: test one more "
            "shifted-late block (e.g. 22-29) to confirm."
        )
    elif d_ok:
        position_verdict = (
            "Bucket 5 (I breaks, D works) --- position matters: block must "
            "anchor near layer 15. Possibly because layer 15 is the first "
            "KV consumer; the producer->consumer transition is a natural "
            "anchor point."
        )
    else:
        position_verdict = (
            "Bucket POSITION-UNEXPECTED --- both I and D broke; cannot "
            "infer whether position matters when the reference itself failed."
        )

    # ----- Bucket 6 consistency flags -----
    anomalies = []
    # G is wider than H and D; if G works without D/H working, monotonicity is violated.
    if g_ok and not d_ok:
        anomalies.append("G works but D does not (G is a superset of D by width)")
    if g_ok and not h_ok:
        anomalies.append("G works but H does not (G is a superset of H by width)")
    if h_ok and not d_ok:
        anomalies.append("H works but D does not (H is a superset of D by width)")
    # Width monotonicity of ppl at r_top
    ppls = {n: ppl_of(n) for n in ("D", "H", "G")}
    if all(math.isfinite(v) for v in ppls.values()):
        # Wider block ppls should not be drastically lower than narrower ones
        # given the stable-region hypothesis; drastic non-monotonicity is worth flagging.
        if ppls["G"] < ppls["D"] * 0.5:
            anomalies.append(
                "G ppl is much lower than D ppl (width should not help this much in the stable region)"
            )

    if anomalies:
        bucket6 = "\n\nBucket 6 SIGNAL --- unexpected pattern:\n  - " + "\n  - ".join(anomalies)
    else:
        bucket6 = ""

    return (
        f"Width axis: {width_verdict}\n\n"
        f"Position axis: {position_verdict}"
        f"{bucket6}"
    )


def run_block_looping_3c_map(args, model, decoder_layers, inputs):
    """Round 3c: 6 blocks (3 round-3b references + 3 new probes) x
    {r=2, r=4, r=8} = 18 cells, vanilla PLE only.

    Thin wrapper around ``_run_block_looping_sweep`` with the 3c default
    block set, output filename, mode string, and bucket interpreter.
    The 3c multi-r summary and stability table are emitted as an
    ``extra_analysis`` hook and recorded under ``analysis_extra`` in
    the output JSON.
    """
    _run_block_looping_sweep(
        args, model, decoder_layers, inputs,
        default_blocks=DEFAULT_BLOCKS_3C,
        default_output_filename="results_round3c_extended_blocks.json",
        mode_name="block-looping-3c",
        interpreter_fn=_interpret_block_map_3c,
        extra_analysis_fn=_multi_r_summary,
    )
