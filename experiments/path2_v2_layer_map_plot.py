"""Phase 2 layer-map heat map + CSV.

Reads the report dict produced by ``experiments/path2_v2_phase2.aggregate``
and writes:

  - ``phase2_layer_map.csv``  --- one row per layer; the load-bearing
                                  artefact for Phase 3.
  - ``phase2_layer_map.png``  --- matplotlib stacked-bar figure.

Both functions are pure-Python; matplotlib is only imported by
``render_png`` so the CSV is always written regardless of plot deps.

Standalone usage:

    python experiments/path2_v2_layer_map_plot.py \\
        --report results/path_2_depth_recurrence_v2/phase2/phase2_summary.json \\
        --out-dir results/path_2_depth_recurrence_v2/phase2/

Per the user's CLAUDE.md, both files land alongside the JSON under
``results/path_2_depth_recurrence_v2/phase2/``.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional


# Stable column order; Phase 3 reads these names. Don't reorder unless
# you also update Phase 3's ingest.
CSV_FIELDS = [
    "layer", "name",
    "accuracy_smart_v2", "accuracy_smart_v2_ci_lo", "accuracy_smart_v2_ci_hi",
    "accuracy_legacy",
    "loop_rate", "truncation_rate", "parse_rate",
    "mean_gen_tokens",
    "n_problems",
    "attention_type", "is_kv_consumer", "depth_tertile",
    "round2c_base_ppl", "base_ppl", "delta_ppl",
    "bucket", "bucket_reason",
]


def _layer_index_from_name(name: str) -> int:
    # "L00" -> 0
    return int(name[1:]) if name.startswith("L") else -1


def _row_for_cell(name: str, cell: dict) -> dict:
    s = cell.get("summary") or {}
    meta = cell.get("metadata") or {}
    ci = s.get("accuracy_smart_v2_ci95") or [None, None]
    return {
        "layer": _layer_index_from_name(name),
        "name": name,
        "accuracy_smart_v2": s.get("accuracy_smart_v2"),
        "accuracy_smart_v2_ci_lo": ci[0] if ci else None,
        "accuracy_smart_v2_ci_hi": ci[1] if ci else None,
        "accuracy_legacy": s.get("accuracy_legacy"),
        "loop_rate": s.get("loop_rate"),
        "truncation_rate": s.get("truncation_rate"),
        "parse_rate": s.get("parse_rate"),
        "mean_gen_tokens": s.get("mean_gen_tokens"),
        "n_problems": s.get("n_problems"),
        "attention_type": meta.get("attention_type"),
        "is_kv_consumer": meta.get("is_kv_consumer"),
        "depth_tertile": meta.get("depth_tertile"),
        "round2c_base_ppl": cell.get("round2c_base_ppl"),
        "base_ppl": cell.get("base_ppl"),
        "delta_ppl": cell.get("delta_ppl"),
        "bucket": cell.get("bucket"),
        "bucket_reason": cell.get("bucket_reason"),
    }


def write_csv(*, report: dict, path: Path) -> None:
    """Emit the per-layer CSV. One row per cell, sorted by layer."""
    cells = report.get("cells", {})
    rows = [_row_for_cell(name, c) for name, c in cells.items()]
    rows.sort(key=lambda r: r["layer"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ---------------------------------------------------------------------------
# PNG render (matplotlib lazy import)
# ---------------------------------------------------------------------------

# Bucket -> color used for both the bar tint and the layer index label.
_BUCKET_COLORS = {
    "preserved": "#2ca02c",   # green
    "degraded":  "#ff7f0e",   # orange
    "broken":    "#d62728",   # red
}


def render_png(*, report: dict, path: Path) -> None:
    """Render the per-layer stacked-bar figure.

    x-axis: layer index 0..N-1.
    y-axis: probability (acc, loop_rate, trunc_rate, all in [0, 1]).
    Three bars per layer, side by side. Layer index label below the bar
    is colored by bucket. Horizontal line at LAYER_PRESERVED_ACC_MIN
    marks the bucket boundary.
    """
    # Lazy import: keep CSV path usable on a CPU laptop without matplotlib.
    import matplotlib
    matplotlib.use("Agg")  # safe in headless contexts (RunPod, CI).
    import matplotlib.pyplot as plt
    from probes.phase2_gates import LAYER_PRESERVED_ACC_MIN, LAYER_DEGRADED_ACC_MIN

    cells = report.get("cells", {})
    rows = [_row_for_cell(name, c) for name, c in cells.items()]
    rows.sort(key=lambda r: r["layer"])

    layers = [r["layer"] for r in rows]
    acc = [r["accuracy_smart_v2"] or 0.0 for r in rows]
    loop = [r["loop_rate"] or 0.0 for r in rows]
    trunc = [r["truncation_rate"] or 0.0 for r in rows]
    buckets = [r["bucket"] or "broken" for r in rows]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    width = 0.27
    xs = list(range(len(layers)))
    ax.bar([x - width for x in xs], acc, width=width,
           color="#1f77b4", label="accuracy_smart_v2")
    ax.bar(xs, loop, width=width, color="#d62728", label="loop_rate")
    ax.bar([x + width for x in xs], trunc, width=width,
           color="#9467bd", label="truncation_rate")

    ax.axhline(LAYER_PRESERVED_ACC_MIN, color="#2ca02c", linestyle="--",
               linewidth=1.0,
               label=f"preserved threshold ({LAYER_PRESERVED_ACC_MIN:.2f})")
    ax.axhline(LAYER_DEGRADED_ACC_MIN, color="#ff7f0e", linestyle=":",
               linewidth=1.0,
               label=f"degraded threshold ({LAYER_DEGRADED_ACC_MIN:.2f})")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L:02d}" for L in layers],
                       rotation=90, fontsize=8)
    for tick_label, bucket in zip(ax.get_xticklabels(), buckets):
        tick_label.set_color(_BUCKET_COLORS.get(bucket, "black"))

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("probability")
    decision = report.get("phase3_launch", {})
    pre = report.get("preflight", {}).get("L17_r1_token_match", {})
    title = (
        "Phase 2 layer reasoning map "
        "(N=50/layer, r=8, C2 prompt, IT model)\n"
        f"Wilson 95 % CI ~ +/-13 pp/layer | "
        f"buckets: preserved={len(report.get('buckets', {}).get('preserved', []))}, "
        f"degraded={len(report.get('buckets', {}).get('degraded', []))}, "
        f"broken={len(report.get('buckets', {}).get('broken', []))} | "
        f"phase3 mode={decision.get('mode')} | "
        f"pre-flight matches={pre.get('matches')}/{pre.get('shared')}"
    )
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--report", required=True, type=Path,
                   help="Path to phase2_summary.json.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Where to write phase2_layer_map.{csv,png}.")
    p.add_argument("--no-plots", action="store_true",
                   help="Write CSV only.")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    report = json.loads(args.report.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "phase2_layer_map.csv"
    write_csv(report=report, path=csv_path)
    print(f"Wrote {csv_path}")
    if args.no_plots:
        return 0
    png_path = args.out_dir / "phase2_layer_map.png"
    try:
        render_png(report=report, path=png_path)
        print(f"Wrote {png_path}")
    except Exception as e:
        print(f"WARN: PNG render failed: {e}")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
