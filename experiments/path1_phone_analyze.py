"""Path 1 plan 8 v2 - merge Mac+iPhone JSONLs into a single results bundle.

Reads:
    results/path_1_cot_tokens/plan8/cells_phone/{c2_mac,a3_mac,c2_iphone,a3_iphone}__*.jsonl
    results/path_1_cot_tokens/plan8/cells_phone/{cell}_sustained.jsonl
    results/path_1_cot_tokens/plan8/cells_phone/accuracy_parity.json (if present)

Writes (per project convention - same dir, _phone suffix to avoid clobber
with the legacy GPU-proxy results_plan8.json - see plan section 2):
    results/path_1_cot_tokens/plan8/results_plan8_phone.json
    results/path_1_cot_tokens/plan8/plan8_phone_pareto.png
    results/path_1_cot_tokens/plan8/plan8_phone_sustained.png

The user-memory rule "save results under results/<path>/<plan>/ and emit
PNGs alongside JSON" is honored.

Outcomes A-F (plan section "Pre-registered interpretation") are scored
from the merged numbers and printed.

Run:
    python experiments/path1_phone_analyze.py
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Iterable

# matplotlib is the only non-stdlib dep; lazy import so --json-only works
# without it.

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results" / "path_1_cot_tokens" / "plan8"
CELLS_PHONE_DIR = RESULTS_DIR / "cells_phone"
RESULTS_JSON = RESULTS_DIR / "results_plan8_phone.json"
PARETO_PNG = RESULTS_DIR / "plan8_phone_pareto.png"
SUSTAINED_PNG = RESULTS_DIR / "plan8_phone_sustained.png"

CELLS = ("C2-Mac", "A3-Mac", "C2-iPhone", "A3-iPhone")

# Plan section "Outcomes" thresholds.
OUTCOME_A_MAC_MEDIAN_S = 3.0
OUTCOME_A_PHONE_MEDIAN_S = 6.0
OUTCOME_C_PHONE_P95_S = 10.0
OUTCOME_E_DIVERGENCE = 3.0  # >3x gap = anomalous
SUSTAINED_DROP_THRESHOLD = 0.30  # 30% tps drop within probe = throttle


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _safe_mean(xs: Iterable[float]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.fmean(vals) if vals else None


def _cell_filename_glob(cell: str) -> str:
    return cell.lower().replace("-", "_") + "__*.jsonl"


def _load_cell_rows(cell: str) -> list[dict]:
    rows: list[dict] = []
    for shard in sorted(CELLS_PHONE_DIR.glob(_cell_filename_glob(cell))):
        rows.extend(_load_jsonl(shard))
    rows.sort(key=lambda r: r.get("idx", 0))
    return rows


def _thermal_priority(s: str | None) -> int:
    if not s:
        return 0
    return {"nominal": 0, "fair": 1, "serious": 2, "critical": 3}.get(s, 0)


def _summarize_cell(cell: str) -> dict:
    rows = _load_cell_rows(cell)
    n = len(rows)
    if n == 0:
        return {"cell": cell, "n": 0}

    walls_s = [r["wallclock_ms"] / 1000.0 for r in rows if "wallclock_ms" in r]
    correct = sum(int(r.get("correct", 0)) for r in rows)
    gen_tokens = [r.get("gen_tokens") for r in rows if r.get("gen_tokens") is not None]
    prompt_tokens = [r.get("prompt_tokens") for r in rows if r.get("prompt_tokens") is not None]

    if cell.endswith("Mac"):
        joules = [r.get("joules_mac") for r in rows if r.get("joules_mac") is not None]
        peak_temps = [r.get("peak_temp_c_mac") for r in rows if r.get("peak_temp_c_mac") is not None]
        energy_summary = {
            "mean_joules": _safe_mean(joules),
            "peak_die_temp_c": max(peak_temps) if peak_temps else None,
            "n_with_power": len(joules),
        }
    else:
        battery_deltas = [
            r.get("battery_delta_pct_iphone")
            for r in rows
            if r.get("battery_delta_pct_iphone") is not None
        ]
        # Worst-thermal observed across rows in this cell.
        worst_thermal = "nominal"
        for r in rows:
            s = r.get("peak_thermal_state_iphone")
            if _thermal_priority(s) > _thermal_priority(worst_thermal):
                worst_thermal = s or worst_thermal
        energy_summary = {
            "mean_battery_delta_pct": _safe_mean(battery_deltas),
            "total_battery_delta_pct": sum(battery_deltas) if battery_deltas else None,
            "worst_thermal_state": worst_thermal,
            "n_with_battery": len(battery_deltas),
        }

    return {
        "cell": cell,
        "n": n,
        "accuracy": correct / n if n else None,
        "correct": correct,
        "median_wallclock_s": _percentile(walls_s, 50),
        "p95_wallclock_s": _percentile(walls_s, 95),
        "min_wallclock_s": min(walls_s) if walls_s else None,
        "max_wallclock_s": max(walls_s) if walls_s else None,
        "mean_gen_tokens": _safe_mean(gen_tokens),
        "mean_prompt_tokens": _safe_mean(prompt_tokens),
        **energy_summary,
    }


def _load_sustained(cell: str) -> dict:
    safe = cell.lower().replace("-", "_")
    path = CELLS_PHONE_DIR / f"{safe}_sustained.jsonl"
    rows = _load_jsonl(path)
    if not rows:
        return {"cell": cell, "buckets": [], "summary": None}
    summary = next((r for r in rows if r.get("summary")), None)
    buckets = [r for r in rows if not r.get("summary")]
    # Compute time-to-throttle (Mac: peak_temp_c_mac > 90 if available;
    # iPhone: first transition into "serious" or "critical").
    time_to_throttle: float | None = None
    if cell.endswith("Mac") and summary:
        # Mac sustained doesn't carry per-bucket temperature; we only know
        # the run-wide peak. If peak exceeds 90C, we mark the run as
        # near-throttle but cannot pinpoint the second.
        peak = summary.get("peak_temp_c_mac")
        if peak is not None and peak >= 90:
            time_to_throttle = 0.0
    else:
        for b in buckets:
            if _thermal_priority(b.get("thermal_state_at_bucket")) >= 2:
                time_to_throttle = b.get("elapsed_s")
                break
    # Tokens/sec drop check
    tps = [b["tokens_per_sec"] for b in buckets if "tokens_per_sec" in b]
    drop_pct: float | None = None
    if len(tps) >= 10:
        first_q = sum(tps[: max(1, len(tps) // 4)]) / max(1, len(tps) // 4)
        last_q = sum(tps[-max(1, len(tps) // 4):]) / max(1, len(tps) // 4)
        if first_q > 0:
            drop_pct = (first_q - last_q) / first_q
    return {
        "cell": cell,
        "buckets": buckets,
        "summary": summary,
        "tps_first_quarter": (
            statistics.fmean(tps[: max(1, len(tps) // 4)])
            if tps
            else None
        ),
        "tps_last_quarter": (
            statistics.fmean(tps[-max(1, len(tps) // 4):])
            if tps
            else None
        ),
        "tps_drop_pct": drop_pct,
        "time_to_throttle_s": time_to_throttle,
    }


def _label_outcome(per_cell: dict, sustained: dict) -> tuple[str, str]:
    c2_mac = per_cell.get("C2-Mac", {})
    c2_phone = per_cell.get("C2-iPhone", {})
    a3_mac = per_cell.get("A3-Mac", {})
    a3_phone = per_cell.get("A3-iPhone", {})

    # Outcome F: quantization gate failure (sanity gate 1)
    accs = [
        per_cell.get(c, {}).get("accuracy") for c in ("C2-Mac", "C2-iPhone")
        if per_cell.get(c, {}).get("n", 0) > 0
    ]
    if accs and any(a is not None and a < 0.65 for a in accs):
        return ("F", "MLX 4-bit accuracy below 0.65 - re-quantize at Q5/Q6")

    # Outcome E: Mac/phone divergence > 3x
    mac_med = c2_mac.get("median_wallclock_s")
    phone_med = c2_phone.get("median_wallclock_s")
    if mac_med and phone_med and phone_med / mac_med > OUTCOME_E_DIVERGENCE:
        return (
            "E",
            f"phone median ({phone_med:.2f}s) / mac median ({mac_med:.2f}s) "
            f"= {phone_med/mac_med:.1f}x exceeds 3x silicon-gap budget",
        )

    # Outcome C: phone p95 > 10s
    phone_p95 = c2_phone.get("p95_wallclock_s")
    if phone_p95 and phone_p95 > OUTCOME_C_PHONE_P95_S:
        return (
            "C",
            f"iPhone C2 p95 ({phone_p95:.2f}s) > {OUTCOME_C_PHONE_P95_S}s "
            f"- fails TTFT threshold on 2021 silicon",
        )

    # Outcome B: phone throttles in sustained probe
    phone_sus = sustained.get("C2-iPhone", {})
    drop = phone_sus.get("tps_drop_pct")
    ttt = phone_sus.get("time_to_throttle_s")
    if (drop is not None and drop > SUSTAINED_DROP_THRESHOLD) or \
       (ttt is not None and ttt < 60):
        return (
            "B",
            f"iPhone sustained drop={drop} ttt={ttt}s - phone OK occasional, "
            "not sustained chat",
        )

    # Outcome D: A3 prompt-cost ratio bigger on phone than mac
    a3_phone_med = a3_phone.get("median_wallclock_s")
    a3_mac_med = a3_mac.get("median_wallclock_s")
    if all(x is not None for x in (mac_med, phone_med, a3_mac_med, a3_phone_med)):
        mac_overhead = a3_mac_med / mac_med if mac_med else None
        phone_overhead = a3_phone_med / phone_med if phone_med else None
        if mac_overhead and phone_overhead and phone_overhead > 1.5 * mac_overhead:
            return (
                "D",
                f"A3 overhead phone={phone_overhead:.2f}x mac={mac_overhead:.2f}x "
                f"- prompt-token cost dominates on phone",
            )

    # Outcome A: comfortably on-device
    if (mac_med and mac_med < OUTCOME_A_MAC_MEDIAN_S
            and phone_med and phone_med < OUTCOME_A_PHONE_MEDIAN_S
            and (drop is None or drop < SUSTAINED_DROP_THRESHOLD)):
        return ("A", "C2 deployable on Apple silicon (Mac+iPhone) within thresholds")

    return ("AMBIGUOUS", "no outcome predicate matched cleanly; see per-cell numbers")


def _emit_pareto(per_cell: dict, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[plot] matplotlib not installed; skipping pareto plot")
        return False

    fig, ax = plt.subplots(figsize=(7, 5))
    for cell, color, marker in (
        ("C2-Mac", "tab:blue", "o"),
        ("A3-Mac", "tab:cyan", "s"),
        ("C2-iPhone", "tab:red", "o"),
        ("A3-iPhone", "tab:orange", "s"),
    ):
        c = per_cell.get(cell, {})
        if not c.get("n"):
            continue
        x = c.get("mean_joules") or c.get("mean_battery_delta_pct")
        y = c.get("accuracy")
        if x is None or y is None:
            continue
        xerr = c.get("median_wallclock_s")
        ax.scatter(x, y, color=color, marker=marker, s=120, label=cell, edgecolors="black")
        ax.annotate(
            f"  {cell}\n  med={xerr:.1f}s",
            (x, y), textcoords="offset points", xytext=(8, -8), fontsize=8,
        )

    ax.set_xlabel("mean energy per problem (J on Mac, %battery on iPhone)")
    ax.set_ylabel("accuracy on GSM8K (n=50)")
    ax.set_title("Path 1 plan 8 v2 - on-device Pareto (accuracy vs energy)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return True


def _emit_sustained(sustained: dict, out_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("[plot] matplotlib not installed; skipping sustained plot")
        return False

    has_any = False
    fig, ax = plt.subplots(figsize=(7, 4))
    for cell, color in (("C2-Mac", "tab:blue"), ("C2-iPhone", "tab:red")):
        s = sustained.get(cell, {})
        buckets = s.get("buckets") or []
        if not buckets:
            continue
        xs = [b["elapsed_s"] for b in buckets]
        ys = [b["tokens_per_sec"] for b in buckets]
        ax.plot(xs, ys, color=color, label=cell)
        ttt = s.get("time_to_throttle_s")
        if ttt is not None:
            ax.axvline(ttt, color=color, linestyle="--", alpha=0.5,
                       label=f"{cell} throttle @ {ttt:.0f}s")
        has_any = True

    if not has_any:
        plt.close(fig)
        print("[plot] no sustained data; skipping plot")
        return False

    ax.set_xlabel("elapsed time (s)")
    ax.set_ylabel("tokens / sec")
    ax.set_title("Path 1 plan 8 v2 - sustained-thermal probe")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return True


def _print_table(per_cell: dict) -> None:
    cols = ["cell", "n", "acc", "med_s", "p95_s", "energy", "thermal"]
    fmt = "{:<11s} {:>3s} {:>5s} {:>7s} {:>7s} {:>10s} {:>9s}"
    print(fmt.format(*cols))
    print("-" * 60)
    for c in CELLS:
        row = per_cell.get(c, {})
        if not row.get("n"):
            print(fmt.format(c, "0", "-", "-", "-", "-", "-"))
            continue
        n_str = str(row["n"])
        acc = row.get("accuracy")
        acc_s = f"{acc:.3f}" if acc is not None else "-"
        med = row.get("median_wallclock_s")
        p95 = row.get("p95_wallclock_s")
        if c.endswith("Mac"):
            energy = row.get("mean_joules")
            energy_s = f"{energy:.1f}J" if energy is not None else "-"
            therm = row.get("peak_die_temp_c")
            therm_s = f"{therm:.1f}C" if therm is not None else "-"
        else:
            energy = row.get("mean_battery_delta_pct")
            energy_s = f"{energy:.3f}%" if energy is not None else "-"
            therm_s = str(row.get("worst_thermal_state") or "-")
        print(fmt.format(
            c, n_str, acc_s,
            f"{med:.2f}" if med else "-",
            f"{p95:.2f}" if p95 else "-",
            energy_s, therm_s,
        ))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--json-only", action="store_true",
                   help="Skip PNGs (for environments without matplotlib)")
    args = p.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_cell = {c: _summarize_cell(c) for c in CELLS}
    sustained = {c: _load_sustained(c) for c in CELLS if c.startswith("C2") or c.startswith("A3")}

    parity_path = CELLS_PHONE_DIR / "accuracy_parity.json"
    parity = json.loads(parity_path.read_text()) if parity_path.exists() else None

    label, rationale = _label_outcome(per_cell, sustained)

    out = {
        "plan": "path_1_cot_tokens/plan8_v2_phone",
        "model_id": "google/gemma-4-E2B-it (MLX 4-bit)",
        "outcome": {"label": label, "rationale": rationale},
        "per_cell": per_cell,
        "sustained": {
            c: {
                "summary": s.get("summary"),
                "tps_first_quarter": s.get("tps_first_quarter"),
                "tps_last_quarter": s.get("tps_last_quarter"),
                "tps_drop_pct": s.get("tps_drop_pct"),
                "time_to_throttle_s": s.get("time_to_throttle_s"),
            }
            for c, s in sustained.items() if s.get("buckets") or s.get("summary")
        },
        "accuracy_parity": parity,
    }

    print()
    print("=== Path 1 plan 8 v2 phone results ===")
    _print_table(per_cell)
    print()
    print(f"OUTCOME: {label}")
    print(f"   {rationale}")
    print()

    RESULTS_JSON.write_text(json.dumps(out, indent=2))
    print(f"[json] wrote {RESULTS_JSON}")

    if not args.json_only:
        _emit_pareto(per_cell, PARETO_PNG)
        _emit_sustained(sustained, SUSTAINED_PNG)

    return 0


if __name__ == "__main__":
    sys.exit(main())
