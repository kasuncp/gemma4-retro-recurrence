"""Phase 2 layer-map bucket classifier.

Pure-Python evaluator. Each layer cell summary is mapped to one of
three buckets that drive Phase 3's launch decision:

  preserved  --- accuracy within ~10 pp of the unmodified baseline AND
                 no pathology (loop_rate < 0.50, truncation_rate < 0.30).
                 Phase 3 anchors block sweeps at these layers.
  degraded   --- accuracy noticeably lower than baseline but still
                 reasoning (>= 40 %) AND no pathology. Used as fallback
                 anchors when there are not enough preserved layers.
  broken     --- accuracy collapsed (< 40 %) OR pathology fires.

Bands come from plans/path_2_depth_recurrence_v2/phase2_layer_map.md;
keep this module in sync with that document.

Anchor reference: Phase 1's 1A measured accuracy_smart_v2 = 0.755 on
N=200 of GSM8K under the C2 prompt at r=1 on the IT model. The
preserved-layer threshold (0.65) is ~10 pp below this, which is the
threshold above which Phase 4's r-sweep can hope to recover the
remaining gap with retrofit training. N=50 has a Wilson 95 % CI of
~ ±13 pp at p ~ 0.7, so the bucket boundaries deliberately do not
promise stable ranking inside the preserved bucket --- Phase 3 at
N=200 does the fine ranking.

Usage:

    from probes.phase2_gates import classify_layer
    bucket, reason = classify_layer(summary_dict)
"""

from typing import Tuple


# Bucket bands. See plan rationale and CI math in module docstring.
LAYER_PRESERVED_ACC_MIN = 0.65            # ~10 pp below 1A anchor (0.755)
LAYER_DEGRADED_ACC_MIN = 0.40             # > 40 % means "model still
                                          #   reasons, just worse"
LAYER_BROKEN_LOOP_MAX = 0.50              # >= 50 % loop_rate is collapse
LAYER_BROKEN_TRUNC_MAX = 0.30             # uses Phase-1 truncation
                                          #   metric (cap-hit AND no
                                          #   extractable answer)


# Phase 3 launch decision thresholds. Counts of layers in each bucket
# determine which mode Phase 3 launches in.
PHASE3_NORMAL_MIN_PRESERVED = 3           # >= 3 preserved -> normal
PHASE3_NARROW_MIN_PRESERVED = 1           # 1..2 preserved  -> narrow
PHASE3_RELAXED_MIN_DEGRADED = 3           # 0 preserved + >= 3 degraded


def classify_layer(s: dict) -> Tuple[str, str]:
    """Return (bucket, human-readable reason) for one layer's summary.

    ``s`` is the GSM8K summary dict produced by
    ``probes.eval_v3.summarise_gsm8k``. The fields we read are:
    ``n_problems``, ``accuracy_smart_v2``, ``loop_rate``,
    ``truncation_rate``. Missing rows count as ``broken``.
    """
    if s.get("n_problems", 0) == 0:
        return "broken", "no rows"
    acc = s["accuracy_smart_v2"]
    loop = s["loop_rate"]
    trunc = s["truncation_rate"]
    if loop >= LAYER_BROKEN_LOOP_MAX or trunc >= LAYER_BROKEN_TRUNC_MAX:
        return "broken", f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}"
    if acc >= LAYER_PRESERVED_ACC_MIN:
        return "preserved", f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}"
    if acc >= LAYER_DEGRADED_ACC_MIN:
        return "degraded", f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}"
    return "broken", f"acc={acc:.2f} loop={loop:.2f} trunc={trunc:.2f}"


def decide_phase3_launch(buckets: dict) -> dict:
    """Map bucket counts -> (mode, anchors, rationale).

    ``buckets`` is ``{"preserved": [layer_label,...], "degraded": [...],
    "broken": [...]}`` where layer_label is e.g. ``"L15"``. Mode is one
    of ``normal | narrow | relaxed | halt``.
    """
    preserved = list(buckets.get("preserved", []))
    degraded = list(buckets.get("degraded", []))

    if len(preserved) >= PHASE3_NORMAL_MIN_PRESERVED:
        return {
            "mode": "normal",
            "anchors": preserved,
            "rationale": (
                f">= {PHASE3_NORMAL_MIN_PRESERVED} preserved layers; "
                "Phase 3 block sweep can anchor at preserved layers."
            ),
        }
    if len(preserved) >= PHASE3_NARROW_MIN_PRESERVED:
        return {
            "mode": "narrow",
            "anchors": preserved,
            "rationale": (
                f"only {len(preserved)} preserved layer(s); Phase 3 "
                "narrows the sweep to preserved-anchor +/- 2 neighbours."
            ),
        }
    if len(degraded) >= PHASE3_RELAXED_MIN_DEGRADED:
        return {
            "mode": "relaxed",
            "anchors": degraded,
            "rationale": (
                f"0 preserved AND >= {PHASE3_RELAXED_MIN_DEGRADED} "
                "degraded; Phase 3 lowers bucket gate and uses degraded "
                "layers as anchors. Reasoning recurrence is a partial-"
                "preservation regime; document before launch."
            ),
        }
    return {
        "mode": "halt",
        "anchors": [],
        "rationale": (
            "0 preserved AND 0 degraded; recurrence on the IT model at "
            "r=8 is fundamentally fragile per-layer. Skip Phase 3/4 and "
            "route directly to Phase 6 (retrofit training)."
        ),
    }
