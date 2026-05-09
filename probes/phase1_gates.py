"""Phase 1 gate evaluators.

Each gate is a pure function ``(metrics_dict) -> (passed: bool, message: str)``.
Pulled out of the runner so it's CPU-unit-testable against fixtures.

Bands come from plans/path_2_depth_recurrence_v2/phase1_hook_sanity.md;
keep this file in sync with that document. If a band changes here,
update the plan and re-run the test fixtures.
"""

from typing import Tuple

# --- Cell 1A: IT C2 baseline gate -------------------------------------------
CELL_1A_ACC_BAND = (0.65, 0.82)         # Path 1 anchor 71.6 %
CELL_1A_LOOP_MAX = 0.05
CELL_1A_TRUNC_MAX = 0.05
CELL_1A_PARSE_MIN = 0.95


def gate_1A(s: dict) -> Tuple[bool, str]:
    if s.get("n_problems", 0) == 0:
        return False, "no rows"
    acc = s["accuracy_smart_v2"]
    if not (CELL_1A_ACC_BAND[0] <= acc <= CELL_1A_ACC_BAND[1]):
        return False, (
            f"accuracy_smart_v2={acc:.3f} outside band {CELL_1A_ACC_BAND}; "
            "harness regressed vs Path 1 (71.6 % anchor on N=500). Most "
            "likely: chat template not applied --- inspect first prompt for "
            "<start_of_turn>model suffix."
        )
    if s["loop_rate"] > CELL_1A_LOOP_MAX:
        return False, (
            f"loop_rate={s['loop_rate']:.3f} > {CELL_1A_LOOP_MAX}; "
            "C2 prompt should be loop-free on IT (Path 1 anchor 0 %)."
        )
    if s["truncation_rate"] > CELL_1A_TRUNC_MAX:
        return False, (
            f"truncation_rate={s['truncation_rate']:.3f} > {CELL_1A_TRUNC_MAX}; "
            "EOS firing wrong, or max_new_tokens too small."
        )
    if s["parse_rate"] < CELL_1A_PARSE_MIN:
        return False, (
            f"parse_rate={s['parse_rate']:.3f} < {CELL_1A_PARSE_MIN}; "
            "smart_v2 returned None too often."
        )
    return True, (
        f"acc={acc:.3f} loop={s['loop_rate']:.3f} "
        f"trunc={s['truncation_rate']:.3f} parse={s['parse_rate']:.3f}"
    )


# --- Cell 1B: IT 8-shot bridges -------------------------------------------
#
# Round 5 (legacy = 54.8 %) and Path 1 plan 5 (smart_v2 = 30.0 %) used
# DIFFERENT prompts; one prompt cannot bridge both anchors. Phase 1
# therefore runs two cells, one per anchor:
#
#   * baseline-8shot-control   --- Path 1 plan 5 prompt (single user
#       turn + "#### N" marker) --- gate_1B_path1 checks smart_v2.
#   * baseline-8shot-round5    --- Round 5 prompt (alternating turns +
#       "The answer is N." marker + stop_strings) --- gate_1B_round5
#       checks accuracy_legacy.
#
# Loop_rate band has no lower bound: Path 1 plan 3's 10-12 % loop rate
# was measured with a stricter detector than has_repetition_loop in
# probes.extractors. We keep the upper bound to catch real pathology.

CELL_1B_SMART_BAND = (0.20, 0.40)        # Path 1 plan 5 anchor 30.0 %
CELL_1B_PATH1_LOOP_MAX = 0.25
CELL_1B_PATH1_TRUNC_MAX = 0.20

CELL_1B_LEGACY_BAND = (0.44, 0.64)       # Round 5 anchor 54.8 %
CELL_1B_ROUND5_LOOP_MAX = 0.25
CELL_1B_ROUND5_TRUNC_MAX = 0.20


def gate_1B_path1(s: dict) -> Tuple[bool, str]:
    """Bridges to Path 1 plan 5's smart_v2 anchor (30.0 %, single-turn,
    ``#### N`` exemplars). Legacy is recorded but not gated --- this
    prompt does not reproduce round 5's 54.8 %, by design."""
    if s.get("n_problems", 0) == 0:
        return False, "no rows"
    smart = s["accuracy_smart_v2"]
    if not (CELL_1B_SMART_BAND[0] <= smart <= CELL_1B_SMART_BAND[1]):
        return False, (
            f"accuracy_smart_v2={smart:.3f} outside band {CELL_1B_SMART_BAND}; "
            "Path 1 plan 5 anchor was 30.0 %."
        )
    loop = s["loop_rate"]
    if loop > CELL_1B_PATH1_LOOP_MAX:
        return False, (
            f"loop_rate={loop:.3f} > {CELL_1B_PATH1_LOOP_MAX}; "
            "harness regressed --- 8-shot CoT should not pathologically loop "
            "more than a quarter of the time on the IT model."
        )
    if s["truncation_rate"] > CELL_1B_PATH1_TRUNC_MAX:
        return False, (
            f"truncation_rate={s['truncation_rate']:.3f} > {CELL_1B_PATH1_TRUNC_MAX}"
        )
    return True, (
        f"acc_smart={smart:.3f} acc_legacy={s['accuracy_legacy']:.3f} "
        f"loop={loop:.3f} trunc={s['truncation_rate']:.3f}"
    )


def gate_1B_round5(s: dict) -> Tuple[bool, str]:
    """Bridges to Path 2 round 5's legacy anchor (54.8 %, multi-turn,
    'The answer is N.' marker, stop_strings). Smart_v2 is recorded but
    not gated --- it will likely run higher than this band, since the
    round 5 prompt produces well-formed answers both extractors hit."""
    if s.get("n_problems", 0) == 0:
        return False, "no rows"
    legacy = s["accuracy_legacy"]
    if not (CELL_1B_LEGACY_BAND[0] <= legacy <= CELL_1B_LEGACY_BAND[1]):
        return False, (
            f"accuracy_legacy={legacy:.3f} outside band {CELL_1B_LEGACY_BAND}; "
            "cross-round bridge to round 5 (54.8 %) broken --- diff Wei "
            "exemplars + chat-template wrapping against round 5 manifest "
            "(probes.mode_round5._format_gsm8k_prompt_chat)."
        )
    loop = s["loop_rate"]
    if loop > CELL_1B_ROUND5_LOOP_MAX:
        return False, (
            f"loop_rate={loop:.3f} > {CELL_1B_ROUND5_LOOP_MAX}; "
            "harness regressed."
        )
    if s["truncation_rate"] > CELL_1B_ROUND5_TRUNC_MAX:
        return False, (
            f"truncation_rate={s['truncation_rate']:.3f} > "
            f"{CELL_1B_ROUND5_TRUNC_MAX}; stop_strings may not be wired."
        )
    return True, (
        f"acc_legacy={legacy:.3f} acc_smart={s['accuracy_smart_v2']:.3f} "
        f"loop={loop:.3f} trunc={s['truncation_rate']:.3f}"
    )


# Compatibility alias so external callers / older tests can still import
# gate_1B; defaults to the path1 (smart_v2) variant since that's the
# anchor the original gate prioritised after the legacy check.
gate_1B = gate_1B_path1


# --- Cell 1C: token-match gate ----------------------------------------------
def gate_1C(info: dict) -> Tuple[bool, str]:
    """``info`` is the dict returned by token_match.compare.

    Expected shape:
      {"shared": int, "matches": int}  -- success
      {"shared": int, "mismatches": [int, ...]}  -- failure
      {"reason": "no shared idxs", ...}  -- error
    """
    if "reason" in info:
        return False, f"token_match aborted: {info['reason']}"
    matches = info.get("matches", 0)
    shared = info.get("shared", 0)
    if matches == shared and shared > 0:
        return True, f"matches={matches}/{shared}"
    mismatches = info.get("mismatches", [])
    return False, (
        f"{len(mismatches)}/{shared} idxs differ; first 3: "
        f"{mismatches[:3]} --- hook is mutating generation state at r=1. "
        "Halt before Phase 2."
    )


# --- Cell 1D: W5-r8 process smoke gate --------------------------------------
CELL_1D_LOOP_MAX = 0.95


def gate_1D(s: dict, *, expected_n: int) -> Tuple[bool, str]:
    """``s`` is the standard summary; we additionally require all
    ``expected_n`` rows to be present (the runner deletes the cell on
    a fresh run, so this catches "process crashed midway").
    """
    if s.get("n_problems", 0) != expected_n:
        return False, (
            f"got {s.get('n_problems', 0)}/{expected_n} rows; "
            "process crashed during cell execution."
        )
    loop = s["loop_rate"]
    if loop > CELL_1D_LOOP_MAX:
        return False, (
            f"loop_rate={loop:.3f} > {CELL_1D_LOOP_MAX}; recurrence at r=8 "
            "produces total generation collapse on this block. Round 5 saw "
            "97-100 % (with 8-shot prompt confound); seeing this on C2 too "
            "means it's structural. Halt; this changes Phase 4's r-sweep."
        )
    if s["mean_gen_tokens"] == 0:
        return False, (
            "mean_gen_tokens=0; every generation immediately emitted EOS "
            "--- hook collapsed activations to a degenerate fixed point."
        )
    return True, (
        f"n={s['n_problems']} loop={loop:.3f} "
        f"mean_tok={s['mean_gen_tokens']:.0f}"
    )


# --- Cell 1E: base perplexity drift gate ------------------------------------
CELL_1E_DRIFT_MAX = 1e-4


def gate_1E(unmodified_ppl: float, r1_ppl: float) -> Tuple[bool, str]:
    if unmodified_ppl is None or r1_ppl is None:
        return False, "missing baseline or r=1 ppl in round-1 JSON"
    if unmodified_ppl <= 0:
        return False, f"unmodified_ppl={unmodified_ppl} non-positive"
    drift = abs(r1_ppl - unmodified_ppl) / unmodified_ppl
    if drift >= CELL_1E_DRIFT_MAX:
        return False, (
            f"rel_drift={drift:.2e} >= {CELL_1E_DRIFT_MAX:.0e}; "
            "block-loop hook regressed at the perplexity level. Diff "
            "install_block_loop_hooks against round 1 results.json."
        )
    return True, (
        f"unmod_ppl={unmodified_ppl:.4f} r1_ppl={r1_ppl:.4f} "
        f"drift={drift:.2e}"
    )


# --- Aggregate over all cells -----------------------------------------------
def all_passed(report: dict) -> bool:
    """``report`` is the full phase 1 summary; return True iff every cell
    has ``gate_passed == True``."""
    cells = report.get("cells", {})
    if not cells:
        return False
    return all(c.get("gate_passed") is True for c in cells.values())
