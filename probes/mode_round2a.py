"""Round 2a --- PLE variants probe and zero-PLE diagnostic.

Runs {vanilla, scaled, once} x {r=1, 4, 8} at one target layer to measure
how different PLE re-injection policies affect looped perplexity. The
``run_zero_diagnostic`` helper is a cut-down variant (plan2a-add1) that
distinguishes "three-way tie is real" from "patch broken" by comparing
vanilla r=1 to zero r=1 at the same layer.
"""

import math
import sys

from .data import compute_perplexity
from .env import ROUND1_UNMODIFIED_PPL, _results_path, _write_results_json
from .hooks import make_looped_forward_ple
from .introspect import inspect_ple_strategy, print_ple_inspection


def run_zero_diagnostic(args, model, decoder_layers, inputs):
    """plan2a-add1: distinguish 'three-way tie is real' from 'patch broken'.

    Round 2a produced bitwise-identical perplexities for vanilla / scaled / once
    at every r. That is ambiguous: either PLE at TARGET_LAYER genuinely doesn't
    move the needle under our loop, or the patch never intercepted PLE at all.

    This function runs the minimal cell that distinguishes the two: vanilla r=1
    and zero r=1 at the same TARGET_LAYER, in the same execution (so evaluation
    state is identical). If they differ, the patch is working.
    """
    output_json = args.output_json or _results_path("results_round2a_addendum.json")

    target_layer = decoder_layers[args.target_layer]
    inspection = inspect_ple_strategy(target_layer)
    print_ple_inspection(inspection)

    if inspection["strategy"] != "A":
        print(
            f"ERROR: PLE wiring is Strategy {inspection['strategy']}, not A. "
            "Cannot run zero-PLE diagnostic via kwarg interception."
        )
        sys.exit(2)

    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    orig_forward = target_layer.forward
    location_recorder = {}
    try:
        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_v1, ppl_v1 = compute_perplexity(model, inputs)

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="zero", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_z1, ppl_z1 = compute_perplexity(model, inputs)
    finally:
        target_layer.forward = orig_forward

    nll_diff = abs(nll_z1 - nll_v1)
    status = "WORKING" if nll_diff > 1e-6 else "BROKEN"
    ple_location = location_recorder.get("ple_location", "unknown")

    print("=== Zero-PLE diagnostic ===")
    print(f"vanilla r=1:  ppl = {ppl_v1}")
    print(f"zero    r=1:  ppl = {ppl_z1}")
    print()
    print(f"Difference: {nll_diff:.6f}  (absolute NLL difference)")
    print(f"Patch status: {status}")
    print(f"PLE location: {ple_location}")

    output = {
        "config": {
            "mode": "ple-variants",
            "diagnostic": "zero-ple",
            "model_id": args.model_id,
            "target_layer": args.target_layer,
            "r": 1,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "ple_kwarg": ple_kwarg,
        },
        "inspection": {
            "strategy": inspection["strategy"],
            "layer_class": inspection["layer_class"],
            "source_file": inspection["source_file"],
            "start_lineno": inspection["start_lineno"],
            "signature": inspection["signature"],
            "ple_kwarg": inspection["ple_kwarg"],
        },
        "cells": [
            {"ple_mode": "vanilla", "r": 1, "mean_nll": nll_v1, "ppl": ppl_v1},
            {"ple_mode": "zero", "r": 1, "mean_nll": nll_z1, "ppl": ppl_z1},
        ],
        "nll_difference": nll_diff,
        "patch_status": status,
        "ple_location": ple_location,
    }
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")


def run_ple_variants_mode(args, model, decoder_layers, inputs):
    """Round-2a: vanilla / scaled / once x r in {1,4,8} at TARGET_LAYER."""
    r_values = args.r_values if args.r_values is not None else [1, 4, 8]
    output_json = args.output_json or _results_path("results_round2a.json")

    target_layer = decoder_layers[args.target_layer]
    inspection = inspect_ple_strategy(target_layer)
    print_ple_inspection(inspection)

    if inspection["strategy"] != "A":
        msg = (
            f"\nPLE wiring is Strategy {inspection['strategy']}, not the expected "
            f"Strategy A (PLE passed as kwarg into the decoder layer's forward).\n"
        )
        if inspection["strategy"] == "B":
            msg += (
                "Strategy B means PLE is applied in the outer model forward, "
                "OUTSIDE the per-layer forward. Round 1's hook fundamentally "
                "could not control PLE: round-1 'vanilla' was actually 'once' "
                "in disguise. Stop and report this finding (see plan2a).\n"
            )
        else:  # "C"
            msg += (
                "Strategy C means PLE is referenced inside the layer in a way "
                "that is not a simple kwarg (e.g. computed from input_ids, "
                "fused into a projection, ...). Source dump above. Stop and "
                "report (see plan2a).\n"
            )
        print("--- Decoder layer forward source ---")
        print(inspection["source"])
        print("--- end source ---")
        print(msg)
        out_payload = {
            "mode": "ple-variants",
            "aborted": True,
            "inspection": {k: v for k, v in inspection.items() if k != "source"},
            "inspection_source": inspection["source"],
            "reason": msg.strip(),
        }
        _write_results_json(output_json, out_payload)
        print(f"Wrote diagnostic-only {output_json}")
        sys.exit(2)

    ple_kwarg = inspection["ple_kwarg"]
    print(f"Strategy A confirmed. PLE kwarg = {ple_kwarg!r}.\n")

    print("Running unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.4f}")

    orig_forward = target_layer.forward
    cells = []
    cell_index = {}  # (mode, r) -> ppl, for table rendering
    location_recorder = {}  # plan2a-add2: capture detected PLE arrival path

    try:
        # ---- Step 4: regression checks BEFORE collecting experiment data ----
        print("\n=== Regression checks (plan2a Step 4 + add2) ===")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="vanilla", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_v1, ppl_v1 = compute_perplexity(model, inputs)
        print(f"  vanilla r=1: ppl = {ppl_v1:.6f}")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="scaled", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_s1, ppl_s1 = compute_perplexity(model, inputs)
        print(f"  scaled  r=1: ppl = {ppl_s1:.6f}")

        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="once", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_o1, ppl_o1 = compute_perplexity(model, inputs)
        print(f"  once    r=1: ppl = {ppl_o1:.6f}")

        # plan2a-add2 check 4: inline zero-PLE diagnostic. The original bug
        # made every mode behave like vanilla, so zero r=1 was bitwise-equal
        # to vanilla r=1. After the fix, zero must produce a different ppl.
        target_layer.forward = make_looped_forward_ple(
            orig_forward, r=1, ple_mode="zero", ple_kwarg=ple_kwarg,
            location_recorder=location_recorder,
        )
        nll_z1, ppl_z1 = compute_perplexity(model, inputs)
        print(f"  zero    r=1: ppl = {ppl_z1:.6f}")

        # Tolerances: vanilla-vs-round1 we allow numerical drift across HW;
        # scaled/once vs vanilla at r=1 must be bitwise equal (scale=1.0).
        check1 = math.isclose(ppl_v1, ROUND1_UNMODIFIED_PPL, rel_tol=1e-3)
        check2 = ppl_s1 == ppl_v1
        check3 = ppl_o1 == ppl_v1
        check4 = abs(nll_z1 - nll_v1) > 1e-6  # zero must differ from vanilla
        print(f"  check 1 (vanilla r=1 ~= round1 unmodified={ROUND1_UNMODIFIED_PPL:.4f}): {check1}")
        print(f"  check 2 (scaled r=1 == vanilla r=1):  {check2}")
        print(f"  check 3 (once   r=1 == vanilla r=1):  {check3}")
        print(f"  check 4 (zero   r=1 != vanilla r=1):  {check4}   "
              f"[NLL diff = {abs(nll_z1 - nll_v1):.6f}]")
        if not (check1 and check2 and check3 and check4):
            target_layer.forward = orig_forward
            print("\nERROR: regression checks failed --- aborting before main run.")
            if not check4:
                print("  check 4 failure means the PLE patch is BROKEN: zero-mode "
                      "produced bitwise-identical perplexity to vanilla. The hook "
                      "is not actually intercepting per_layer_input.")
            sys.exit(3)

        # Reuse r=1 measurements (identical across modes by construction).
        for ple_mode, (nll1, ppl1) in [
            ("vanilla", (nll_v1, ppl_v1)),
            ("scaled", (nll_s1, ppl_s1)),
            ("once", (nll_o1, ppl_o1)),
        ]:
            cell_index[(ple_mode, 1)] = ppl1
            cells.append({"ple_mode": ple_mode, "r": 1, "mean_nll": nll1, "ppl": ppl1})

        # ---- Step 5: 9-cell experiment (excluding r=1 already done) ----
        print("\n=== Main sweep (plan2a Step 5) ===")
        for ple_mode in ("vanilla", "scaled", "once"):
            for r in r_values:
                if r == 1:
                    continue
                target_layer.forward = make_looped_forward_ple(
                    orig_forward, r=r, ple_mode=ple_mode, ple_kwarg=ple_kwarg,
                    location_recorder=location_recorder,
                )
                mean_nll, ppl = compute_perplexity(model, inputs)
                cell_index[(ple_mode, r)] = ppl
                cells.append({"ple_mode": ple_mode, "r": r, "mean_nll": mean_nll, "ppl": ppl})
                print(f"  {ple_mode:7s} r={r:2d}: mean NLL = {mean_nll:.4f}   perplexity = {ppl:.4f}")
    finally:
        target_layer.forward = orig_forward

    ple_location = location_recorder.get("ple_location", "unknown")
    print(f"\nPLE location detected: {ple_location}")

    # ---- Step 6: printed table ----
    print("\n=== PLE variants (layer {}) ===".format(args.target_layer))
    header = "              " + "    ".join(f"r={r:<5d}" for r in r_values)
    print(header)
    for ple_mode in ("vanilla", "scaled", "once"):
        row_vals = "  ".join(f"{cell_index[(ple_mode, r)]:8.2f}" for r in r_values)
        print(f"  {ple_mode:8s}  {row_vals}")

    output = {
        "config": {
            "mode": "ple-variants",
            "model_id": args.model_id,
            "target_layer": args.target_layer,
            "r_values": r_values,
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
        "regression_checks": {
            "vanilla_r1_matches_round1": check1,
            "scaled_r1_equals_vanilla_r1": check2,
            "once_r1_equals_vanilla_r1": check3,
            "zero_r1_differs_from_vanilla_r1": check4,
            "vanilla_r1_ppl": ppl_v1,
            "zero_r1_ppl": ppl_z1,
            "zero_vs_vanilla_nll_diff": abs(nll_z1 - nll_v1),
            "round1_unmodified_ppl": ROUND1_UNMODIFIED_PPL,
        },
        "addendum2": {
            "ple_location": ple_location,
            "zero_diagnostic_passed": check4,
            "fix_applied": "handle_positional_ple_args",
        },
        "cells": cells,
    }
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")
