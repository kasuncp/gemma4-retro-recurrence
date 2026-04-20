"""Round 1 --- original single-layer naive looping sanity check."""

from .data import compute_perplexity
from .env import _results_path, _write_results_json
from .hooks import make_looped_forward


def run_original_mode(args, model, decoder_layers, inputs):
    """Round-1 sanity check, preserving exact prior behaviour."""
    r_values = args.r_values if args.r_values is not None else [1, 2, 4, 8]
    output_json = args.output_json or _results_path("results.json")

    print("\nRunning unmodified baseline (no hook) ...")
    unmod_nll, unmod_ppl = compute_perplexity(model, inputs)
    print(f"unmodified:  mean NLL = {unmod_nll:.4f}   perplexity = {unmod_ppl:.2f}")

    orig_forward = decoder_layers[args.target_layer].forward
    results = {}
    try:
        for r in r_values:
            decoder_layers[args.target_layer].forward = make_looped_forward(orig_forward, r)
            mean_nll, ppl = compute_perplexity(model, inputs)
            results[r] = {"mean_nll": mean_nll, "ppl": ppl}
            print(f"r={r:2d}:  mean NLL = {mean_nll:.4f}   perplexity = {ppl:.2f}")
    finally:
        decoder_layers[args.target_layer].forward = orig_forward

    print("\n=== Summary ===")
    baseline_ppl = results[r_values[0]]["ppl"]
    summary = []
    for r, res in results.items():
        ratio = res["ppl"] / baseline_ppl
        line = f"r={r:2d}:  ppl={res['ppl']:7.2f}   (x{ratio:.2f} vs r={r_values[0]})"
        print(line)
        summary.append({"r": r, "ppl": res["ppl"], "ratio": ratio})

    drift = abs(results[1]["ppl"] - unmod_ppl) / unmod_ppl if 1 in results else None
    if drift is not None:
        print(
            f"\nHook sanity: r=1 ppl={results[1]['ppl']:.4f} vs unmodified "
            f"ppl={unmod_ppl:.4f} (relative drift={drift:.2e})"
        )
        if drift > 1e-3:
            print("WARNING: r=1 hooked perplexity differs from unmodified - hook may be buggy.")

    output = {
        "config": {
            "mode": "original",
            "model_id": args.model_id,
            "target_layer": args.target_layer,
            "r_values": r_values,
            "num_sequences": args.num_sequences,
            "max_length": args.max_length,
            "dtype": args.dtype,
        },
        "unmodified": {"mean_nll": unmod_nll, "ppl": unmod_ppl},
        "results": {str(r): v for r, v in results.items()},
        "summary": summary,
        "hook_drift": drift,
    }
    _write_results_json(output_json, output)
    print(f"\nWrote {output_json}")
