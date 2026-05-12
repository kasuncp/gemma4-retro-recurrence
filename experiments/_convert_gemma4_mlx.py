"""Convert google/gemma-4-E2B-it to MLX 4-bit.

History (kept here so the next person doesn't repeat the same mistake):

    Gemma 4 has num_hidden_layers=35 with num_kv_shared_layers=20. mlx-lm's
    gemma4_text only allocates K/V projection storage for layers 0..14;
    layers 15..34 share K/V from layer 13 (sliding) or layer 14 (full).
    The HF "google/gemma-4-E2B-it" checkpoint, however, *stores* k_proj /
    v_proj tensors for layers 15..34 too - and those tensors are NOT
    byte-equal to the alias source (cos ~0, rel_l2 ~1.5).

    An earlier version of this script (matching the project name
    "gemma-4-retrofitted-recurrence") hypothesised that the HF checkpoint
    was a retrofit replacing K/V sharing with per-layer K/V projections,
    and overrode num_kv_shared_layers=0 to force MLX to allocate per-layer
    K/V everywhere. That hypothesis was empirically falsified during plan
    sanity gate 1: with num_kv_shared_layers=0 the model emits multi-script
    gibberish even on "Hello"; with num_kv_shared_layers=20 (HF default) it
    produces coherent, correct output ("What is 7 plus 5?" -> "12"). The
    upper-block k_proj / v_proj tensors HF stores are inert at inference -
    presumably an artifact of the export, not a retrofitted projection.

What this script actually does now: standard mlx-lm convert, with
strict=False at HF-load time so the inert upper-block k_proj / v_proj
tensors are silently dropped instead of tripping the strict check. The
saved MLX model has the layout mlx-lm's gemma4_text natively expects
(num_kv_shared_layers=20, double-wide MLP on layers 15..34), and loads
with plain `mlx_lm.load()` - no runtime patches.

Verify after conversion:
    .venv/bin/python experiments/path1_phone_mac.py --accuracy-only \\
        --cells C2-Mac --n 50
Plan sanity gate 1: accuracy >= 65%.
"""

from __future__ import annotations

import sys

import mlx_lm.convert as _convert
import mlx_lm.utils as _utils

_orig_load_model = _utils.load_model


def _patched_load_model(model_path, lazy=False, strict=True, model_config=None,
                        get_model_classes=None):
    # strict=False drops HF's inert upper-block k_proj / v_proj tensors that
    # mlx-lm's gemma4_text doesn't allocate (layers 15..34 share K/V).
    kwargs = dict(model_path=model_path, lazy=lazy, strict=False,
                  model_config=model_config)
    if get_model_classes is not None:
        kwargs["get_model_classes"] = get_model_classes
    return _orig_load_model(**kwargs)


_utils.load_model = _patched_load_model


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hf-path", default="google/gemma-4-E2B-it")
    parser.add_argument("--mlx-path", default="./mlx_models/gemma-4-E2B-it-mlx-q4")
    parser.add_argument("-q", "--quantize", action="store_true", default=True)
    parser.add_argument("--q-bits", type=int, default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    args = parser.parse_args()

    print(f"[convert] patched: strict=False (drops inert HF upper-block k_proj/v_proj)")
    print(f"[convert] hf={args.hf_path} -> mlx={args.mlx_path} q-bits={args.q_bits}")
    _convert(
        hf_path=args.hf_path,
        mlx_path=args.mlx_path,
        quantize=args.quantize,
        q_bits=args.q_bits,
        q_group_size=args.q_group_size,
    )
    print(f"[convert] done -> {args.mlx_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
