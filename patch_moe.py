"""Patch transformers/integrations/moe.py to wrap the custom_op registration
in a try/except so it silently no-ops on torch<2.5.

torch<2.5 rejects bare ``torch.Tensor`` type annotations in
``torch.library.custom_op``. The grouped_mm custom kernel is only used for
MoE expert dispatch acceleration; Gemma4 inference works correctly without it.

Usage:
    python patch_moe.py          # auto-detects transformers install location
    python patch_moe.py /path/to/moe.py   # explicit path
"""
import os
import sys


OLD = (
    "if is_torch_available():\n"
    "    torch.library.custom_op(\"transformers::grouped_mm_fallback\", _grouped_mm_fallback, mutates_args=())\n"
    "    torch.library.register_fake(\"transformers::grouped_mm_fallback\", _grouped_mm_fallback_fake)\n"
    "    torch.library.register_autograd(\n"
    "        \"transformers::grouped_mm_fallback\",\n"
    "        _grouped_mm_fallback_backward,\n"
    "        setup_context=_grouped_mm_fallback_setup_context,\n"
    "    )\n"
)

NEW = (
    "if is_torch_available():\n"
    "    try:\n"
    "        torch.library.custom_op(\"transformers::grouped_mm_fallback\", _grouped_mm_fallback, mutates_args=())\n"
    "        torch.library.register_fake(\"transformers::grouped_mm_fallback\", _grouped_mm_fallback_fake)\n"
    "        torch.library.register_autograd(\n"
    "            \"transformers::grouped_mm_fallback\",\n"
    "            _grouped_mm_fallback_backward,\n"
    "            setup_context=_grouped_mm_fallback_setup_context,\n"
    "        )\n"
    "    except Exception:\n"
    "        pass  # torch<2.5: bare torch.Tensor annotations unsupported in custom_op\n"
)


def find_moe_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    try:
        import transformers
        return os.path.join(os.path.dirname(transformers.__file__),
                            "integrations", "moe.py")
    except ImportError:
        return None


def main():
    path = find_moe_path()
    if not path or not os.path.isfile(path):
        print(f"moe.py not found (path={path!r}) — skipping patch")
        return 0

    with open(path) as f:
        src = f.read()

    if OLD not in src:
        print(f"moe.py already patched or layout changed — skipping")
        return 0

    with open(path, "w") as f:
        f.write(src.replace(OLD, NEW, 1))
    print(f"patched {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
