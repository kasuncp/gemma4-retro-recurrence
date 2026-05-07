"""Environment, model loading, path helpers, and global constants.

Importing this module triggers ``probes/__init__.py``, which ensures the
HF_HOME redirect has run before torch/transformers load here.
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B"
EXPECTED_NUM_LAYERS = 35
DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

# Round 1's anchor for regression checks that compare against the
# originally-published unmodified perplexity.
ROUND1_UNMODIFIED_PPL = 12.53165455614523

# All result JSONs --- both what this script WRITES and the default paths it
# READS for cross-round inputs --- live under results/. The directory is
# created on demand at write time. Users with legacy files at the project
# root can still point at them via --output-json / --importance-json /
# --round2c-json flags.
RESULTS_DIR = Path("results")


def _results_path(filename):
    """Return a path under the results directory for a given filename."""
    return str(RESULTS_DIR / filename)


def _write_results_json(path, payload):
    """Write ``payload`` to ``path`` as JSON, creating parent dirs as needed."""
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))


def print_env():
    print("=" * 60)
    print(f"torch:        {torch.__version__}")
    import transformers
    print(f"transformers: {transformers.__version__}")
    print(f"CUDA avail:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"GPU:          {props.name} ({props.total_memory / 1e9:.1f} GB)")
        print(f"capability:   sm_{props.major}{props.minor}")
    print(f"HF_HOME:      {os.environ.get('HF_HOME', '(default)')}")
    has_token = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    print(f"HF token set: {has_token}")
    print("=" * 60)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.")
        sys.exit(1)
    if not has_token:
        print(
            "WARNING: no HF_TOKEN / HUGGING_FACE_HUB_TOKEN set. Gemma is a "
            "gated model --- load will fail unless you've cached the weights "
            "or logged in via `huggingface-cli login`."
        )


def load_model(model_id, dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map="cuda",
        )
    except (ValueError, KeyError) as e:
        print(f"AutoModelForCausalLM failed ({e}); trying multimodal loader.")
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=dtype, device_map="cuda",
        )
    model.train(False)  # inference mode
    return tokenizer, model
