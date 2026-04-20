"""PLE x Loop Recurrence probes for Gemma 4 E2B --- shared infrastructure.

Submodule layout:
    env           --- model / tokenizer loading, path helpers, global constants.
    data          --- Wikitext input prep and perplexity computation.
    introspect    --- decoder-layer discovery, PLE wiring inspection, layer metadata.
    hooks         --- loop hooks (single-layer, PLE-aware, pair-loop).
    stats         --- correlation and rank helpers used by analysis passes.
    mode_round1   --- original single-layer looping sanity check.
    mode_round2a  --- PLE-variants probe + zero-PLE diagnostic.
    mode_round2b  --- per-layer PLE importance scan and layer-location sweep.
    mode_round2c  --- full 35-layer vanilla/once looping-tolerance map.
    mode_round3a  --- pair-of-layers looping probe.

The HF cache redirect below runs at package-import time --- BEFORE any
submodule imports torch/transformers --- so that Gemma's ~5 GB download
survives RunPod pod restarts via persistent /workspace storage. Submodules
may rely on HF_HOME being set before they load.
"""

import os
from pathlib import Path

_PERSISTENT = Path("/workspace")
if _PERSISTENT.is_dir() and "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(_PERSISTENT / ".cache" / "huggingface")
