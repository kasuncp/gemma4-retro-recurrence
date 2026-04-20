#!/usr/bin/env bash
# One-shot runner for the PLE x recurrence probes on RunPod.
#
# Usage:
#   ./run.sh                          # round 2a (default)
#   ./run.sh --mode original          # round 1 sanity check
#   ./run.sh --mode ple-variants      # round 2a (explicit)
#   ./run.sh --target-layer 22 ...    # any flags pass through to the python script
#
# Steps:
#   1. Source .env (creates a stub if missing).
#   2. Validate HF_TOKEN.
#   3. Install/upgrade transformers, datasets, accelerate.
#   4. Run ple_sanity_check.py with whatever args you passed (default: --mode ple-variants).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- 1. Load .env ----------
if [[ ! -f .env ]]; then
    if [[ -f .env.example ]]; then
        cp .env.example .env
        echo "Created .env from .env.example. Edit it to set HF_TOKEN, then re-run."
        exit 1
    else
        cat > .env <<'EOF'
# Hugging Face access token for gated Gemma weights.
# Get one at https://huggingface.co/settings/tokens after accepting the
# Gemma license at https://huggingface.co/google/gemma-4-E2B
HF_TOKEN=
EOF
        echo "Created .env stub. Edit it to set HF_TOKEN, then re-run."
        exit 1
    fi
fi

# Export every non-comment, non-blank assignment from .env into this shell.
set -a
# shellcheck disable=SC1091
source .env
set +a

# ---------- 2. Validate token ----------
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set in .env. Gemma is a gated model."
    echo "Get a token at https://huggingface.co/settings/tokens and add it to .env."
    exit 1
fi
# transformers / huggingface_hub read either name; keep both populated.
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

# ---------- 3. Install deps ----------
PYTHON="${PYTHON:-python3}"
echo "Using $($PYTHON --version) at $(command -v $PYTHON)"

# Skip pip install on re-runs once the marker exists, unless FORCE_INSTALL=1.
INSTALL_MARKER=".deps-installed"
if [[ ! -f "$INSTALL_MARKER" || "${FORCE_INSTALL:-0}" == "1" ]]; then
    echo "Installing/upgrading transformers, datasets, accelerate ..."
    "$PYTHON" -m pip install -U pip
    "$PYTHON" -m pip install -U transformers datasets accelerate
    touch "$INSTALL_MARKER"
else
    echo "Deps already installed (delete $INSTALL_MARKER or set FORCE_INSTALL=1 to reinstall)."
fi

# ---------- 4. Run probe ----------
# Default to round 2a; if the user passed any args, use those verbatim instead.
if [[ $# -eq 0 ]]; then
    set -- --mode ple-variants
fi

echo
echo "=== Running: $PYTHON ple_sanity_check.py $* ==="
exec "$PYTHON" ple_sanity_check.py "$@"
