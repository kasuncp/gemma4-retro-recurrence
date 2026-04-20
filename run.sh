#!/usr/bin/env bash
# One-shot runner for the PLE x recurrence probes on RunPod.
#
# Usage:
#   ./run.sh                          # round 2a, wrapped in tmux (default)
#   ./run.sh --mode original          # round 1 sanity check
#   ./run.sh --mode ple-variants      # round 2a (explicit)
#   ./run.sh --target-layer 22 ...    # any flags pass through to the python script
#   ./run.sh --no-tmux                # run inline without a tmux wrapper
#
# By default the run is launched inside a detached-friendly tmux session named
# "gemma-recurrence" so closing your SSH terminal will NOT kill the experiment.
# Detach: Ctrl+B then D. Re-attach: tmux attach -t gemma-recurrence.
#
# Steps:
#   0. (Re-)enter a tmux session unless --no-tmux is passed.
#   1. Source .env (creates a stub if missing).
#   2. Validate HF_TOKEN.
#   3. Install/upgrade transformers, datasets, accelerate.
#   4. Run ple_sanity_check.py with whatever args you passed (default: --mode ple-variants).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION_NAME="${TMUX_SESSION:-gemma-recurrence}"

# ---------- 0. tmux wrapper ----------
# Detect --no-tmux in args and strip it from what we forward to the python script.
USE_TMUX=1
FORWARDED_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--no-tmux" ]]; then
        USE_TMUX=0
    else
        FORWARDED_ARGS+=("$arg")
    fi
done

if [[ "$USE_TMUX" == "1" && -z "${TMUX:-}" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux not found --- installing ..."
        SUDO=""
        if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
            SUDO="sudo"
        fi
        # Try every supported package manager. Don't let `set -e` abort us
        # mid-attempt --- we want to print a useful hint on failure.
        install_ok=0
        if command -v apt-get >/dev/null 2>&1; then
            $SUDO apt-get update -qq && $SUDO apt-get install -y -qq tmux && install_ok=1 || true
        elif command -v yum >/dev/null 2>&1; then
            $SUDO yum install -y -q tmux && install_ok=1 || true
        elif command -v dnf >/dev/null 2>&1; then
            $SUDO dnf install -y -q tmux && install_ok=1 || true
        elif command -v apk >/dev/null 2>&1; then
            $SUDO apk add --no-cache tmux && install_ok=1 || true
        elif command -v brew >/dev/null 2>&1; then
            brew install tmux && install_ok=1 || true
        else
            echo "ERROR: no supported package manager found to install tmux."
            echo "Install tmux manually, or re-run with --no-tmux."
            exit 1
        fi
        # Verify the install actually produced a usable tmux binary.
        if [[ "$install_ok" != "1" ]] || ! command -v tmux >/dev/null 2>&1; then
            echo "ERROR: tmux install failed (network down? repo missing the package?)."
            echo "Re-run with --no-tmux to proceed without a tmux wrapper:"
            echo "  ./run.sh --no-tmux ${FORWARDED_ARGS[*]+${FORWARDED_ARGS[*]}}"
            echo "Or use nohup as a lightweight alternative:"
            echo "  nohup ./run.sh --no-tmux ${FORWARDED_ARGS[*]+${FORWARDED_ARGS[*]}} > run.log 2>&1 &"
            exit 1
        fi
        echo "tmux installed: $(tmux -V)"
    fi

    # If a session with this name already exists, just attach.
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Existing tmux session '$SESSION_NAME' found --- attaching."
        echo "Detach: Ctrl+B then D.  Kill the session: tmux kill-session -t $SESSION_NAME"
        exec tmux attach -t "$SESSION_NAME"
    fi

    # Build a robustly-quoted inner command so args with spaces survive.
    # Guard against empty FORWARDED_ARGS: `printf '%q '` with no args emits a
    # literal '' (empty quoted string), which would become an empty arg
    # passed through to argparse and rejected as an unknown argument.
    if [[ ${#FORWARDED_ARGS[@]} -eq 0 ]]; then
        QUOTED_ARGS=""
    else
        QUOTED_ARGS=$(printf '%q ' "${FORWARDED_ARGS[@]}")
    fi
    QUOTED_DIR=$(printf '%q' "$SCRIPT_DIR")
    INNER_CMD="cd $QUOTED_DIR && ./run.sh ${QUOTED_ARGS}--no-tmux; status=\$?; echo; echo \"=== run finished (exit=\$status). Type exit or Ctrl+D to close session. ===\"; exec bash"

    echo "Launching tmux session '$SESSION_NAME' ..."
    echo "  Detach (run keeps going): Ctrl+B then D"
    echo "  Re-attach later:          tmux attach -t $SESSION_NAME"
    echo "  Kill session:             tmux kill-session -t $SESSION_NAME"
    exec tmux new-session -s "$SESSION_NAME" "$INNER_CMD"
fi

# Once we're past the wrapper, only the python-script args remain.
set -- "${FORWARDED_ARGS[@]+"${FORWARDED_ARGS[@]}"}"

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

RUN_ARGS=("$@")

echo
echo "=== Running: $PYTHON ple_sanity_check.py ${RUN_ARGS[*]} ==="
set +e
"$PYTHON" ple_sanity_check.py "${RUN_ARGS[@]}"
RUN_STATUS=$?
set -e

if [[ $RUN_STATUS -ne 0 ]]; then
    echo
    echo "=== Python exited with status $RUN_STATUS --- skipping git push. ==="
    exit $RUN_STATUS
fi

# ---------- 5. Commit & push results ----------
# Only explicit result JSONs. Never `git add -A` --- .env lives here and must
# not be pushed. Push failure is warned-on but not fatal (push manually once
# credentials are sorted).
echo
echo "=== Committing and pushing results ==="

if ! command -v git >/dev/null 2>&1; then
    echo "git not available --- skipping push."
    exit 0
fi
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Not inside a git repo --- skipping push."
    exit 0
fi

FILES_TO_ADD=()
# New-style: everything under results/. The python script always writes
# there now (flag --output-json can override). Legacy bare filenames are
# still staged if they're present at the repo root, so existing result
# JSONs don't get orphaned mid-migration.
if [[ -d results ]]; then
    # shellcheck disable=SC2207
    while IFS= read -r f; do
        FILES_TO_ADD+=("$f")
    done < <(find results -maxdepth 1 -type f -name '*.json' 2>/dev/null)
fi
for f in results.json results_round2a.json results_round2a_addendum.json results_round2a_fixed.json results_round2b_importance.json results_round2b_location.json results_round2c_full_map.json results_round3a_pair_looping.json; do
    [[ -f "$f" ]] && FILES_TO_ADD+=("$f")
done

if [[ ${#FILES_TO_ADD[@]} -eq 0 ]]; then
    echo "No result JSONs on disk --- nothing to commit."
    exit 0
fi

git add "${FILES_TO_ADD[@]}"

if git diff --cached --quiet; then
    echo "Result JSONs unchanged --- nothing to commit."
    exit 0
fi

TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname || echo "unknown-host")
COMMIT_MSG="results: ${RUN_ARGS[*]:-default} @ ${TIMESTAMP} (${HOSTNAME_SHORT})"
git commit -m "$COMMIT_MSG"

echo "Pushing to origin ..."
if git push; then
    echo "=== Push successful. ==="
else
    echo "WARNING: git push failed. Commit is local at $(git rev-parse HEAD)."
    echo "Push manually once credentials / network are sorted:  git push"
    exit 1
fi
