#!/usr/bin/env bash
# One-shot runner for the PLE x recurrence probes and the Path 1 CoT gate on
# RunPod.
#
# Usage:
#   ./run.sh                                     # probe, round 2a, wrapped in tmux (default)
#   ./run.sh --mode original                     # probe, round 1 sanity check
#   ./run.sh --mode ple-variants                 # probe, round 2a (explicit)
#   ./run.sh --mode reasoning-eval-r5            # probe, round 5 (latest)
#   ./run.sh --target-layer 22 ...               # probe: any flags pass through
#   ./run.sh --script path1                      # Path 1 CoT gate (all 4 cells, sequential)
#   ./run.sh --script path1 --n 20               # Path 1: quick 20-example preview (separate results dir)
#   ./run.sh --script path1 --batch-size 8       # Path 1: batch 8 prompts/call to lift GPU util
#   ./run.sh --script path1 --summarize          # Path 1 CoT gate: aggregate shards
#   ./run.sh --script path1 --cells it:cot       # Path 1 CoT gate: single cell
#   ./run.sh --script path1-plan2                # Path 1 plan 2: length + self-consistency sweeps
#   ./run.sh --script path1-plan2 --n 20 --cells A3 B3   # Path 1 plan 2: quick preview
#   ./run.sh --no-tmux                           # run inline without a tmux wrapper
#   ./run.sh --dry-run --script path1            # offline smoke: prints dispatch summary, no deps/python/git
#
# --script selects the python entry point:
#   probe  (default) -> ple_sanity_check.py     (Path 2 depth-recurrence probes)
#   path1            -> path1_cot_gate.py       (Path 1 CoT gate, plan 1)
#   path1-plan2      -> path1_length_and_sc.py  (Path 1 plan 2: length + self-consistency)
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
#   4. Run the selected python script with whatever args you passed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SESSION_NAME="${TMUX_SESSION:-gemma-recurrence}"

# ---------- Experiment registry ----------
# To add a new experiment script, add one entry to each of the four
# parallel arrays below — do NOT sprinkle cases elsewhere in this file.
# Entries must stay in lockstep (same index across arrays).
#
# Convention every entry point must satisfy:
#   1. Accept --no-resume and resume by reading existing output if the
#      script can run for >30 minutes. Short single-shot runs are exempt.
#   2. Write exclusively under the declared result_root. The watcher
#      rsyncs that dir; results written elsewhere will never reach the laptop.
#   3. Exit 0 on success, non-zero on failure. run.sh drops .DONE / .FAILED
#      markers at EXPERIMENT_RESULT_DIR automatically; scripts don't write them.
#   4. Register here: key, script file, default args, result_root, result_depth.
#        result_depth=flat       -> stage-5 commits depth-1 *.json only
#        result_depth=recursive  -> stage-5 commits *.json + *.jsonl recursive
#
# We use parallel indexed arrays instead of `declare -A` so this file runs
# on both macOS stock bash 3.2 (no assoc-array support) and pod bash 4+.
EXPERIMENT_KEYS=(probe                path1                       path1-plan2)
EXPERIMENT_SCRIPTS=(ple_sanity_check.py  path1_cot_gate.py           path1_length_and_sc.py)
EXPERIMENT_DEFAULTS=("--mode ple-variants"  ""                        "")
EXPERIMENT_ROOTS=(results             results/path_1_cot_tokens    results/path_1_cot_tokens/plan2)
EXPERIMENT_DEPTHS=(flat               recursive                    recursive)

# Return the index of $1 in EXPERIMENT_KEYS, or non-zero if not found.
# Echoes the index on success.
_registry_idx() {
    local key="$1" i
    for i in "${!EXPERIMENT_KEYS[@]}"; do
        if [[ "${EXPERIMENT_KEYS[$i]}" == "$key" ]]; then
            echo "$i"
            return 0
        fi
    done
    return 1
}

# ---------- 0. tmux wrapper ----------
# Detect --no-tmux and --script (run.sh's own flags) and strip them from what
# we forward to the python script.
USE_TMUX=1
DRY_RUN=0
TARGET_SCRIPT="probe"
FORWARDED_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-tmux)
            USE_TMUX=0
            shift
            ;;
        --dry-run)
            # Offline smoke-test mode: skip deps install, skip Python, skip
            # stage 5 (git commit/push). Prints a dispatch summary that tests
            # can assert on. Implies --no-tmux. See tests/run/test_dry_run.sh.
            DRY_RUN=1
            USE_TMUX=0
            shift
            ;;
        --script)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --script needs a value (one of: ${EXPERIMENT_KEYS[*]})."
                exit 2
            fi
            TARGET_SCRIPT="$2"
            shift 2
            ;;
        --script=*)
            TARGET_SCRIPT="${1#*=}"
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! _experiment_idx=$(_registry_idx "$TARGET_SCRIPT"); then
    echo "ERROR: unknown --script '$TARGET_SCRIPT' (expected one of: ${EXPERIMENT_KEYS[*]})."
    exit 2
fi

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
    QUOTED_SCRIPT=$(printf '%q' "$TARGET_SCRIPT")
    INNER_CMD="cd $QUOTED_DIR && ./run.sh --script $QUOTED_SCRIPT ${QUOTED_ARGS}--no-tmux; status=\$?; echo; echo \"=== run finished (exit=\$status). Type exit or Ctrl+D to close session. ===\"; exec bash"

    echo "Launching tmux session '$SESSION_NAME' ..."
    echo "  Detach (run keeps going): Ctrl+B then D"
    echo "  Re-attach later:          tmux attach -t $SESSION_NAME"
    echo "  Kill session:             tmux kill-session -t $SESSION_NAME"
    exec tmux new-session -s "$SESSION_NAME" "$INNER_CMD"
fi

# Once we're past the wrapper, only the python-script args remain.
set -- "${FORWARDED_ARGS[@]+"${FORWARDED_ARGS[@]}"}"

# ---------- Dry-run short-circuit ----------
# Emit a dispatch summary and exit. Useful for offline smoke testing that
# registry lookups + default args + result roots resolve correctly without
# needing HF_TOKEN, python deps, or a GPU.  See tests/run/test_dry_run.sh.
if [[ "$DRY_RUN" == "1" ]]; then
    _dry_script="${EXPERIMENT_SCRIPTS[$_experiment_idx]}"
    _dry_defaults="${EXPERIMENT_DEFAULTS[$_experiment_idx]}"
    _dry_root="${EXPERIMENT_ROOTS[$_experiment_idx]}"
    _dry_depth="${EXPERIMENT_DEPTHS[$_experiment_idx]}"
    _dry_effective_args=("$@")
    if [[ ${#_dry_effective_args[@]} -eq 0 && -n "$_dry_defaults" ]]; then
        # shellcheck disable=SC2086
        set -- $_dry_defaults
        _dry_effective_args=("$@")
    fi
    echo "dry-run: target_script=$TARGET_SCRIPT"
    echo "dry-run: py_script=$_dry_script"
    echo "dry-run: default_args=$_dry_defaults"
    echo "dry-run: result_root=$_dry_root"
    echo "dry-run: result_depth=$_dry_depth"
    echo "dry-run: effective_args=${_dry_effective_args[*]+${_dry_effective_args[*]}}"
    echo "dry-run: ok"
    exit 0
fi

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

# ---------- 4. Run the selected python script ----------
# Defaults differ per entry point. If the user passed any args, use those
# verbatim instead.
PY_SCRIPT="${EXPERIMENT_SCRIPTS[$_experiment_idx]}"
_default_args="${EXPERIMENT_DEFAULTS[$_experiment_idx]}"
if [[ $# -eq 0 && -n "$_default_args" ]]; then
    # shellcheck disable=SC2086  # deliberate word-split: defaults are simple flag tokens
    set -- $_default_args
fi
unset _default_args

RUN_ARGS=("$@")

echo
echo "=== Running: $PYTHON $PY_SCRIPT ${RUN_ARGS[*]} ==="
set +e
"$PYTHON" "$PY_SCRIPT" "${RUN_ARGS[@]}"
RUN_STATUS=$?
set -e

# ---------- 4a. Write completion marker for the agent ----------
# EXPERIMENT_RESULT_DIR is exported by `runpod.sh launch` when the run is
# agent-driven. If it's set, drop .DONE/.FAILED so runpod.sh watch can detect
# the terminal state. If the dir doesn't exist (python crashed early), create
# it so the marker write succeeds.
if [[ -n "${EXPERIMENT_RESULT_DIR:-}" ]]; then
    mkdir -p "$EXPERIMENT_RESULT_DIR"
    if [[ $RUN_STATUS -eq 0 ]]; then
        touch "$EXPERIMENT_RESULT_DIR/.DONE"
        echo "marker: wrote $EXPERIMENT_RESULT_DIR/.DONE"
    else
        touch "$EXPERIMENT_RESULT_DIR/.FAILED"
        echo "marker: wrote $EXPERIMENT_RESULT_DIR/.FAILED"
    fi
fi

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
# Registry-driven: stage files under the declared result_root, using the
# declared depth.  flat=depth-1 *.json only; recursive=*.json+*.jsonl.
_result_root="${EXPERIMENT_ROOTS[$_experiment_idx]}"
_result_depth="${EXPERIMENT_DEPTHS[$_experiment_idx]}"

if [[ -d "$_result_root" ]]; then
    # shellcheck disable=SC2207
    if [[ "$_result_depth" == "flat" ]]; then
        while IFS= read -r f; do
            FILES_TO_ADD+=("$f")
        done < <(find "$_result_root" -maxdepth 1 -type f -name '*.json' 2>/dev/null)
    else
        while IFS= read -r f; do
            FILES_TO_ADD+=("$f")
        done < <(find "$_result_root" -type f \( -name '*.json' -o -name '*.jsonl' \) 2>/dev/null)
    fi
fi

# Legacy: pre-migration probe runs dropped JSONs at the repo root. Pick them
# up so old result files don't get orphaned. Only relevant for 'probe'.
if [[ "$TARGET_SCRIPT" == "probe" ]]; then
    for f in results.json results_round2a.json results_round2a_addendum.json results_round2a_fixed.json results_round2b_importance.json results_round2b_location.json results_round2c_full_map.json results_round3a_pair_looping.json; do
        [[ -f "$f" ]] && FILES_TO_ADD+=("$f")
    done
fi
unset _result_root _result_depth

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
COMMIT_MSG="results(${TARGET_SCRIPT}): ${RUN_ARGS[*]:-default} @ ${TIMESTAMP} (${HOSTNAME_SHORT})"
git commit -m "$COMMIT_MSG"

echo "Pushing to origin ..."
if git push; then
    echo "=== Push successful. ==="
else
    echo "WARNING: git push failed. Commit is local at $(git rev-parse HEAD)."
    echo "Push manually once credentials / network are sorted:  git push"
    exit 1
fi
