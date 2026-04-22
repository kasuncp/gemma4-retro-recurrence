#!/usr/bin/env bash
# Multi-GPU launcher for the round-5 reasoning eval.
#
# Splits the 9 round-5 configs into 3 shards, each pinned to one GPU via
# CUDA_VISIBLE_DEVICES. Each shard writes its own JSONL checkpoint dir.
# After all shards complete, JSONLs are merged and one final pass
# produces the canonical results/results_round5_reasoning.json with
# summary tables, agreement matrix, and sanity checks.
#
# Usage:
#   ./run_r5_multigpu.sh                 # auto-wraps in tmux session 'gemma-r5'
#   ./run_r5_multigpu.sh --no-tmux       # run inline (no tmux)
#
# By default the run is launched inside a detached-friendly tmux session
# named "gemma-r5" so closing your SSH terminal will NOT kill the shards.
# Detach: Ctrl+B then D.  Re-attach: tmux attach -t gemma-r5.
# If tmux is not installed, the script tries to install it automatically
# (apt / yum / dnf / apk / brew).
#
# Shard assignment (kept deliberate):
#   * baseline + W5-r1 share GPU0. Plan5 requires W5-r1 generations to
#     match baseline token-for-token; cross-GPU bitwise determinism is
#     not guaranteed even on matched hardware, so these must co-locate.
#   * Remaining r=8 configs are distributed so the heaviest GPU runs
#     ~5.2 "baseline-equivalent config-time units" vs the serial total
#     of ~14, giving ~2.7x wall-clock speedup on 3 GPUs.
#   * Time units assume replay cost scales with block width:
#     baseline/W5-r1 = 1, W2=1.4, W3/S10/S20 = 1.6, W4 = 1.8,
#     W5-r8 / W5-r8-noPLE = 2.0. These are approximate; edit the shards
#     below if your observed timings disagree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------- tmux wrapper (mirrors run.sh) ----------
# Auto-wraps this script in a detached-friendly tmux session so SSH
# drops don't kill the multi-hour shards. Installs tmux if missing.
SESSION_NAME="${TMUX_SESSION:-gemma-r5}"

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
        if [[ "$install_ok" != "1" ]] || ! command -v tmux >/dev/null 2>&1; then
            echo "ERROR: tmux install failed (network down? repo missing the package?)."
            echo "Re-run with --no-tmux to proceed without a tmux wrapper:"
            echo "  ./run_r5_multigpu.sh --no-tmux"
            exit 1
        fi
        echo "tmux installed: $(tmux -V)"
    fi

    # If a session with this name already exists, just attach.
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Existing tmux session '$SESSION_NAME' found --- attaching."
        echo "Detach: Ctrl+B then D.  Kill: tmux kill-session -t $SESSION_NAME"
        exec tmux attach -t "$SESSION_NAME"
    fi

    # Build a robustly-quoted inner command so args with spaces survive.
    # Guard against empty FORWARDED_ARGS: `printf '%q '` with no args
    # emits a literal '' that argparse would reject as an unknown arg.
    if [[ ${#FORWARDED_ARGS[@]} -eq 0 ]]; then
        QUOTED_ARGS=""
    else
        QUOTED_ARGS=$(printf '%q ' "${FORWARDED_ARGS[@]}")
    fi
    QUOTED_DIR=$(printf '%q' "$SCRIPT_DIR")
    INNER_CMD="cd $QUOTED_DIR && ./run_r5_multigpu.sh ${QUOTED_ARGS}--no-tmux; status=\$?; echo; echo \"=== run finished (exit=\$status). Type exit or Ctrl+D to close session. ===\"; exec bash"

    echo "Launching tmux session '$SESSION_NAME' ..."
    echo "  Detach (run keeps going): Ctrl+B then D"
    echo "  Re-attach later:          tmux attach -t $SESSION_NAME"
    echo "  Kill session (all shards): tmux kill-session -t $SESSION_NAME"
    exec tmux new-session -s "$SESSION_NAME" "$INNER_CMD"
fi

# Past the wrapper: no meaningful flags remain for this script.
set -- "${FORWARDED_ARGS[@]+"${FORWARDED_ARGS[@]}"}"

# --- Shard definitions ---
# 3-GPU default. For 2 GPUs, comment the 3-GPU block and uncomment the
# 2-GPU block. The manifest check requires every shard to be launched
# with identical args, so keep the common args (below) in sync when
# editing.
SHARD_0=(baseline W5-r1 W2-r8 W3-r8)
SHARD_1=(W4-r8 W5-r8)
SHARD_2=(S10-W3 S20-W3 W5-r8-noPLE)
# 2-GPU alternative:
# SHARD_0=(baseline W5-r1 W2-r8 W3-r8 S10-W3)
# SHARD_1=(W4-r8 W5-r8 S20-W3 W5-r8-noPLE)
# (and remove SHARD_2 + adjust the loop bounds below)

NUM_GPUS=3

# --- Common args forwarded verbatim to every python invocation ---
# Any arg that enters the manifest (dtype, max_gen_tokens, model ids)
# must match across all shards or the merged manifest check will fail.
COMMON_ARGS=(
    --mode reasoning-eval-r5
    --run-arc
)

# --- Paths ---
MERGED_DIR="results/round5_partial"
PARTIAL_DIR_PREFIX="results/round5_partial_gpu"
LOG_DIR="results/round5_logs"

# --- Env setup (mirrors run.sh step 1+2) ---
if [[ ! -f .env ]]; then
    echo "ERROR: .env not found. Run ./run.sh once to bootstrap .env, or"
    echo "       create it manually with HF_TOKEN=<your token>."
    exit 1
fi
set -a
# shellcheck disable=SC1091
source .env
set +a
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN is not set in .env."
    exit 1
fi
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN}}"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN}}"

PYTHON="${PYTHON:-python3}"
mkdir -p "$LOG_DIR"

# --- GPU visibility preflight ---
if command -v nvidia-smi >/dev/null 2>&1; then
    VISIBLE_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
    if [[ "$VISIBLE_GPUS" -lt "$NUM_GPUS" ]]; then
        echo "ERROR: NUM_GPUS=$NUM_GPUS but only $VISIBLE_GPUS GPU(s) visible via nvidia-smi."
        echo "       Edit the shard definitions at the top of this script."
        exit 1
    fi
fi

# Route each shard's args through a helper so bash's array-passing
# quirks don't silently drop configs.
launch_shard() {
    local gpu_id=$1
    shift
    local configs=("$@")
    local ckpt_dir="${PARTIAL_DIR_PREFIX}${gpu_id}"
    local log_file="$LOG_DIR/shard_gpu${gpu_id}.log"

    # Status goes to stderr so the final `echo $!` is the only thing on
    # stdout --- otherwise $(launch_shard ...) captures the status lines
    # too and the calling `wait $pid` chokes on a multi-line blob.
    echo "[gpu${gpu_id}] configs: ${configs[*]}" >&2
    echo "[gpu${gpu_id}] ckpt:    ${ckpt_dir}" >&2
    echo "[gpu${gpu_id}] log:     ${log_file}" >&2
    CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON" ple_sanity_check.py \
        "${COMMON_ARGS[@]}" \
        --configs "${configs[@]}" \
        --checkpoint-dir "$ckpt_dir" \
        > "$log_file" 2>&1 &
    echo $!
}

echo "=== Launching $NUM_GPUS shards ==="
PIDS=()
PID0=$(launch_shard 0 "${SHARD_0[@]}");    PIDS+=("$PID0")
PID1=$(launch_shard 1 "${SHARD_1[@]}");    PIDS+=("$PID1")
PID2=$(launch_shard 2 "${SHARD_2[@]}");    PIDS+=("$PID2")

echo
echo "=== Shards launched. Tailing aggregated progress every 30s ==="
echo "    Individual logs: $LOG_DIR/shard_gpu{0,1,2}.log"
echo "    Kill all:        kill ${PIDS[*]}"
echo

# Poll: as long as any shard is alive, print a one-line summary per shard.
# `kill -0 <pid>` returns 0 iff the process still exists.
while :; do
    alive=0
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            alive=$((alive + 1))
        fi
    done
    for i in 0 1 2; do
        log="$LOG_DIR/shard_gpu${i}.log"
        last=$(tail -n 1 "$log" 2>/dev/null || echo "(no output yet)")
        printf "  [gpu%d] %s\n" "$i" "$last"
    done
    if [[ "$alive" -eq 0 ]]; then
        break
    fi
    echo "  --- $alive shard(s) still running ---"
    sleep 30
done

# --- Collect exit codes ---
FAIL=0
for i in 0 1 2; do
    pid="${PIDS[$i]}"
    if ! wait "$pid"; then
        status=$?
        echo "ERROR: shard gpu${i} (pid=$pid) exited with status $status."
        echo "       Tail of $LOG_DIR/shard_gpu${i}.log:"
        tail -n 20 "$LOG_DIR/shard_gpu${i}.log" | sed 's/^/         /'
        FAIL=1
    else
        echo "OK: shard gpu${i} (pid=$pid) finished cleanly."
    fi
done
if [[ "$FAIL" -ne 0 ]]; then
    echo
    echo "One or more shards failed. Merge + finalize SKIPPED."
    echo "Fix the failing shard(s) and re-run; completed configs will"
    echo "resume from their JSONLs on the next launch."
    exit 1
fi

# --- Merge JSONLs into the canonical checkpoint dir ---
# Each shard has disjoint --configs, so JSONL filenames are disjoint.
# Manifests are expected to be identical across shards (same COMMON_ARGS).
echo
echo "=== Merging shard JSONLs into $MERGED_DIR ==="
mkdir -p "$MERGED_DIR"

# Manifest: copy from shard 0; verify the other shards match.
cp "${PARTIAL_DIR_PREFIX}0/manifest.json" "$MERGED_DIR/manifest.json"
for i in 1 2; do
    if ! diff -q "${PARTIAL_DIR_PREFIX}0/manifest.json" "${PARTIAL_DIR_PREFIX}${i}/manifest.json" >/dev/null; then
        echo "ERROR: manifest.json differs between shard 0 and shard ${i}."
        echo "       Shards were launched with inconsistent args. Aborting merge."
        exit 1
    fi
done

# JSONLs: copy with collision check. Two shards producing the same
# filename means the shard definitions overlap --- a config bug, not a
# data bug. Bail loudly so the user fixes the shard split.
for i in 0 1 2; do
    d="${PARTIAL_DIR_PREFIX}${i}"
    shopt -s nullglob
    for f in "$d"/*.jsonl; do
        base=$(basename "$f")
        target="$MERGED_DIR/$base"
        if [[ -e "$target" ]]; then
            if ! diff -q "$f" "$target" >/dev/null; then
                echo "ERROR: $base exists in multiple shards with differing contents."
                echo "       Shard definitions overlap. Check SHARD_0/1/2 at the top of this script."
                exit 1
            fi
        fi
        cp "$f" "$target"
    done
    shopt -u nullglob
done

echo "  merged $(find "$MERGED_DIR" -maxdepth 1 -name '*.jsonl' | wc -l | tr -d ' ') JSONL files."

# --- Finalize: run the mode once more against the merged dir. ---
# Every config's JSONL is complete, so _run_dataset sees remaining=[]
# for each config and emits zero generations. The python code still
# loads the model on GPU 0 (~1-2 min per pass, 2 passes) to run its
# strategy-A inspection and range checks before falling through to
# the summary + sanity checks + final results JSON write.
echo
echo "=== Finalizing: summary + sanity checks + results JSON ==="
echo "    (Pins to GPU 0. Model load happens even though no generation runs.)"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" ple_sanity_check.py \
    "${COMMON_ARGS[@]}" \
    --checkpoint-dir "$MERGED_DIR"

echo
echo "=== Done ==="
echo "  Canonical output: results/results_round5_reasoning.json"
echo "  Merged JSONLs:    $MERGED_DIR/"
echo "  Per-shard JSONLs: ${PARTIAL_DIR_PREFIX}{0,1,2}/  (safe to delete)"
