# RunPod Experiment Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous watch loop + extended `runpod.sh` subcommands that provision a RunPod GPU, launch a tmux-wrapped experiment, periodically rsync partial results back to the laptop, and cleanly tear the pod down on completion or emergency-tear-down on credit exhaustion.

**Architecture:** Three pieces. (1) Extend `runpod.sh` with idempotent primitives — `cost`, `bootstrap`, `launch`, `tmux-alive`, `marker`, `sync-down`, `watch`. (2) The new `watch` subcommand is a pure-bash state machine reading `experiment.yaml`; no LLM in the tick. (3) Three one-shot `claude -p` touchpoints (launch validation, crash triage, final summary) invoked manually by the operator — no changes needed to support them beyond the extended `runpod.sh` they call.

**Tech Stack:** bash, `curl` + `jq` (already deps of `runpod.sh`), `yq` (new dep — YAML reader), `rsync` (new dep), `ssh`/`scp` (already used). Tests are plain bash scripts with a fixture-driven `api()` shim — no bats or external test framework.

**Design spec:** `docs/superpowers/specs/2026-04-23-runpod-agent-design.md`.

---

## File Structure

**Files to modify:**
- `runpod.sh` — add helpers (`_read_config`, `_need_yq`, `_unix_now`) and subcommands (`cmd_cost`, `cmd_bootstrap`, `cmd_launch`, `cmd_tmux_alive`, `cmd_marker`, `cmd_sync_down`, `cmd_watch`); extend `cmd_up` to write `started_at`; extend dispatch + help.
- `run.sh` — one block near the end to touch `.DONE`/`.FAILED` in `$EXPERIMENT_RESULT_DIR` when that env var is set.
- `.gitignore` — add `experiment.yaml`, `watch.log`, `.runpod-state.json.bak`.

**Files to create:**
- `experiment.example.yaml` — template for the user to copy to `experiment.yaml`.
- `tests/runpod/test_cost.sh` — bash unit test for `cmd_cost` threshold logic using a stubbed `api()`.
- `tests/runpod/fixtures/pod_normal.json` — canned `GET /pods/{id}` response.
- `tests/runpod/fixtures/user_normal.json` — canned `GET /user` response.
- `tests/runpod/fixtures/user_low_balance.json` — balance below emergency threshold.

---

## Task 1: Scaffolding — example config + gitignore + yq helper

**Files:**
- Create: `experiment.example.yaml`
- Modify: `.gitignore`
- Modify: `runpod.sh` (add helpers near top, before `cmd_up`)

- [ ] **Step 1: Write `experiment.example.yaml`**

```yaml
# Copy to experiment.yaml (gitignored) and edit for your run.
run:
  flags: "--script path1 --n 500 --batch-size 8"
  result_dir: "results/path_1_cot_tokens/plan1"
  # local_result_dir defaults to result_dir if omitted.
  # local_result_dir: "results/path_1_cot_tokens/plan1"

pod:
  gpu_types: "NVIDIA GeForce RTX 4090"
  gpu_count: 1
  cloud_type: SECURE
  container_disk_gb: 50
  volume_gb: 50

git:
  url: "https://github.com/kasuncp/gemma4-retro-recurrence.git"
  ref: "main"

budget:
  cap_usd: 30.0          # soft warn when spend exceeds this
  emergency_usd: 0.50    # hard teardown when account balance drops below this
  max_hours: 12          # hard teardown on wall time

watch:
  tick_seconds: 420      # 7 minutes
```

- [ ] **Step 2: Update `.gitignore`**

Append these lines (preserving existing content):

```
experiment.yaml
watch.log
.runpod-state.json
.runpod-state.json.bak
```

(`.runpod-state.json` should not be committed — it holds ssh endpoints and pod ids.)

- [ ] **Step 3: Add `_need_yq`, `_unix_now`, and `_read_config` helpers to `runpod.sh`**

Insert after the `need curl; need jq; need ssh; need scp` line (currently line 39) and before `api()`:

```bash
_need_yq() {
    command -v yq >/dev/null || die "missing dependency: yq — install with 'brew install yq' (macOS) or 'apt-get install yq' (Linux). Required for reading experiment.yaml."
}

_unix_now() { date +%s; }

# Read one yaml field from experiment.yaml. Usage: _read_config '.run.flags'
# Returns empty string if the field is null/absent. Errors out if file missing.
_read_config() {
    local field="$1" file="${EXPERIMENT_CONFIG:-experiment.yaml}"
    [[ -f "$file" ]] || die "config file not found: $file"
    _need_yq
    local v; v=$(yq -r "$field // \"\"" "$file" 2>/dev/null) || die "yq failed to parse $file (field $field)"
    [[ "$v" == "null" ]] && v=""
    echo "$v"
}
```

- [ ] **Step 4: Commit**

```bash
git add experiment.example.yaml .gitignore runpod.sh
git commit -m "Add experiment.yaml template + yq-based config helper"
```

---

## Task 2: `cmd_cost` — cost/budget primitive with test fixtures

**Files:**
- Modify: `runpod.sh` (add `cmd_cost` below `cmd_ssh`, roughly at line 201)
- Create: `tests/runpod/fixtures/pod_normal.json`
- Create: `tests/runpod/fixtures/user_normal.json`
- Create: `tests/runpod/fixtures/user_low_balance.json`
- Create: `tests/runpod/test_cost.sh`

Design note: `cmd_cost` reads pod + user from the RunPod API, computes `spentUsd` from `started_at`, and optionally evaluates thresholds if `BUDGET_CAP_USD` / `BUDGET_EMERGENCY_USD` are in the environment. `cmd_watch` will populate these from `experiment.yaml` before calling `cmd_cost`. This keeps `cmd_cost` testable without a config file.

- [ ] **Step 1: Create fixture files**

`tests/runpod/fixtures/pod_normal.json`:

```json
{
  "id": "abc123",
  "name": "probe-test",
  "costPerHr": 0.39,
  "desiredStatus": "RUNNING",
  "publicIp": "1.2.3.4",
  "portMappings": { "22": 40022 },
  "machine": { "dataCenterId": "US-CA-1" }
}
```

`tests/runpod/fixtures/user_normal.json`:

```json
{
  "id": "user_xyz",
  "email": "test@example.com",
  "clientBalance": 41.23
}
```

`tests/runpod/fixtures/user_low_balance.json`:

```json
{
  "id": "user_xyz",
  "email": "test@example.com",
  "clientBalance": 0.30
}
```

- [ ] **Step 2: Write `tests/runpod/test_cost.sh` (the failing test)**

```bash
#!/usr/bin/env bash
# Tests for cmd_cost threshold logic.
# Stubs api() to return fixtures; stubs _load_state to return fake ids.
# Stubs _unix_now to return a deterministic time so elapsed hours are exact.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PASS=0; FAIL=0
_pass() { PASS=$((PASS+1)); echo "  PASS: $1"; }
_fail() { FAIL=$((FAIL+1)); echo "  FAIL: $1"; echo "    got: $2"; echo "    want: $3"; }

# Set up shimmed environment. Source runpod.sh but block the dispatch at bottom
# by setting RUNPOD_SHIM=1 (we'll guard it in runpod.sh).
export RUNPOD_API_KEY="fake-key-for-tests"
export RUNPOD_SHIM=1

# Stub api() before sourcing.
api() {
    local path="$2"
    case "$path" in
        /pods/*) cat "$SCRIPT_DIR/fixtures/${POD_FIXTURE:-pod_normal}.json" ;;
        /user)   cat "$SCRIPT_DIR/fixtures/${USER_FIXTURE:-user_normal}.json" ;;
        *) echo "test stub: unexpected path $path" >&2; return 1 ;;
    esac
}
export -f api

# Stub state loader + timing.
_load_state() { POD_ID="abc123"; POD_HOST="1.2.3.4"; POD_PORT="40022"; POD_STARTED_AT="${POD_STARTED_AT:-1000000000}"; }
_unix_now() { echo "${FAKE_NOW:-1000003600}"; }  # default: 1 hour after start
export -f _load_state
export -f _unix_now

source "$REPO_ROOT/runpod.sh"

# --- Scenario: normal (1 hour elapsed, $0.39/hr, $41 balance, cap $30, emergency $0.50)
echo "scenario: normal"
POD_FIXTURE=pod_normal USER_FIXTURE=user_normal FAKE_NOW=1000003600 \
    BUDGET_CAP_USD=30 BUDGET_EMERGENCY_USD=0.50 \
    out=$(cmd_cost) || { echo "cmd_cost failed"; exit 1; }
spent=$(jq -r '.spentUsd' <<<"$out")
hard=$(jq -r '.hardStopTriggered' <<<"$out")
soft=$(jq -r '.softWarnTriggered' <<<"$out")
[[ "$spent" == "0.39" ]] && _pass "spent=0.39" || _fail "spent" "$spent" "0.39"
[[ "$hard"  == "false"  ]] && _pass "hardStop=false" || _fail "hardStop" "$hard" "false"
[[ "$soft"  == "false"  ]] && _pass "softWarn=false" || _fail "softWarn" "$soft" "false"

# --- Scenario: low balance (hard stop on account balance)
echo "scenario: low balance"
POD_FIXTURE=pod_normal USER_FIXTURE=user_low_balance FAKE_NOW=1000003600 \
    BUDGET_CAP_USD=30 BUDGET_EMERGENCY_USD=0.50 \
    out=$(cmd_cost) || { echo "cmd_cost failed"; exit 1; }
hard=$(jq -r '.hardStopTriggered' <<<"$out")
[[ "$hard" == "true" ]] && _pass "hardStop=true on low balance" || _fail "hardStop" "$hard" "true"

# --- Scenario: spend over cap (soft warn fires, hard does not)
echo "scenario: spend over cap (100 hours elapsed)"
POD_FIXTURE=pod_normal USER_FIXTURE=user_normal FAKE_NOW=1000360000 \
    BUDGET_CAP_USD=30 BUDGET_EMERGENCY_USD=0.50 \
    out=$(cmd_cost) || { echo "cmd_cost failed"; exit 1; }
soft=$(jq -r '.softWarnTriggered' <<<"$out")
hard=$(jq -r '.hardStopTriggered' <<<"$out")
[[ "$soft" == "true"  ]] && _pass "softWarn=true on overspend" || _fail "softWarn" "$soft" "true"
[[ "$hard" == "false" ]] && _pass "hardStop=false on overspend" || _fail "hardStop" "$hard" "false"

echo
echo "Tests: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
```

Make it executable: `chmod +x tests/runpod/test_cost.sh`.

- [ ] **Step 3: Run the test and verify it fails**

Run: `bash tests/runpod/test_cost.sh`
Expected: exits non-zero. The `cmd_cost` function doesn't exist yet, and sourcing runs the dispatch at the bottom of `runpod.sh` (unknown subcommand error).

- [ ] **Step 4: Add a `RUNPOD_SHIM` guard at the bottom of `runpod.sh`**

Find the dispatch block (currently `sub="${1:-}"; shift || true` at ~line 217) and wrap it in an if-guard:

```bash
# ---------- dispatch ----------
if [[ -z "${RUNPOD_SHIM:-}" ]]; then
    sub="${1:-}"; shift || true
    case "$sub" in
        up)     cmd_up "$@" ;;
        exec)   cmd_exec "$@" ;;
        run)    cmd_run "$@" ;;
        push)   cmd_push "$@" ;;
        pull)   cmd_pull "$@" ;;
        ssh)    cmd_ssh "$@" ;;
        status) cmd_status "$@" ;;
        logs)   cmd_logs "$@" ;;
        down)   cmd_down "$@" ;;
        ""|help|-h|--help)
            sed -n '2,25p' "$0"; exit 0 ;;
        *) die "unknown subcommand: $sub (try --help)" ;;
    esac
fi
```

This lets `test_cost.sh` source the file to load functions without dispatching.

- [ ] **Step 5: Extend `_load_state` to also load `started_at`**

Find `_load_state` (currently ~line 147) and add:

```bash
_load_state() {
    [[ -f "$STATE_FILE" ]] || die "no state file ($STATE_FILE). Run 'up' first."
    POD_ID=$(jq -r '.id' "$STATE_FILE")
    POD_HOST=$(jq -r '.publicIp // empty' "$STATE_FILE")
    POD_PORT=$(jq -r '.sshPort // empty' "$STATE_FILE")
    POD_STARTED_AT=$(jq -r '.started_at // empty' "$STATE_FILE")
    [[ -n "$POD_ID" ]] || die "malformed state file"
}
```

- [ ] **Step 6: Implement `cmd_cost`**

Add after `cmd_status` (~line 206):

```bash
cmd_cost() {
    _load_state
    [[ -n "$POD_STARTED_AT" ]] || die "state file has no started_at; was the pod brought up with the current runpod.sh?"
    local pod user
    pod=$(api GET "/pods/$POD_ID") || die "pod GET failed"
    user=$(api GET "/user") || die "user GET failed"

    local cost_per_hr balance now elapsed_s
    cost_per_hr=$(jq -r '.costPerHr // 0' <<<"$pod")
    balance=$(jq -r '.clientBalance // 0' <<<"$user")
    now=$(_unix_now)
    elapsed_s=$(( now - POD_STARTED_AT ))

    # bc has 2-decimal rounding issues; use awk with printf for consistent output.
    local spent budget_remaining cap emergency hard soft
    cap="${BUDGET_CAP_USD:-}"
    emergency="${BUDGET_EMERGENCY_USD:-}"
    spent=$(awk -v c="$cost_per_hr" -v s="$elapsed_s" 'BEGIN{printf "%.2f", c*(s/3600.0)}')

    if [[ -n "$cap" ]]; then
        budget_remaining=$(awk -v a="$cap" -v b="$spent" 'BEGIN{printf "%.2f", a-b}')
        soft=$(awk -v s="$spent" -v c="$cap" 'BEGIN{print (s>c)?"true":"false"}')
    else
        budget_remaining="null"; soft="false"
    fi
    if [[ -n "$emergency" ]]; then
        hard=$(awk -v b="$balance" -v e="$emergency" 'BEGIN{print (b<e)?"true":"false"}')
    else
        hard="false"
    fi

    jq -n \
        --argjson cph "$cost_per_hr" \
        --argjson spent "$spent" \
        --argjson bal "$balance" \
        --arg cap "${cap:-null}" \
        --arg rem "$budget_remaining" \
        --arg em "${emergency:-null}" \
        --argjson hard "$hard" \
        --argjson soft "$soft" \
        --argjson elapsed "$elapsed_s" \
        '{
            costPerHr: $cph,
            spentUsd: $spent,
            accountBalance: $bal,
            budgetCapUsd: ($cap|tonumber? // null),
            budgetRemaining: ($rem|tonumber? // null),
            emergencyUsd: ($em|tonumber? // null),
            elapsedSeconds: $elapsed,
            hardStopTriggered: $hard,
            softWarnTriggered: $soft
        }'
}
```

- [ ] **Step 7: Re-run the test and verify it passes**

Run: `bash tests/runpod/test_cost.sh`
Expected: `Tests: 5 passed, 0 failed` and exit 0.

- [ ] **Step 8: Commit**

```bash
git add runpod.sh tests/runpod/
git commit -m "Add cmd_cost with fixture-driven threshold tests"
```

---

## Task 3: Persist `started_at` in the state file on `up`

**Files:**
- Modify: `runpod.sh` (within `cmd_up`, at state-file write)

- [ ] **Step 1: Modify `cmd_up` to add `started_at`**

Find the line that writes the initial state file in `cmd_up` (currently ~line 122):

```bash
echo "$resp" | jq '{id, name, costPerHr, desiredStatus}' > "$STATE_FILE"
```

Replace with:

```bash
local started_at; started_at=$(_unix_now)
echo "$resp" | jq --argjson ts "$started_at" '{id, name, costPerHr, desiredStatus} + {started_at: $ts}' > "$STATE_FILE"
```

Find the later line that updates the state file after ssh becomes reachable (~line 136):

```bash
jq --arg h "$host" --arg p "$port" '. + {publicIp: $h, sshPort: ($p|tonumber)}' \
    "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
```

Leave this unchanged — the `+` merge preserves `started_at`.

- [ ] **Step 2: Verify by inspection**

No test changes needed — `test_cost.sh` already stubs `_load_state` with a fake `POD_STARTED_AT`. This task only affects real pods.

Dry-check: read through the modified `cmd_up` to confirm both state writes include `started_at` (initial write adds it; reachability write preserves it via `+`).

- [ ] **Step 3: Commit**

```bash
git add runpod.sh
git commit -m "Persist started_at in state file for cost accounting"
```

---

## Task 4: `cmd_bootstrap` — clone/update repo on pod, copy .env

**Files:**
- Modify: `runpod.sh`

- [ ] **Step 1: Implement `cmd_bootstrap`**

Add after `cmd_cost`:

```bash
cmd_bootstrap() {
    [[ $# -ge 1 ]] || die "bootstrap needs <git-url> [ref]"
    local url="$1" ref="${2:-main}"
    _load_state; _refresh_ssh

    # Derive repo dir name from the URL (strip .git).
    local repo_name; repo_name=$(basename "$url" .git)
    local repo_dir="/workspace/$repo_name"

    echo "bootstrapping $url ($ref) -> $repo_dir"
    _ssh "set -e
        mkdir -p /workspace
        if [ -d '$repo_dir/.git' ]; then
            cd '$repo_dir'
            git fetch --quiet origin
            git reset --quiet --hard 'origin/$ref' 2>/dev/null || git reset --quiet --hard '$ref'
        else
            git clone --quiet '$url' '$repo_dir'
            cd '$repo_dir'
            git checkout --quiet '$ref' 2>/dev/null || true
        fi
        chmod +x run.sh runpod.sh 2>/dev/null || true
        echo 'bootstrap ok: '\$(git -C '$repo_dir' rev-parse --short HEAD)
    "

    # Copy local .env up if it exists. Without it run.sh will abort on missing HF_TOKEN.
    if [[ -f .env ]]; then
        echo "uploading .env -> $repo_dir/.env"
        scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
            -i "$SSH_KEY" -P "$POD_PORT" .env "root@$POD_HOST:$repo_dir/.env"
    else
        echo "warn: local .env not found — run.sh will fail on the pod without HF_TOKEN"
    fi

    # Remember repo_dir so later subcommands know where to cd.
    jq --arg rd "$repo_dir" '. + {repo_dir: $rd}' "$STATE_FILE" > "$STATE_FILE.tmp" \
        && mv "$STATE_FILE.tmp" "$STATE_FILE"
}
```

- [ ] **Step 2: Extend `_load_state` to load `POD_REPO_DIR`**

Modify `_load_state`:

```bash
_load_state() {
    [[ -f "$STATE_FILE" ]] || die "no state file ($STATE_FILE). Run 'up' first."
    POD_ID=$(jq -r '.id' "$STATE_FILE")
    POD_HOST=$(jq -r '.publicIp // empty' "$STATE_FILE")
    POD_PORT=$(jq -r '.sshPort // empty' "$STATE_FILE")
    POD_STARTED_AT=$(jq -r '.started_at // empty' "$STATE_FILE")
    POD_REPO_DIR=$(jq -r '.repo_dir // empty' "$STATE_FILE")
    [[ -n "$POD_ID" ]] || die "malformed state file"
}
```

- [ ] **Step 3: Manual smoke (skip if no pod available)**

With a pod already up, run:

```bash
./runpod.sh bootstrap https://github.com/kasuncp/gemma4-retro-recurrence.git main
```

Expected: prints `bootstrap ok: <sha>` and (if `.env` exists locally) the scp message. Re-run to verify idempotence — second run should still succeed and print the same sha if `main` didn't move.

If no pod is available, skip this step and rely on the integration smoke test in Task 10.

- [ ] **Step 4: Commit**

```bash
git add runpod.sh
git commit -m "Add cmd_bootstrap to clone/update repo and copy .env to pod"
```

---

## Task 5: `cmd_launch` + `run.sh` marker write

**Files:**
- Modify: `runpod.sh`
- Modify: `run.sh`

- [ ] **Step 1: Implement `cmd_launch`**

Add after `cmd_bootstrap`:

```bash
cmd_launch() {
    [[ $# -eq 2 ]] || die "launch needs <run.sh-flags> <result-dir>"
    local flags="$1" result_dir="$2"
    _load_state; _refresh_ssh
    [[ -n "$POD_REPO_DIR" ]] || die "no repo_dir in state; run bootstrap first"

    local session="gemma-recurrence"
    local launch_cmd
    # Quote flags + result_dir defensively. printf %q handles embedded spaces.
    launch_cmd=$(printf 'cd %q && EXPERIMENT_RESULT_DIR=%q ./run.sh %s' \
        "$POD_REPO_DIR" "$result_dir" "$flags")

    _ssh "set -e
        if tmux has-session -t '$session' 2>/dev/null; then
            echo 'launch: session already running, no-op'
            exit 0
        fi
        mkdir -p '$POD_REPO_DIR/$result_dir'
        tmux new-session -d -s '$session' \"$launch_cmd\"
        echo 'launch: session started'
    "
}
```

- [ ] **Step 2: Modify `run.sh` to write markers**

Find the tail of `run.sh`. Currently the file ends after the git push block with an `exit` or fall-through. Locate the run of the python script (the block that sets `RUN_STATUS`, currently ~line 209):

```bash
set +e
"$PYTHON" "$PY_SCRIPT" "${RUN_ARGS[@]}"
RUN_STATUS=$?
set -e
```

**Immediately after** that block (before any git-push code), insert the marker write:

```bash
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
```

- [ ] **Step 3: Local sanity check of the marker block (no pod)**

Run (from the repo root, with env var set and a fake python command):

```bash
EXPERIMENT_RESULT_DIR=/tmp/runpod-agent-test bash -c '
    mkdir -p /tmp/runpod-agent-test
    RUN_STATUS=0
    if [[ -n "${EXPERIMENT_RESULT_DIR:-}" ]]; then
        mkdir -p "$EXPERIMENT_RESULT_DIR"
        if [[ $RUN_STATUS -eq 0 ]]; then
            touch "$EXPERIMENT_RESULT_DIR/.DONE"
        else
            touch "$EXPERIMENT_RESULT_DIR/.FAILED"
        fi
    fi
    ls -la "$EXPERIMENT_RESULT_DIR"
'
```

Expected: output shows `.DONE` file. Then re-run with `RUN_STATUS=1` inline to verify `.FAILED`. Clean up: `rm -rf /tmp/runpod-agent-test`.

- [ ] **Step 4: Commit**

```bash
git add runpod.sh run.sh
git commit -m "Add cmd_launch and EXPERIMENT_RESULT_DIR marker write in run.sh"
```

---

## Task 6: `cmd_tmux_alive` + `cmd_marker`

**Files:**
- Modify: `runpod.sh`

- [ ] **Step 1: Implement `cmd_tmux_alive`**

Add after `cmd_launch`:

```bash
cmd_tmux_alive() {
    _load_state; _refresh_ssh
    # Exit 0 if the tmux session exists AND its current pane is running
    # something other than bash. Session with only a bash prompt means the
    # python run finished (clean or via failure); the tail of run.sh drops
    # to `exec bash` per its current wrapper.
    local result
    result=$(_ssh "
        if ! tmux has-session -t 'gemma-recurrence' 2>/dev/null; then
            echo 'no-session'; exit 0
        fi
        cmd=\$(tmux display-message -t 'gemma-recurrence' -p '#{pane_current_command}' 2>/dev/null)
        echo \"cmd=\$cmd\"
    ")
    case "$result" in
        *"no-session"*) return 1 ;;
        *"cmd=bash"*)   return 1 ;;  # dropped to shell = run exited
        *"cmd="*)       return 0 ;;  # anything non-bash = still running
        *)              return 1 ;;
    esac
}
```

- [ ] **Step 2: Implement `cmd_marker`**

Add after `cmd_tmux_alive`:

```bash
cmd_marker() {
    [[ $# -eq 1 ]] || die "marker needs <result-dir>"
    local result_dir="$1"
    _load_state; _refresh_ssh
    [[ -n "$POD_REPO_DIR" ]] || die "no repo_dir in state; run bootstrap first"

    local marker
    marker=$(_ssh "
        if [ -f '$POD_REPO_DIR/$result_dir/.DONE' ]; then
            echo DONE
        elif [ -f '$POD_REPO_DIR/$result_dir/.FAILED' ]; then
            echo FAILED
        else
            echo NONE
        fi
    " | tr -d '[:space:]')

    if [[ "$marker" == "DONE" || "$marker" == "FAILED" ]]; then
        echo "$marker"
        return 0
    fi

    # No marker. Is tmux still alive?
    if cmd_tmux_alive 2>/dev/null; then
        echo "RUNNING"
    else
        echo "CRASHED"
    fi
}
```

- [ ] **Step 3: Manual sanity check (skip if no pod)**

With a pod up and bootstrapped:

```bash
./runpod.sh marker results/path_1_cot_tokens/plan1
```

Expected before launch: `RUNNING` if no tmux session (wait — actually with no tmux session and no marker, tmux-alive returns 1, so output is `CRASHED`). Adjust expectation:
- Pre-launch, no tmux session, no marker → `CRASHED`
- During run, tmux session active, no marker → `RUNNING`
- After clean run, `.DONE` present → `DONE`

This is the intended semantics: `CRASHED` means "no completion marker and nothing is running to produce one." The agent treats `CRASHED` before launch as a sign it should invoke launch, not a crash.

**Note for task 8 (cmd_watch):** `cmd_watch` should not call `cmd_marker` before it has called `cmd_launch`. The watch loop starts only after launch.

- [ ] **Step 4: Commit**

```bash
git add runpod.sh
git commit -m "Add cmd_tmux_alive and cmd_marker"
```

---

## Task 7: `cmd_sync_down` — rsync pull

**Files:**
- Modify: `runpod.sh`

- [ ] **Step 1: Add `rsync` to the dep check**

Find the top-of-file dep-check line (currently line 39):

```bash
need curl; need jq; need ssh; need scp
```

Replace with:

```bash
need curl; need jq; need ssh; need scp; need rsync
```

- [ ] **Step 2: Implement `cmd_sync_down`**

Add after `cmd_marker`:

```bash
cmd_sync_down() {
    [[ $# -eq 2 ]] || die "sync-down needs <remote-subdir> <local-dir>"
    local remote_sub="$1" local_dir="$2"
    _load_state; _refresh_ssh
    [[ -n "$POD_REPO_DIR" ]] || die "no repo_dir in state; run bootstrap first"

    mkdir -p "$local_dir"
    # rsync's --stats provides a transferred-bytes count we can parse. -az keeps
    # perms/mtimes and compresses over the wire. --partial lets resumable
    # transfers survive a mid-pull disconnect.
    local ssh_cmd="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i $SSH_KEY -p $POD_PORT"
    local src="root@$POD_HOST:$POD_REPO_DIR/$remote_sub/"
    local stats
    stats=$(rsync -az --partial --stats -e "$ssh_cmd" "$src" "$local_dir/" 2>&1) \
        || die "rsync failed: $stats"

    # Extract "Number of regular files transferred: N" — the first such line.
    local transferred
    transferred=$(awk -F': ' '/Number of regular files transferred/ {print $2; exit}' <<<"$stats")
    local bytes
    bytes=$(awk -F': ' '/Total transferred file size/ {print $2; exit}' <<<"$stats")
    echo "sync-down: ${transferred:-0} files, ${bytes:-0 bytes} from $remote_sub"
}
```

- [ ] **Step 3: Manual sanity check (skip if no pod)**

With a pod up, bootstrapped, and a result dir already populated:

```bash
./runpod.sh sync-down results/path_1_cot_tokens/plan1 results/path_1_cot_tokens/plan1
```

Expected: `sync-down: N files, <bytes> from results/path_1_cot_tokens/plan1`. Re-run immediately — expect `sync-down: 0 files` (nothing changed).

- [ ] **Step 4: Commit**

```bash
git add runpod.sh
git commit -m "Add cmd_sync_down (rsync-based pull)"
```

---

## Task 8: `cmd_watch` — main loop

**Files:**
- Modify: `runpod.sh`

Design note: `cmd_watch` reads `experiment.yaml` once at start, exports thresholds as env vars for `cmd_cost`, and loops on tick_seconds. All logging goes to stdout AND `./watch.log`. Exit codes: 0 = CLEAN, 2 = EMERGENCY, 3 = CRASHED.

- [ ] **Step 1: Implement `cmd_watch`**

Add after `cmd_sync_down`:

```bash
cmd_watch() {
    # Load config.
    local flags result_dir local_result_dir cap emergency max_hours tick
    flags=$(_read_config '.run.flags')
    result_dir=$(_read_config '.run.result_dir')
    local_result_dir=$(_read_config '.run.local_result_dir')
    [[ -z "$local_result_dir" ]] && local_result_dir="$result_dir"
    cap=$(_read_config '.budget.cap_usd')
    emergency=$(_read_config '.budget.emergency_usd')
    max_hours=$(_read_config '.budget.max_hours')
    tick=$(_read_config '.watch.tick_seconds')

    [[ -n "$flags" && -n "$result_dir" ]] || die "experiment.yaml missing required fields (run.flags, run.result_dir)"
    [[ -n "$cap" && -n "$emergency" && -n "$max_hours" && -n "$tick" ]] \
        || die "experiment.yaml missing required budget.{cap_usd,emergency_usd,max_hours} or watch.tick_seconds"

    export BUDGET_CAP_USD="$cap"
    export BUDGET_EMERGENCY_USD="$emergency"

    _load_state
    local started_at="$POD_STARTED_AT"
    local log="./watch.log"
    local warned=0
    local emergency_reason=""

    _log() { local msg="$*"; local ts; ts=$(date '+%H:%M:%S'); echo "[$ts] $msg" | tee -a "$log"; }
    _final_pull() {
        local budget_s="$1"
        timeout "$budget_s" "$0" sync-down "$result_dir" "$local_result_dir" 2>&1 | tee -a "$log" || \
            _log "WARN: final sync-down exceeded ${budget_s}s budget or failed"
    }

    _log "watch start: flags=[$flags] result_dir=$result_dir cap=\$$cap emergency=\$$emergency max_hours=$max_hours tick=${tick}s"

    while true; do
        # 1. cost + threshold checks.
        local cost_json
        cost_json=$("$0" cost 2>&1) || { _log "cost call failed; sleeping"; sleep "$tick"; continue; }
        local spent bal hard soft
        spent=$(jq -r '.spentUsd' <<<"$cost_json")
        bal=$(jq -r '.accountBalance' <<<"$cost_json")
        hard=$(jq -r '.hardStopTriggered' <<<"$cost_json")
        soft=$(jq -r '.softWarnTriggered' <<<"$cost_json")

        if [[ "$hard" == "true" ]]; then
            emergency_reason="account balance (\$$bal) below emergency threshold (\$$emergency)"
            break
        fi

        local wall_hours
        wall_hours=$(awk -v n="$(_unix_now)" -v s="$started_at" 'BEGIN{printf "%.2f", (n-s)/3600.0}')
        if awk -v w="$wall_hours" -v m="$max_hours" 'BEGIN{exit !(w>m)}'; then
            emergency_reason="wall time ${wall_hours}h exceeded max_hours ${max_hours}h"
            break
        fi

        if [[ "$soft" == "true" && "$warned" == "0" ]]; then
            _log "BUDGET WARN: spent=\$$spent exceeds cap=\$$cap; continuing"
            warned=1
        fi

        # 2. sync-down.
        "$0" sync-down "$result_dir" "$local_result_dir" 2>&1 | tee -a "$log"

        # 3. marker check.
        local marker
        marker=$("$0" marker "$result_dir" 2>/dev/null | tr -d '[:space:]')
        _log "spent=\$$spent bal=\$$bal wall=${wall_hours}h marker=$marker"

        case "$marker" in
            DONE|FAILED)
                _log "terminal: $marker — final pull + teardown"
                _final_pull 120
                "$0" down 2>&1 | tee -a "$log"
                _log "CLEAN exit (marker=$marker)"
                return 0
                ;;
            CRASHED)
                _log "CRASHED — tmux died with no marker; pulling and leaving pod UP for inspection"
                _final_pull 120
                _log "pod still up; run: ./runpod.sh ssh  |  ./runpod.sh logs  |  T2 triage prompt"
                return 3
                ;;
            RUNNING) ;;
            *) _log "WARN: unexpected marker output: [$marker]" ;;
        esac

        sleep "$tick"
    done

    # Reached only via emergency break.
    _log "EMERGENCY: $emergency_reason"
    _final_pull 60
    "$0" down 2>&1 | tee -a "$log"
    _log "EMERGENCY exit (reason: $emergency_reason)"
    printf '\a'  # terminal bell
    return 2
}
```

- [ ] **Step 2: Quick dry-run of config parsing**

Create a minimal fake config and verify `_read_config` works:

```bash
cat > /tmp/test-exp.yaml <<EOF
run:
  flags: "--script probe"
  result_dir: "results"
budget:
  cap_usd: 5.0
  emergency_usd: 0.50
  max_hours: 0.5
watch:
  tick_seconds: 60
git:
  url: "x"
  ref: "main"
EOF
EXPERIMENT_CONFIG=/tmp/test-exp.yaml bash -c '
    source ./runpod.sh  # RUNPOD_SHIM unset will try to dispatch — use the guard
' 2>&1 | head -5  # expect dispatch usage since we pass no subcommand
```

That test is ugly because `runpod.sh` dispatches. Better: just run `yq -r '.run.flags' /tmp/test-exp.yaml` to confirm yq can parse it. Expected: `--script probe`. Then `rm /tmp/test-exp.yaml`.

- [ ] **Step 3: Commit**

```bash
git add runpod.sh
git commit -m "Add cmd_watch: state-machine loop with budget + marker polling"
```

---

## Task 9: Dispatch wiring + help text

**Files:**
- Modify: `runpod.sh` (top-of-file comment block AND bottom dispatch case)

- [ ] **Step 1: Extend the help comment at the top of `runpod.sh`**

Find the Usage block (lines 7-16). Add new lines before the "Config" block:

```bash
#   ./runpod.sh cost                     # print cost/budget JSON (reads BUDGET_*_USD env)
#   ./runpod.sh bootstrap <url> [ref]    # clone or pull the repo on the pod, copy local .env up
#   ./runpod.sh launch "<flags>" <dir>   # start the experiment in a tmux session on the pod
#   ./runpod.sh tmux-alive               # exit 0 if experiment tmux session is alive
#   ./runpod.sh marker <dir>             # print DONE|FAILED|RUNNING|CRASHED
#   ./runpod.sh sync-down <remote> <local>  # rsync results from pod to laptop
#   ./runpod.sh watch                    # main loop: reads experiment.yaml; ticks until done
```

- [ ] **Step 2: Extend the dispatch case**

Find the dispatch (inside the `if [[ -z "${RUNPOD_SHIM:-}" ]]` block from Task 2). Add cases for the new subcommands:

```bash
    case "$sub" in
        up)         cmd_up "$@" ;;
        exec)       cmd_exec "$@" ;;
        run)        cmd_run "$@" ;;
        push)       cmd_push "$@" ;;
        pull)       cmd_pull "$@" ;;
        ssh)        cmd_ssh "$@" ;;
        status)     cmd_status "$@" ;;
        logs)       cmd_logs "$@" ;;
        down)       cmd_down "$@" ;;
        cost)       cmd_cost "$@" ;;
        bootstrap)  cmd_bootstrap "$@" ;;
        launch)     cmd_launch "$@" ;;
        tmux-alive) cmd_tmux_alive "$@" ;;
        marker)     cmd_marker "$@" ;;
        sync-down)  cmd_sync_down "$@" ;;
        watch)      cmd_watch "$@" ;;
        ""|help|-h|--help)
            sed -n '2,35p' "$0"; exit 0 ;;
        *) die "unknown subcommand: $sub (try --help)" ;;
    esac
```

Note the help line changed from `'2,25p'` to `'2,35p'` to include the new usage lines added in Step 1.

- [ ] **Step 3: Verify help output**

Run: `./runpod.sh help`
Expected: prints the Usage block including all new subcommands.

Run: `./runpod.sh` (no args)
Expected: same help output.

Run: `./runpod.sh nonsense`
Expected: `error: unknown subcommand: nonsense (try --help)`.

- [ ] **Step 4: Commit**

```bash
git add runpod.sh
git commit -m "Wire new subcommands into dispatch + update usage help"
```

---

## Task 10: Integration smoke test + operator runbook

**Files:**
- Create: `docs/superpowers/specs/runbook-runpod-agent.md` (operator quick-reference)

This task does NOT write new code. It runs the end-to-end flow against a real RunPod pod with a short experiment to confirm the agent works before trusting it with a multi-hour run. If you don't have RunPod access right now, stop after Step 1; the runbook alone is valuable.

- [ ] **Step 1: Write the runbook**

Create `docs/superpowers/specs/runbook-runpod-agent.md`:

```markdown
# RunPod Experiment Agent — Operator Runbook

## Prerequisites

- `$RUNPOD_API_KEY` in env (or `.env` you source before invoking).
- SSH keypair at `~/.ssh/id_ed25519` (override via `$SSH_KEY`).
- `yq`, `rsync`, `tmux`, `jq`, `curl` on your laptop.
- `experiment.yaml` at repo root (copy from `experiment.example.yaml`).

## Starting a run

1. Edit `experiment.yaml` — set `run.flags`, `run.result_dir`, `budget.*`.
2. Launch (use a Claude session for validation):

   ```
   claude -p "Read ./experiment.yaml. Validate required fields and bounds. Then run:
     ./runpod.sh up
     ./runpod.sh bootstrap <url> <ref>   (values from experiment.yaml)
     ./runpod.sh launch '<run.flags>' '<run.result_dir>'
   On success print: pod id, ssh cmd, costPerHr, hours of runway to budget.cap_usd
   (soft warn), hours to account-balance hitting emergency_usd (hard stop), and
   whether max_hours will fire first. Do NOT start the watch loop."
   ```

3. Start the watch loop in local tmux:

   ```
   tmux new -s rp-watch './runpod.sh watch'
   ```

   Detach: Ctrl+B D. Re-attach: `tmux attach -t rp-watch`. Log: `./watch.log`.

## On exit

- **exit 0 (CLEAN):** pod torn down, results under `./<local_result_dir>/`. Run Claude summary:

  ```
  claude -p "Read ./<local_result_dir>/. List produced files with sizes. For
  any *.json results, extract headline metrics (scan top-level keys). Write a
  short markdown summary suitable for a commit message. Then stage the new
  result files with 'git add'. Do NOT commit. Do NOT push."
  ```

- **exit 2 (EMERGENCY):** pod torn down, reason logged. Read `./watch.log`. No LLM needed — the numbers tell the story.

- **exit 3 (CRASHED):** pod still up. Triage via Claude:

  ```
  claude -p "runpod.sh watch exited with CRASHED (tmux dead, pod up). Run:
    ./runpod.sh logs
    ./runpod.sh exec 'dmesg | tail -100'
    ./runpod.sh exec 'nvidia-smi'
  Diagnose and recommend retry-same-pod / retry-fresh-pod / abandon.
  Do NOT run runpod.sh down."
  ```

  You decide what to do with the pod.

## Recovery

- Laptop rebooted mid-watch: `tmux attach -t rp-watch` if the tmux was on a durable host; otherwise re-run `./runpod.sh watch`. All state is on disk.
- Teardown from another machine: copy `.runpod-state.json` + `experiment.yaml` + ssh key, then `./runpod.sh down`.
```

- [ ] **Step 2: Commit the runbook**

```bash
git add docs/superpowers/specs/runbook-runpod-agent.md
git commit -m "Add operator runbook for runpod agent"
```

- [ ] **Step 3: End-to-end smoke test (requires live RunPod access)**

Create a smoke `experiment.yaml` designed to finish in under 10 minutes:

```yaml
run:
  flags: "--script probe --mode original"
  result_dir: "results"
pod:
  gpu_types: "NVIDIA GeForce RTX 4090"
  gpu_count: 1
  cloud_type: SECURE
  container_disk_gb: 50
  volume_gb: 50
git:
  url: "https://github.com/kasuncp/gemma4-retro-recurrence.git"
  ref: "main"
budget:
  cap_usd: 2.0
  emergency_usd: 0.50
  max_hours: 0.5
watch:
  tick_seconds: 120
```

Run:

```bash
./runpod.sh up
./runpod.sh bootstrap https://github.com/kasuncp/gemma4-retro-recurrence.git main
./runpod.sh launch "--script probe --mode original" "results"
tmux new -s rp-watch './runpod.sh watch'
```

Expected outcomes:
- watch prints tick lines every 120s with `marker=RUNNING`
- within ~5-15 min: `marker=DONE`, final pull, `runpod.sh down`, exit 0
- `.runpod-state.json` is gone after down; `results/` has at least one updated JSON locally
- `watch.log` has the tick history

If it hangs past 30 min without `marker=DONE`, ssh in and inspect: `./runpod.sh ssh` → `tmux attach -t gemma-recurrence`.

- [ ] **Step 4: Failure-injection smoke (optional, requires a running pod from Step 3 that hasn't finished)**

Interrupt the tmux session on the pod mid-run to exercise CRASHED:

```bash
./runpod.sh exec 'tmux kill-session -t gemma-recurrence || true'
```

Wait for the next watch tick. Expected: `marker=CRASHED`, final pull, exit 3, pod still up. Then:

```bash
./runpod.sh down   # clean up manually
```

- [ ] **Step 5: Commit any fixes discovered during smoke**

If the smoke test surfaces bugs, fix them inline and commit:

```bash
git add -u
git commit -m "Fix <specific bug found during smoke test>"
```

If it passes cleanly, no commit needed for this step.

---

## Self-Review Checklist (completed)

**Spec coverage:** Every subcommand listed in the spec's "Extended `runpod.sh` surface" table has a task (cost=T2, bootstrap=T4, launch=T5, tmux-alive=T6, marker=T6, sync-down=T7, watch=T8). Config schema is T1; marker write in run.sh is T5; state-file started_at is T3; dispatch wiring is T9; smoke test is T10. Claude touchpoints (T1/T2/T3 in the spec) are prompts, not code — documented in T10's runbook.

**Placeholder scan:** No TBD, TODO, or unspecified validation. Every bash block is complete and runnable. Error cases in the code (`die` calls) are intentional, not placeholders.

**Type consistency:** Function names match across tasks (`cmd_cost`, `cmd_bootstrap`, `cmd_launch`, `cmd_tmux_alive`, `cmd_marker`, `cmd_sync_down`, `cmd_watch`). State-file keys match: `started_at`, `repo_dir`, `publicIp`, `sshPort`, `id`. Env var names match: `EXPERIMENT_RESULT_DIR`, `BUDGET_CAP_USD`, `BUDGET_EMERGENCY_USD`, `RUNPOD_SHIM`, `EXPERIMENT_CONFIG`.
