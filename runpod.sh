#!/usr/bin/env bash
# runpod.sh — thin wrapper around RunPod REST API + SSH for spin-up / exec / tear-down.
#
# Requires: curl, jq, ssh, scp. An ssh keypair at $SSH_KEY (default ~/.ssh/id_ed25519).
# Auth: export RUNPOD_API_KEY=... (get one from https://console.runpod.io/user/settings)
#
# Usage:
#   ./runpod.sh go                       # one-shot: up + bootstrap + launch + detached watcher (all from experiment.yaml)
#   ./runpod.sh up                       # create a pod, wait for SSH, save id to state file
#   ./runpod.sh exec "cmd"               # run a single command on the current pod
#   ./runpod.sh run script.sh            # upload and execute a local script
#   ./runpod.sh push <local> <remote>    # scp into pod
#   ./runpod.sh pull <remote> <local>    # scp out of pod
#   ./runpod.sh ssh                      # open an interactive shell
#   ./runpod.sh status                   # show current pod info
#   ./runpod.sh logs                     # tail /workspace/startup.log if present
#   ./runpod.sh down                     # terminate (delete) the pod
#   ./runpod.sh cost                     # print cost/budget JSON (reads BUDGET_*_USD env)
#   ./runpod.sh bootstrap <url> [ref]    # clone or pull the repo on the pod, copy local .env up
#   ./runpod.sh launch "<flags>" <dir>   # start the experiment in a tmux session on the pod
#   ./runpod.sh tmux-alive               # exit 0 if experiment tmux session is alive
#   ./runpod.sh marker <dir>             # print DONE|FAILED|RUNNING|CRASHED
#   ./runpod.sh sync-down <remote> <local>  # rsync results from pod to laptop
#   ./runpod.sh watch                    # main loop: reads experiment.yaml; ticks until done
#
# Config (env vars with defaults — override inline or in a .env file you source):
# Auto-source a sibling .env so `./runpod.sh go` works as a single command
# without requiring the caller to `source .env` first.
_rp_script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -f "$_rp_script_dir/.env" ]]; then
    set -a; . "$_rp_script_dir/.env"; set +a
fi
unset _rp_script_dir
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY (directly or in sibling .env)}"
: "${POD_NAME:=probe-$(date +%Y%m%d-%H%M%S)}"
: "${POD_IMAGE:=runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
: "${POD_GPU_TYPES:=NVIDIA GeForce RTX 4090}"   # comma-separated, tried in order
: "${POD_GPU_COUNT:=1}"
: "${POD_CLOUD_TYPE:=SECURE}"                    # SECURE | COMMUNITY
: "${POD_CONTAINER_DISK_GB:=50}"
: "${POD_VOLUME_GB:=50}"
: "${POD_VOLUME_MOUNT:=/workspace}"
: "${POD_NETWORK_VOLUME_ID:=}"                   # optional, for persistent workspace
: "${POD_DATA_CENTERS:=}"                        # optional, comma-separated; empty = any
: "${POD_PORTS:=22/tcp,8888/http}"               # 22/tcp is required for this script
: "${POD_INTERRUPTIBLE:=false}"                  # true = spot pricing
# Can't use `:={}}` — bash reads the default up to the first `}`, stripping
# the trailing `}` off any braced literal. Default unconditionally below.
: "${POD_ENV_JSON:=}"
[[ -z "$POD_ENV_JSON" ]] && POD_ENV_JSON='{}'  # extra env as JSON object, e.g. '{"WANDB_API_KEY":"..."}'
: "${SSH_KEY:=$HOME/.ssh/id_ed25519}"
: "${STATE_FILE:=.runpod-state.json}"
: "${API_BASE:=https://rest.runpod.io/v1}"

die() { echo "error: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null || die "missing dependency: $1"; }
need curl; need jq; need ssh; need scp; need rsync

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

api() {
    # api METHOD PATH [json-body]
    local method="$1" path="$2" body="${3:-}"
    local tmp; tmp=$(mktemp)
    local http_code
    if [[ -n "$body" ]]; then
        http_code=$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "$body" "$API_BASE$path")
    else
        http_code=$(curl -sS -o "$tmp" -w '%{http_code}' -X "$method" \
            -H "Authorization: Bearer $RUNPOD_API_KEY" \
            "$API_BASE$path")
    fi
    if [[ "$http_code" -ge 400 ]]; then
        echo "HTTP $http_code from $method $path:" >&2
        cat "$tmp" >&2; echo >&2
        rm -f "$tmp"; return 1
    fi
    cat "$tmp"; rm -f "$tmp"
}

# GraphQL fallback for endpoints not exposed by REST v1 (e.g. account balance).
# Usage: _graphql '<query string>'
_graphql() {
    local query="$1"
    local body; body=$(jq -cn --arg q "$query" '{query: $q}')
    local tmp; tmp=$(mktemp)
    local http_code
    http_code=$(curl -sS -o "$tmp" -w '%{http_code}' -X POST \
        -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "$body" \
        "https://api.runpod.io/graphql")
    if [[ "$http_code" -ge 400 ]]; then
        echo "HTTP $http_code from GraphQL:" >&2
        cat "$tmp" >&2; echo >&2
        rm -f "$tmp"; return 1
    fi
    if jq -e '.errors' >/dev/null 2>&1 <"$tmp"; then
        echo "GraphQL errors:" >&2
        jq '.errors' "$tmp" >&2
        rm -f "$tmp"; return 1
    fi
    cat "$tmp"; rm -f "$tmp"
}

# ---------- subcommands ----------

cmd_up() {
    [[ -f "$SSH_KEY.pub" ]] || die "public key not found at $SSH_KEY.pub (generate with: ssh-keygen -t ed25519)"
    local pubkey; pubkey=$(cat "$SSH_KEY.pub")

    # Build env object: merge POD_ENV_JSON with PUBLIC_KEY (required for SSH).
    local env_obj
    env_obj=$(jq -c --arg pk "$pubkey" '. + {PUBLIC_KEY: $pk}' <<<"$POD_ENV_JSON") \
        || die "POD_ENV_JSON is not valid JSON: $POD_ENV_JSON"

    # Turn comma lists into JSON arrays.
    local gpu_arr ports_arr dc_arr
    gpu_arr=$(jq -R -c 'split(",") | map(gsub("^\\s+|\\s+$"; ""))' <<<"$POD_GPU_TYPES")
    ports_arr=$(jq -R -c 'split(",") | map(gsub("^\\s+|\\s+$"; ""))' <<<"$POD_PORTS")
    if [[ -n "$POD_DATA_CENTERS" ]]; then
        dc_arr=$(jq -R -c 'split(",") | map(gsub("^\\s+|\\s+$"; ""))' <<<"$POD_DATA_CENTERS")
    else
        dc_arr="null"
    fi

    local body
    body=$(jq -n \
        --arg name "$POD_NAME" \
        --arg image "$POD_IMAGE" \
        --arg cloud "$POD_CLOUD_TYPE" \
        --arg mount "$POD_VOLUME_MOUNT" \
        --arg netvol "$POD_NETWORK_VOLUME_ID" \
        --argjson gpuCount "$POD_GPU_COUNT" \
        --argjson containerDisk "$POD_CONTAINER_DISK_GB" \
        --argjson volumeGb "$POD_VOLUME_GB" \
        --argjson interruptible "$POD_INTERRUPTIBLE" \
        --argjson gpuTypes "$gpu_arr" \
        --argjson ports "$ports_arr" \
        --argjson dataCenters "$dc_arr" \
        --argjson env "$env_obj" \
        '{
            name: $name,
            imageName: $image,
            cloudType: $cloud,
            computeType: "GPU",
            gpuCount: $gpuCount,
            gpuTypeIds: $gpuTypes,
            gpuTypePriority: "availability",
            containerDiskInGb: $containerDisk,
            volumeInGb: $volumeGb,
            volumeMountPath: $mount,
            ports: $ports,
            interruptible: $interruptible,
            env: $env
        }
        + (if $netvol != "" then {networkVolumeId: $netvol} else {} end)
        + (if $dataCenters != null then {dataCenterIds: $dataCenters, dataCenterPriority: "availability"} else {} end)')

    echo "creating pod: $POD_NAME ..."
    local resp; resp=$(api POST /pods "$body") || die "pod creation failed (common causes: no capacity for selected GPU/region, invalid network volume id)"
    local pod_id; pod_id=$(jq -r '.id' <<<"$resp")
    [[ "$pod_id" != "null" && -n "$pod_id" ]] || die "no pod id in response: $resp"
    local started_at; started_at=$(_unix_now)
    echo "$resp" | jq --argjson ts "$started_at" '{id, name, costPerHr, desiredStatus} + {started_at: $ts}' > "$STATE_FILE"
    echo "pod id: $pod_id — waiting for SSH..."

    # Poll until port 22 is mapped + reachable.
    local host port attempts=60
    while (( attempts-- > 0 )); do
        sleep 5
        local info; info=$(api GET "/pods/$pod_id") || continue
        host=$(jq -r '.publicIp // empty' <<<"$info")
        port=$(jq -r '.portMappings["22"] // empty' <<<"$info")
        if [[ -n "$host" && -n "$port" ]]; then
            if ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
                   -o ConnectTimeout=5 -o BatchMode=yes \
                   -i "$SSH_KEY" -p "$port" "root@$host" true 2>/dev/null; then
                jq --arg h "$host" --arg p "$port" '. + {publicIp: $h, sshPort: ($p|tonumber)}' \
                    "$STATE_FILE" > "$STATE_FILE.tmp" && mv "$STATE_FILE.tmp" "$STATE_FILE"
                echo "ready: ssh -p $port root@$host"
                return 0
            fi
        fi
        echo "  still waiting... (host=${host:-?} port=${port:-?})"
    done
    die "timed out waiting for SSH"
}

_load_state() {
    [[ -f "$STATE_FILE" ]] || die "no state file ($STATE_FILE). Run 'up' first."
    POD_ID=$(jq -r '.id' "$STATE_FILE")
    POD_HOST=$(jq -r '.publicIp // empty' "$STATE_FILE")
    POD_PORT=$(jq -r '.sshPort // empty' "$STATE_FILE")
    POD_STARTED_AT=$(jq -r '.started_at // empty' "$STATE_FILE")
    POD_REPO_DIR=$(jq -r '.repo_dir // empty' "$STATE_FILE")
    [[ -n "$POD_ID" ]] || die "malformed state file"
}

_refresh_ssh() {
    # Re-query the API if host/port missing (e.g., after stop/start).
    if [[ -z "$POD_HOST" || -z "$POD_PORT" ]]; then
        local info; info=$(api GET "/pods/$POD_ID")
        POD_HOST=$(jq -r '.publicIp // empty' <<<"$info")
        POD_PORT=$(jq -r '.portMappings["22"] // empty' <<<"$info")
        [[ -n "$POD_HOST" && -n "$POD_PORT" ]] || die "pod has no SSH endpoint yet"
    fi
}

_ssh() {
    _load_state; _refresh_ssh
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -i "$SSH_KEY" -p "$POD_PORT" "root@$POD_HOST" "$@"
}

cmd_exec() {
    [[ $# -ge 1 ]] || die "exec needs a command string"
    _ssh "$*"
}

cmd_run() {
    [[ $# -ge 1 && -f "$1" ]] || die "run needs a local script path"
    local script="$1"; shift
    _load_state; _refresh_ssh
    local remote="/tmp/$(basename "$script").$$"
    scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -i "$SSH_KEY" -P "$POD_PORT" "$script" "root@$POD_HOST:$remote"
    _ssh "chmod +x $remote && $remote $*"
}

cmd_push() {
    [[ $# -eq 2 ]] || die "push needs <local> <remote>"
    _load_state; _refresh_ssh
    scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -i "$SSH_KEY" -P "$POD_PORT" "$1" "root@$POD_HOST:$2"
}

cmd_pull() {
    [[ $# -eq 2 ]] || die "pull needs <remote> <local>"
    _load_state; _refresh_ssh
    scp -r -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -i "$SSH_KEY" -P "$POD_PORT" "root@$POD_HOST:$1" "$2"
}

cmd_ssh() { _ssh; }

cmd_status() {
    _load_state
    api GET "/pods/$POD_ID" | jq '{id, name, desiredStatus, costPerHr, adjustedCostPerHr, publicIp, portMappings, machine: .machine.dataCenterId}'
}

cmd_cost() {
    _load_state
    [[ -n "$POD_STARTED_AT" ]] || die "state file has no started_at; was the pod brought up with the current runpod.sh?"
    local pod user
    pod=$(api GET "/pods/$POD_ID") || die "pod GET failed"
    # /v1/user doesn't exist on the REST API; use GraphQL myself{clientBalance}.
    user=$(_graphql 'query { myself { clientBalance } }') || die "user GraphQL query failed"

    local cost_per_hr balance now elapsed_s
    cost_per_hr=$(jq -r '.costPerHr // 0' <<<"$pod")
    balance=$(jq -r '.data.myself.clientBalance // 0' <<<"$user")
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
        # runpod/pytorch images don't ship tmux or rsync; cmd_launch needs
        # tmux and cmd_sync_down needs rsync. Install once per pod.
        missing=''
        for pkg in tmux rsync; do
            command -v \$pkg >/dev/null 2>&1 || missing=\"\$missing \$pkg\"
        done
        if [ -n \"\${missing# }\" ]; then
            echo \"installing:\$missing\"
            DEBIAN_FRONTEND=noninteractive apt-get update -qq >/dev/null
            DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \$missing >/dev/null
        fi
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

cmd_tmux_alive() {
    _load_state; _refresh_ssh
    # Alive = tmux session exists AND the pane's bash has a child process.
    # An idle bash prompt has no children; any running subprocess (python,
    # pt_main_thread, etc.) means the experiment is still going. Previously
    # we checked #{pane_current_command}, but that stays 'bash' even while
    # bash's children do real work, so every tick falsely reported CRASHED.
    local result
    result=$(_ssh "
        if ! tmux has-session -t 'gemma-recurrence' 2>/dev/null; then
            echo 'no-session'; exit 0
        fi
        pane_pid=\$(tmux display-message -t 'gemma-recurrence' -p '#{pane_pid}' 2>/dev/null)
        if [ -z \"\$pane_pid\" ]; then
            echo 'no-pid'; exit 0
        fi
        if pgrep -P \"\$pane_pid\" >/dev/null 2>&1; then
            echo 'running'
        else
            echo 'idle'
        fi
    ")
    case "$result" in
        *"no-session"*) return 1 ;;
        *"no-pid"*)     return 1 ;;
        *"idle"*)       return 1 ;;  # bash prompt with no children = run exited
        *"running"*)    return 0 ;;  # any child = still running
        *)              return 1 ;;
    esac
}

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

cmd_logs() { _ssh "tail -n 200 -f /workspace/startup.log"; }

cmd_go() {
    # Single-command orchestration: up + bootstrap + launch + detached
    # watcher. All parameters come from experiment.yaml; no args.
    local url ref flags result_dir
    url=$(_read_config '.git.url')
    ref=$(_read_config '.git.ref')
    flags=$(_read_config '.run.flags')
    result_dir=$(_read_config '.run.result_dir')
    [[ -n "$url" && -n "$ref" && -n "$flags" && -n "$result_dir" ]] \
        || die "experiment.yaml missing required fields (git.url, git.ref, run.flags, run.result_dir)"

    # Refuse to stack. Explicit teardown is safer than silently orphaning.
    [[ ! -f "$STATE_FILE" ]] \
        || die "$STATE_FILE exists; a pod is already registered. './runpod.sh down' first (or delete the stale state file)."
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t rp-watch 2>/dev/null; then
        die "tmux session 'rp-watch' already running. 'tmux kill-session -t rp-watch' first."
    fi
    if command -v screen >/dev/null 2>&1 && screen -ls 2>/dev/null | grep -q '\.rp-watch[[:space:]]'; then
        die "screen session 'rp-watch' already running. 'screen -S rp-watch -X quit' first."
    fi

    "$0" up
    "$0" bootstrap "$url" "$ref"
    "$0" launch "$flags" "$result_dir"

    # Detach the watcher. Prefer tmux; fall back to the pre-installed macOS
    # screen. The detached session inherits this shell's env + CWD, so the
    # watch loop finds experiment.yaml and .runpod-state.json in the same
    # directory the user invoked us from.
    local self; self="$(cd "$(dirname "$0")" && pwd -P)/$(basename "$0")"
    if command -v tmux >/dev/null 2>&1; then
        tmux new-session -d -s rp-watch "$self watch"
        printf '\nGO complete. Watcher detached in tmux session rp-watch.\n'
        printf '  attach:  tmux attach -t rp-watch\n'
        printf '  tail:    tail -f ./watch.log\n'
        printf '  stop:    ./runpod.sh down  &&  tmux kill-session -t rp-watch\n'
    elif command -v screen >/dev/null 2>&1; then
        screen -dmS rp-watch bash -c "exec $(printf '%q' "$self") watch"
        printf '\nGO complete. Watcher detached in screen session rp-watch.\n'
        printf '  attach:  screen -r rp-watch\n'
        printf '  tail:    tail -f ./watch.log\n'
        printf '  stop:    ./runpod.sh down  &&  screen -S rp-watch -X quit\n'
    else
        echo "WARN: launch complete but no tmux or screen installed;"
        echo "      run './runpod.sh watch' in another shell yourself."
    fi
}

cmd_down() {
    _load_state
    echo "terminating pod $POD_ID..."
    api DELETE "/pods/$POD_ID" >/dev/null && echo "deleted" && rm -f "$STATE_FILE"
}

# ---------- dispatch ----------
if [[ -z "${RUNPOD_SHIM:-}" ]]; then
    sub="${1:-}"; shift || true
    case "$sub" in
        go)         cmd_go "$@" ;;
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
            sed -n '2,25p' "$0"; exit 0 ;;
        *) die "unknown subcommand: $sub (try --help)" ;;
    esac
fi