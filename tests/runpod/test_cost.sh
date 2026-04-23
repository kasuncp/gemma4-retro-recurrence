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

source "$REPO_ROOT/runpod.sh"

# Stub api(), state loader, and timing after source so they override sourced definitions.
api() {
    local path="$2"
    case "$path" in
        /pods/*) cat "$SCRIPT_DIR/fixtures/${POD_FIXTURE:-pod_normal}.json" ;;
        /user)   cat "$SCRIPT_DIR/fixtures/${USER_FIXTURE:-user_normal}.json" ;;
        *) echo "test stub: unexpected path $path" >&2; return 1 ;;
    esac
}
export -f api
_load_state() { POD_ID="abc123"; POD_HOST="1.2.3.4"; POD_PORT="40022"; POD_STARTED_AT="${POD_STARTED_AT:-1000000000}"; }
_unix_now() { echo "${FAKE_NOW:-1000003600}"; }  # default: 1 hour after start
export -f _load_state
export -f _unix_now

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
