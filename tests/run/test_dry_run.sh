#!/usr/bin/env bash
# Offline smoke test for run.sh's registry-driven dispatch.
#
# Runs ./run.sh --dry-run --script X for every registered script key and
# asserts that the dispatch summary names the right Python file, result
# root, and depth. Does NOT load Python, install deps, or touch git —
# takes <1s and is safe to run with the live experiment ticking.
#
# Keep this test in lockstep with the EXPERIMENT_* arrays in run.sh. When
# you add an experiment there, add a row to EXPECTED below.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PASS=0; FAIL=0
_pass() { PASS=$((PASS+1)); echo "  PASS: $1"; }
_fail() { FAIL=$((FAIL+1)); echo "  FAIL: $1"; echo "    got:  $2"; echo "    want: $3"; }

# Expected dispatch rows. One per registered script:
#   key | py_script | default_args | result_root | result_depth
EXPECTED=(
    "probe|ple_sanity_check.py|--mode ple-variants|results|flat"
    "path1|path1_cot_gate.py||results/path_1_cot_tokens|recursive"
    "path1-plan2|path1_length_and_sc.py||results/path_1_cot_tokens/plan2|recursive"
    "path1-plan4|path1_arc_easy.py||results/path_1_cot_tokens/plan4|recursive"
)

# Extract one `dry-run: <field>=<value>` line from run.sh's output.
_extract() {
    local out="$1" field="$2"
    grep "^dry-run: $field=" <<<"$out" | head -n1 | sed "s|^dry-run: $field=||"
}

for row in "${EXPECTED[@]}"; do
    IFS='|' read -r key py defaults root depth <<<"$row"
    echo "scenario: --script $key"

    out=$(cd "$REPO_ROOT" && ./run.sh --dry-run --script "$key" 2>&1) \
        || { _fail "$key: run.sh exited non-zero" "$out" "exit=0"; continue; }

    got_py=$(_extract "$out" py_script)
    got_defaults=$(_extract "$out" default_args)
    got_root=$(_extract "$out" result_root)
    got_depth=$(_extract "$out" result_depth)
    got_ok=$(_extract "$out" ok || true)  # 'ok' has no '=' — match via suffix
    # The 'ok' line is literally 'dry-run: ok', no '='. Detect separately.
    grep -q "^dry-run: ok$" <<<"$out" && ok_present=1 || ok_present=0

    [[ "$got_py"       == "$py"       ]] && _pass "$key py_script=$py"           || _fail "$key py_script"      "$got_py"       "$py"
    [[ "$got_defaults" == "$defaults" ]] && _pass "$key default_args=[$defaults]" || _fail "$key default_args" "$got_defaults" "$defaults"
    [[ "$got_root"     == "$root"     ]] && _pass "$key result_root=$root"        || _fail "$key result_root"  "$got_root"     "$root"
    [[ "$got_depth"    == "$depth"    ]] && _pass "$key result_depth=$depth"      || _fail "$key result_depth" "$got_depth"    "$depth"
    [[ "$ok_present"   == "1"         ]] && _pass "$key ok-line present"          || _fail "$key ok"           "$out"          "dry-run: ok"
done

# Forwarded args: extra flags after --script must pass through to effective_args.
echo "scenario: forwarded args"
out=$(cd "$REPO_ROOT" && ./run.sh --dry-run --script path1 --n 5 --batch-size 8 2>&1)
got_eff=$(_extract "$out" effective_args)
[[ "$got_eff" == "--n 5 --batch-size 8" ]] \
    && _pass "effective_args forwards extras" \
    || _fail "effective_args" "$got_eff" "--n 5 --batch-size 8"

# Unknown script must be rejected BEFORE the dry-run summary prints.
echo "scenario: unknown script rejected"
out=$(cd "$REPO_ROOT" && ./run.sh --dry-run --script frobnicate 2>&1) \
    && rc=0 || rc=$?
grep -q "unknown --script 'frobnicate'" <<<"$out" \
    && _pass "unknown script errors with message" \
    || _fail "unknown script error message" "$out" "contains: unknown --script 'frobnicate'"
[[ "$rc" -ne 0 ]] \
    && _pass "unknown script non-zero exit ($rc)" \
    || _fail "unknown script exit" "$rc" "non-zero"

echo
echo "Tests: $PASS passed, $FAIL failed"
[[ $FAIL -eq 0 ]]
