# RunPod experiment agent — design

**Date:** 2026-04-23
**Topic:** Autonomous orchestration of RunPod experiment runs — extends `runpod.sh` + adds a local watch loop + three Claude touchpoints.

## Goal

Run a long (hours-long) experiment on a RunPod GPU with minimal babysitting. The agent provisions a pod, clones the repo, launches `run.sh` in a tmux session, periodically pulls partial results back to the laptop, and tears the pod down cleanly on completion — or emergency-tears-down if account credits or the per-run budget fall under a hard threshold.

## Non-goals

- Multi-pod orchestration. One experiment → one pod.
- Auto-retry after crash. The agent reports and stops; the operator decides whether to relaunch.
- Replacing `run.sh`. The existing tmux + deps + python-launch flow stays; only a one-line marker-file addition is needed.
- Pushing results from inside the pod. The pod's `git push` in `run.sh` is intentionally a no-op (no credentials on the pod); the agent pulls results via rsync and commits locally.

## Constraints and context

- Repo: `github.com/kasuncp/gemma4-retro-recurrence.git`.
- `run.sh` already wraps the experiment in tmux session `gemma-recurrence`, installs deps, and runs `ple_sanity_check.py` (probe) or `path1_cot_gate.py` / `path1_length_and_sc.py` (path1 variants).
- `runpod.sh` already has: `up`, `exec`, `run`, `push`, `pull`, `ssh`, `status`, `logs`, `down`, with state in `.runpod-state.json`.
- RunPod REST API exposes `costPerHr` per pod and `clientBalance` on the user endpoint. There is no per-pod "credits remaining" signal; the agent computes spend from `costPerHr × elapsed_since_up`.
- Results layout on the pod matches the local layout: `results/<path>/<plan>/…` (per the project's existing convention).

## Architecture overview

Three pieces, composed:

1. **Extended `runpod.sh`** — new subcommands that wrap the RunPod API and ssh/rsync into idempotent primitives (cost, bootstrap, launch, tmux-alive, marker, sync-down, watch).
2. **`runpod.sh watch` loop** — a pure-bash state machine that runs in a local tmux session on the operator's laptop. Ticks every ~7 min. No LLM in the loop.
3. **Three Claude touchpoints** — one-shot `claude -p` invocations at launch (validate config, provision, print cost-to-exhaustion), on crash (triage logs + recommend), and on clean completion (summarize results, stage for commit).

Configuration is file-driven: a single `experiment.yaml` at repo root (gitignored) is the source of truth for one agent run. It is re-read on every watch tick so the agent is resumable after laptop reboots or loop restarts — no state lives in Claude context between touchpoints.

## Extended `runpod.sh` surface

Existing subcommands unchanged. New ones:

| Subcommand | Behavior |
|---|---|
| `runpod.sh cost` | `GET /pods/{id}` for `costPerHr` + `GET /user` for `clientBalance`. Computes `spentUsd = costPerHr × hours_since_up` using `started_at` added to state file. Prints one JSON object: `{costPerHr, spentUsd, accountBalance, budgetCapUsd, budgetRemaining, hardStopTriggered, softWarnTriggered}`. `hardStopTriggered = (accountBalance < emergency_usd)` — account balance is the ONLY hard-stop signal. `softWarnTriggered = (spentUsd > budgetCapUsd)` — the per-run budget is a soft warn; the agent warns once and keeps running. |
| `runpod.sh bootstrap <git-url> [ref]` | On the pod: `git clone` (or `git -C <dir> fetch && git reset --hard <ref>`) into `/workspace/<repo>`, `scp` the local `.env` up to the repo dir, `chmod +x run.sh runpod.sh`. Idempotent. |
| `runpod.sh launch "<flags>" <result-dir>` | Starts tmux session `gemma-recurrence` on the pod running `./run.sh <flags>` inside the repo dir. Exports `EXPERIMENT_RESULT_DIR=<result-dir>` in the session so `run.sh` can write the marker without re-deriving the path from flags. No-op if the session already exists. |
| `runpod.sh tmux-alive` | Exits 0 if the pod has a tmux session `gemma-recurrence` with at least one pane whose command is not `bash` (i.e. the run is still executing). Exits 1 otherwise. |
| `runpod.sh marker <result-dir>` | `ssh` to check for `<result-dir>/.DONE` or `<result-dir>/.FAILED`. Prints one word: `DONE`, `FAILED`, `RUNNING` (no marker + tmux alive), or `CRASHED` (no marker + tmux dead). |
| `runpod.sh sync-down <remote-dir> <local-dir>` | `rsync -az --partial` over ssh. Prints a one-line summary of changed file count + byte count. |
| `runpod.sh watch` | The main loop. Reads `experiment.yaml` via `yq` (or a tiny bash parser if `yq` is unavailable on the laptop — decide in implementation). Runs until a terminal state; see §"Watch state machine". |

Existing subcommands that need a tiny change:

- `cmd_up` adds `started_at` (unix seconds from `date +%s`) to `.runpod-state.json` after the pod is ready. Used by `cmd_cost` to compute spend.

Single edit to `run.sh`:

- After the python script runs, if `EXPERIMENT_RESULT_DIR` is set in the environment, `touch "$EXPERIMENT_RESULT_DIR/.DONE"` on exit 0 and `touch "$EXPERIMENT_RESULT_DIR/.FAILED"` otherwise. The dir is exported by `runpod.sh launch`, so `run.sh` doesn't need to re-derive it from flags — this keeps the coupling in one place. If the env var is absent (`run.sh` invoked manually outside the agent), skip the marker write. Keeping the existing git-push block is fine; it's intentionally a no-op when the pod lacks credentials.

## `experiment.yaml` schema

Gitignored. One file per agent run. Fields:

```yaml
run:
  flags: "--script path1 --n 500 --batch-size 8"   # passed verbatim to run.sh
  result_dir: "results/path_1_cot_tokens/plan1"    # where .DONE appears; rsync source on pod
  local_result_dir: "results/path_1_cot_tokens/plan1"  # rsync destination; defaults to result_dir

pod:
  gpu_types: "NVIDIA GeForce RTX 4090"   # maps to POD_GPU_TYPES
  gpu_count: 1
  cloud_type: SECURE
  container_disk_gb: 50
  volume_gb: 50
  # image, name, ports default to runpod.sh defaults if omitted

git:
  url: "https://github.com/kasuncp/gemma4-retro-recurrence.git"
  ref: "main"

budget:
  cap_usd: 30.0           # soft stop: per-run spend cap; agent warns once and continues
  emergency_usd: 0.50     # hard stop: teardown when account balance (clientBalance) < this
  max_hours: 12           # hard stop on wall time

watch:
  tick_seconds: 420       # ~7 minutes
```

Validation rules (enforced by T1 and by `watch` on startup):

- `run.flags`, `run.result_dir`, `git.url` are required.
- `budget.emergency_usd` must be `>= 0.50` (below this the final pull is unlikely to finish in time).
- `budget.cap_usd > 0`.
- `budget.max_hours > 0`.
- `watch.tick_seconds` in [60, 1800].

## Watch state machine

`runpod.sh watch` loops indefinitely. One tick:

```
1. cost = `runpod.sh cost`
   if cost.hardStopTriggered:            -> EMERGENCY   # account balance < emergency_usd
   if wall_hours > budget.max_hours:     -> EMERGENCY
   if cost.softWarnTriggered and not warned:
     print "BUDGET WARN: spent=$X exceeds cap=$Y; continuing"
     set warned = true
     (no teardown — budget cap is soft; only account balance and max_hours are hard)

2. `runpod.sh sync-down <remote:result_dir> <local:local_result_dir>`
   Log changed-file count to watch.log.

3. state = `runpod.sh marker <result_dir>`
   DONE    -> CLEAN
   FAILED  -> CLEAN (with exit=1 noted)
   CRASHED -> CRASHED
   RUNNING -> sleep(tick_seconds); next tick
```

Hard-stop signals (teardown): account balance < `emergency_usd`, OR wall time > `max_hours`.
Soft signals (warn only): `spentUsd > cap_usd`. The agent prints one warning line per threshold-cross, then continues running until the experiment finishes or a hard-stop fires.

Terminal states:

| State | Actions | Exit code |
|---|---|---|
| CLEAN | `timeout 120 runpod.sh sync-down …` (final pull) → `runpod.sh down` → print summary → exit. | 0 |
| EMERGENCY | `timeout 60 runpod.sh sync-down …` (best-effort last pull) → `runpod.sh down` → print reason (which threshold fired, cost snapshot) → exit. | 2 |
| CRASHED | `timeout 120 runpod.sh sync-down …` → DO NOT teardown → print "pod still up; inspect with `runpod.sh logs` / `runpod.sh ssh`; run T2 prompt" → exit. | 3 |

Logging: everything goes to `./watch.log` (append) with timestamps. Each tick emits one line to stdout so `tmux attach -t rp-watch` shows live progress.

## Claude touchpoints

Three one-shot `claude -p` invocations. No persistent conversation.

### T1 — Launch (operator runs this to start)

Prompt template:

```
Read ./experiment.yaml. Validate required fields and bounds. Then run:
  ./runpod.sh up
  ./runpod.sh bootstrap <git.url> <git.ref>
  ./runpod.sh launch '<run.flags>' '<run.result_dir>'
On success, print:
  - pod id + ssh command (from .runpod-state.json)
  - current costPerHr and POD_GPU_TYPES that was actually allocated
  - hours of runway before budget.cap_usd is crossed (soft warn)
  - hours of runway before account balance hits budget.emergency_usd (hard stop)
  - whether max_hours will fire before either of the above
Do NOT start the watch loop. Exit.
```

Value: catches config typos, sanity-checks that `result_dir` matches the implied output path for `run.flags` (e.g. flags with `--script path1` imply `results/path_1_cot_tokens/planN`), gives a pre-launch cost gut-check.

### T2 — Crash triage (operator runs after `watch` exits 3)

Prompt template:

```
runpod.sh watch exited with CRASHED (tmux dead, pod up). Run:
  ./runpod.sh logs               # last 500 lines of /workspace/startup.log
  ./runpod.sh exec 'dmesg | tail -100'
  ./runpod.sh exec 'nvidia-smi'
  ./runpod.sh exec 'tail -200 /workspace/<repo>/run.log || true'
Diagnose: OOM, python exception, or host issue?
Recommend: retry on same pod, retry on fresh pod, or abandon.
Do NOT run runpod.sh down.
```

Value: log-grep + GPU-state check + one-paragraph verdict. The human decides the next move based on the diagnosis.

### T3 — Final summary (operator runs after `watch` exits 0)

Prompt template:

```
Read ./<local_result_dir>/. List produced files with sizes. For any *.json
results, extract headline metrics (scan top-level keys). Write a short
markdown summary suitable for a commit message. Then stage the new result
files with `git add`. Do NOT commit. Do NOT push.
```

Value: pre-written commit body + ready-to-review diff. The operator runs `git commit` and `git push` manually.

## Operator runbook

```
# One-time setup: put experiment.yaml at repo root. Add it to .gitignore.

# 1. Launch (interactive)
$ claude -p "<T1 prompt>"
# -> prints pod id, ssh cmd, cost-to-exhaustion estimates

# 2. Start watch in local tmux (survives terminal close)
$ tmux new -s rp-watch './runpod.sh watch'
# -> ~every 7 min: [HH:MM] spent=$X remain=$Y bal=$Z marker=RUNNING
# Detach: Ctrl+B D. Re-attach: tmux attach -t rp-watch.

# 3. When watch exits:
#    exit 0: run T3, review, commit, push manually.
#    exit 2: emergency teardown already fired. Read watch.log. No LLM.
#    exit 3: run T2 to diagnose; YOU choose retry/teardown.
```

## Resumability

All persistent state lives on disk: `.runpod-state.json` (pod id, ssh endpoint, `started_at`), `experiment.yaml` (config), `watch.log` (history), and the result files themselves. Nothing is held in Claude context between T1/T2/T3. Consequences:

- Laptop crash mid-run: re-attach `rp-watch` (if it was on a server tmux) or run `./runpod.sh watch` again (if local) — it re-reads state and picks up.
- Watch loop killed + restarted: no double-provisioning, no lost ticks, idempotent subcommands.
- Repo checkout on a different machine: copy `.runpod-state.json` + `experiment.yaml` + ssh key, resume.

## Testing strategy

- **Unit-level (bash):** mock `api()` in `runpod.sh` with a shim that reads canned JSON from fixture files; assert `cmd_cost` output shape and threshold logic for several scenarios (normal, soft-warn on spend > cap, hard-stop on balance < emergency, max_hours exceeded).
- **Integration (live):** a "smoke" experiment config that runs `run.sh --script probe` on a small model with `--n 4` or equivalent, budget cap $1, max_hours 0.25. End-to-end: T1 → watch → CLEAN → T3. Run once before trusting the agent with a real multi-hour run.
- **Failure injection:** manually kill the tmux session on the pod during a smoke run to exercise the CRASHED path. Manually set `emergency_usd` to something above the current balance to exercise EMERGENCY without waiting for real credit exhaustion.

## Out of scope for this spec

- A desktop notification system beyond the terminal bell.
- Multi-experiment queuing.
- Automatic relaunch on crash (explicitly rejected in brainstorm Q6).
- An LLM-in-the-loop tick (explicitly rejected in Q5).

## Open questions

None blocking implementation. One low-priority decision for the writing-plans phase: whether to vendor a tiny bash YAML parser or require `yq` on the laptop. Current lean: try `yq` first, fall back to a 20-line awk parser for just the fields we use.
