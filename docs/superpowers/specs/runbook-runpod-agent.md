# RunPod Experiment Agent — Operator Runbook

## Prerequisites

- `$RUNPOD_API_KEY` in env (or `.env` you source before invoking).
- SSH keypair at `~/.ssh/id_ed25519` (override via `$SSH_KEY`).
- `yq`, `rsync`, `tmux`, `jq`, `curl` on your laptop.
- `experiment.yaml` at repo root (copy from `experiment.example.yaml`).

## Starting a run

1. Edit `experiment.yaml` — set `run.flags`, `run.result_dir`, `budget.*`.
2. Launch. One command does everything (up + bootstrap + launch + detached watcher, all parameters from `experiment.yaml`; `.env` is auto-sourced):

   ```
   ./runpod.sh go
   ```

   The watcher runs in a detached tmux session (falls back to `screen` if tmux isn't installed). Attach with `tmux attach -t rp-watch` (Ctrl+B D to detach) or `screen -r rp-watch`. Live log: `tail -f ./watch.log`.

3. Optional — validate with a Claude session instead of reading the numbers yourself:

   ```
   claude -p "Read ./experiment.yaml. Validate required fields and bounds. Then run:
     ./runpod.sh go
   On success print: pod id, ssh cmd, costPerHr, hours of runway to budget.cap_usd
   (soft warn), hours to account-balance hitting emergency_usd (hard stop), and
   whether max_hours will fire first."
   ```

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
