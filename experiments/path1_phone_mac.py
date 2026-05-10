"""Path 1 plan 8 v2 - Mac side of phone-class on-device measurement.

Runs the C2-Mac and A3-Mac cells defined in plans/path_1_cot_tokens/plan8.md
on a MacBook Pro M3 using MLX-LM with the 4-bit quantized
google/gemma-4-E2B-it weights.

Why this script exists alongside the legacy experiments/path1_plan8.py:
    The legacy script measures on a 3090 via llama.cpp/GGUF as a phone-proxy.
    This script measures on actual Apple silicon. Per plan section 5, prompt
    builders, exemplars, GSM8K loader, and the answer-extraction regex are
    imported from the legacy script (not re-derived) so byte-equivalence
    sanity check 5 has teeth.

What this script does NOT do:
    - It does not run the iPhone cells. The iPhone runs the Path1Bench
      Xcode app which loads a prompts.jsonl emitted here. See
      Path1Bench/README.md.
    - It does not perform the MLX 4-bit weight conversion. Run once:
          python -m mlx_lm.convert --hf-path google/gemma-4-E2B-it -q --q-bits 4 \
              --mlx-path ./mlx_models/gemma-4-E2B-it-mlx-q4
    - It does not download GSM8K. The legacy load_problems calls
      datasets.load_dataset("openai/gsm8k").

Power measurement requires powermetrics, which needs root. Run:
    sudo -E python experiments/path1_phone_mac.py --cells C2-Mac A3-Mac
Without sudo, the script proceeds with timing+thermal only and joules=null.

Typical CLI:
    # 1) sanity: model loads, generates one problem
    python experiments/path1_phone_mac.py --smoke

    # 2) emit prompts.jsonl for the iOS app to bundle
    python experiments/path1_phone_mac.py --emit-prompts

    # 3) sanity check 5: byte-equivalence vs legacy prompts (Experiment 5 jsonl)
    python experiments/path1_phone_mac.py --byte-equivalence

    # 4) sanity gate 1: quantization accuracy parity (no power needed)
    python experiments/path1_phone_mac.py --accuracy-only --cells C2-Mac

    # 5) main run: timing + power + thermal for both cells
    sudo -E python experiments/path1_phone_mac.py --cells C2-Mac A3-Mac

    # 6) sustained-thermal probe (5 minutes continuous gen)
    sudo -E python experiments/path1_phone_mac.py --sustained --cells C2-Mac
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import plistlib
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Allow `python experiments/path1_phone_mac.py` from project root by ensuring
# the project root is on sys.path before importing the legacy module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Plan 8 v2, section 5: import (do not re-derive) from the legacy script.
from experiments.path1_plan8 import (  # noqa: E402
    EXEMPLARS_COT,
    EXEMPLARS_DIRECT,
    FALLBACK_RE,
    GOLD_RE,
    MODEL_ID,
    build_a3_prompt,
    build_c2_prompt,
    extract_answer,
    load_problems,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MLX_DIR = _PROJECT_ROOT / "mlx_models" / "gemma-4-E2B-it-mlx-q4"
RESULTS_DIR = _PROJECT_ROOT / "results" / "path_1_cot_tokens" / "plan8"
CELLS_PHONE_DIR = RESULTS_DIR / "cells_phone"
PROMPTS_JSONL = CELLS_PHONE_DIR / "prompts.jsonl"
# Mirror copy under the iOS bundle so XcodeGen picks it up as a Resource.
IOS_RESOURCES_DIR = _PROJECT_ROOT / "Path1Bench" / "Path1Bench" / "Resources"
IOS_PROMPTS_JSONL = IOS_RESOURCES_DIR / "prompts.jsonl"

CELL_C2 = "C2-Mac"
CELL_A3 = "A3-Mac"
ALL_CELLS = (CELL_C2, CELL_A3)
DEVICE_TAG = "mac_m3"

DEFAULT_N = 50
MAX_NEW_TOKENS = 512
INTER_PROBLEM_COOLDOWN_S = 30
POWERMETRICS_INTERVAL_MS = 100  # 10 Hz
SUSTAINED_DEFAULT_DURATION_S = 300  # 5 minutes
SUSTAINED_TPS_BUCKET_S = 1.0


# ---------------------------------------------------------------------------
# Power + thermal sampling via powermetrics
# ---------------------------------------------------------------------------

def _is_root() -> bool:
    return hasattr(os, "geteuid") and os.geteuid() == 0


def _epoch_from_powermetrics_timestamp(ts: Any) -> float | None:
    """powermetrics plist 'timestamp' is a datetime in local TZ. Convert to POSIX."""
    if isinstance(ts, _dt.datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_dt.timezone.utc).astimezone()
        return ts.timestamp()
    return None


@dataclass
class PowerSample:
    t_epoch: float
    cpu_w: float
    gpu_w: float
    ane_w: float
    peak_die_temp_c: float | None  # max across reported SMC sensors


@dataclass
class PowerCapture:
    """Manages a single powermetrics subprocess for the lifetime of a run.

    On stop(), reads back the plist log and exposes a list of PowerSample.
    Per-problem joules + peak temp are computed by joining samples to
    [t_start, t_end] windows recorded by the caller.
    """
    log_path: Path
    proc: subprocess.Popen | None = None
    available: bool = False
    samples: list[PowerSample] = field(default_factory=list)

    def start(self) -> bool:
        if shutil.which("powermetrics") is None:
            print("[power] powermetrics not on PATH; skipping power capture")
            return False
        if not _is_root():
            print("[power] not root - skipping power capture (rerun with `sudo -E`)")
            return False
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "powermetrics",
            "--samplers", "cpu_power,gpu_power,ane_power,smc",
            "-i", str(POWERMETRICS_INTERVAL_MS),
            "-f", "plist",
        ]
        f = open(self.log_path, "wb")
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"[power] failed to spawn powermetrics: {exc}")
            return False
        # Give powermetrics a moment to spin up so the first samples are valid.
        time.sleep(1.0)
        self.available = True
        return True

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def parse_log(self) -> list[PowerSample]:
        if not self.available or not self.log_path.exists():
            return []
        data = self.log_path.read_bytes()
        # powermetrics -f plist emits multiple plist documents separated by NUL.
        out: list[PowerSample] = []
        for chunk in data.split(b"\0"):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                plist = plistlib.loads(chunk)
            except Exception:
                continue
            ts_epoch = _epoch_from_powermetrics_timestamp(plist.get("timestamp"))
            if ts_epoch is None:
                continue
            cpu_mw = float(plist.get("processor", {}).get("package_power_mw", 0.0))
            gpu_mw = float(plist.get("gpu", {}).get("package_power_mw", 0.0))
            ane_mw = float(plist.get("processor", {}).get("ane_energy", 0.0)) or \
                float(plist.get("processor", {}).get("ane_power_mw", 0.0))
            # SMC peak die temp across all reported sensors (when available)
            peak_temp: float | None = None
            smc = plist.get("smc", {})
            if isinstance(smc, dict):
                for k, v in smc.items():
                    if not isinstance(v, (int, float)):
                        continue
                    if "temperature" in k.lower() or k.lower().endswith("_c"):
                        peak_temp = max(peak_temp or v, float(v))
            out.append(PowerSample(
                t_epoch=ts_epoch,
                cpu_w=cpu_mw / 1000.0,
                gpu_w=gpu_mw / 1000.0,
                ane_w=ane_mw / 1000.0,
                peak_die_temp_c=peak_temp,
            ))
        out.sort(key=lambda s: s.t_epoch)
        self.samples = out
        return out

    def joules_in_window(self, t0_epoch: float, t1_epoch: float) -> float | None:
        """Trapezoidal-integrate total package power over [t0, t1]. Returns None if no samples."""
        if not self.samples:
            return None
        win = [s for s in self.samples if t0_epoch <= s.t_epoch <= t1_epoch]
        if len(win) < 2:
            return None
        joules = 0.0
        for a, b in zip(win[:-1], win[1:]):
            dt = b.t_epoch - a.t_epoch
            if dt <= 0 or dt > 1.0:
                continue  # gap or duplicate
            avg_w = (a.cpu_w + a.gpu_w + a.ane_w + b.cpu_w + b.gpu_w + b.ane_w) / 2.0
            joules += avg_w * dt
        return joules

    def peak_temp_in_window(self, t0_epoch: float, t1_epoch: float) -> float | None:
        win = [s.peak_die_temp_c for s in self.samples
               if t0_epoch <= s.t_epoch <= t1_epoch and s.peak_die_temp_c is not None]
        return max(win) if win else None


# ---------------------------------------------------------------------------
# MLX-LM wrapper
# ---------------------------------------------------------------------------

@dataclass
class MLXRunner:
    weights_path: Path
    model: Any = None
    tokenizer: Any = None

    def load(self) -> None:
        try:
            from mlx_lm import load as _mlx_load  # type: ignore
        except ImportError as exc:
            raise SystemExit(
                "mlx_lm is not installed. Run:\n"
                "    pip install mlx-lm\n"
                f"(import failed: {exc})"
            ) from exc
        if not self.weights_path.exists():
            raise SystemExit(
                f"MLX weights not found at {self.weights_path}.\n"
                "Convert with:\n"
                f"    python -m mlx_lm.convert --hf-path {MODEL_ID} -q --q-bits 4 "
                f"--mlx-path {self.weights_path}"
            )
        print(f"[mlx] loading {self.weights_path}")
        self.model, self.tokenizer = _mlx_load(str(self.weights_path))
        print("[mlx] model + tokenizer loaded")

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def greedy_generate(self, prompt: str, max_tokens: int) -> tuple[str, int, int, float]:
        """Returns (completion_text, n_gen_tokens, n_prompt_tokens, gen_secs)."""
        from mlx_lm import stream_generate  # type: ignore

        t0 = time.perf_counter()
        pieces: list[str] = []
        n_gen = 0
        n_prompt = 0
        for resp in stream_generate(self.model, self.tokenizer, prompt, max_tokens=max_tokens):
            pieces.append(getattr(resp, "text", "") or "")
            n_gen = int(getattr(resp, "generation_tokens", n_gen + 1))
            n_prompt = int(getattr(resp, "prompt_tokens", n_prompt))
        return "".join(pieces), n_gen, n_prompt, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Per-cell run
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer, cell: str, question: str) -> str:
    if cell.startswith("C2"):
        return build_c2_prompt(tokenizer, question)
    if cell.startswith("A3"):
        return build_a3_prompt(tokenizer, question)
    raise ValueError(f"unknown cell {cell!r}")


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _existing_idxs(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            done.add(int(json.loads(line)["idx"]))
        except Exception:
            continue
    return done


def run_cell(
    runner: MLXRunner,
    cell: str,
    n: int,
    cooldown_s: int,
    *,
    capture_power: bool,
    no_resume: bool,
) -> Path:
    """Run one cell on n GSM8K problems with cold-start per problem."""
    problems, golds = load_problems(0, n, n)
    out_path = CELLS_PHONE_DIR / f"{cell.lower().replace('-', '_')}__0000_{n:04d}.jsonl"
    if no_resume and out_path.exists():
        out_path.unlink()
    done = _existing_idxs(out_path)
    todo = [(i, p, g) for i, (p, g) in enumerate(zip(problems, golds)) if i not in done]
    print(f"[{cell}] {len(done)} done, {len(todo)} to run -> {out_path.name}")

    power: PowerCapture | None = None
    if capture_power:
        log_path = CELLS_PHONE_DIR / f"{cell.lower().replace('-', '_')}_powermetrics.plist"
        power = PowerCapture(log_path=log_path)
        if not power.start():
            power = None

    try:
        for i, prob, gold in todo:
            prompt = _build_prompt(runner.tokenizer, cell, prob["question"])
            t_epoch_start = time.time()
            t_start = time.perf_counter()
            completion, n_gen, n_prompt, gen_secs = runner.greedy_generate(
                prompt, MAX_NEW_TOKENS,
            )
            wall_ms = (time.perf_counter() - t_start) * 1000.0
            t_epoch_end = time.time()

            pred, hashed = extract_answer(completion)
            row = {
                "idx": i,
                "cell": cell,
                "device": DEVICE_TAG,
                "gold": gold,
                "pred": pred,
                "correct": int(pred is not None and pred == gold),
                "hash_hit": int(hashed),
                "completion": completion,
                "gen_tokens": n_gen,
                "prompt_tokens": n_prompt,
                "wallclock_ms": wall_ms,
                "gen_secs": gen_secs,
                "joules_mac": None,
                "peak_temp_c_mac": None,
                "battery_delta_pct_iphone": None,
                "peak_thermal_state_iphone": None,
                "t_epoch_start": t_epoch_start,
                "t_epoch_end": t_epoch_end,
            }
            _append_jsonl(out_path, row)
            print(
                f"[{cell}] idx={i:3d} wall={wall_ms/1000.0:5.2f}s "
                f"gen={n_gen:3d} prompt={n_prompt:4d} "
                f"correct={row['correct']}"
            )
            if cooldown_s > 0 and i != todo[-1][0]:
                time.sleep(cooldown_s)
    finally:
        if power is not None:
            power.stop()
            power.parse_log()
            # Backfill joules + peak_temp into the JSONL we just wrote.
            _backfill_power(out_path, power)

    return out_path


def _backfill_power(jsonl_path: Path, power: PowerCapture) -> None:
    """Read jsonl, fill joules_mac + peak_temp_c_mac per row using PowerCapture windows, rewrite."""
    if not jsonl_path.exists():
        return
    rows = []
    for line in open(jsonl_path):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    for r in rows:
        t0 = r.get("t_epoch_start")
        t1 = r.get("t_epoch_end")
        if t0 is None or t1 is None:
            continue
        r["joules_mac"] = power.joules_in_window(t0, t1)
        r["peak_temp_c_mac"] = power.peak_temp_in_window(t0, t1)
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[power] backfilled {len(rows)} rows in {jsonl_path.name}")


# ---------------------------------------------------------------------------
# Sustained-thermal probe
# ---------------------------------------------------------------------------

def run_sustained(
    runner: MLXRunner,
    cell: str,
    duration_s: float,
    *,
    capture_power: bool,
) -> Path:
    """Continuously generate (looping over problems) for duration_s seconds.
    Records tokens/sec at SUSTAINED_TPS_BUCKET_S buckets and peak power/temp."""
    problems, _ = load_problems(0, 50, 50)
    out_path = CELLS_PHONE_DIR / f"{cell.lower().replace('-', '_')}_sustained.jsonl"
    if out_path.exists():
        out_path.unlink()

    power: PowerCapture | None = None
    if capture_power:
        log_path = CELLS_PHONE_DIR / f"{cell.lower().replace('-', '_')}_sustained_powermetrics.plist"
        power = PowerCapture(log_path=log_path)
        if not power.start():
            power = None

    print(f"[{cell}] sustained probe: {duration_s:.0f}s")
    bucket_start = time.perf_counter()
    bucket_token_count = 0
    bucket_idx = 0
    total_tokens = 0
    overall_start = bucket_start
    pi = 0
    try:
        while time.perf_counter() - overall_start < duration_s:
            prob = problems[pi % len(problems)]
            prompt = _build_prompt(runner.tokenizer, cell, prob["question"])
            t_epoch_start = time.time()
            _, n_gen, _, gen_secs = runner.greedy_generate(prompt, MAX_NEW_TOKENS)
            t_epoch_end = time.time()
            total_tokens += n_gen
            bucket_token_count += n_gen
            now = time.perf_counter()
            while now - bucket_start >= SUSTAINED_TPS_BUCKET_S:
                # flush a bucket point
                _append_jsonl(out_path, {
                    "cell": cell,
                    "device": DEVICE_TAG,
                    "bucket_idx": bucket_idx,
                    "elapsed_s": bucket_idx * SUSTAINED_TPS_BUCKET_S,
                    "tokens_in_bucket": bucket_token_count,
                    "total_tokens": total_tokens,
                    "tokens_per_sec": bucket_token_count / SUSTAINED_TPS_BUCKET_S,
                    "t_epoch_start": t_epoch_start,
                    "t_epoch_end": t_epoch_end,
                })
                bucket_idx += 1
                bucket_start += SUSTAINED_TPS_BUCKET_S
                bucket_token_count = 0
            pi += 1
    finally:
        if power is not None:
            power.stop()
            power.parse_log()
            # Append a single trailing summary row with peak temp + total joules.
            t0 = overall_start
            # Convert perf_counter delta back to wall epoch:
            t_overall_epoch_start = time.time() - (time.perf_counter() - overall_start)
            t_overall_epoch_end = time.time()
            _append_jsonl(out_path, {
                "cell": cell,
                "device": DEVICE_TAG,
                "summary": True,
                "duration_s": duration_s,
                "total_tokens": total_tokens,
                "joules_total_mac": power.joules_in_window(t_overall_epoch_start, t_overall_epoch_end),
                "peak_temp_c_mac": power.peak_temp_in_window(t_overall_epoch_start, t_overall_epoch_end),
            })

    print(f"[{cell}] sustained probe done: {total_tokens} total tokens in {duration_s:.0f}s")
    return out_path


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def emit_prompts_jsonl(runner: MLXRunner, n: int) -> Path:
    """Write per-(idx, cell) prompts to PROMPTS_JSONL and a copy to the iOS bundle.

    The iOS app loads the same file as a bundled resource so the prompt bytes
    are identical on both devices (plan section 5)."""
    problems, golds = load_problems(0, n, n)
    PROMPTS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, (prob, gold) in enumerate(zip(problems, golds)):
        for cell in ("C2", "A3"):
            prompt = _build_prompt(runner.tokenizer, cell + "-Mac", prob["question"])
            tokens = runner.encode(prompt)
            rows.append({
                "idx": i,
                "cell": cell,  # device-agnostic name
                "prompt": prompt,
                "prompt_tokens_hf": len(tokens),
                "gold": gold,
                "question": prob["question"],
            })

    with open(PROMPTS_JSONL, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    IOS_RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROMPTS_JSONL, IOS_PROMPTS_JSONL)
    print(f"[prompts] wrote {len(rows)} rows to {PROMPTS_JSONL}")
    print(f"[prompts] mirrored to {IOS_PROMPTS_JSONL}")
    return PROMPTS_JSONL


def byte_equivalence_check(runner: MLXRunner, n: int) -> int:
    """Sanity check 5: confirm prompts match Experiment 5 (legacy) jsonl bytes.

    If experiments/path1_plan8.py's prompts.jsonl is found alongside its
    results, compare row-by-row. Else fall back to a self-consistency check
    (same output across two builds), which still catches non-determinism."""
    problems, _ = load_problems(0, n, n)
    legacy_paths = [
        _PROJECT_ROOT / "results" / "path_1_cot_tokens" / "plan5" / "prompts.jsonl",
        _PROJECT_ROOT / "results" / "path_1_cot_tokens" / "plan8" / "prompts.jsonl",
    ]
    legacy_path = next((p for p in legacy_paths if p.exists()), None)
    diffs = 0

    if legacy_path is not None:
        print(f"[byte-eq] comparing against legacy {legacy_path}")
        legacy_rows: dict[tuple[int, str], str] = {}
        for line in open(legacy_path):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "prompt" in row and "cell" in row and "idx" in row:
                key = (int(row["idx"]), str(row["cell"]).split("-")[0])
                legacy_rows[key] = row["prompt"]
        if not legacy_rows:
            print("[byte-eq] legacy file present but has no prompts; falling back to self-consistency")
            legacy_path = None

    for i, prob in enumerate(problems):
        for cell in ("C2", "A3"):
            current = _build_prompt(runner.tokenizer, cell + "-Mac", prob["question"])
            other = _build_prompt(runner.tokenizer, cell + "-Mac", prob["question"])  # 2nd build for self-consistency
            if current != other:
                print(f"[byte-eq] non-deterministic prompt at idx={i} cell={cell}")
                diffs += 1
                continue
            if legacy_path is not None:
                expected = legacy_rows.get((i, cell))
                if expected is not None and expected != current:
                    print(f"[byte-eq] DIFF idx={i} cell={cell}")
                    print(f"  current[:120] = {current[:120]!r}")
                    print(f"  legacy[:120]  = {expected[:120]!r}")
                    diffs += 1
    if diffs == 0:
        print(f"[byte-eq] PASS (n={n}, both cells)")
    else:
        print(f"[byte-eq] FAIL: {diffs} mismatches")
    return diffs


def accuracy_only(runner: MLXRunner, cell: str, n: int) -> dict:
    """Sanity gate 1: quantization accuracy parity. Plan threshold: >= 65%."""
    problems, golds = load_problems(0, n, n)
    correct = 0
    for i, (prob, gold) in enumerate(zip(problems, golds)):
        prompt = _build_prompt(runner.tokenizer, cell, prob["question"])
        completion, _, _, _ = runner.greedy_generate(prompt, MAX_NEW_TOKENS)
        pred, _ = extract_answer(completion)
        ok = int(pred is not None and pred == gold)
        correct += ok
        print(f"[acc] {cell} idx={i:3d} gold={gold} pred={pred} {'OK' if ok else 'X'}")
    acc = correct / max(1, len(problems))
    verdict = "PASS" if acc >= 0.65 else "FAIL (re-quantize at Q5/Q6)"
    out = {"cell": cell, "n": len(problems), "correct": correct, "accuracy": acc, "verdict": verdict}
    print(f"[acc] {cell}: {correct}/{len(problems)} = {acc:.3f} - {verdict}")
    return out


# ---------------------------------------------------------------------------
# Smoke test (no MLX dependency for the import-only path)
# ---------------------------------------------------------------------------

def smoke_imports() -> None:
    """Verify legacy imports succeeded and prompt builders are callable.
    Does NOT touch MLX or download GSM8K. Cheap to run."""
    assert len(EXEMPLARS_COT) == 8
    assert len(EXEMPLARS_DIRECT) == 8
    assert callable(build_c2_prompt)
    assert callable(build_a3_prompt)
    assert callable(extract_answer)
    assert GOLD_RE.search("the answer is #### 42").group(1) == "42"
    print("[smoke-imports] OK: legacy reuse intact")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--cells", nargs="+", choices=list(ALL_CELLS), default=list(ALL_CELLS))
    p.add_argument("--n", type=int, default=DEFAULT_N)
    p.add_argument("--weights", default=str(DEFAULT_MLX_DIR), help="Path to MLX 4-bit weights dir")
    p.add_argument("--cooldown", type=int, default=INTER_PROBLEM_COOLDOWN_S,
                   help="Seconds between problems for thermal recovery")
    p.add_argument("--no-power", action="store_true",
                   help="Skip powermetrics capture (timing+thermal only)")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite existing per-cell JSONLs")
    # Modes (mutually exclusive top-level intents)
    p.add_argument("--smoke", action="store_true", help="Smoke check legacy imports only")
    p.add_argument("--smoke-mlx", action="store_true",
                   help="Load MLX + run one problem; skip if mlx_lm missing")
    p.add_argument("--emit-prompts", action="store_true",
                   help="Emit prompts.jsonl for the iOS bundle")
    p.add_argument("--byte-equivalence", action="store_true",
                   help="Run sanity check 5: prompt-format byte-equivalence")
    p.add_argument("--accuracy-only", action="store_true",
                   help="Run sanity gate 1: quantization accuracy parity")
    p.add_argument("--sustained", action="store_true",
                   help="Run sustained-thermal probe instead of n-problem cell")
    p.add_argument("--duration", type=float, default=SUSTAINED_DEFAULT_DURATION_S,
                   help="Sustained probe duration in seconds")
    return p.parse_args()


def _ensure_dirs() -> None:
    CELLS_PHONE_DIR.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    _ensure_dirs()
    capture_power = not args.no_power

    if args.smoke:
        smoke_imports()
        return 0

    runner = MLXRunner(weights_path=Path(args.weights))
    runner.load()

    if args.smoke_mlx:
        prob, gold = load_problems(0, 1, 1)
        prompt = build_c2_prompt(runner.tokenizer, prob[0]["question"])
        text, n_gen, n_prompt, dt = runner.greedy_generate(prompt, 64)
        print(f"[smoke-mlx] gen={n_gen} prompt={n_prompt} secs={dt:.2f}")
        print(f"[smoke-mlx] completion[:200]={text[:200]!r}")
        return 0

    if args.emit_prompts:
        emit_prompts_jsonl(runner, args.n)
        return 0

    if args.byte_equivalence:
        diffs = byte_equivalence_check(runner, args.n)
        return 0 if diffs == 0 else 1

    if args.accuracy_only:
        results = []
        for cell in args.cells:
            results.append(accuracy_only(runner, cell, args.n))
        out_path = CELLS_PHONE_DIR / "accuracy_parity.json"
        out_path.write_text(json.dumps(results, indent=2))
        print(f"[acc] wrote {out_path}")
        return 0

    if args.sustained:
        for cell in args.cells:
            run_sustained(runner, cell, args.duration, capture_power=capture_power)
        return 0

    # Default: per-cell n-problem run
    for cell in args.cells:
        run_cell(runner, cell, args.n, args.cooldown,
                 capture_power=capture_power, no_resume=args.no_resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
