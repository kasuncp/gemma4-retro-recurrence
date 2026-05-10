# Path 1 — experiment 8: phone-class on-device measurement

## Context

Every Path 1 number reported so far is on a single 3090 in bfloat16. The series' thesis is *reasoning per unit energy on a phone* — and that translation has not been measured. C2's "1.6 TFLOPs / 3.3 s wall-clock per problem" is meaningless without the phone-side mapping.

Experiment 8 answers the on-device half of the equation:

> **What is the per-problem latency, energy, and thermal footprint of running C2 on representative phone-class hardware?**

This is the experiment that turns a datacenter-FLOP plateau number into a deployment claim.

## Available hardware

This run uses two Apple-silicon devices already in hand:

- **MacBook Pro M3** — actively cooled, gives clean joules via `powermetrics`. Acts as the **upper-bound / irreducible-compute measurement** (no thermal throttling, full memory bandwidth).
- **iPhone 13 Pro Max** — A15 Bionic (2021), 6 GB RAM, passively cooled. Acts as the **realistic phone-class deployment measurement**.

Snapdragon / Android measurement is **deferred** to a follow-up; it requires hardware not currently available. The Apple-only run is sufficient to answer the core question — *can C2 run on a phone* — and provides one of the two platform points the original plan called for.

### Caveats (must appear in the writeup, not buried)

1. **iPhone 13 PM is 2021 silicon (A15), not 2026 flagship.** A current-gen iPhone (A18 Pro) or Snapdragon 8 Elite would be ~2× faster on inference and have a more generous thermal envelope. iPhone-13-PM numbers should be read as a **conservative lower bound** on what a current flagship achieves — not as the modal phone experience in 2026.
2. **iOS does not expose joules to apps.** No `/sys/class/power_supply/` analog; no per-component power API without private entitlements. Energy on iPhone is measured indirectly (battery-delta over a fixed run, or via a tethered USB-C power meter if available). Mac is the clean joules number; iPhone is the *realistic-thermal* number with a coarser energy estimate.
3. **iOS thermal telemetry is discrete.** `ProcessInfo.thermalState` returns four levels (`.nominal / .fair / .serious / .critical`) — enough for throttle detection but not for a continuous temperature curve. Mac gets continuous SMC-sampled temperatures via `powermetrics`.
4. **MacBook Pro M3 is actively cooled.** This is a *feature* for the irreducible-compute number but means the Mac is NOT phone-thermal-representative. The "phone-thermal" reading comes from iPhone alone; the Mac is the "compute floor without thermal interference" number.
5. **6 GB RAM on iPhone 13 PM is tight.** Gemma-4 E2B at MLX 4-bit fits, but Q5/Q6 may OOM under iOS memory pressure. If Q4 fails the accuracy parity gate, Q5 may not be reachable on this device — escalate to a fresher iPhone or accept the accuracy loss.

## Implementation decisions (locked)

These were settled in the pre-implementation review on 2026-05-09. Recorded so the implementation pass doesn't re-litigate them.

1. **Branch.** Implementation lands on a fresh branch `path_1_plan_8_v2`, cut from `main`. Not on the current `path_2_phase_2` branch (unrelated work in flight) and not on the legacy `path_1_plan_8` branch (carries the obsolete GPU-proxy attempt; mixing the two implementations in one history is confusing).
2. **Coexistence with the legacy plan8 GPU-proxy artifacts.** A prior implementation of this plan number — a 3090 / GGUF / `llama.cpp` proxy of phone measurement — is on disk: `experiments/path1_plan8.py`, `experiment_plan8.yaml`, `results/path_1_cot_tokens/plan8/results_plan8.json`, `results/path_1_cot_tokens/plan8/cells/`. **Leave them in place untouched.** The new on-device deliverables use the names already given in *Deliverables* (`path1_phone_mac.py`, `Path1Bench/`, `path1_phone_analyze.py`) and do not collide. To avoid clobbering the legacy JSON, the new analyze script writes `results_plan8_phone.json` (not `results_plan8.json`) and per-cell files under `results/path_1_cot_tokens/plan8/cells_phone/` (not `cells/`).
3. **iOS app delivery scope.** Full Xcode project committed under `Path1Bench/` — `Path1Bench.xcodeproj/project.pbxproj`, `Package.resolved`, Swift sources, `Info.plist`. The user's only manual setup is opening the project in Xcode and selecting a signing team. Building, signing, and deploying to the iPhone are user-driven Xcode steps; they cannot be automated from this CLI session, and the implementation must not pretend otherwise.
4. **Model identifier.** `google/gemma-4-E2B-it` (the project's model — same as `experiments/path1_plan8.py` `MODEL_ID`). MLX 4-bit weights produced by:

   ```bash
   python -m mlx_lm.convert --hf-path google/gemma-4-E2B-it -q --q-bits 4 \
       --mlx-path ./mlx_models/gemma-4-E2B-it-mlx-q4
   ```

   The same `mlx_models/gemma-4-E2B-it-mlx-q4/` directory is the source for both the Mac script and the iOS app's bundled resources. (Earlier mentions of "Gemma-3" in this doc were typos and have been corrected.)
5. **Code reuse from the legacy script.** `path1_phone_mac.py` reuses the C2/A3 prompt builders, `EXEMPLARS_COT` / `EXEMPLARS_DIRECT`, the GSM8K loader, and the answer-extraction regex from `experiments/path1_plan8.py` — imported, not re-derived. This is what gives sanity check #5 (prompt-format byte-equivalence) any teeth. iOS-side prompts are not re-implemented in Swift; the Mac script emits a `prompts.jsonl` (one row per problem with the fully-assembled prompt string) that the iOS app loads as a bundled resource. Same bytes on both devices, by construction.
6. **iOS SwiftPM dependencies.** MLX Swift via `https://github.com/ml-explore/mlx-swift` and the model loader from `https://github.com/ml-explore/mlx-swift-examples`. Pin both via `Package.resolved`. Before committing the resolved file, confirm the chosen tag supports Gemma-3/4 architecture; if not, pin to a `main` revision and note that choice in `Path1Bench/README.md`.
7. **Execution timing.** Implementation must wait until the path 2 phase 2 measurement run currently using these devices completes. The implementation pass cannot reuse the in-flight working tree, must not compete for the model weights cache, and must not start the new branch until the user confirms the device is free.

## Scope — what this plan is and isn't

**In scope:**
- Two Apple-silicon devices (above).
- Two cells per device, mirroring the four-paths comparison axes:
  - **C2 cell** (zero-shot plain prompt, chat template, greedy, max_new_tokens = 512) — the deployment-target cell.
  - **A3 cell** (8-shot Wei et al. CoT, greedy, max_new_tokens = 512) — for reference, to see how the prompt-token difference matters on-device.
- Three measurements per cell:
  - **Wall-clock per problem** (median + p95 over 50 problems).
  - **Energy per problem** — clean joules on Mac, battery-delta-derived approximation on iPhone.
  - **Thermal envelope** — continuous SMC temperature on Mac; `ProcessInfo.thermalState` transitions on iPhone; sustained 5-min throttle probe on both.
- Quantization: **MLX 4-bit** for both. Same model weights both sides; same C2 / A3 prompts.

**Explicitly out of scope:**
- No Android / Snapdragon measurement. Deferred to a follow-up plan (plan8b) once hardware is accessible.
- No accuracy re-validation on quantized weights as the main goal — that's a separate question (does MLX 4-bit preserve C2's 71.6 %?). Pin a small accuracy parity check (n = 50 GSM8K) in the sanity gate.
- No multi-modal probes. Text-only.
- No Path 2/3/4 measurements. This plan benchmarks Path 1 only.
- No older-iPhone or older-Mac sweep. Two device points, period.

## Environment

**MacBook Pro M3:**
- OS: macOS 14+ (Sonoma or later).
- Runner: MLX-LM with 4-bit quantization. Pin the `mlx-lm` and `mlx` package versions.
- Quantization: MLX 4-bit conversion of `gemma-4-E2B-it`. Pin the converted weight directory hash.
- Power: `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 100` (10 Hz). Capture CPU / GPU / ANE separately.
- Temperature: `powermetrics --samplers smc -i 1000` (1 Hz). Records per-die temperatures.
- Power source: device on AC; capture power-source state in metadata so the run is reproducible. (Battery-vs-AC changes the M3's frequency caps.)

**iPhone 13 Pro Max:**
- OS: iOS 17+.
- Runner: **MLX Swift** in a small benchmark app, sideloaded via Xcode. Free Apple Developer account = 7-day signing (re-sign weekly), paid $99/yr account = normal signing. MLX-Swift supports Gemma; same 4-bit quantized weights as the Mac.
- Quantization: MLX 4-bit (same artifact as Mac).
- Power (best-effort, in priority order):
  - **Primary:** Battery-percentage delta across a fixed-duration sustained run. Calibrate against an idle baseline.
  - **Secondary (if available):** Tethered USB-C / Lightning power meter (e.g. ChargerLAB KM003C) measuring draw at the connector while phone is held at ≥ 95 % charge so charging current is small and stable. Subtract idle baseline.
  - **Tertiary:** Xcode Instruments → Energy Log "energy impact" score. Coarse, not in joules; used only as a directional cross-check.
- Thermals: `ProcessInfo.processInfo.thermalState` polled every 1 s via `NotificationCenter.default.addObserver(forName: ProcessInfo.thermalStateDidChangeNotification...)`. Record state-transition timestamps.
- Run conditions: airplane mode on, screen brightness fixed at 50 %, all background app refresh disabled, app holds idle timer disabled (`UIApplication.shared.isIdleTimerDisabled = true`). Document whether the phone is in a case (cases change thermal mass significantly).

## Protocol

### Cells

| Device | Cell | Prompt | Decode | n |
|---|---|---|---|---|
| MacBook Pro M3 | **C2-Mac** | C2 plain | greedy, 512 tok | 50 |
| MacBook Pro M3 | **A3-Mac** | 8-shot Wei et al. | greedy, 512 tok | 50 |
| iPhone 13 PM | **C2-iPhone** | C2 plain | greedy, 512 tok | 50 |
| iPhone 13 PM | **A3-iPhone** | 8-shot Wei et al. | greedy, 512 tok | 50 |

Same 50 problems on every device (deterministic head of the GSM8K test split).

### Per-problem measurement

For each problem:

1. Cold start a fresh inference session (model resident in RAM, generation state reset).
2. Tokenize prompt; record `prompt_tokens`.
3. Start timer + power sampler.
4. Generate up to 512 tokens (or until `<eos>`); record `gen_tokens`, wall-clock, and power samples over the duration.
5. Stop timers; record peak `thermalState` (iPhone) or peak SoC die temperature (Mac).
6. Pause 30 s before next problem to allow thermal recovery.

### Sustained-thermal probe

After the 50-problem batch, run one **sustained 5-minute generation** (continuous queries with no cooldown). Record:
- Tokens/sec curve over 5 minutes (1 Hz sampling).
- Peak temperature (Mac) / max thermalState reached and timestamps of each transition (iPhone).
- Time-to-throttle if it occurs.

This distinguishes "fine for one query" from "fine for sustained assistant use." The contrast between Mac (cooled) and iPhone (passive) is the point: the Mac shows the irreducible compute curve, the iPhone shows what passive cooling does to it.

## Budget

| Step | Time |
|---|---|
| MLX 4-bit conversion + load to each device | ~1 hour total (single artifact, copy to both) |
| iOS app scaffolding (Xcode project, MLX Swift integration, JSONL logger, file-export) | ~4–6 hours one-time |
| 50 problems × 4 cells × ~5 min each | ~17 hours device-time, parallelizable across the two devices (~9 hours each) |
| Sustained-thermal probe (10 min × 4 cells) | ~40 min device-time |
| Analysis + plot | ~3 hours |

**Total: ~1 day device time + 1–1.5 days engineering** (the iOS app is the new cost relative to the Mac-only path).

## Sanity checks

1. **Quantization accuracy parity gate.** Before measuring latency, run the C2 cell on n = 50 GSM8K problems on each device's quantized model. Accuracy must be ≥ 65 % (within ~10 % of bf16's 71.6 %). If lower, MLX 4-bit is too aggressive and Q5 / Q6 is required — note this may not fit in iPhone 13 PM's 6 GB RAM under iOS memory pressure.
2. **Cold-start vs warm-start.** Confirm the second problem in a sequence has the same latency as the first ± 5 %. If warm is much faster, the measurement is dominated by KV-cache reuse; report both numbers.
3. **Power-measurement floor.** Idle for 30 s before each problem to establish baseline; subtract baseline from per-problem joules (Mac) or battery-delta (iPhone).
4. **Throttle detection.**
   - Mac: log peak die temperature; if within 5 °C of the M3's known throttle threshold, flag as "near-throttle."
   - iPhone: any transition `.nominal → .fair` is a soft warning; `.serious` or `.critical` means real throttling — flag the affected problems and segment the median/p95 by thermal state.
5. **Prompt-format byte-equivalence.** Confirm the C2 prompt sent to the on-device model byte-matches the C2 prompt from Experiment 5's jsonl. Off-by-one tokenization differences invalidate the comparison.
6. **iPhone idle-power calibration.** Run a 5-minute idle baseline (app open, no inference) and subtract its battery-% drain rate from the inference run's drain rate. Without this, the iPhone joules estimate is meaningless.
7. **Charge-state confounding (iPhone).** If using the USB-C power-meter approach, confirm battery is ≥ 95 % so charging current is small and stable; otherwise the power meter reads charging draw, not inference draw. If using battery-delta, run the test off-charger from a known starting %.
8. **Mac power-source confounding.** Run on AC. Battery mode caps clocks differently and would underreport peak performance.

## Pre-registered interpretation

Decide outcomes before measuring.

### Outcome A — C2 is comfortably on-device
Median wall-clock < 3 s on Mac and < 6 s on iPhone, p95 acceptable on both, no thermal throttling within 5 minutes. **C2 is a deployable assistant cell on Apple silicon.** The four-paths comparison can use these numbers as the on-device budget envelope.

### Outcome B — Mac fine, iPhone throttles on sustained use
Mac runs cleanly; iPhone hits `.serious` thermalState within 60 s of sustained generation, tokens/sec drops > 30 %. **C2 works on a phone for occasional queries, not for chat-style sustained interaction on this generation of phone.** Document the sustainable rate; the four-paths comparison is at the sustainable rate.

### Outcome C — C2 is too slow at p95 on iPhone
iPhone p95 wall-clock > 10 s. **C2 fails the < 2 s TTFT user-experience threshold on 2021 silicon.** Open question: would A18 Pro / SD 8 Elite clear it? Plan a follow-up on a fresher device. Mitigations: smaller quantization (re-check accuracy), speculative decoding (separate plan), or accept that "C2 on a current flagship" is the deployment claim, not "C2 on any 2021+ phone."

### Outcome D — A3's prompt overhead matters more on iPhone than on Mac
On iPhone, A3's 747 prompt tokens cost a much larger fraction of total wall-clock than on Mac, because prompt processing is more memory-bandwidth-bound and the iPhone is bandwidth-limited. **The Experiment 5 prompt-format finding has a stronger second-order benefit on phone**: the cheaper prompt is dramatically faster TTFT on-device.

### Outcome E — Mac and iPhone diverge by > 3×
Beyond the expected 1.5–2× silicon gap. **Indicates iOS runtime overhead, memory pressure, or thermal dampening above what's explainable by raw silicon.** Document and investigate before drawing conclusions — could be a measurement artifact (charge-state confound, idle-power miscalibration) rather than a real platform difference.

### Outcome F — Quantization gate fails
MLX 4-bit accuracy < 65 % on either device. **Q4 is too aggressive for Gemma-4 E2B**. Re-quantize at Q5 / Q6, re-measure. If Q5 OOMs on iPhone, the deployment claim caveats to "current iPhone or larger-RAM Android device."

## Deliverables

1. `path1_phone_mac.py` — Mac side. Runs MLX-LM benchmark, captures `powermetrics` output, emits JSONL.
2. `Path1Bench/` — iOS app (Xcode project). Embeds MLX Swift, runs benchmark on tap, writes JSONL to app sandbox; export via Files app / AirDrop / Xcode device download.
3. `path1_phone_analyze.py` — merges per-device JSONLs, computes median / p95 latency, mean joules (Mac) / battery-delta (iPhone), peak thermal state; emits `results_plan8_phone.json` and a Pareto plot of accuracy vs energy. Output goes to `results/path_1_cot_tokens/plan8/` per project convention; emit PNGs alongside the JSON. The `_phone` suffix avoids clobbering the legacy GPU-proxy `results_plan8.json` already in that directory (see *Implementation decisions* §2).

Per-cell JSONLs in `results/path_1_cot_tokens/plan8/cells_phone/` (the legacy GPU-proxy run uses `cells/`; the two coexist). Each row records:
```
{idx, gen_tokens, prompt_tokens, wallclock_ms, joules_mac, battery_delta_pct_iphone, peak_temp_c_mac, peak_thermal_state_iphone, correct, device, cell}
```

## Report back

Paste the four-cell latency / energy / thermal table, plus the sustained-probe curves, plus the per-device outcome label. Update the blog post's conclusion table with on-device numbers — making explicit that the iPhone half is on 2021 silicon and a 2026 flagship would likely be ~2× faster.

## What this closes

This is the last Path 1 experiment **on the hardware currently available**. Combined with Experiments 6 and 7, after this the four-paths head-to-head can be planned with a fixed on-device budget (Apple-silicon-derived). A follow-up plan8b can add Snapdragon / Android when hardware is accessible; the methodology and analysis pipeline from this plan transfer directly.
