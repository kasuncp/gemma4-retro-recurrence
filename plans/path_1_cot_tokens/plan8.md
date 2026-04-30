# Path 1 — experiment 8: phone-class on-device measurement

## Context

Every Path 1 number reported so far is on a single 3090 in bfloat16. The series' thesis is *reasoning per unit energy on a phone* — and that translation has not been measured. C2's "1.6 TFLOPs / 3.3 s wall-clock per problem" is meaningless without the phone-side mapping.

Experiment 8 answers the on-device half of the equation:

> **What is the per-problem latency, energy, and thermal footprint of running C2 on representative phone-class hardware?**

This is the experiment that turns a datacenter-FLOP plateau number into a deployment claim.

## Scope — what this plan is and isn't

**In scope:**
- Two reference devices, covering the dominant 2026 phone SoC families:
  - **Snapdragon 8 Gen 3 / 8 Elite** (Android, recent flagship). Most portable for Gemma deployment via MLC-LLM or llama.cpp Vulkan.
  - **Apple M2/M3** (iPad or Mac mini — phone-adjacent compute envelope, MLX backend). Apple Silicon GPU is the cleanest "phone-class but measurable" platform.
- Two cells per device, mirroring the four-paths comparison axes:
  - **C2 cell** (zero-shot plain prompt, chat template, greedy, max_new_tokens = 512) — the deployment-target cell.
  - **A3 cell** (8-shot Wei et al. CoT, greedy, max_new_tokens = 512) — for reference, to see how the prompt-token difference matters on-device.
- Three measurements per cell:
  - **Wall-clock per problem** (median + p95 over 50 problems).
  - **Energy per problem** (joules — average power × duration).
  - **Thermal envelope** (peak SoC temperature during sustained 5-minute generation; throttle behavior).
- Quantization: **Q4_K_M GGUF** for llama.cpp / Vulkan path; **MLX 4-bit** for Apple. Same model weights both sides; same C2 / A3 prompts.

**Explicitly out of scope:**
- No iPhone or Pixel measurement *in this plan* (smaller battery + harder thermal constraints; merits its own follow-up if Experiment 8 looks promising).
- No accuracy re-validation on quantized weights — that's a separate question (does Q4_K_M preserve C2's 71.6%?). Pin a small accuracy parity check (n=50 GSM8K) in the sanity gate.
- No multi-modal probes. Text-only.
- No Path 2/3/4 measurements. This plan benchmarks Path 1 only; the four-paths comparison rig adds the others later.

## Environment

**Snapdragon device:**
- OS: Android 14+
- Runner: MLC-LLM (preferred — Vulkan + Gemma support is mature) or llama.cpp Vulkan backend. Pin git SHA.
- Quantization: Q4_K_M GGUF of `gemma-4-E2B-it`. Pin the GGUF file SHA-256.
- Power: Android Battery Historian + on-device perf counters (`/sys/class/power_supply/battery/current_now` × `voltage_now`).
- Temperature: `/sys/class/thermal/thermal_zone*/temp` polled at 1 Hz.

**Apple device:**
- OS: macOS 14+ (Mac mini M2 / M3 — closest to phone thermal envelope) or iPadOS 17+ (M2/M4 iPad, fanless).
- Runner: MLX-LM with 4-bit quantization. Pin git SHA.
- Quantization: MLX 4-bit conversion of `gemma-4-E2B-it`.
- Power: `powermetrics` (macOS) — sample at 1 Hz, capture CPU/GPU/ANE power separately.
- Temperature: `pmset -g thermlog` and `powermetrics --samplers smc`.

## Protocol

### Cells

| Device | Cell | Prompt | Decode | n |
|---|---|---|---|---|
| SD 8 Gen 3 | **C2-SD** | C2 plain | greedy, 512 tok | 50 |
| SD 8 Gen 3 | **A3-SD** | 8-shot Wei et al. | greedy, 512 tok | 50 |
| Apple M2/M3 | **C2-M** | C2 plain | greedy, 512 tok | 50 |
| Apple M2/M3 | **A3-M** | 8-shot Wei et al. | greedy, 512 tok | 50 |

n = 50 chosen to give tight median / p95 estimates without taking days of phone time. The same 50 problems on every device (deterministic head of the GSM8K test split).

### Per-problem measurement

For each problem:

1. Cold start a fresh inference session.
2. Tokenize prompt; record `prompt_tokens`.
3. Start timer + power sampler.
4. Generate up to 512 tokens (or until `<eos>`); record `gen_tokens`, wall-clock, and accumulated joules.
5. Stop timers; record peak SoC temperature reached during generation.
6. Pause 30s before next problem to allow thermal recovery (otherwise problem 2 onwards reports throttled numbers).

### Sustained-thermal probe

After the 50-problem batch, run one **sustained 5-minute generation** (continuous queries with no cooldown). Record:
- Tokens/sec curve over 5 minutes (1-Hz sampling) — does throttling kick in?
- Peak temperature.
- Time-to-throttle if throttling occurs.

This distinguishes "fine for one query" from "fine for sustained assistant use."

## Budget

| Step | Time |
|---|---|
| Quantize + load to each device | ~2 hours per device |
| 50 problems × 4 cells × ~5 min each | ~17 hours device-time, but parallelizable across devices (~9 hours each) |
| Sustained-thermal probe (10 min × 4 cells) | ~40 min device-time |
| Analysis + plot | ~3 hours |

**Total: ~2 days of device time + 1 day of engineering**, assuming both devices are accessible.

If only one device is available, halve the cells and budget — but the cross-platform comparison is then deferred.

## Sanity checks

1. **Quantization accuracy parity gate.** Before measuring latency, run the C2 cell on n = 50 GSM8K problems on each quantized model. Accuracy must be ≥ 65% (i.e., within ~10% of the bf16 71.6%). If lower, Q4_K_M is too aggressive for this model and a Q5/Q6 quantization is required. Document.
2. **Cold-start vs warm-start.** Confirm that the second problem in a sequence has the same latency as the first ± 5%. If warm is much faster, the measurement is dominated by KV-cache reuse and the per-problem number isn't representative of an assistant's first-query experience. Either fix or report both.
3. **Power-measurement floor.** Idle power for 30s before each problem to establish baseline; subtract baseline from per-problem joules.
4. **Throttle detection.** Document the temperature threshold at which throttling kicks in for each device; if peak generation temperature is within 5°C of that threshold, flag the result as "near-throttle."
5. **Prompt-format byte-equivalence.** Confirm the C2 prompt sent to the on-device model byte-matches the C2 prompt from Experiment 5's jsonl. Off-by-one tokenization differences invalidate the comparison.

## Pre-registered interpretation

Decide outcomes before measuring.

### Outcome A — C2 is comfortably on-device
Median wall-clock < 3 s, p95 < 6 s, energy < 5 J per problem, no thermal throttling within 5 minutes. **C2 is a deployable assistant cell on a phone.** The four-paths comparison can use these numbers as the on-device budget envelope.

### Outcome B — C2 works for one query but throttles on sustained use
Median latency acceptable, but the 5-minute sustained probe shows tokens/sec dropping > 30% within 60 seconds. **C2 is fine for occasional queries, not for chat-style sustained interaction.** Document the sustainable rate; the four-paths comparison is at the sustainable rate, not the cold-start rate.

### Outcome C — C2 is too slow at p95
p95 wall-clock > 10 s. **C2 fails the < 2 s time-to-first-token user-experience threshold for an assistant.** Mitigations: smaller quantization (which lowers accuracy — measure), speculative decoding (separate plan), or move to a smaller model (which exits Path 1 entirely).

### Outcome D — A3's prompt overhead matters
On-device, A3 (with 747 prompt tokens) is significantly slower than C2 (with 69 prompt tokens) — say, by > 50% wall-clock. **The Experiment 5 prompt-format finding has a second-order benefit on phone**: the cheaper prompt isn't just more accurate, it's also dramatically faster TTFT on-device. Strengthens the Path 1 conclusion.

### Outcome E — Cross-platform divergence
Apple and Snapdragon results differ by > 2× on either latency or energy. **The on-device story is platform-specific.** Document and flag for the four-paths comparison: which platform was assumed.

## Deliverables

Two scripts (one per platform), shared analysis:

1. `path1_phone_snapdragon.py` — Android side. Connects via ADB, runs MLC-LLM benchmark, captures power/thermal logs.
2. `path1_phone_apple.py` — Apple side. Runs MLX benchmark locally, captures `powermetrics` output.
3. `path1_phone_analyze.py` — merges per-device JSONLs, computes median/p95 latency, mean joules, peak temp; emits `results_plan8.json` and a Pareto plot of accuracy vs energy.

Per-cell JSONLs in `results/path_1_cot_tokens/plan8/cells/`. Each row records `{idx, gen_tokens, wallclock_ms, joules, peak_temp_c, correct}`.

## Report back

Paste the four-cell latency / energy / temp table and the per-device outcome label. Update the blog post's conclusion table with on-device numbers (replacing or augmenting the 3090 numbers).

## What this closes

This is the last Path 1 experiment. Combined with Experiments 6 and 7, after this the four-paths head-to-head can be planned in earnest: a fixed on-device budget (latency, joules), a fixed benchmark suite, and Path 1's representative cell pinned with full per-platform measurement.
