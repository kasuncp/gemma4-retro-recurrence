# RFC 001 — Path 1 Findings: Index and Reader's Guide

| Field | Value |
|---|---|
| **RFC** | 001 |
| **Title** | Path 1 Findings: Index and Reader's Guide |
| **Status** | Informational |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Supersedes** | — |
| **Superseded by** | — |

## Abstract

This RFC indexes the body of knowledge produced by the Path 1 investigation of the four-paths "test-time compute on a phone" study. Path 1 asks whether Gemma 4 E2B can trade inference-time compute for reasoning accuracy — using only prompting and generation — without touching the model weights. This document serves as the entry point to six related RFCs and three in-flight experiment plans, to be read when picking up the investigation in a fresh context window.

## Scope of the investigation

Path 1 is one of four paths being evaluated head-to-head in a shared measurement rig (see `four-paths-one-phone-visual-guide.md` for the program-level context). Path 1 specifically is the **zero-engineering baseline**: inference-only techniques, no training, no architectural changes, no quantization. It is the path every other path (depth-recurrence, quantized sibling, Mixture-of-Depths) has to beat at matched compute.

The question Path 1 answers for the head-to-head: **what is the best accuracy E2B can achieve on reasoning benchmarks by spending more inference-time compute alone?**

## RFC set

| RFC | Title | Status | Summary |
|---|---|---|---|
| **RFC 001** | Index and Reader's Guide | Informational | This document |
| [**RFC 002**](./rfc-002-cot-gate-finding.md) | The CoT Gate Finding | Finding | CoT beats direct answering on E2B-it by +30 pts (p < 10⁻⁸) |
| [**RFC 003**](./rfc-003-inference-compute-plateau.md) | The Inference-Compute Plateau | Finding | Beyond the CoT lift, more compute does not help on GSM8K |
| [**RFC 004**](./rfc-004-repetition-loop-failure-mode.md) | The Repetition-Loop Failure Mode | Finding | 13% of IT-CoT completions collapse into repetition; IT-specific |
| [**RFC 005**](./rfc-005-methodology.md) | Methodology and Harness Decisions | Methodological | Pinned setup, eval protocol, harness-bug history, sanity-check practices |
| [**RFC 006**](./rfc-006-open-questions-and-next-steps.md) | Open Questions and Next Steps | Planning | Three pre-registered experiments to finalize Path 1's characterization (items 1 and 3 closed) |
| [**RFC 007**](./rfc-007-repetition-loop-isolation.md) | Repetition-Loop Isolation: Quantitative Verdict | Finding | Rep-loops cost 0–3 pts on top of a hard reasoning ceiling on GSM8K; B-leaning-A |
| [**RFC 008**](./rfc-008-arc-easy-cross-benchmark.md) | ARC-Easy Cross-Benchmark Validity | Finding | Plateau confirmed on ARC-Easy (B3-A3 = +2.2 pts, p=0.080); CoT does not lift over direct on ARC-Easy |

## Experiment plans referenced

Pre-registered in experiment-plan format. Each is hand-off-ready for the implementer agent. Read [RFC 006](./rfc-006-open-questions-and-next-steps.md) first for context on why these three and not others.

| Plan | Title | Status |
|---|---|---|
| `path1_plan1.md` (historical) | The CoT gate | **Executed** — see [RFC 002](./rfc-002-cot-gate-finding.md) |
| [`path1_plan2.md`](./path1_plan2.md) | Length sweep and self-consistency | **Executed** — see [RFC 003](./rfc-003-inference-compute-plateau.md) |
| [`path1_plan3.md`](./path1_plan3.md) | Repetition-loop isolation (analysis-only) | **Executed** — see [RFC 007](./rfc-007-repetition-loop-isolation.md) |
| [`path1_plan4.md`](./path1_plan4.md) | ARC-Easy cross-benchmark validity | **Executed** — see [RFC 008](./rfc-008-arc-easy-cross-benchmark.md) |
| [`path1_plan5.md`](./path1_plan5.md) | 0-shot vs 8-shot CoT | Ready to execute |

## Reading order recommendations

### If you're a new context window picking up the investigation
Read in this order:
1. **RFC 001** (this document) for the lay of the land
2. **RFC 005** (methodology) — before reading any findings, understand the harness, the fixes that were needed, and the sanity-check discipline we adopted
3. **RFC 002** (gate) and **RFC 003** (plateau) — the two primary findings
4. **RFC 004** (repetition loops) — cross-cutting failure-mode observation
5. **RFC 006** (open questions) and the three plan files — what remains to do

### If you only have time for the headline
Read **RFC 002**, **RFC 003**, and the "Path 1 representative" section of **RFC 006**. That's enough to drop Path 1 into the four-paths head-to-head correctly.

### If you're the implementer agent dispatching the next experiments
Read **RFC 005** for harness constraints, then go straight to the `path1_plan*.md` files. Each plan is self-contained with sanity checks, budgets, and pre-registered interpretation.

## Path 1 representative for the four-paths head-to-head

Based on findings to date, the Path 1 entry is:

> **Gemma 4 E2B-it, 8-shot CoT (Wei et al. 2022 exemplars), greedy decode, `max_new_tokens = 512`. GSM8K accuracy: 30.0% ± 4.1 (95% CI) at n = 500. ARC-Easy accuracy: 82.8% ± 3.3 at n = 500.**

GSM8K is cell `A3_len512` in plan 2. ARC-Easy is cell `A3_arc` in plan 4 (matched protocol with hand-crafted 8-shot CoT exemplars; see [RFC 008](./rfc-008-arc-easy-cross-benchmark.md)). On ARC-Easy specifically, an 8-shot direct cell (`direct_arc`) lands at the same accuracy at ~30× less compute — disclosed as a footnote on the head-to-head entry rather than the primary representative.

## One-sentence takeaways

These are the claims Path 1 contributes to the overall investigation — the things other paths have to improve on or explain.

- **Chain-of-thought is necessary on E2B-it for non-trivial GSM8K accuracy.** Without it, the model scores at ~1% on 8-shot direct prompting. **The same CoT-vs-direct gate does not fire on ARC-Easy** — direct prompting matches CoT at 8-shot (see [RFC 008](./rfc-008-arc-easy-cross-benchmark.md)). The +30-pt CoT lift is GSM8K-specific.
- **Beyond that CoT lift, inference-time compute does not move the needle on GSM8K or ARC-Easy.** Neither longer generation (up to 1024 tokens) nor self-consistency voting (up to k = 10 chains) produces a statistically significant gain over the greedy 512-token baseline. The plateau is cross-benchmark — same +2.2-pt SC-vs-greedy lift on both rigs, both insignificant.
- **The repetition-loop failure mode is a small contributor to the GSM8K ceiling and is essentially absent on ARC-Easy.** Plan-3 isolation lands "B-leaning-A" (rep-loops cost 0–3 pts on top of a hard reasoning ceiling on GSM8K, see [RFC 007](./rfc-007-repetition-loop-isolation.md)); plan-4 measures ARC-Easy rep-loop rate at 0.6% (vs GSM8K's 10.2%, see [RFC 008](./rfc-008-arc-easy-cross-benchmark.md)). The four-paths scoring axis does not need a separate generation-stability metric.

## Provenance

Findings produced across plan 1 (n=100 gate) and plan 2 (n=500 length and self-consistency sweeps). Raw JSONL per-problem data preserved at:

- `results/path_1_cot_tokens/plan1/it__cot__0000_0100.jsonl`, `base__cot__0000_0100.jsonl`, `it__direct__0000_0100.jsonl`, `base__direct__0000_0100.jsonl`
- `results/path_1_cot_tokens/plan2/A1_len128__0000_0500.jsonl` through `A4_len1024__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/B1_k1__0000_0500.jsonl` through `B4_k10__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan1/results_gate.json`, `plan2/results_plan2.json`

All manifests pinned with model commit SHAs (see RFC 005).

## D3 visualizations

Five standalone HTML artifacts render the key results. All have D3 v7 inlined — they work offline without any CDN:

- `fig1_gate_accuracy.html` — plan 1 gate, 4 cells with Wilson CIs
- `fig2_failure_modes.html` — plan 1 CoT failure-mode breakdown, stacked bars
- `fig3_sweeps.html` — plan 2 axis A and axis B, side-by-side
- `fig4_pareto.html` — plan 2 accuracy vs tokens, log-scale
- `fig5_chain_vs_vote.html` — plan 2 chain-level vs voted accuracy decomposition

---

*This RFC last updated: end of active investigation thread, before continuation in fresh context.*
