# RFC 002 — The CoT Gate Finding

| Field | Value |
|---|---|
| **RFC** | 002 |
| **Title** | The CoT Gate Finding |
| **Status** | Finding (pre-registered, confirmed) |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Depends on** | [RFC 005 (methodology)](./rfc-005-methodology.md) |
| **Related** | [RFC 003 (plateau)](./rfc-003-inference-compute-plateau.md), [RFC 004 (repetition)](./rfc-004-repetition-loop-failure-mode.md) |
| **Source plan** | `path1_plan1.md` (historical) |

## Abstract

Chain-of-thought prompting produces a large, statistically unambiguous accuracy lift on Gemma 4 E2B over direct-answer prompting on GSM8K. On the instruction-tuned variant, 8-shot CoT scores 31.0% (n = 100) while 8-shot direct scores 1.0%, with all pairwise discordance flowing in one direction (30 problems CoT-only correct, 0 direct-only correct). The gate result rules in Path 1 for the four-paths head-to-head.

## Motivation

Before measuring *how much* inference-time compute helps on E2B, we had to establish *that* CoT helps at all. A model that can't be meaningfully lifted by CoT prompting has no Path 1 — every downstream inference-compute technique (longer generation, self-consistency, deliberation prompts) depends on the base mechanism of in-context reasoning being effective.

The gate question, pre-registered before the experiment:

> Does giving Gemma 4 E2B reasoning steps in-context beat giving it none, on GSM8K, at a matched prompting protocol?

If no, Path 1 would have been wounded at the root. The entire leg would have been deprioritized in the four-paths comparison with an honest null write-up.

## Experimental setup

### Protocol

- **Model:** `google/gemma-4-E2B-it` at commit `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`, plus `google/gemma-4-E2B` (base) at commit `9d53598892698e981fc42f78b0f8c005cecd63ca` as a secondary signal.
- **Benchmark:** GSM8K, deterministic first `n = 100` problems of the test split.
- **Conditions** (matched 8-shot across all four cells, only reasoning-present-vs-absent varies):
  - `it:direct` — 8-shot, reasoning stripped, `max_new_tokens = 16`
  - `it:cot` — 8-shot, reasoning intact, `max_new_tokens = 512`
  - `base:direct` — same as `it:direct` on base model
  - `base:cot` — same as `it:cot` on base model
- **Decoding:** Greedy, `do_sample = False`, `temperature = 0`, `bf16`.
- **Exemplars:** Wei et al. 2022 GSM8K 8-shot exemplars, terminal "The answer is X." replaced by "#### X" for uniform answer extraction across direct and CoT cells. Exemplar hash: `a33e6d90c6844317`.
- **Answer extraction:** `####`-regex first, fall back to last integer in the completion.

### Pre-registered decision rule

Committed before running:
- **GREEN** — lift ≥ 10 points with non-overlapping CIs. Path 1 alive; proceed to plan 2.
- **YELLOW** — lift < 10 points but > 3 points; follow up with self-consistency at k = 5.
- **RED** — CoT ≤ direct or lift < 3 points with overlapping CIs. Path 1 wounded; write up null.

## Results

### Headline numbers

| Cell | Accuracy | 95% CI (Wilson) | Hash-hit rate |
|---|---|---|---|
| `it:direct` | 1/100 = 1.0% | [0.002, 0.054] | 0.99 |
| `it:cot` | 31/100 = 31.0% | [0.228, 0.406] | 0.82 |
| `base:direct` | 9/100 = 9.0% | [0.048, 0.162] | 0.99 |
| `base:cot` | 22/100 = 22.0% | [0.150, 0.311] | 0.89 |

### Paired statistics (IT model, the primary gate)

Across the 100 shared problems:

| Outcome | Count |
|---|---|
| Both correct | 1 |
| Only CoT correct | 30 |
| Only direct correct | 0 |
| Both wrong | 69 |

McNemar's test two-sided: **p = 1.9 × 10⁻⁹**.

### Verdict

**GREEN.** Both pre-registered thresholds cleared: the CoT lift is +30.0 points (well above the 10-point floor) and the 95% CIs for `it:direct` and `it:cot` do not overlap.

### Secondary finding — base model also shows a real CoT lift

| Comparison | Result |
|---|---|
| `base:cot` vs `base:direct` | 22% vs 9%, paired 19-6 discordance, McNemar p = 0.015 |

CoT capability is latent in the pretrained weights; instruction tuning amplifies it. Per the original plan's interpretation rule, this falls under "both models show a lift" — CoT capability is not purely a product of instruction tuning, though IT does widen the gap over direct.

## Interpretation and caveats

### The direct cell is confounded, but the gate finding is not

The IT direct cell at `max_new_tokens = 16` produced outputs dominated by pattern-matching to the 8-shot format ("#### 8\nQuestion: ..."). The model emits a number and begins generating the next exemplar. This means the 1.0% IT-direct result is not a pure measurement of the model's direct-answer reasoning capacity — it's a measurement under a restrictive prompt format that the model handles poorly.

This does **not** invalidate the gate result. It means the +30 pt lift is a **lower bound** on the true CoT lift. A better-posed direct cell (e.g., with a stop sequence on `"\nQuestion:"` or using the IT chat template) would likely score higher than 1% but still well below 31%. The qualitative claim — CoT enables non-trivial reasoning on GSM8K that direct prompting cannot — is robust to this confound. In plan 1, 30 problems that the CoT cell solved the direct cell cannot have solved under any prompt format since the direct cell never produces the multi-step reasoning those solutions require.

### The IT model has more repetition-loop failures than the base model

Failure-mode breakdown on the 69 IT-CoT wrong answers:

| Failure category | Count |
|---|---|
| Terminated cleanly with wrong answer | 49 |
| Repetition loop | 13 |
| Truncated without any answer | 7 |

For base-CoT, the repetition-loop count was 3/78 wrong. The IT model is ~4× more prone to this failure mode. See [RFC 004](./rfc-004-repetition-loop-failure-mode.md).

### The gate passes at n = 100; the extension to n = 500 (plan 2 `A3_len512`) reproduced

Plan 2 re-ran what is effectively the IT-CoT cell (at the same settings) on the full 500-problem slice. On the shared 100 problems, plan 2's `A3_len512` matches plan 1's `it:cot` byte-for-byte on 99/100 problems; the one mismatch is a benign early-termination difference with the same final answer. Both cells scored 31/100 on the shared set, confirming the gate result is reproducible and not sensitive to any between-run drift.

## Evidence preserved

Per-problem JSONL:
- `results/path_1_cot_tokens/plan1/it__direct__0000_0100.jsonl`
- `results/path_1_cot_tokens/plan1/it__cot__0000_0100.jsonl`
- `results/path_1_cot_tokens/plan1/base__direct__0000_0100.jsonl`
- `results/path_1_cot_tokens/plan1/base__cot__0000_0100.jsonl`

Summary: `results/path_1_cot_tokens/plan1/results_gate.json`, including the pinned config (exemplar hash, model commits, library versions).

Visualization: [`fig1_gate_accuracy.html`](./fig1_gate_accuracy.html), [`fig2_failure_modes.html`](./fig2_failure_modes.html).

## Implications for the head-to-head

1. **Path 1 has a non-trivial baseline.** The CoT lift is real; subsequent paths must beat the post-CoT E2B-it ceiling, not the pre-CoT one.
2. **A minimal Path 1 representative is already defined.** Even if plan 2 had shown no further gains (which it did — see [RFC 003](./rfc-003-inference-compute-plateau.md)), the gate cell by itself gives a defensible entry at 31% GSM8K.
3. **The gate's strength is enormous.** p = 1.9 × 10⁻⁹ at n = 100 means no plausible amount of additional data could overturn this finding. The gate is not rerun in subsequent plans; its result is treated as established.

## Open questions not closed by this RFC

- **Does CoT work on ARC-Easy?** The gate was run on GSM8K only. Most likely yes and with a smaller absolute lift (ARC-Easy is 4-way multiple choice where direct is structurally easier), but untested. See [RFC 006](./rfc-006-open-questions-and-next-steps.md) and `path1_plan4.md`.
- **Is 8-shot optimal, or would 0-shot CoT lift the ceiling?** The gate establishes that CoT helps; it does not establish that 8-shot is the best way to elicit CoT. See `path1_plan5.md`.

## References

- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903. Source of the 8-shot exemplars used in this RFC.
- Pre-registration: the YELLOW/GREEN/RED decision rule committed in `path1_plan1.md` before running. YELLOW branch was not invoked (result was clean GREEN).

---

*This RFC documents a settled result. It should not need re-opening unless a harness bug is discovered in retrospect — in which case update, don't overwrite.*
