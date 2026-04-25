# RFC 003 — The Inference-Compute Plateau on GSM8K

| Field | Value |
|---|---|
| **RFC** | 003 |
| **Title** | The Inference-Compute Plateau on GSM8K |
| **Status** | Finding (pre-registered, confirmed as honest null) |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Depends on** | [RFC 002 (gate)](./rfc-002-cot-gate-finding.md), [RFC 005 (methodology)](./rfc-005-methodology.md) |
| **Related** | [RFC 004 (repetition)](./rfc-004-repetition-loop-failure-mode.md) |
| **Source plan** | [`path1_plan2.md`](./path1_plan2.md) |

## Abstract

On Gemma 4 E2B-it with 8-shot CoT prompting, GSM8K accuracy does not improve beyond what greedy decoding at 128 tokens already achieves. An eight-cell sweep across `max_new_tokens ∈ {128, 256, 512, 1024}` (greedy) and `k ∈ {1, 3, 5, 10}` sampled chains with majority voting produced accuracies in a 2.6-point band from 29.6% to 32.2% (n = 500 each). Every pairwise comparison fails McNemar significance. The best cell (B4 at k = 10) exceeds the greedy 512-token reference by +2.2 points (p = 0.193). Path 1 has a ceiling, and the ceiling does not move with inference-time compute.

## Motivation

[RFC 002](./rfc-002-cot-gate-finding.md) established that CoT beats direct prompting by a wide margin. Plan 2 asked the follow-up:

> Given that CoT helps, how much more accuracy can we buy by spending more inference-time compute on the same prompt — and what is the shape of the compute-vs-accuracy curve?

Two mechanisms were pre-registered:
1. **Generation length** — greedy decode with progressively longer `max_new_tokens`.
2. **Self-consistency** — sample k chains at temperature > 0, majority-vote the parsed answers.

These were selected before execution and are the two levers Path 1 has that do not require engineering or architectural changes.

## Experimental setup

### Protocol

- **Model:** `google/gemma-4-E2B-it`, commit `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf`.
- **Benchmark:** GSM8K, first `n = 500` problems of the test split (superset of plan 1's first 100).
- **Prompting:** 8-shot Wei et al. CoT exemplars, exemplar hash `a33e6d90c6844317` (byte-identical to plan 1).
- **Dtype:** bf16.
- **Seed:** `torch.manual_seed(0)` set once at script start.

### Axis A — generation-length sweep (greedy)

| Cell | `max_new_tokens` | Decode |
|---|---|---|
| `A1_len128` | 128 | Greedy |
| `A2_len256` | 256 | Greedy |
| **`A3_len512`** (reference) | 512 | Greedy |
| `A4_len1024` | 1024 | Greedy |

A3 is the Path 1 reference cell. It matches plan 1's `it:cot` protocol exactly and serves as the cross-plan sanity check.

### Axis B — self-consistency sweep (sampled)

| Cell | k chains | Decode |
|---|---|---|
| `B1_k1` | 1 | Sampled (T = 0.7, top_p = 0.95), 512-token cap |
| `B2_k3` | 3 | Same, majority vote |
| `B3_k5` | 5 | Same, majority vote |
| `B4_k10` | 10 | Same, majority vote |

Majority vote: plurality of parsed-integer answers, ties broken first-chain-wins.

### Pre-registered interpretation bands

Committed before looking at results.

**Axis A:**
- KNEE-AT-512 — A3 and A4 within 3 points; A1/A2 below by > 5.
- MONOTONIC-LIFT — A4 ≥ A3 + 5 with non-overlapping CIs.
- SATURATED-EARLIER — A2 and A3 within 2 points.

**Axis B (SC vs greedy A3):**
- SC-WINS — B3 or B4 beats A3 by ≥ 8 points with non-overlapping CIs.
- SC-FLAT — B3 within 3 points of A3.
- SC-HURTS — B3 < A3 by ≥ 3 points.

## Results

### Axis A — generation length

| Cell | Accuracy | 95% CI | Mean gen tokens | Truncation rate |
|---|---|---|---|---|
| `A1_len128` | 149/500 = 29.8% | [0.260, 0.340] | 124.6 | 92.2% (hit the cap) |
| `A2_len256` | 150/500 = 30.0% | [0.261, 0.342] | 233.1 | 76.0% |
| **`A3_len512`** | 150/500 = 30.0% | [0.261, 0.342] | 403.4 | 63.4% |
| `A4_len1024` | 150/500 = 30.0% | [0.261, 0.342] | 723.8 | 62.4% |

**Paired McNemar vs A3:** all p ≥ 1.00.

### Axis B — self-consistency

| Cell | Voted accuracy | 95% CI | Chain-level avg | Vote-degeneracy |
|---|---|---|---|---|
| `B1_k1` | 154/500 = 30.8% | [0.269, 0.350] | 0.308 | 99.2% (trivial at k = 1) |
| `B2_k3` | 159/500 = 31.8% | [0.279, 0.360] | 0.297 | 40.6% |
| `B3_k5` | 148/500 = 29.6% | [0.258, 0.337] | 0.279 | 31.6% |
| `B4_k10` | 161/500 = 32.2% | [0.283, 0.364] | 0.290 | 23.8% |

**Paired McNemar vs A3:**

| Comparison | Only A3 | Only cfg | Both | p |
|---|---|---|---|---|
| A3 vs B1 | 26 | 30 | 124 | 0.689 |
| A3 vs B2 | 23 | 32 | 127 | 0.281 |
| A3 vs B3 | 34 | 32 | 116 | 0.902 |
| A3 vs B4 | 24 | 35 | 126 | 0.193 |

None significant.

### Cross-plan reproducibility

A3_len512 restricted to the first 100 problems: **99/100 completions byte-identical** with plan 1's `it__cot__0000_0100.jsonl`. Correctness flags match 100/100. The one text mismatch is a benign early-termination difference with the same final answer. Harness drift ruled out.

## Interpretation

### Axis A verdict: **SATURATED-EARLIER**, earlier than the pre-registered band assumed

The pre-registered KNEE-AT-512 interpretation assumed saturation around 512 tokens. The actual saturation is at 128 tokens. This is both more surprising and more consequential:

- Of 500 A3 problems, 317 (63%) hit the 512-token cap — i.e., the model was not done reasoning when it was cut off — **yet accuracy is identical to A4 at 1024 tokens**. Extending the cap lets those 317 problems continue generating, but the additional generation does not produce any additional correct answers.
- A1 at 128 tokens has 92% truncation (almost no problem finishes reasoning) and still scores 29.8%.

**The strongest claim supported by this data: the problems E2B-it can solve on GSM8K are answered by the first ~128 tokens of reasoning. The problems it cannot solve are not solved by more tokens — those tokens either enable repetition-loop failures (see [RFC 004](./rfc-004-repetition-loop-failure-mode.md)) or produce verbose incorrect paths.**

### Axis B verdict: **SC-FLAT**

B4 at k = 10 is the best cell at 32.2%, a +2.2 point lift over A3 (30.0%). The paired McNemar p = 0.193 is well short of any conventional significance threshold. This falls squarely in the pre-registered SC-FLAT band (± 3 points).

One subtlety worth surfacing: **sampled chains are worse than greedy.** Chain-level accuracy averages (across all k × n chains) are 30.8% (k=1), 29.7% (k=3), 27.9% (k=5), 29.0% (k=10). Voting across k chains recovers some of this: +2.1 pts at k=3, +1.7 at k=5, +3.2 at k=10. But the vote only partially compensates for the sampling-vs-greedy penalty. It does not meaningfully exceed greedy.

At T = 0.7, sampled chains are producing real diversity — vote-degeneracy is 24% at k = 10, well below the "sampler is deterministic" warning threshold of 0.95. So the plateau is not caused by the sampler failing to explore. The plateau appears to be a genuine ceiling.

### The plateau is a real property of E2B-it, not a harness artifact

Two strong reasons to trust this conclusion:

1. **The A3-plan1 cross-check.** Plan 2's A3 reproduces plan 1's IT-CoT on 99/100 shared problems. The harness is stable.
2. **The Axis A uniformity is implausible as artifact.** Four cells, scoring 149, 150, 150, 150 out of 500, is not a statistical fluke — it is four cells converging on nearly identical correct-problem sets. `A4` has 150 correct, A3 has 150 correct, and the McNemar discordance between them is 0/0. This reproducibility across lengths is the signature of a real ceiling, not noise.

## Consequences for the four-paths head-to-head

1. **Path 1's GSM8K entry is A3 at 30.0% ± 4.1.** The cheaper A1 cell at 128 tokens matches A3's accuracy at one-fourth the compute, but A3 is the defensible primary entry because it has headroom for longer-tail problems that occasionally need it.
2. **Paths 2, 3, and 4 face a fixed, known baseline.** The target to beat is 30% on GSM8K. Any path that meaningfully exceeds 30% at comparable compute is a genuine win.
3. **The "just use the quantized sibling" risk to the overall thesis (Path 3) is heightened.** With Path 1 pinned at 30%, the baseline is low enough that a properly quantized E4B has a plausible shot at dominating. This matches the up-front flag in the `four-paths-one-phone-visual-guide.md`.

## Limitations and residual uncertainty

Three specific things plan 2 did not close:

- **This finding is GSM8K-only.** The plateau may or may not hold on ARC-Easy. Pre-registered plan 4 closes this.
- **The prompt format is pinned at 8-shot Wei et al.** 0-shot CoT might land at a different ceiling. Pre-registered plan 5 closes this.
- **The repetition-loop failure mode was characterized at n = 100.** Its full impact across the n = 500 × 8-cell surface is quantified in pre-registered plan 3.

See [RFC 006](./rfc-006-open-questions-and-next-steps.md) for the pre-registered follow-ups.

## Evidence preserved

Per-problem JSONL, n = 500 each:
- `results/path_1_cot_tokens/plan2/A1_len128__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/A2_len256__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/A3_len512__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/A4_len1024__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/B1_k1__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/B2_k3__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/B3_k5__0000_0500.jsonl`
- `results/path_1_cot_tokens/plan2/B4_k10__0000_0500.jsonl`

Axis B JSONLs include the `chains` list — all k per-chain completions per problem — so chain-level analysis can be recomputed without rerunning the sweep.

Summary: `results/path_1_cot_tokens/plan2/results_plan2.json`.
Pinned config: in the summary JSON — includes model commits, library versions, exemplar hash.
Script: `path1_length_and_sc.py` with resume-on-restart.

Visualizations: [`fig3_sweeps.html`](./fig3_sweeps.html), [`fig4_pareto.html`](./fig4_pareto.html), [`fig5_chain_vs_vote.html`](./fig5_chain_vs_vote.html).

## References

- Wang, X. et al. (2022). *Self-Consistency Improves Chain-of-Thought Reasoning in Language Models.* arXiv:2203.11171. Original self-consistency method.
- [RFC 002](./rfc-002-cot-gate-finding.md) — the gate result this plateau finding builds on.
- [RFC 004](./rfc-004-repetition-loop-failure-mode.md) — the failure mode that partially explains the plateau shape.

---

*This RFC documents the primary finding from plan 2. It is supplemented, but not replaced, by the results of the three pre-registered follow-up plans.*
