# RFC 004 — The Repetition-Loop Failure Mode

| Field | Value |
|---|---|
| **RFC** | 004 |
| **Title** | The Repetition-Loop Failure Mode |
| **Status** | Observation (cross-cutting, under investigation) |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Depends on** | [RFC 002 (gate)](./rfc-002-cot-gate-finding.md), [RFC 003 (plateau)](./rfc-003-inference-compute-plateau.md) |
| **Completed by** | [`path1_plan3.md`](./path1_plan3.md) — pre-registered quantitative analysis |

## Abstract

A meaningful fraction of Gemma 4 E2B-it's chain-of-thought completions on GSM8K collapse into degenerate repetition — the model falls into a short repeating sub-sequence and continues emitting it until it hits the generation cap. At n = 100 the rate is 13% of completions. The base model does not exhibit this failure mode at the same rate (3%). This is a generation-stability problem distinct from a reasoning-capability problem and has implications for how Path 1's ceiling should be interpreted against the other paths in the head-to-head.

## Observation

While classifying failures in plan 1's n = 100 IT-CoT cell, a distinct failure signature emerged: completions that contain a short phrase repeating three or more times in a row until the 512-token generation cap is reached. Examples from the actual data:

- `idx 7`: `"Normally it takes 20 minutes. Normally it takes 20 minutes. Normally it takes 20 minutes. ..."`
- `idx 12`: `"costs $3 a year. costs $3 a year. costs $3 a year. ..."`
- `idx 14`: `"20% enrolled. 20% enrolled. 20% enrolled. ..."`

These are qualitatively different from other failures in the distribution. The model is not reasoning badly — it is not reasoning at all. It has entered a decoding attractor and cannot escape it within the generation budget.

## Measurement method

### Detection regex

The canonical detector used across all RFCs and the figures:

```python
re.search(r'(.{10,60})\1{2,}', completion)
```

A substring of 10–60 characters that repeats at least three times immediately after itself. This is a conservative definition: it will miss paraphrased repetition ("I count 5 apples. I see 5 apples. There are 5 apples.") and will not detect near-verbatim loops shorter than 10 characters.

### Outcome taxonomy used in all failure-mode analysis

Every non-correct completion is classified into exactly one of:

1. **`correct`** — parsed answer matches gold.
2. **`terminated_wrong`** — generation ended cleanly (with a `####` marker or a natural sentence terminator) but the parsed answer is wrong. This is a genuine reasoning error.
3. **`repetition_loop`** — detection regex triggers. Subsumes any case where the generation is stuck in a verbatim loop, regardless of whether a `####` was also emitted earlier.
4. **`truncated_no_answer`** — hit the generation cap without emitting a parseable answer, and the detection regex did not trigger. This is a third, distinct failure mode: the model ran out of room while still producing coherent-looking text.

## Quantitative findings from existing data

### Plan 1, n = 100

| Model + prompt | Correct | Terminated wrong | Repetition loop | Truncated no answer |
|---|---|---|---|---|
| IT, 8-shot CoT | 31 | 49 | **13** | 7 |
| Base, 8-shot CoT | 22 | 67 | **3** | 8 |

The IT model is ~4× more prone to repetition collapse than the base model at the same prompt and decoding settings.

### Plan 2, n = 500 (partial data — a full systematic pass is pre-registered in plan 3)

From spot analysis during plan 2:
- `A3_len512`: 317/500 completions hit the 512-token cap. The fraction of those that are verbatim rep-loops vs coherent-but-long reasoning was not enumerated in plan 2; it is one of the targets of plan 3.
- `B1_k1`: per-chain median gen tokens = 77, much shorter than A3's median. Sampled chains terminate earlier. Chain-level rep-loop rate appears lower than A3's by inspection, but was not quantified.

## Why this matters — two interpretations of Path 1's ceiling

Path 1's ceiling on GSM8K is 30% (per [RFC 003](./rfc-003-inference-compute-plateau.md)). This ceiling has two possible decompositions:

### Hypothesis A: Reasoning-bounded ceiling
The 30% ceiling reflects the model's reasoning capacity. The 13% of completions that rep-loop are a separate surface phenomenon that does not interact with the accuracy ceiling — every rep-loop completion would also have been a reasoning failure.

### Hypothesis B: Stability-influenced ceiling
Some subset of rep-loop completions would have produced a correct answer if the model had not fallen into the loop. Under this hypothesis, the 30% ceiling is artificially low by the "accuracy cost of rep-loops" — a number that could range from 0 to 13% depending on how many rep-loop problems were recoverable.

These have very different implications for the four-paths head-to-head:

- Under hypothesis A, Paths 2–4 must beat Path 1 on genuine reasoning capability. A quantized E4B (Path 3) would need more reasoning capacity; depth-recurrence (Path 2) would need healing to add reasoning depth.
- Under hypothesis B, Paths 2–4 can pick up some accuracy by reducing rep-loop rate while holding reasoning constant. Mixture-of-Depths (Path 4) specifically has a plausible mechanism here — per-token compute routing might prevent the model from getting stuck in low-compute attractors.

**We do not yet know which hypothesis is correct.** Plan 3 is designed to decide this with two counterfactual accuracies on the existing data (lenient and upper-bound adjusted), neither of which replaces the headline 30% but both of which bound the stability-cost contribution.

## Why the IT model is more susceptible

One speculative note, not a confirmed finding: instruction-tuned models have more distinctive "voice" patterns (response templates, hedging phrases, characteristic sentence structures) than their base counterparts. If a greedy decode lands on one of those patterns, the same token sequence can extend itself deterministically — each repetition of the pattern increases the probability of re-entering the same pattern. The base model, being less committed to any particular voice, is less likely to be trapped this way.

This hypothesis predicts that:

- Lower temperature → more rep-loops (consistent with: A3 greedy has rep-loops; sampled B1/B2/B3/B4 appear to have fewer per chain).
- Problems whose correct solution has a natural phrase-level repetition structure (e.g., "First we compute A. Then we compute B. Then we compute C.") are more likely to trigger loops.

Both predictions are testable on existing plan 2 data, and plan 3 is structured to test the second one (the "sticky problems" analysis).

## Interaction with self-consistency

An important observation from plan 2 axis B: chain-level rep-loop rates matter for how well self-consistency can work.

- If individual chains rep-loop independently, a majority vote across k chains can still recover the correct answer — as long as a plurality of non-rep-loop chains agree on it.
- If rep-loops concentrate on the same problems (some problems are attractors for sampling too), the vote will often be between correct answers from non-rep-looped chains and a rep-loop-answer from multiple rep-looped chains. Whichever has plurality wins; if the rep-loops outnumber, the vote is wrong even when the reasoning is recoverable.

Plan 3 will quantify:
1. Per-chain rep-loop rate in axis B vs A3's rate.
2. Problems where self-consistency voted incorrectly specifically because rep-loop chains outvoted correct chains. This is the **"rep-loop-plurality-wrong"** category in the plan 3 analysis.

If rep-loop-plurality-wrong is a non-negligible fraction of vote errors, it is a concrete mechanism by which SC is being dragged down — and a mechanism Path 4 (Mixture-of-Depths) could plausibly address.

## Caveats on the regex detector

Three limitations the writeup should flag:

- **False positives.** Legitimate arithmetic traces can contain verbatim repeats ("10 × 3 = 30. 30 + 10 = 40. 40 × 3 = 120. 120 + 40 = 160. ..."). Spot-check 30 flagged completions manually before reporting the rate; if false-positive rate is > 20%, tighten the regex to `(.{15,80})\1{2,}` and re-analyze.
- **False negatives.** Paraphrased repetition is missed entirely. The true rep-loop rate is likely 10–20% higher than the detector catches.
- **Language-specific.** Works for English GSM8K; may need recalibration for any future multilingual benchmark.

These caveats belong in the methodology section of any external write-up, not in the headline claims.

## Evidence preserved

All per-problem JSONL listed in RFC 002 and RFC 003 is sufficient to re-run the classification from scratch. No new data is needed. The plan 3 script (`path1_rep_loop_analysis.py`, per `path1_plan3.md`) produces:

- `results_plan3.json` — per-cell counts in the four-category taxonomy
- `path1_rep_loops_by_cell.csv` — ready for downstream plotting
- Top-10 sticky-problem IDs with their GSM8K text

## Open questions not closed by this RFC

- **Quantitative contribution to the ceiling.** See plan 3 pre-registered outcomes A / B / C (small / medium / large stability contribution).
- **Sampling-vs-greedy asymmetry.** Whether T = 0.7 sampling genuinely reduces rep-loop rate, and if so, by how much. See plan 3 step 4.
- **Interaction with prompt format.** Does 0-shot CoT (plan 5) have the same rep-loop rate as 8-shot, or is the exemplar context itself priming the loops? Could be worth a follow-up analysis on plan 5 data once it exists.

## References

- [RFC 002](./rfc-002-cot-gate-finding.md) — source of the n = 100 classification.
- [RFC 003](./rfc-003-inference-compute-plateau.md) — the plateau finding this RFC partially explains.
- [`path1_plan3.md`](./path1_plan3.md) — the analysis that closes this RFC's quantitative gaps.
- Holtzman, A. et al. (2020). *The Curious Case of Neural Text Degeneration.* arXiv:1904.09751 — background on repetition attractors in greedy decoding.

---

*This RFC is a cross-cutting observation that arose during plans 1 and 2 and will be sharpened quantitatively by plan 3. If plan 3 reveals any of the pre-registered outcomes (A / B / C / D / E), update this RFC with the verdict.*
