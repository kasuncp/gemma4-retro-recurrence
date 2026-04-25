# RFC 006 — Open Questions and Next Steps

| Field | Value |
|---|---|
| **RFC** | 006 |
| **Title** | Open Questions and Next Steps |
| **Status** | Planning (plan 3 and plan 4 executed, plan 5 pending) |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Depends on** | [RFC 002](./rfc-002-cot-gate-finding.md), [RFC 003](./rfc-003-inference-compute-plateau.md), [RFC 004](./rfc-004-repetition-loop-failure-mode.md), [RFC 005](./rfc-005-methodology.md) |
| **Related plans** | [`path1_plan3.md`](./path1_plan3.md), [`path1_plan4.md`](./path1_plan4.md), [`path1_plan5.md`](./path1_plan5.md) |

## Abstract

After plans 1 and 2, Path 1 has a defensible representative for the four-paths head-to-head (E2B-it at A3, 30.0% on GSM8K). Three specific questions remain, each of which could materially change how Path 1 is interpreted or positioned. Each has a pre-registered experiment plan ready to execute. This RFC explains what each question is, why it matters, and how the three plans relate to one another.

## State of Path 1 as of this RFC

### What is known

- CoT beats direct prompting by a wide margin on E2B-it GSM8K ([RFC 002](./rfc-002-cot-gate-finding.md)).
- Inference-time compute beyond greedy 512 tokens does not improve GSM8K accuracy ([RFC 003](./rfc-003-inference-compute-plateau.md)).
- IT-CoT completions collapse into repetition loops ~13% of the time; the base model rarely does ([RFC 004](./rfc-004-repetition-loop-failure-mode.md)).
- The Path 1 GSM8K representative is `A3_len512` at 30.0% ± 4.1.

### What is not known

Three specific gaps the remaining plans close.

## Question 1 — Does the plateau generalize to ARC-Easy?

**Full plan:** [`path1_plan4.md`](./path1_plan4.md). **Status:** executed — see [RFC 008](./rfc-008-arc-easy-cross-benchmark.md). Outcome A (plateau confirmed) primary, outcome D (CoT does not beat direct on ARC-Easy) secondary. The remainder of this section is preserved for context; the verdict lives in RFC 008.

### Why this matters

Everything in [RFC 003](./rfc-003-inference-compute-plateau.md) is a GSM8K-specific finding. GSM8K has two failure modes ARC-Easy does not:

1. Repetition loops ([RFC 004](./rfc-004-repetition-loop-failure-mode.md)) are a generation-level failure; ARC-Easy scores via log-likelihood over four answer choices and is immune.
2. Truncation matters on GSM8K (63% of A3 completions hit the 512-token cap) and does not matter on ARC-Easy.

Either of these mechanisms could mean ARC-Easy shows a self-consistency or length response that GSM8K hides. If it does, [RFC 003](./rfc-003-inference-compute-plateau.md)'s plateau claim needs a GSM8K-specific asterisk, and the Path 1 ARC-Easy representative might be a different cell than the Path 1 GSM8K representative.

### The experiment

Two new cells on ARC-Easy at n = 500, both with the same model and pinning as plan 2:
- `A3-arc` — mirror of plan 2's A3 (greedy, 512 tokens)
- `B3-arc` — mirror of plan 2's B3 (k = 5 self-consistency)

Optional third cell `direct-arc` for completeness (mirrors plan 1's direct cell on the new benchmark). Total budget ≈ 3.5 hr on a single 3090.

### What outcomes mean

- **Plateau confirmed** — B3-arc − A3-arc within ±3 points. Path 1's plateau story holds cross-benchmark. Strengthens the head-to-head claim.
- **Self-consistency lifts on ARC-Easy only** — if B3-arc − A3-arc ≥ 5 points. Path 1's head-to-head entries differ per benchmark. Also tells us [RFC 004](./rfc-004-repetition-loop-failure-mode.md)'s rep-loops were suppressing SC lift on GSM8K, which is a real finding in its own right.
- **Benchmark saturation** — both cells > 90%. ARC-Easy is not differentiating paths; consider adding ARC-Challenge to the rig.
- **CoT does not beat direct on ARC-Easy** — only diagnosable if direct-arc is run. Surprising and would reopen the gate question.

## Question 2 — Is 8-shot the best prompt format, or is the ceiling prompt-format-specific?

**Full plan:** [`path1_plan5.md`](./path1_plan5.md).

### Why this matters

Plans 1 and 2 pinned the prompt format at 8-shot Wei et al. exemplars and swept every other axis inside that format. The 30% ceiling could be a reasoning ceiling on the model, or it could be a 8-shot-specific ceiling that a different prompt format would clear.

The relevant prior: IT models frequently match or beat 8-shot accuracy at 0-shot because their instruction-tuning explicitly teaches step-by-step reasoning. The Wei et al. exemplars are from 2022 (pre-IT-ubiquity) and may anchor the model's style to the exemplars rather than to the question.

If 0-shot CoT lifts accuracy above 30%, the plan 2 plateau is partly artificial and Path 1 needs a re-baseline.

### The experiment

Two new cells on GSM8K at n = 500, same model and pinning as plan 2:
- `C1_zeroshot_simple` — 0-shot with `"Let's think step by step."` prefix, via chat template
- `C2_zeroshot_plain` — 0-shot, no prefix, via chat template

Compared against A3 from plan 2 as the 8-shot reference (no re-run). Total budget ≈ 1.5 hr.

### What outcomes mean

- **8-shot is optimal (C1 ≈ C2 ≈ A3 at 30%)** — the ceiling is real. Path 1's plan 2 story is strengthened.
- **0-shot beats 8-shot (C1 ≥ A3 + 5 pts)** — the plan 2 ceiling is partly artificial. Path 1's representative switches from A3 to C1. A follow-up plan would need to re-run the length and SC sweeps on C1's prompt format to re-establish the new ceiling shape. This is an expensive but honest outcome.
- **Plain 0-shot ≈ 0-shot simple ≈ 8-shot** — the IT model CoTs on its own. Pick the cheapest format (plain 0-shot) as the Path 1 representative; saves exemplar tokens on every call.

### Design note

Run plan 5 after plan 3 but before or alongside plan 4. If plan 5 outcome B fires (0-shot beats 8-shot), plan 4's ARC-Easy cells should be re-specified to use the new preferred format before running — running plan 4 first on the current 8-shot ARC spec and then learning plan 5 shifted the baseline would waste the ARC compute.

## Question 3 — How much of the plateau is reasoning vs stability?

**Full plan:** [`path1_plan3.md`](./path1_plan3.md). Pure analysis, no GPU needed.

### Why this matters

Suppose 5 of the 13 rep-loop problems in plan 1's IT-CoT cell were solvable by the model's reasoning capacity, but the generator entered a repetition attractor before producing the answer. Under this hypothesis, the 30% ceiling is (roughly) 25% reasoning-bounded plus 5% stability-cost.

This matters for the four-paths head-to-head because it changes what the other paths have to accomplish:

- **If stability cost is ~0** (all rep-loops were also reasoning failures): Paths 2–4 must beat Path 1 on genuine reasoning capacity.
- **If stability cost is ~5 pts**: Paths 2–4 can pick up "free" accuracy by reducing rep-loops while holding reasoning constant. Path 4 (Mixture-of-Depths) has the most natural mechanism for this — per-token compute routing could help escape attractors.
- **If stability cost is > 8 pts**: The four-paths comparison has a confound. A path that reduces rep-loops while leaving reasoning unchanged will appear to "win" on accuracy. The scoring axis may need to control for this.

### The experiment (analysis only)

Systematic re-analysis of existing JSONL from plans 1 and 2 — no new generations. Classify every completion and every axis-B chain into four outcome buckets (correct / terminated-wrong / rep-loop / truncated-no-answer) and compute two counterfactual adjusted accuracies (lenient and upper-bound).

### What outcomes mean

- **Small factor** (< 3 pts of ceiling): Path 1's ceiling is predominantly a reasoning ceiling. Other paths must exceed it on reasoning.
- **Medium factor** (3–8 pts): Ceiling is part reasoning, part stability. Writeup notes the decomposition; Path 4 framing in particular gets a concrete stability-aid angle.
- **Large factor** (> 8 pts): Reframe the head-to-head. Consider adding a generation-stability metric alongside accuracy.

### Order of operations

Plan 3 is free (no GPU) and fast (~2 minutes on CPU). **Run plan 3 first.** Its outcome may inform how you interpret plans 4 and 5.

## Recommended run order

1. **Plan 3 (rep-loop analysis)** — runs on existing data, 2 minutes, results inform the other two.
2. **Plan 5 (0-shot vs 8-shot)** — cheapest GPU plan, 1.5 hr. If it shifts the Path 1 representative, plan 4 needs to be re-specified before running.
3. **Plan 4 (ARC-Easy)** — 3.5 hr. Uses whatever representative plan 5 settles on (A3 or C1 or similar).

This order is not strictly sequential — plans 3 and 4 are logically independent — but it is the one that minimizes wasted compute and maximizes information value per hour of GPU time.

## What this RFC does **not** propose

Several plausible-sounding experiments are deliberately not recommended:

- **Temperature sweep on self-consistency.** Expected effect size is within CI of existing results; sampler is already producing real diversity (vote-degeneracy at k = 10 is 24%). Low signal-per-compute.
- **Deliberation/reflection prompts ("verify your answer", etc.).** Plan 1 explicitly scoped these out to keep the gate clean, and they belong to a different leg. Adding them now is scope creep.
- **k > 10 on self-consistency.** Diminishing returns — extrapolating, k = 20 gets ≈ +3 pts instead of +2.2. Still insignificant.
- **n > 500 for tighter CIs.** n = 500 already gives ±4 pp half-width at p = 0.30. Going to n = 1000 halves that, but no pre-registered decision rule hinges on that precision.

These are explicitly noted here so a future investigator does not spend compute on them without a specific reason to revisit the decision.

## End-state condition for Path 1

Path 1 is considered "characterized for the head-to-head" when the following are all true:

1. [RFC 002](./rfc-002-cot-gate-finding.md) gate established — **done**.
2. [RFC 003](./rfc-003-inference-compute-plateau.md) plateau characterized on GSM8K — **done**.
3. Plan 3 outcome labeled (small / medium / large rep-loop contribution) — **done**, B-leaning-A, see [RFC 007](./rfc-007-repetition-loop-isolation.md).
4. Plan 4 outcome labeled (plateau confirmed / SC-lifts-on-ARC / saturation / other) — **done**, A primary + D secondary, see [RFC 008](./rfc-008-arc-easy-cross-benchmark.md).
5. Plan 5 outcome labeled (8-shot optimal / 0-shot better / all equal) — **pending**.
6. Path 1 representative cell(s) confirmed for both GSM8K and ARC-Easy — **GSM8K = A3_len512 at 30.0%; ARC-Easy = A3_arc at 82.8%**. Plan 5 may shift the GSM8K representative if 0-shot beats 8-shot.

When all six are checked, Path 1 is ready for the head-to-head and a consolidated write-up. No further Path 1 experiments should be scheduled without either a new plan document or an explicit revision to this RFC.

## Successor RFCs expected

After plans 3, 4, and 5 complete, three new RFCs (007, 008, 009 or similar) will document their findings in the same style as RFCs 002–004. This RFC will then be updated with a "Superseded by" reference to those new findings.

## References

- [RFC 001](./rfc-001-index.md) — index.
- [RFC 002](./rfc-002-cot-gate-finding.md), [RFC 003](./rfc-003-inference-compute-plateau.md), [RFC 004](./rfc-004-repetition-loop-failure-mode.md) — the findings these plans build on.
- [RFC 005](./rfc-005-methodology.md) — normative methodology all three plans must follow.
- [`path1_plan3.md`](./path1_plan3.md), [`path1_plan4.md`](./path1_plan4.md), [`path1_plan5.md`](./path1_plan5.md) — executable plans.

---

*This RFC is actively planning-stage. Update it as the three plans complete, and supersede it with new finding RFCs once all three have executed.*
