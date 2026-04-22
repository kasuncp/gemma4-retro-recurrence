# 10 — Cross-round synthesis and open questions

This document pulls the project's findings together and separates what the
data actually supports from what remains open.

## What the 5 rounds have established

### 1. Naive looping of pretrained Gemma 4 E2B is survivable at small r

Single-layer looping in the valley region degrades log-linearly with r
(Round 1: layer 17 r=8 → 3.4× baseline ppl). No NaN, no cliff, no
catastrophic instability from naive PLE re-injection. This made the
project viable.

### 2. PLE injection policy has small, layer-dependent effects

- `scaled` ≡ `vanilla` in all measurements (Round 2a, confirmed at every r
  tested). Drop `scaled` from future designs.
- `once` vs `vanilla` is **layer-dependent**:
  - At layer 5 (early, non-consumer): `once` is 48% worse at r=8.
  - At layer 17 (valley, consumer): `once` is 14% better at r=8.
  - At layer 28 (late consumer): `once` is 8% worse at r=8.
- Across all 35 layers at r=8 (Round 2c), `once` helps at 16, hurts at 19.
  No systematic pattern. Dropped from Rounds 3+ testing.

### 3. PLE signal is extremely spiky per-layer

Four layers carry most of the PLE contribution at r=1:

| Layer | nll_diff from zeroing PLE |
|-------|-------------:|
| L3 | +1.83 |
| L33 | +0.74 |
| L8 | +0.65 |
| L13 | +0.35 |

The KV-consumer region (layers 15–32) is almost entirely PLE-inert. L3 is
the most extreme PLE spike and sits in the prelude where it would be
included in any retrofit.

### 4. Only layers 15–19 support single-layer looping at r=8

Out of 35 layers, exactly 5 (15, 16, 17, 18, 19) stay within 10× baseline
at r=8 under single-layer vanilla looping. Everything else degrades
100×–10²⁰×.

### 5. The KV producer→consumer boundary (L14→L15) is a hard wall for blocks

- Any block crossing the boundary (C: 12–19, E: 13–22) breaks even at r=2.
- Any block fully inside the consumer zone *and* anchored at L15 works up
  to width 10 (blocks A, B, D, G, H).
- An all-consumer block shifted late (I: 20–27) is catastrophic.
- A distant all-consumer block (F: 25–32) drifts under iteration (fine at
  r=2, broken by r=4).

The constraint is:

> **Recurrent block must start at L15 and cannot extend below it.**
> **Block width can safely reach 10 layers (15–24).** **L25+ has not been
> tested as part of a valley-anchored block.**

### 6. Perplexity stability does NOT imply reasoning stability

This is the project's most consequential finding:

| Block | Width | Perplexity at r=8 | ARC-Easy at r=8 |
|-------|------:|------------------:|---------------:|
| A | 5 | 30.1 | **40.0%** |
| D | 8 | 33.3 | 26.5% |
| G | 10 | 36.0 | 24.5% |
| (baseline) | — | 12.5 | 83.5% |

Wider blocks are marginally worse on perplexity but **substantially
worse on reasoning**. The gradient runs opposite to what "more compute =
better reasoning" would predict. Post hoc explanations exist (wider
blocks accumulate more deviation per token; perplexity is dominated by
local tokens that the looped block preserves; ARC requires maintaining
consistent world-state across a problem statement) but none of them have
been tested.

### 7. The D-r1 and W5-r1 sanity checks are critical

Round 4 recorded `gsm8k_d_r1_vs_baseline: 250/250 matches`, meaning the
block-looping hook during *generation* is bitwise-identical to the
unmodified model at r=1. This is the only thing that lets us rule out
"hook bug during generation" as the cause of Round 4's ARC collapse. Any
future generation-mode experiment should preserve this sanity check.

## What the 5 rounds have NOT established

### What would actually happen with training

The entire project so far is probe-only. Pretrained-only recurrence with
no training is **not** what the paper's method actually proposes — the
paper's key claim is that recurrence plus continued pretraining (Muon
optimizer, recurrence curriculum, healing on minimal-distribution-shift
data, then task-specific refinement) can match or beat the parent model
on math benchmarks.

The Round 4 ARC-Easy collapse (83.5% → 24.5–26.5%) says that **blocks D
and G are bad starting points for training without healing**. It does
*not* say those blocks cannot be trained into useful recurrent models.
The paper's OLMo and Llama results showed similar large drops from
surgery that healing recovered.

But Round 4 *also* says A (the narrow 5-layer block) is a much better
starting point — almost twice the preserved reasoning capability. If
training were to commence now with no further probing, A would be the
defensible choice, not D or G.

### Whether narrow blocks help anywhere, or just degrade less

A-r8 (width 5) preserves 40% ARC-Easy vs baseline 83.5%. That's a loss,
not a gain. The question Round 5 Outcome A would answer is: *does the
trend extend?* If widths 4, 3, 2 produce progressively higher accuracy —
at some point crossing baseline — then narrow-block recurrence in E2B
actually *adds capability*. If widths 4, 3, 2 all plateau in the 35–45%
range, then narrow is just "less bad", not "good", and the paper's
thesis doesn't really hold for E2B without architectural surgery.

### Whether PLE re-injection is a major driver of reasoning damage

Round 2c's across-35-layers data said PLE policy at r=8 has no systematic
effect on perplexity. But ARC-Easy was never measured under different
PLE modes — all Round 4 configs used vanilla (PLE every iteration).
Round 5 Outcome B (the noPLE ablation) is the first direct measurement
of whether PLE injection during recurrence affects reasoning.

### The pair-looping result

Round 3a flagged a suspected hook bug and wasn't debugged. So we don't
actually know whether pair-looping has the valley-rescue behavior that
block-looping turned out to have. This is fine — Round 3b provided
equivalent information via a different hook implementation — but if
there's ever a reason to need clean pair-looping data (e.g., to pin down
where *within* a block the rescue happens), the hook would need to be
fixed or rewritten.

## Structural hypotheses that the data is consistent with but doesn't prove

1. **The L15 anchor requirement is about consistent KV state across
   iterations.** If the block starts at L15, each iteration re-consumes
   the same prelude-computed KVs. If the block starts at L20, iteration 2
   consumes KVs computed against iteration 1's drifted residual stream.
   This is testable by logging KV state divergence over iterations in
   block I (20–27) vs block D (15–22).

2. **The valley is specifically "consumer layers that use earlier-
   produced KVs without introducing fresh ones."** Everything outside
   that regime is fragile for different reasons: producer layers
   (0–14) generate KVs that downstream consumers depend on; very late
   layers (31–34) are near the output and drift specifically affects
   final prediction heads; the final global layer (34) is unique.

3. **Wider blocks damage reasoning more than narrower ones because
   attention-mechanism composition compounds multiplicatively.** 10
   attention mechanisms looped 8 times = 80 attention applications per
   forward pass vs 5 × 8 = 40. The hidden state trajectory through
   self-attention space is longer, not just deeper. This is consistent
   with reasoning depending on long-range token interactions that
   degrade under this trajectory lengthening while local-context
   predictions (perplexity) stay okay.

None of these hypotheses are established. They're rough frames that the
data doesn't contradict.

## Live decision point

The project is at a fork. Three possible next steps, listed in the plan 5
document's exit-scenario analysis:

### (A) Commit to narrow blocks and train

If Round 5's width sweep confirms the monotonic pattern (narrower →
better reasoning), train a valley-narrow block (width 3–5) with the
paper's healing + task-specific curriculum. This is the "paper method,
with E2B-specific block choice" path.

### (B) Fix PLE injection and retry wide blocks

If Round 5's PLE ablation shows noPLE dramatically helps, revise the
hook to handle PLE correctly (iter-1-only, or learned gate) and re-run
Round 4 with the fix. This tests whether the Round 4 wide-block collapse
was a specific artifact of re-injected PLE rather than a fundamental
limit of wide-block recurrence on E2B.

### (C) Pivot to non-Gemma base or different recurrence unit

If Round 5 shows all strategies land in the 25–40% ARC range regardless,
E2B's architecture is structurally hostile to this method. Options:

- **FFN-only recurrence:** Loop only the MLP of each block, skipping
  self-attention. This avoids the attention-mechanism composition
  hypothesis above.
- **Different base model:** Use one of the paper's validated models
  (OLMo-2-1B, Llama-3.2-1B) to confirm the project's tooling and then
  return to E2B with clearer understanding of which architectural
  features caused the problem.

## Most load-bearing single decisions looking forward

1. **Model variant for Round 5.** Base vs -it matters for the harness
   bug fix. Switching to -it requires re-verifying all the Round 1–3
   perplexity measurements carry over (the -it model has different
   weights after instruction-tuning, so probing results are not
   guaranteed to transfer).

2. **Whether to run Round 5 on the existing no-training setup or first
   implement the minimal training infrastructure.** Every round so far
   has been strictly probe-only. Training requires Muon optimizer (paper
   Section 4.3.1), healing data curriculum, truncated backpropagation.
   The farther the project goes into probing, the more evidence
   accumulates that probes alone cannot answer the remaining question
   ("does training recover what surgery destroyed?").

3. **Whether to expand beyond ARC-Easy.** ARC-Easy is a good "is the
   model still functional?" canary but doesn't measure math/logical
   reasoning directly. HellaSwag (language modeling), ARC-Challenge
   (harder reasoning), and a properly-harnessed GSM8K would give
   complementary signals. Round 5 only adds HellaSwag/ARC-Challenge
   conditional on Outcome A.

## Tool/infrastructure state

As of end of Round 4:

- `ple_sanity_check.py` has 6 modes: `original`, `ple-variants`,
  `ple-importance-scan`, `layer-location`, `full-looping-map`,
  `pair-looping-map`, `block-looping`, `reasoning-eval`.
- The positional-arg-aware PLE hook (Round 2a addendum 2) is used
  throughout.
- The block-looping hook handles multi-layer blocks with per-layer PLE;
  D-r1 / W5-r1 sanity checks confirm it's no-op at r=1 in both perplexity
  and generation modes.
- `results_round*_*.json` files are consistent enough to be read by
  downstream scripts; Round 5 plan reuses the schema with two additions
  (`truncation_rate`, `w5_r8_vs_round4_a_r8`).
- The base model is `google/gemma-4-E2B` (unchanged across all rounds).
  Round 5 may switch to `-it`; if so, all prior rounds would need to
  be re-verified under the new weights.

## One-paragraph summary

Five rounds of probing established that retrofitting the paper's
recurrence method onto Gemma 4 E2B is architecturally constrained in
three ways: (1) the recurrent block must start at L15, the first KV
consumer; (2) it cannot extend below L15 or cross the producer/consumer
boundary; (3) it can span up to 10 layers before stability degrades.
Within those constraints, perplexity is fine — valley-anchored blocks of
width 5–10 produce r=8 perplexities in the 30–40 range vs 12.5 baseline.
But Round 4's downstream evaluation reversed the story: wider blocks
that preserve perplexity collapse ARC-Easy reasoning to near-random,
while the narrowest tested block (width 5) retains about half of
baseline reasoning. This inversion is the opening for Round 5, which
will sweep block widths 2–5 at r=8 to see whether the trend continues
and whether a PLE-injection fix recovers wider blocks. GSM8K data from
Round 4 is harness-bugged (4.8% baseline) and will be re-collected in
Round 5 with instruction-tuned-model + chat template + 8-shot CoT +
512-token generation. No training has been attempted; that decision is
deferred to after Round 5 results come in.
