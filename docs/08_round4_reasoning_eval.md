# 08 — Round 4: Reasoning benchmark check

**Plan:** `plan4.md`
**Result file:** `results_round4_reasoning.json` (3.9 MB — includes per-problem
records)
**Status:** Complete. The findings changed the project's direction.

## Question

Seven rounds of perplexity probing established a safe recurrent-block design
for E2B (valley-anchored, up to width 10). But the paper's thesis is that
depth-recurrence differentially helps *reasoning* benchmarks while language-
modeling metrics stay roughly flat. Does that hold here, or does
perplexity-stability fail to imply reasoning-stability on E2B?

## Setup

- **Model:** `google/gemma-4-E2B` (base, not -it — this turns out to matter
  for GSM8K; see failure analysis below)
- **Benchmarks:**
  - GSM8K: first 250 test problems, zero-shot, "Question: ... Answer:"
    prompt, greedy decoding, max_gen_tokens=256, flexible numeric answer
    extraction
  - ARC-Easy: first 200 test questions, zero-shot, multiple-choice format,
    parse first A/B/C/D character from response
- **7 configurations:**

| Config | Block | r | Description |
|--------|-------|---|-------------|
| baseline | — | 1 | Unmodified E2B |
| D-r1 | 15–22 | 1 | Sanity check — must match baseline bitwise |
| A-r8 | 15–19 | 8 | Narrow block (width 5) |
| D-r4 | 15–22 | 4 | Width 8, mid r |
| D-r8 | 15–22 | 8 | Width 8, paper-comparable r |
| G-r4 | 15–24 | 4 | Width 10, mid r |
| G-r8 | 15–24 | 8 | Width 10, paper-comparable r |

## Headline results

![Round 4 reasoning accuracy](figs/fig08_round4_accuracy.png)

| Config | GSM8K | ARC-Easy | Δ vs baseline (ARC) |
|--------|-------:|---------:|--------------------:|
| baseline | 4.8% (12/250) | **83.5%** (167/200) | — |
| D-r1 | 4.8% (12/250) | 83.5% (167/200) | 0 (sanity pass) |
| A-r8 | 1.2% (3/250) | **40.0%** (80/200) | **−43.5%** |
| D-r4 | 1.2% (3/250) | 24.5% (49/200) | −59.0% |
| D-r8 | 2.0% (5/250) | 26.5% (53/200) | −57.0% |
| G-r4 | 2.0% (5/250) | 26.5% (53/200) | −57.0% |
| G-r8 | 1.2% (3/250) | 24.5% (49/200) | −59.0% |

### The D-r1 sanity check

The sanity check is the single most important number in the table.

- GSM8K `d_r1_vs_baseline`: **250/250 match**
- ARC-Easy `d_r1_vs_baseline`: **167/167 match** (on every baseline-correct
  problem)

This confirms the block-looping hook is a no-op at r=1 during *generation*
(not just during the perplexity probes of earlier rounds). It rules out
"hook bug during generation" as an explanation for the recurrent configs'
poor results.

## Two very different stories: GSM8K vs ARC-Easy

### GSM8K baseline is a harness bug, not a model measurement

**The baseline scores 12/250 = 4.8%**, which is far below Gemma 4 E2B's
expected GSM8K accuracy. Inspection of the 238 wrong baseline responses
reveals two dominant failure modes:

| Failure mode | Count | Example |
|-------------|-------|---------|
| **Textbook continuation** — generates more "Question:/Answer:" pairs instead of solving the asked question | 172 | Q: Josh decides to try flipping a house. He buys for $80k, puts in $50k. → Gen: " $100,000\nQuestion: A house is valued at $100,000. The house is sold for $120,000. What is the percentage of profit?\nAnswer: 20%\nQuestion: ..." |
| **Truncation** — hits max_gen_tokens=256 mid-computation with degenerate loops | 17+ | Gen ends: "...150 miles.\nStep 8:\nThe distance covered by each train in the two days is 150 miles.\nStep 9:\nThe distance covered by..." |
| Short/other | 49 | Early stops with wrong numbers |

This is not "the pretrained base model can't do math." It's the prompt
format (`Question: ... Answer:`) triggering textbook-style continuations,
combined with `max_gen_tokens=256` truncating the chain-of-thought attempts
that do get started. Gemma-class models' published GSM8K numbers are all
with 8-shot chain-of-thought prompting using the Wei et al. exemplars —
not zero-shot continuation.

**Consequence:** the GSM8K column in the table measures harness, not model.
All 7 configs are between 1.2% and 4.8%, which is below the noise floor
for a real model comparison. No GSM8K conclusions about recurrence are
warranted from this data. Round 5 was written specifically to fix this.

### ARC-Easy is a real measurement

The baseline's 83.5% on ARC-Easy is consistent with expectations for Gemma
4 E2B on multiple-choice QA. The recurrent configs' scores are real:

- All wide-block configs (D-r4, D-r8, G-r4, G-r8) collapse to
  **~24–26% accuracy** — that's approaching random guessing (25% on
  four-choice questions).
- The narrow block **A-r8 holds 40%** — substantially above random,
  substantially below baseline.

## Agreement analysis — what recurrence is doing to ARC-Easy

![ARC agreement with baseline](figs/fig09_round4_arc_agreement.png)

For each recurrent config, how do its 200 problem responses break down
against the baseline?

| vs baseline | both correct | only baseline right | only recurrent right | both wrong |
|-------------|-------------:|--------------------:|---------------------:|-----------:|
| D-r4 | 37 | 130 | 12 | 21 |
| D-r8 | 44 | 123 | 9 | 24 |
| G-r4 | 44 | 123 | 9 | 24 |
| G-r8 | 41 | 126 | 8 | 25 |
| **A-r8** | **70** | **97** | **10** | **23** |
| D-r1 | 167 | 0 | 0 | 33 |

### Observations

1. **All recurrent configs get some problems the baseline missed**
   (8–12 "only recurrent right" per 200). So recurrence is not strictly
   a lossy channel — it does unlock different capability on some
   problems. But the rate is low (~5%).

2. **The wide-block configs (D, G) mostly lose what the baseline knew.**
   Between 123–130 of 167 baseline-correct problems become wrong.

3. **A-r8 preserves 70 of 167 baseline-correct problems** (42%) while
   the wide-block configs preserve only 22–26% of baseline-correct
   problems. This is the key asymmetry — narrowing the block from 8–10
   layers to 5 nearly doubles the preserved-capability rate.

4. **Agreement among recurrent configs is high.** For example,
   A-r8 vs G-r8: they agree on 149 of 200 problems (39 both correct, 110
   both wrong). D-r4 vs D-r8: 184 of 200 agreement. The recurrent configs
   are not exploring different failure modes from each other; they're
   failing on largely the same problems.

## The finding that changed the project

For the first four rounds, narrow-vs-wide block choice looked like a
trade-off being decided in favor of wide (D and G both had clean
perplexity stories; A was a fallback). Round 4 inverts that:

| Block | r=8 ppl (round 3) | ARC-Easy (round 4) | ARC preserved |
|-------|------------------:|-------------------:|--------------:|
| A (15–19, width 5) | 30.1 | **40.0%** | 42% of baseline-correct |
| D (15–22, width 8) | 33.3 | 26.5% | 26% of baseline-correct |
| G (15–24, width 10) | 36.0 | 24.5% | 25% of baseline-correct |

The widest block has slightly worse perplexity *and* substantially worse
reasoning. On ARC-Easy, the narrower the block, the better — which is the
opposite of what a "more compute per token = better reasoning" framing
would predict. The plan 5 document names this inversion directly:

> "ARC-Easy is the trustworthy benchmark (baseline 83.5% in round 4
> matches expectations). The one signal from round 4: A-r8 `[15,19]`
> scored 40% vs ~25% for wider blocks."

## Interpretation buckets

From the plan's 6 scenarios:

- Bucket 1 (recurrence helps reasoning >3%): no, all configs are worse.
- Bucket 2 (within ±2% of baseline): no, all configs drop by 43–59%.
- Bucket 3 (mild degradation, −2% to −10%): no, drops are much larger.
- **Bucket 4 (severe degradation, >10% drop): yes — but with a strong
  layering within the "red light".** A-r8 is much less bad than D/G.
- Bucket 5 (D-r1 doesn't match baseline): no, sanity passes perfectly.
- **Bucket 6 (configs disagree on which problems, regardless of
  aggregate): partial** — the 8–12 "only recurrent right" problems show
  recurrence does shift *which* problems are solvable, but not enough
  to compensate for what's lost.

## What this round established

1. **Perplexity stability does NOT imply reasoning stability** on E2B.
   Blocks D and G have good perplexity (33–36 at r=8) but collapse
   ARC-Easy to ~25%.
2. **Narrower blocks preserve reasoning better.** A (width 5) retains
   40% ARC-Easy; D (width 8) and G (width 10) retain ~25%. The gradient
   runs opposite to perplexity rankings.
3. **The generation-mode hook is validated** (D-r1 matches baseline
   token-for-token), so the collapse isn't a hook bug.
4. **GSM8K was uninformative** due to a harness bug: base model + zero-
   shot + `Question:/Answer:` prompt format + max_gen_tokens=256 ≈ mostly
   textbook-continuation and truncated chains. Round 5's first task is
   to fix this.

## Why wide blocks might hurt reasoning more than they hurt perplexity

Speculative — the data doesn't resolve this, but it shapes Round 5's
design:

- **Perplexity is a per-token average** and is dominated by the 95% of
  tokens that are linguistically easy. A block that preserves 95% of
  surface language fluency while destroying 100% of multi-step logical
  propagation would score well on perplexity and badly on ARC.
- **ARC-Easy requires maintaining a consistent world-model across a
  problem statement** (40–80 tokens of question) to compare against
  choice text. If wider blocks scramble long-range dependencies more than
  narrower ones, this is where it would show up first — while perplexity
  (mostly local-context) stays low.
- **Each extra layer looped adds a pass through an attention mechanism**
  that wasn't trained to be re-entered. Block D loops 8 attention
  mechanisms r=8 times = 64 attention applications per forward pass;
  A loops 5 × 8 = 40. That's a quantitative, not just qualitative,
  difference in how far the hidden state travels off the pretrained
  manifold per token.

Round 5's width sweep (widths 2, 3, 4, 5 at r=8) is designed to test
whether the trend continues monotonically: if A at width 5 is 40%, does
width 3 reach higher? Does width 2? Does approaching width 1 recover
the baseline?
