# Gemma 4 E2B — Round 4: Reasoning Benchmark Check

## Context

Seven rounds of probes (1 through 3c) have fully characterized the perplexity landscape for looping pretrained Gemma 4 E2B. We have two architecturally validated recurrent block candidates:

- **Block D (layers 15–22, width 8)**: ppl 33.3 at r=8, near-flat across r.
- **Block G (layers 15–24, width 10)**: ppl 36.0 at r=8, also near-flat across r.

Design rules established:
- Block must anchor at layer 15 (first KV consumer). Shifted-late blocks (block I at 20–27) break catastrophically despite identical architectural properties on paper.
- Block must not cross the KV producer/consumer boundary at layer 14/15.
- Block width can safely reach 10 layers; 11+ untested.

**The gap:** all of this is measured by Wikitext-2 perplexity. The target use case is mobile reasoning (math, logic, personal-assistant tasks). The paper's entire thesis is that depth-recurrence differentially helps *reasoning* benchmarks while language-modeling metrics stay roughly flat. A recurrent block that preserves linguistic surface competence may or may not preserve (let alone enhance) reasoning capability. Before committing serious training compute, we need one metric that actually measures what we care about.

This round runs a small zero-shot reasoning evaluation over the top recurrent configurations. No training, no fine-tuning — just "do these architecturally-stable configurations retain reasoning ability out of the box?" The result determines whether to proceed to retrofit training with confidence or to rethink.

## Scope

**In scope:**
- Extend `ple_sanity_check.py` with `--mode=reasoning-eval`.
- Evaluate 6–7 configurations on **GSM8K** zero-shot (chosen as the standard reasoning benchmark most-used in the paper).
- Evaluate a subset of configurations on **ARC-Easy** as a secondary check (broader knowledge, lighter reasoning).
- Use a fixed set of 200–500 problems per benchmark, same problems across all configurations, to make comparisons directly meaningful.
- Produce exact-match accuracy per configuration, plus per-problem agreement statistics so we can see *which* problems each configuration gets right.

**Out of scope:**
- No training.
- No MATH benchmark (harder, requires more eval-time compute; GSM8K is a better cost/signal trade for a first pass).
- No MMLU (it's mostly knowledge recall, less reasoning, and the paper showed MMLU is preserved under retrofit — less diagnostic for our question).
- No few-shot evaluation for the first pass (zero-shot is a cleaner signal about intrinsic reasoning).
- No sampling (use greedy decoding for reproducibility).
- No chain-of-thought prompting tricks. Plain "Question: ... Answer:" format.

## Configurations to evaluate

| Config | Description | Why |
|---|---|---|
| **baseline** | unmodified E2B, r=1 equivalent | Reference ceiling |
| **D-r4** | block D (15–22) at r=4 | Cheapest viable recurrent setup |
| **D-r8** | block D (15–22) at r=8 | Paper-comparable r at D's width |
| **G-r4** | block G (15–24) at r=4 | Widest block, moderate r |
| **G-r8** | block G (15–24) at r=8 | Widest block, paper-comparable r |
| **A-r8** | block A (15–19) at r=8 | Narrow block control — does wider block help reasoning even if perplexity is similar? |
| **D-r1** | block D at r=1 (no looping) | Sanity — must match baseline bitwise for same reasoning answers |

7 configurations. If compute is tight, drop A-r8. If very tight, drop G-r4.

## Benchmark details

### GSM8K

- Load `openai/gsm8k`, split=`test`.
- Use the first 250 problems for the main measurement. (Full test set is 1319 problems; 250 gives enough signal for ~±3% accuracy resolution at typical small-model performance levels and keeps runtime tractable.)
- Prompt format:
  ```
  Question: {problem_text}
  Answer:
  ```
  No few-shot, no chain-of-thought prompt. Let the model generate whatever it wants until end-of-sequence or a reasonable token limit (256).
- Scoring: extract the final numeric answer from the generated text and compare to the gold answer. Use the "flexible extract" style from the lm-eval harness — look for the last number in the response, normalize it (strip commas, handle fractions as floats), compare numerically.
- Gold answers in GSM8K are indicated by `####` in the reference solution; parsing is `float(ref.split('####')[-1].strip().replace(',', ''))`.

### ARC-Easy (secondary)

- Load `allenai/ai2_arc` config `ARC-Easy`, split=`test`.
- First 200 questions.
- Multiple-choice format. Score by comparing the generated answer letter (A/B/C/D) against the correct choice.
- Prompt format:
  ```
  Question: {q}
  Options:
  A. {opt_a}
  B. {opt_b}
  C. {opt_c}
  D. {opt_d}
  Answer:
  ```
- Parse first `A|B|C|D` character in the response.

If GSM8K results are clear-cut (strong directional signal) we might skip ARC. If GSM8K is ambiguous, ARC provides a second axis.

## Implementation

### Reusing the existing recurrent-block hook

The block-looping hook from round 3b already produces the right forward-pass behaviour. For evaluation, we need to generate tokens, not just compute perplexity. Three considerations:

1. **KV cache management during generation.** Previous rounds used `use_cache=False` because we were only doing single-pass perplexity. For generation, we do want a KV cache for the non-looped parts of the model (prelude and coda), but the *inside* of the looped block should not accumulate cache across iterations of the loop — same as before.

   The cleanest way: keep `use_cache=False` globally during generation. This will slow things down (full recomputation per token) but the eval set is small and this avoids subtle bugs. 250 GSM8K problems × 256 tokens × 7 configs × ~0.5s/token recompute = ~1.5 hours. Acceptable for a one-time probe.

2. **Looping behaviour during auto-regressive generation.** When generating token N+1, the forward pass still goes through the looped block r times. Each generated token gets r times the recurrent-block compute. This matches the paper's inference setup.

3. **Greedy decoding only.** No temperature, no sampling. Same output every run. Makes reproducibility trivial and makes per-problem agreement statistics meaningful.

### Per-problem agreement matrix

After collecting results, for each (config_A, config_B) pair, report:
- `both_correct`: how many problems both got right
- `only_A_correct`: how many A got right but B didn't
- `only_B_correct`: reverse
- `both_wrong`: how many both got wrong

This matters because *"G-r8 gets 47% and baseline gets 42%"* is much weaker evidence than *"G-r8 correctly answers 15 problems that baseline got wrong, while only missing 8 that baseline got right"*. The former could be noise, the latter tells us recurrence is unlocking genuinely different capability.

### Infrastructure notes

- Generation loop will use HuggingFace's `model.generate()`. Wrap the model such that the block-looping hook is active before calling generate.
- Cache hooks between configurations: install hook → run GSM8K → restore original forward → install next hook → repeat.
- Save raw per-problem outputs (question, gold, generated, parsed_answer, correct) to JSON so we can re-analyze without re-running.

## Deliverables

1. Updated `ple_sanity_check.py` with `--mode=reasoning-eval`.
2. `results_round4_reasoning.json` with:
   - Per-configuration GSM8K accuracy.
   - Per-configuration ARC-Easy accuracy (if run).
   - Full per-problem results (not just aggregates) so we can compute agreement matrices.
   - A pairwise agreement matrix between every pair of configurations.
3. A printed summary table:
   ```
   === GSM8K zero-shot results (N=250) ===
   config      accuracy   vs baseline   unique_correct_vs_baseline
   baseline    XX.X%      +0.0%         —
   D-r4        XX.X%      +X.X%         XX
   D-r8        XX.X%      +X.X%         XX
   G-r4        XX.X%      +X.X%         XX
   G-r8        XX.X%      +X.X%         XX
   A-r8        XX.X%      +X.X%         XX
   D-r1        XX.X%      must match    0
   ```
4. A one-paragraph interpretation identifying which scenario bucket we land in (below).

## Interpretation buckets

Setting the thresholds conservatively. Baseline E2B on GSM8K zero-shot is expected around 30–40% based on the model card benchmarks (37.5% on AIME 2026 was reported, GSM8K should be similar or higher). Thresholds below are relative to whatever baseline measures.

**Bucket 1 — Recurrence helps reasoning (>3% absolute improvement over baseline).**
Strong green light. Block G-r8 or D-r8 outperforms baseline. This is remarkable because we haven't trained anything — we're just looping pretrained weights. If pretrained-only recurrence already improves reasoning, retrofit training should improve it significantly more. Strongest possible case to commit training compute.

**Bucket 2 — Recurrence preserves reasoning (within ±2% of baseline).**
Green light. The looped block doesn't destroy reasoning ability even though it degraded perplexity 2.7× (block G at r=8). This means training has a stable starting point: reasoning capability isn't broken; training just needs to teach the model to exploit the recurrent compute. This matches the paper's finding that pretrained non-recurrent weights preserve reasoning after surgery, before any recurrence-specific training.

**Bucket 3 — Recurrence mildly degrades reasoning (−2% to −10%).**
Yellow light. Reasoning took a hit but not catastrophic. Healing training should recover this, as the paper's section 4.4 showed for language-modeling benchmarks after surgery. Proceed to training but expect to need a healing phase before task-specific training.

**Bucket 4 — Recurrence severely degrades reasoning (>10% drop).**
Red light. The perplexity-stable blocks are *not* reasoning-stable. Retrofit training would have to simultaneously repair reasoning *and* teach recurrence, which is much harder than the paper's setup. Possible responses: (a) commit to block D instead of G (narrower, potentially less damaging); (b) use much lower r for training (r=2 or 4); (c) reconsider whether paper-style retrofit is the right approach for E2B.

**Bucket 5 — D-r1 doesn't match baseline.**
Hook bug in generation mode. Stop and debug before interpreting other configurations. (r=1 looping must be a no-op.)

**Bucket 6 — Configurations disagree strongly on which problems they get right, regardless of aggregate accuracy.**
Interesting finding regardless of bucket 1–4. If G-r8 gets 20 problems right that baseline misses but misses 20 different problems baseline gets right, recurrence is shifting *which* problems are solvable rather than improving overall. This is worth pushing on — may indicate that training needs to blend recurrent and non-recurrent compute paths rather than committing fully to recurrence.

## Report back

Paste:
- The summary table with accuracies and deltas.
- The D-r1 vs baseline sanity check result.
- The interpretation paragraph identifying the bucket.
- If Bucket 6 signal appears, the pairwise agreement matrix for G-r8 vs baseline.

Do not proceed to training or multi-block exploration until this result is in. Next round depends entirely on which bucket we land in.