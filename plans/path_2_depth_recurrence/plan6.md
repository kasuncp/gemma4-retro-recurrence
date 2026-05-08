# Round 6 Plan — Reasoning Eval v3 (Path 1 prompt corrections)

**Project:** Retrofitted recurrence on Gemma 4 E2B
**Status:** Ready for implementation
**Depends on:** Round 5 harness; Path 1 results in `results/path_1_cot_tokens/` (especially plans 5, 8, 9 — covered in `outputs/local-first-mobile-ai-part-2-explained.md`)

---

## Why this round exists

Path 1 finished after Path 2 was already through round 5. Three Path 1 findings invalidate the round 4 / round 5 reasoning measurement and require a re-run before any retrofit-training conclusions can be trusted.

1. **8-shot CoT actively hurts on E2B-it.** Path 1 measured 30.0% (8-shot) vs 71.6% (zero-shot via chat template, "C2") on the same 500 GSM8K problems. McNemar's test: p ≈ 5×10⁻⁴³. Round 5's 54.8% baseline is *below* the model's actual ceiling by ~17–23 points.
2. **Naive last-integer extraction throws away ~6.4 percentage points** of correct-but-buried answers. The model produces things like `**$16.00**` and `= 16.` that flexible-extract grabs as `0` or `00`. Path 1 plan 9 specifies a smart_v2 extractor that recovers most of these.
3. **ARC-Easy is saturated.** Path 1 measured ~83% across every prompt format — no separation, no signal. ARC-Challenge separates C2 from 8-shot CoT by 62 points (75.6% vs 13.6%) and is the benchmark that should have been used.

A fourth observation that motivates the retry independent of the corrections: **round 5 partials show 97–100% truncation on every recurrent config** (W2-r8, W4-r8, W5-r8, S10-W3, S20-W3 all between 0.8% and 3.1% accuracy with truncation rates from 0.97 to 1.00). That's a repetition-loop collapse, not a reasoning loss. The corrected harness will let us tell whether recurrence on E2B-it can answer at all under a competitive prompt, or whether the loop-collapse pathology is structural.

The shape of the question hasn't changed: *does recurrence help reasoning out of the box, before training?* The yardstick has.

---

## Scope

**In scope:**
- Add `--mode=reasoning-eval-v3` to `ple_sanity_check.py` (do not modify the round 4/5 mode in place — keep the prior runs reproducible).
- Use **Gemma 4 E2B-it** (instruction-tuned), bf16, on the same hardware as round 5.
- **Primary prompt = C2** (just the question, routed through `tokenizer.apply_chat_template` with `add_generation_prompt=True`). Greedy decoding. `max_new_tokens=512`.
- Smart_v2 answer extractor for GSM8K (recipe below). Letter-extractor for ARC-Challenge (already correct in round 4).
- **Repetition-loop detector** (`(.{10,60})\1{2,}` regex) run on every completion, reported per config alongside accuracy and truncation.
- One control config keeps the 8-shot CoT prompt for direct cross-round comparison with round 5's 54.8%.
- Replace ARC-Easy with **ARC-Challenge**.
- Sample size: **500 GSM8K problems** (matches Path 1; ±4pt CI). 200 ARC-Challenge problems.
- Same recurrent-block hooks as round 5 (block-looping with positional-arg PLE handling, `use_cache=False`).

**Out of scope:**
- No training, no LoRA, no fine-tuning.
- No MATH benchmark (Path 1 confirmed E2B caps at 0% regardless of prompt — a capability ceiling, not a prompt ceiling).
- No BBH-lite (separate question; deferred to a possible plan 7).
- No on-device / Snapdragon measurement (separate concern; this round stays on the desktop GPU used by rounds 4–5).
- No new PLE policies. `vanilla` and `noPLE` are the only two used here, identical to round 5.
- No few-shot prompt experiments. Path 1 closed this.
- No sampling / temperature sweep. Greedy only.

---

## Configurations

| Config | Block | r | Prompt | PLE strategy | Why |
|---|---|---|---|---|---|
| **baseline-C2** | — | 1 | C2 | n/a | The new reference. Must hit 70–78% on GSM8K, 70–78% on ARC-Challenge. |
| **baseline-8shot-control** | — | 1 | 8-shot CoT | n/a | Cross-round bridge. Must reproduce round 5's 54.8% within ±2pt to confirm harness equivalence. |
| **W5-r1** | [15,19] | 1 | C2 | vanilla | Token-for-token sanity vs `baseline-C2`. r=1 must be a no-op. |
| **W2-r8** | [15,16] | 8 | C2 | vanilla | Narrowest block — narrowest blocks degraded least in round 5. |
| **W3-r8** | [15,17] | 8 | C2 | vanilla | |
| **W4-r8** | [15,18] | 8 | C2 | vanilla | |
| **W5-r8** | [15,19] | 8 | C2 | vanilla | Round 5's primary block. |
| **D-r8** | [15,22] | 8 | C2 | vanilla | Round 4's recurrent design. |
| **G-r8** | [15,24] | 8 | C2 | vanilla | Round 3c's widest viable. |
| **W5-r8-noPLE** | [15,19] | 8 | C2 | iter-0-only | Path 1 confirmed prompt format dominates over compute knobs — but the noPLE pilot is cheap and addresses a separate axis. |

**10 configs total.** Drop `D-r8` and `G-r8` if GPU-tight; the width sweep + the W5 controls are the primary measurement.

---

## Mandatory validation gates (RUN FIRST, ABORT IF ANY FAIL)

Run these on a 50-problem subset before launching the full sweep. Each gate has an empirical anchor from Path 1.

1. **`baseline-C2` GSM8K @ N=50 ∈ [65%, 82%]**
   Path 1 plan 8 measured 71.6% on N=500. The 50-problem first-quartile sample is noisier (±10pt CI) but should land in this band. Below 65% means the chat template is misapplied or the smart extractor is broken; above 82% means the random subset is unrepresentative — re-shuffle and retry.

2. **`baseline-8shot-control` GSM8K @ N=50 ∈ [44%, 64%]**
   Round 5 measured 54.8% on N=250. ±10pt for N=50. Below the band → harness regression vs round 5; halt and diff the config against `round5_partial_gpu0/manifest.json`.

3. **`baseline-C2` ARC-Challenge @ N=50 ∈ [68%, 82%]**
   Path 1 plan 7 measured 75.6% on N=200.

4. **Smart-extractor unit test passes.**
   The 12 cases in §"Smart_v2 extractor recipe" below must all extract correctly. This is a pure-function test — no model needed. Run it as a pytest before any GPU work.

5. **`W5-r1` GSM8K matches `baseline-C2` token-for-token** on the same 50 problems.
   `r=1` must be a no-op. If the per-problem `generated_text` strings disagree on even one problem, the loop hook has a regression — halt and debug. Path 2 round 5's `gsm8k__W5-r1.jsonl` already passed this check at 54.8% but the harness has changed.

If gate 1 lands at, say, 30%, the most likely cause is the chat template is being concatenated as raw text instead of routed through `apply_chat_template`. Path 1 plan 5 spent its first hour on this exact bug; consult its README before re-running.

---

## C2 prompt recipe (exact)

```python
def build_c2_prompt(tokenizer, question: str) -> torch.Tensor:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # critical — emits the <start_of_turn>model header
        return_tensors="pt",
    )
```

No system prompt. No few-shot exemplars. No "Let's think step by step" suffix (that's C1; C1 scored 67% in Path 1, C2 scored 71.6% — use C2). The model decides whether to reason out loud or answer directly.

Generation:

```python
gen_kwargs = dict(
    max_new_tokens=512,
    do_sample=False,
    temperature=None,
    top_p=None,
    use_cache=False,  # required for the block-looping hook to work correctly
    pad_token_id=tokenizer.eos_token_id,
)
```

**`use_cache=False` is non-negotiable** — the block-looping hook relies on it. This is what makes generation slow (full recompute per token); the round 5 harness already set this and budgeted accordingly.

---

## Smart_v2 extractor recipe

Path 1 plan 9 derived this from reading 26 problems where C2 reasoned correctly but flexible-extract grabbed the wrong number. Apply the rules **in order** — the first match wins.

```python
import re

_NUM = r"-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\$?\d+(?:\.\d+)?"

def smart_v2_extract(text: str) -> float | None:
    """Extract the model's intended numeric answer from a GSM8K completion.

    Apply rules in order; return the first hit. Strip $, commas, and a
    trailing decimal zero ('16.00' -> 16, but '16.5' stays 16.5).
    """
    if not text:
        return None
    t = text.strip()

    rules = [
        # 1. Explicit "Answer: <num>" / "answer is <num>" / "= <num>" near end of text
        rf"(?i)(?:answer\s*[:\-]?\s*|the answer is\s+){_NUM}",
        # 2. Bold-formatted final number: **$16.00** or **16**
        rf"\*\*({_NUM})\*\*",
        # 3. "= <num>" at end of a line, possibly followed by punctuation
        rf"={_NUM}\s*[.\)\]]?\s*$",
        # 4. "$<num>" anywhere in the last sentence
        # 5. Final number on the last non-empty line
        # 6. Last number anywhere (flexible-extract fallback)
    ]
    # ... (full implementation has 6 rules; see Path 1 plan 9 for the exact logic)
    # Each rule returns the matched number cleaned via _normalize_number().

    def _normalize(s: str) -> float | None:
        s = s.replace("$", "").replace(",", "").strip()
        try:
            v = float(s)
            return int(v) if v.is_integer() else v
        except ValueError:
            return None

    # Implementation pattern: find all matches per rule, take the LAST match
    # (later in the text = closer to the final answer), normalize, return.
    ...
```

**Required unit-test cases (must all pass before GPU run):**

```
("The answer is **$16.00**.", 16),
("So the customer saved $50 - $34 = $16.", 16),
("**16**", 16),
("Final answer: 16", 16),
("16.5 dollars", 16.5),
("After 32% off, the price is $34. With tax: $34 × 1.06 = $36.04. The savings: $50 - $34 = **$16.00**.", 16),
("The total cost is 1,250 dollars.", 1250),
("Negative two: -2", -2),
("$3.50 and $4.50, total $8.", 8),
("Step 1: 5+3=8. Step 2: 8*2=16.", 16),
("", None),
("There is no number here.", None),
```

Borrow the full implementation from Path 1 (`results/path_1_cot_tokens/plan9/` — there is a tested extractor module). Do not re-derive.

---

## Repetition-loop detector

Run on every completion before extraction. Report per config:

```python
import re
_LOOP = re.compile(r"(.{10,60})\1{2,}", flags=re.DOTALL)

def has_loop(text: str) -> bool:
    return bool(_LOOP.search(text))
```

Path 1 measured 9–10% loop rate on the *baseline*. Round 5 partials show 97% on recurrent configs. Flag any config where loop_rate > 30% — that's a generation pathology, not a reasoning result, and should be reported separately from accuracy.

---

## Reporting format

Same JSON schema as round 5, with three additions per config:

```json
{
  "name": "W5-r8",
  "block": [15, 19],
  "r": 8,
  "prompt": "C2",
  "ple_strategy": "vanilla",
  "n_problems": 500,
  "accuracy_smart_v2": 0.XXX,
  "accuracy_flexible_extract_legacy": 0.XXX,   // for cross-checking with round 5
  "truncation_rate": 0.XXX,
  "loop_rate": 0.XXX,                          // NEW
  "extractor_lift": 0.XXX,                     // NEW: smart_v2 - flexible_extract
  "wall_clock_seconds": ...
}
```

Plus a sanity-check block:

```json
"sanity_checks": {
  "baseline_c2_gsm8k_50": {"accuracy": 0.XX, "in_band": true, "band": [0.65, 0.82]},
  "baseline_c2_arc_50":   {"accuracy": 0.XX, "in_band": true, "band": [0.68, 0.82]},
  "baseline_8shot_50":    {"accuracy": 0.XX, "in_band": true, "band": [0.44, 0.64]},
  "extractor_unit_tests": {"passed": 12, "total": 12},
  "w5_r1_token_match":    {"matches": 50, "total": 50}
}
```

Print a final summary table at end of run:

```
=== Round 6 GSM8K (N=500, C2 prompt + smart_v2 extractor) ===
config              acc      vs base   trunc   loop   uniq_correct_vs_base
baseline-C2         71.6%    +0.0      0.05    0.09   —
baseline-8shot      30.0%    -41.6     0.30    0.10   X
W5-r1               71.6%    +0.0      0.05    0.09   0  (must equal baseline)
W2-r8               XX.X%    XX        XX      XX     XX
W3-r8               XX.X%    XX        XX      XX     XX
W4-r8               XX.X%    XX        XX      XX     XX
W5-r8               XX.X%    XX        XX      XX     XX
D-r8                XX.X%    XX        XX      XX     XX
G-r8                XX.X%    XX        XX      XX     XX
W5-r8-noPLE         XX.X%    XX        XX      XX     XX
```

(The first three rows are illustrative anchors — actual numbers populate from the run.)

---

## Interpretation buckets — commit before looking

**Bucket A — Recurrence still helps once the baseline is competitive (>+3pt).**
Some recurrent config beats `baseline-C2` (71.6%) by ≥3pt. Genuinely surprising; recurrence on pretrained weights *and* a competitive prompt unlocks reasoning. Strongest possible green light for retrofit training.

**Bucket B — Recurrence preserves (within ±2pt of `baseline-C2`).**
At least one recurrent config stays within noise of 71.6%. Training has a stable starting point; round 7 is the retrofit-training plan.

**Bucket C — Mild degradation (-2 to -10pt).**
Healing-training-recoverable territory. Round 7 = retrofit training with a healing phase.

**Bucket D — Severe degradation (>-10pt).**
The prompt fix didn't rescue recurrence. The round 5 collapse (97% truncation, ~1% accuracy) was structural, not prompt-induced. Round 7 has to be one of:
  - Pivot to a different recurrence formulation (FFN-only, attention-only).
  - Lower r to 2 or 4 and retest — round 5 only swept r=8 on reasoning.
  - Reconsider whether E2B's PLE + hybrid attention can ever support depth-recurrence at all.

**Bucket E — Loop-rate collapse persists (loop_rate > 50% on recurrent configs).**
The pathology is not "reasoning lost" but "generation pathologically loops." Recurrence is putting the residual stream into a fixed point the model can't escape during decoding. This is the most likely outcome given round 5 partials. If E confirms, the next move is **r-sweep with C2** at fixed block W5, varying r ∈ {1, 2, 3, 4, 6, 8} — to find where loop rate crosses 50% and whether that boundary is reasoning-useful below it.

**Bucket F — Validation gate fails on `baseline-C2`.**
Harness still broken. Halt; do not interpret recurrent configs. Diff against Path 1 plan 8's setup line by line.

---

## Why a width sweep wasn't the round 5 approach but is the round 6 fallback

Round 5 already swept widths W2/W3/W4/W5 with 8-shot CoT and saw monotone collapse (all 0.8%–3.1%). If round 6 lands in Bucket E (loop collapse), an r-sweep tells us *more* than a width sweep does, because the round 5 width sweep already saturated the "narrower is better" axis at the lowest meaningful resolution (W2). The unexplored axis is iteration count. Hence the proposed Bucket-E follow-on.

If round 6 lands in Bucket B/C, an r-sweep is also the next probe — to find the largest r at which the recurrent block still preserves reasoning. Either way, round 7 is most likely an r-sweep at fixed block geometry (W5 or D), 6 r-values, C2 prompt.

---

## What this round explicitly does NOT change about round 5's findings

- The Wikitext-2 perplexity map across all 35 layers (rounds 2c, 3a–3c) is unaffected. Path 1's prompt findings are about IT-model downstream tasks; perplexity-on-base measurements are a different axis.
- The "valley anchored at layer 15, max width ~10" architectural conclusion is unaffected.
- Block geometry choices (W5, D, G) come from the perplexity work and remain the right ones to test.

The only thing round 6 corrects is the **measurement of reasoning capability** under those geometries.

---

## Open questions before implementation

1. **Smart extractor source.** Path 1 plan 9 has a working extractor — pull it directly, do not re-derive. Confirm the file path inside `results/path_1_cot_tokens/plan9/` and import vs. copy.
2. **Sample size for tight budget.** If GPU is tight, drop GSM8K to N=300 (CI ±5pt). Don't go below 250 — too noisy for the ±3pt deltas we're trying to detect.
3. **`noPLE` priority.** It's a strategy pilot, not a correction. If GPU is *very* tight, drop `W5-r8-noPLE` first. The width sweep + the cross-round 8-shot control are the load-bearing measurements.

---

## Exit criteria

One of three round-7 paths, decided by the bucket:

- **Bucket A or B** → Round 7 is retrofit-training design (the path 2 thesis works).
- **Bucket C** → Round 7 is healing training + retrofit, longer compute budget.
- **Bucket D, E, or F** → Round 7 is an r-sweep diagnostic (or a pivot to a different recurrence formulation, depending on which sub-pattern emerges).

Do not proceed to training under any bucket without first writing plan 7.
