# Path 1 — plan 1: CoT gate on GSM8K

## Context

Path 1 of the "four paths, one phone" investigation asks whether Gemma 4 E2B can trade inference-time compute for reasoning accuracy, with no weight changes at all — just longer/richer prompting. The three mechanisms in that bucket are (a) longer chain-of-thought generation, (b) self-consistency sampling, (c) deliberation / reflection prompts.

Before sweeping any of them, we need to answer one binary question:

> **Does giving Gemma 4 E2B reasoning steps in-context beat giving it none, on GSM8K, at a matched prompting protocol?**

If no, Path 1 is wounded on E2B at inference-only scale — and every Path 1 sub-mechanism (longer CoT, self-consistency, deliberation) inherits that weakness. We write up the null, flag the leg, and re-weight the head-to-head.

If yes, plan2 sweeps generation length and self-consistency on top of the protocol this plan locks in.

This is the Path 1 analog of `plans/path_2_depth_recurrence/plan1.md` — a narrow gating probe, not a paper.

## Scope — what this probe is and isn't

**In scope:**
- Load pretrained `google/gemma-4-E2B-it` (primary) and `google/gemma-4-E2B` base (sanity).
- Score on the GSM8K test split, deterministic first `n = 500` problems.
- Two prompt conditions per model, 8-shot in both cases:
  - `direct`: the same 8 exemplars with reasoning stripped, only `#### <number>` as the answer.
  - `cot`: the standard 8 exemplars with reasoning intact.
- Greedy decoding (`do_sample=False`, `temperature=0`).
- `max_new_tokens = 16` for `direct`, `max_new_tokens = 512` for `cot`.
- Extract integer answers, compute exact-match accuracy + Wilson 95% CIs.
- Print a 2×2 table, write JSON to `results/path_1_cot_tokens/results_gate.json`, apply the decision rule below.

**Explicitly out of scope (do NOT add these):**
- No self-consistency sampling (`k > 1`). That is plan2.
- No generation-length sweep. That is plan2.
- No ARC-Easy, no MATH, no BBH. GSM8K only.
- No deliberation / reflection / "let's verify" prompts. GSM8K 8-shot exemplars verbatim, one with-reasoning variant and one stripped variant.
- No zero-shot "let's think step by step" template — using 8-shot everywhere is what isolates "does reasoning in-context help" from "does the model follow zero-shot instructions."
- No training. No fine-tuning. No adapter surgery.
- No on-device measurement.
- No plotting beyond the printed table.

If the script grows past ~250 lines or needs more than one file, stop and reconsider.

## Environment

```
pip install -U transformers torch accelerate datasets
```

Hardware: one GPU with ≥16 GB VRAM in bfloat16 (H100/A100/4090/3090). Expected wall time: ~2.5 hours for all four cells.

## Protocol summary

| Cell | Model                    | Condition | Exemplars                              | max_new_tokens |
|------|--------------------------|-----------|----------------------------------------|----------------|
| 1    | `gemma-4-E2B-it`         | direct    | 8-shot, reasoning stripped             | 16             |
| 2    | `gemma-4-E2B-it`         | cot       | 8-shot, reasoning intact               | 512            |
| 3    | `gemma-4-E2B` (base)     | direct    | same exemplars as cell 1               | 16             |
| 4    | `gemma-4-E2B` (base)     | cot       | same exemplars as cell 2               | 512            |

Same 500 GSM8K test problems across all cells. Same 8 exemplars across all cells (use the canonical CoT-paper GSM8K exemplars, or the first 8 of the GSM8K train split — pick one and document which in the JSON output).

## Implementation steps

### Step 1 — Load models and tokenizers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "it":   "google/gemma-4-E2B-it",
    "base": "google/gemma-4-E2B",
}

def load(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    mdl.train(False)  # inference mode — equivalent to the usual .eval() idiom
    return tok, mdl
```

Load one model at a time to keep VRAM headroom; free it before loading the other.

### Step 2 — Prepare GSM8K test slice

```python
from datasets import load_dataset

gsm = load_dataset("gsm8k", "main", split="test")
problems = gsm.select(range(500))   # deterministic first-500 slice
```

Each problem has `question` and `answer` fields. Ground-truth integer is the trailing `#### <number>` of `answer`.

### Step 3 — Build the 8-shot prompts

```python
EXEMPLARS_COT = [        # (question, full reasoning ending in "#### N")
    ("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
     "Natalia sold 48/2 = 24 clips in May.\nAltogether she sold 48 + 24 = 72 clips.\n#### 72"),
    # ... 7 more canonical GSM8K exemplars
]

EXEMPLARS_DIRECT = [     # (question, "#### N")  — same questions, reasoning stripped
    (q, "#### " + a.split("####")[-1].strip())
    for q, a in EXEMPLARS_COT
]

def build_prompt(exemplars, question):
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in exemplars)
    return f"{shots}\n\nQuestion: {question}\nAnswer:"
```

Pin the exact 8 exemplars in the script. Commit them. They must be byte-identical across cells — the only variable is whether reasoning is present.

### Step 4 — Generation loop

```python
@torch.no_grad()
def run_cell(tok, mdl, exemplars, max_new, problems):
    outs = []
    for ex in problems:
        prompt = build_prompt(exemplars, ex["question"])
        ids = tok(prompt, return_tensors="pt").to("cuda")
        gen = mdl.generate(
            **ids,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
        outs.append(text)
    return outs
```

Batching is a nice-to-have but not required; straight-line loop is fine at n=500 and keeps the script simple.

### Step 5 — Answer extraction and scoring

```python
import re, math

GOLD_RE  = re.compile(r"####\s*(-?\d+)")
FALLBACK = re.compile(r"(-?\d+)")

def extract(text: str):
    m = GOLD_RE.search(text)
    if m:
        return int(m.group(1))
    nums = FALLBACK.findall(text)
    return int(nums[-1]) if nums else None

def score(preds, golds):
    n = len(preds)
    correct = sum(1 for p, g in zip(preds, golds) if p is not None and p == g)
    acc = correct / n
    # Wilson 95% CI
    z = 1.96
    denom = 1 + z * z / n
    centre = (acc + z * z / (2 * n)) / denom
    half   = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / denom
    return acc, (centre - half, centre + half), correct
```

Track and report the `####`-hit rate separately — it is a pathology canary for the CoT cells.

### Step 6 — Run the 2×2, print the table, save JSON

```python
import json

golds = [int(re.search(r"####\s*(-?\d+)", p["answer"]).group(1)) for p in problems]

results = {}
for model_key, model_id in MODELS.items():
    tok, mdl = load(model_id)
    for cond, exemplars, max_new in [
        ("direct", EXEMPLARS_DIRECT, 16),
        ("cot",    EXEMPLARS_COT,    512),
    ]:
        texts = run_cell(tok, mdl, exemplars, max_new, problems)
        preds = [extract(t) for t in texts]
        acc, ci, correct = score(preds, golds)
        hashhit = sum(1 for t in texts if GOLD_RE.search(t)) / len(texts)
        results[f"{model_key}:{cond}"] = {
            "accuracy": acc,
            "ci95": ci,
            "correct": correct,
            "n": len(texts),
            "hash_hit_rate": hashhit,
        }
        print(f"{model_key:4s} {cond:6s}  acc={acc:.3f}  "
              f"ci=({ci[0]:.3f},{ci[1]:.3f})  hash={hashhit:.2f}")
    del mdl; torch.cuda.empty_cache()

with open("results/path_1_cot_tokens/results_gate.json", "w") as f:
    json.dump({"config": {...}, "cells": results}, f, indent=2)
```

## Sanity checks to perform before trusting the numbers

1. **Print one full (prompt, completion) pair per cell** before the full run. Confirm the `direct` cell's completion is ≤16 tokens and looks like `#### N`. Confirm the `cot` cell's completion shows reasoning and ends with `#### N`.
2. **Ground-truth parser sanity.** On the 500 gold answers, the `####` regex must hit 500/500. If not, the GSM8K split has drifted from assumption.
3. **Hash-hit rate.** In the `cot` cells, if `####`-extraction lands on <90% of outputs, the prompt format is drifting — debug before trusting fallback-extracted accuracy.
4. **Base model direct-cell degeneracy.** The base model may simply continue generating more fake exemplars instead of answering. That is expected and not a bug — the `max_new_tokens=16` cap plus last-integer fallback will score whatever it emits; flag if >20% of base/direct outputs have no integer at all.
5. **Determinism.** Run the same cell twice on the first 10 problems; greedy output must match byte-for-byte.
6. **Reproducibility.** Pin `transformers`, `torch`, and `datasets` versions in the JSON output. Pin the model commit SHA.

## Decision rule (pre-registered — commit before looking at numbers)

Based on the **`-it` CoT lift**, i.e. cell 2 accuracy − cell 1 accuracy:

- **GREEN — lift ≥ 10 points and 95% CIs non-overlapping.** Path 1 is alive on E2B. Write plan2: generation-length sweep + self-consistency on the `-it` CoT cell.
- **YELLOW — lift < 10 points, or CIs overlap, or both `-it` cells score < 15%.** Gate ambiguous. Do not kill the leg. One follow-up permitted: re-run cell 2 with self-consistency `k = 5, temperature=0.7, majority vote` on the same 500 problems. If that clears the 10-point bar, GREEN; otherwise RED.
- **RED — `-it` CoT ≤ `-it` direct, or lift < 3 points with CIs overlapping.** Path 1 is wounded on E2B at inference-only scale. Write up the null. Deprioritize the leg in the head-to-head and explain why in the plan-1 blog post.

Base-model cells (3–4) are **interpretive, not decisive**:
- Both show a lift → CoT capability is latent in the pretrained weights, instruction tuning amplifies.
- Only `-it` shows a lift → instruction tuning is doing the CoT heavy lifting.
- Neither shows a lift → gate almost certainly RED; cross-check before declaring.

Either base outcome shapes the plan-1 write-up but does not override the `-it` gate.

## Risks and known failure modes

- **Answer-extraction brittleness.** CoT outputs that don't end in `####` fall through to "last integer in output," which can spuriously credit wrong answers. Mitigate by reporting `hash_hit_rate` alongside accuracy; investigate any cell with `hash < 0.9`.
- **Base model format collapse.** The base model with 8-shot `direct` exemplars may keep generating exemplars instead of stopping at the current answer. `max_new_tokens=16` caps the damage; the last-integer fallback will still extract something, but the signal from that cell is weak by design.
- **Prompt-template drift across cells.** The four cells must share exactly the same exemplar questions, differing only in whether reasoning is present. Any incidental formatting difference (newlines, leading spaces) contaminates the comparison. Unit-test the two prompt builders against each other on a fixed question before the run.
- **VRAM on shared boxes.** Loading both models simultaneously OOMs on a 24 GB card. The script must load one, run its two cells, free it, then load the other.
- **`gemma-4-E2B` gated on HF.** Ensure `HF_TOKEN` is set and the user has accepted the Gemma license, or the script fails on model download with a misleading 401.

## Deliverable

A single Python script `path1_cot_gate.py` at repo root (mirroring `ple_sanity_check.py`) that:

1. Runs all four cells end-to-end in bf16 on one GPU.
2. Prints a table:

   ```
   it   direct  acc=0.210  ci=(0.176,0.249)  hash=0.98
   it   cot     acc=0.352  ci=(0.311,0.395)  hash=0.96
   base direct  acc=0.084  ci=(0.063,0.111)  hash=0.71
   base cot     acc=0.198  ci=(0.165,0.236)  hash=0.88
   ```

3. Writes `results/path_1_cot_tokens/results_gate.json` with config (model SHAs, package versions, exemplar set, n), per-cell numbers, and hash-hit rates.
4. Prints the decision-rule verdict (GREEN / YELLOW / RED) on the last line, derived from cells 1–2.

## Report back

When done, paste the printed table plus the verdict line. Do not start on plan2 (length / self-consistency sweeps) or write the plan-1 blog post until the gate result is reviewed.
