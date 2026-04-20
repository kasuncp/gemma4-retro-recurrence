# Gemma 4 E2B — PLE × Loop Recurrence Sanity Check

## Context

This is a 30-minute probe to answer one narrow question before committing to a larger research effort on retrofitting depth-recurrence into Gemma 4 E2B (based on the paper "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence", McLeish et al., 2025, arXiv:2511.07384).

**The concern:** Gemma 4 E2B uses Per-Layer Embeddings (PLE). Each decoder layer has its own small embedding table that gets added into the residual stream at that specific layer. The paper's retrofitting method works by looping a block of layers `r` times at inference. Naively, this means PLE for a given layer gets injected `r` times instead of once — which the pretrained weights were never prepared for.

**The question this probe answers:** *Does the most naive possible loop — running one middle layer `r` times, with PLE applied on every iteration — immediately break the model, or does it degrade gracefully?*

This is the gate before we invest in a larger (1–2 day) probe and, eventually, full training.

## Scope — what this probe is and isn't

**In scope:**
- Load pretrained `google/gemma-4-E2B` (base, not `-it`).
- Pick ONE middle decoder layer (layer 17 out of 35).
- Monkey-patch its forward pass to run `r` times in a row.
- Measure next-token-prediction perplexity on a small slice of Wikitext-2 for `r ∈ {1, 2, 4, 8}`.
- Print results.

**Explicitly out of scope (do NOT add these):**
- No training. No weight updates of any kind.
- No PLE variants (no scaling, no "apply once" logic). Use default PLE behaviour.
- No loop over multiple target layers.
- No GSM8K or downstream task evaluation.
- No block-of-layers looping. One layer only.
- No plotting or fancy output. Just printed numbers.

If implementing this takes more than one file and ~100 lines of code, something has gone wrong — stop and ask for clarification.

## Environment

```
pip install -U transformers torch accelerate datasets
```

Hardware: one GPU with ≥16 GB VRAM in bfloat16. H100/A100/4090/3090 all fine.

## Implementation steps

### Step 1 — Load model and tokenizer

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-4-E2B"  # base pretrained, NOT the -it instruction-tuned variant

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
model.eval()
```

**Verify before proceeding:** print `model` and confirm there are exactly 35 decoder layers. Find the attribute path to the layer list — it's probably `model.model.layers` but confirm (HuggingFace sometimes wraps things differently for multimodal models; if `AutoModelForCausalLM` doesn't work, try `AutoModelForMultimodalLM` and extract the text decoder).

### Step 2 — Prepare eval data

```python
from datasets import load_dataset

ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [t for t in ds["text"] if len(t.strip()) > 500][:50]

inputs = [
    tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to("cuda")
    for t in texts
]
```

50 sequences of up to 512 tokens each. Sufficient for a rough perplexity estimate in a few minutes.

### Step 3 — Define the looping hook

```python
TARGET_LAYER = 17  # middle of 35 layers

decoder_layers = model.model.layers  # adjust path if model structure differs
orig_forward = decoder_layers[TARGET_LAYER].forward

def make_looped_forward(r):
    def looped(hidden_states, *args, **kwargs):
        for _ in range(r):
            out = orig_forward(hidden_states, *args, **kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out
        return out  # return in original format (tuple or tensor)
    return looped
```

**Key detail:** Gemma decoder layers typically return a tuple `(hidden_states, ...)`. Preserve that return format so downstream code doesn't break. If you hit shape errors, inspect `out` once and adjust.

**Do NOT touch PLE.** PLE is applied inside `orig_forward`, so it automatically gets applied on every iteration of the loop. That's the whole point — we want to see what happens with naive looping.

### Step 4 — Evaluate over `r ∈ {1, 2, 4, 8}`

```python
import math

results = {}
for r in [1, 2, 4, 8]:
    decoder_layers[TARGET_LAYER].forward = make_looped_forward(r)

    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for inp in inputs:
            out = model(**inp, labels=inp["input_ids"])
            n_tokens = inp["input_ids"].numel() - 1  # label-shift
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens

    mean_nll = total_nll / total_tokens
    ppl = math.exp(mean_nll)
    results[r] = (mean_nll, ppl)
    print(f"r={r:2d}:  mean NLL = {mean_nll:.4f}   perplexity = {ppl:.2f}")

# Restore original forward
decoder_layers[TARGET_LAYER].forward = orig_forward
```

### Step 5 — Print a summary

```python
print("\n=== Summary ===")
baseline_ppl = results[1][1]
for r, (nll, ppl) in results.items():
    ratio = ppl / baseline_ppl
    print(f"r={r:2d}:  ppl={ppl:7.2f}   (×{ratio:.2f} vs r=1)")
```

## Sanity checks to perform before trusting the numbers

1. **`r=1` should match the unmodified base model.** Run the model without any hook first, compute perplexity on the same 50 sequences, and confirm `r=1` perplexity matches to within floating-point noise. If they disagree, the hook is broken — debug before interpreting further.

2. **Check you're using the base model, not `-it`.** The instruction-tuned model has different behaviour on raw text. `google/gemma-4-E2B` (no suffix) is the one we want.

3. **Confirm bfloat16.** Mixed precision issues can masquerade as "PLE looping breaks everything." If numbers look wildly off, try fp32 once as a control.

## Interpreting the results

Commit to these readings **before** looking at the numbers:

- **`r=2` perplexity within ~1.2× of `r=1`** → pretrained weights gracefully tolerate one extra loop. Proceed to the real probe with cautious optimism.
- **`r=2` perplexity 2×+ worse; `r=4` perplexity 10×+ worse** → PLE cannot absorb repeated injection without training. Pivot the retrofit plan to use an architectural fix (new shared PLE for the recurrent block) rather than relying on training to patch it.
- **`r=2` is fine; `r=8` perplexity is noticeably worse but within ~3× of baseline** → expected and healthy. This is exactly the kind of degradation the healing+training phase is designed to repair. Green light for the full probe.
- **Anything produces NaN or `inf`** → numerical blowup. PLE injection is saturating activations. Same conclusion as the 2× case — need architectural fix, not just training.

## Deliverable

A single Python script `ple_sanity_check.py` that runs in under 30 minutes on one GPU and prints a table like:

```
r= 1:  ppl=  12.34   (×1.00 vs r=1)
r= 2:  ppl=  14.50   (×1.17 vs r=1)
r= 4:  ppl=  22.10   (×1.79 vs r=1)
r= 8:  ppl=  51.30   (×4.16 vs r=1)
```

That table — plus a one-line note on which of the four readings above it matches — is the entire output.

## If things go wrong

- **Model loading fails with "unknown architecture":** try upgrading `transformers` (`pip install -U transformers`). Gemma 4 is new; older versions may not support it.
- **Loading works but `model.model.layers` is wrong path:** print `model` once to find the actual path to the decoder layer list. For multimodal models it might be nested under something like `model.language_model.model.layers`.
- **Layer forward signature mismatch:** inspect `decoder_layers[17].forward` signature and pass args through correctly. Gemma decoder layers have specific kwargs (`attention_mask`, `position_ids`, `past_key_value`, etc.) that must be passed through cleanly.
- **OOM:** reduce `max_length` from 512 to 256, or reduce number of sequences from 50 to 20.

## Report back

When done, paste the printed table and note which interpretation bucket it falls into. Do not do any further work beyond this probe without checking in.