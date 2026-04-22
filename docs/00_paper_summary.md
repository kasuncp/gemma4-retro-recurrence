# 00 — Paper summary and why E2B is a non-trivial target

## The paper

**Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence**
McLeish, Li, Kirchenbauer, Kalra, Bartoldson, Kailkhura, Schwarzschild,
Geiping, Goldstein, Goldblum. arXiv:2511.07384v1, November 2025.
Code: `github.com/mcleish7/retrofitting-recurrence`.

## The method in one picture

The paper converts a pretrained fixed-depth transformer into a depth-recurrent
one by splitting layers into three groups and **looping the middle group**:

```
    Prelude P    →    Recurrent block R (looped r times)    →    Coda C
  (early layers)       (middle layers, applied r times)       (late layers)
```

Formally, for input sequence `x`, number of recurrences `r`, width `h`:

```
e  = P(x)                          # embeddings
s₀ ∼ 𝒩(0, σ²)ⁿˣʰ                    # random initial state
sᵢ = R(e, sᵢ₋₁)  for i ∈ {1..r}    # recurrent iteration
p  = C(sᵣ)                         # output distribution
```

`R` begins with a linear adapter that takes the concatenation of `sᵢ₋₁` and
`e` (so `2h → h`). Tuple notation `(P, R, C)` denotes the number of layers in
each: e.g. `(4, 8, 4)` on TinyLlama means prelude = 4 layers, recurrent block
= 8 layers, coda = 4 layers.

**Critical detail:** when converting, layers are *removed* — the recurrent
block is not the full interior of the original model. The paper uses
`(4, 8, 4)` on TinyLlama (22 layers → 16 used) and drops layers 4–9.

## What the paper proves

1. **Pretrained initialization beats random.** Initializing the recurrent
   model from pretrained TinyLlama / OLMo / Llama-3.2-1B weights converges
   much faster than random initialization (see the paper's Figure 2). At
   matched training FLOPs the gap is large; log-linear extrapolation suggests
   parity would take ~950B tokens of random training to catch up to the
   pretrained init.

2. **A recurrence curriculum reduces training cost.** Scheduling the mean of
   a Poisson-Lognormal `r` distribution from low → high over training is both
   data- and compute-efficient.

3. **Muon optimizer beats AdamW for recurrent models.** AdamW suffers loss
   spikes and NaN during recurrent training; Muon is stable. For the
   non-recurrent parent, the two optimizers are similar.

4. **Retrofitted recurrence can outperform the parent model on math.** Using
   Common Crawl math data, the retrofitted TinyLlama/OLMo/Llama models beat
   their non-recurrent parents on GSM8K and MATH.

5. **A "healing" period with minimal distribution shift recovers basic
   language modeling** that the surgery degrades.

## Why Gemma 4 E2B is a non-trivial target for the paper's method

The paper's three validation models (TinyLlama-1.1B-step-1431k-3T,
OLMo-2-0425-1B, Llama-3.2-1B) are all **plain pre-norm Llama-style decoders**
with uniform architecture across layers. Gemma 4 E2B is not. Four
architectural frictions surface before any compute is spent:

### 1. Per-Layer Embeddings (PLE)

Each decoder layer in E2B has its own small embedding table that gets added
into the residual stream at that specific layer. In the `transformers`
library (confirmed from `Gemma4TextDecoderLayer.forward` in
`models/gemma4/modeling_gemma4.py` line 1354), the layer signature is:

```python
forward(self, hidden_states, per_layer_input=None, shared_kv_states=None,
        position_embeddings=None, attention_mask=None, position_ids=None,
        past_key_values=None, **kwargs)
```

`per_layer_input` is passed **positionally**, not as a kwarg — a fact that
silently broke Round 2a's first hook (see `02_round2a_ple_policies.md`).

The paper's method assumes "the recurrent block is identical parameters being
reused." But if layer `i` has its own PLE table, looping layer `i` r times
means you either apply the same PLE on every iteration (amplifying its
contribution by factor r) or apply it once and omit it afterwards. The paper
offers no prior guidance here. This is Round 2a's question.

### 2. Hybrid local/global attention

E2B interleaves sliding-window (512-token) local attention with full global
attention, and the final decoder layer is mandated global. Rounds 2b/2c
recorded each layer's attention type alongside its PLE importance and KV
role. Concretely in the 35-layer stack (from round 2b metadata):

| Layer index | Attention type |
|---|---|
| 0, 1, 2, 3 | sliding |
| 4 | **global (full)** |
| 5, 6, 7, 8 | sliding |
| 9 | **global** |
| 10, 11, 12, 13 | sliding |
| 14 | **global** |
| 15, 16, 17, 18 | sliding |
| 19 | **global** |
| 20, 21, 22, 23 | sliding |
| 24 | **global** |
| 25, 26, 27, 28 | sliding |
| 29 | **global** |
| 30, 31, 32, 33 | sliding |
| 34 | **global** (mandatory final) |

Pattern: every 5th layer is global. A contiguous recurrent block will either
contain an even mix of local/global, or be pinned to one type depending on
placement.

### 3. Shared KV cache across consumer layers

Layers 15–34 are flagged `is_kv_consumer: true`. The KV producer→consumer
boundary is between layer 14 (last non-consumer) and layer 15 (first
consumer). This boundary turned out to be a **hard wall** for block looping
— see Round 3b.

### 4. Multimodal and thinking-mode instruction tuning

E2B is a VLM with audio modality and a built-in thinking-mode in its
instruction-tuned variant. For our probe work we load the **base** model
(`google/gemma-4-E2B`, not `-it`) to avoid interference.

## Consequences for how the project proceeded

The decision was not to attempt training until these architectural
incompatibilities had been measured. The entire project so far is a
**probe-only** characterization: hook-based runtime modifications with no
weight updates, measuring perplexity (Rounds 1–3c) and then reasoning
benchmark accuracy (Round 4), with hook-drift regression checks at every
step to verify that `r=1` reproduces baseline bitwise.

This has turned out to be the right call. Round 2c revealed a narrow valley
of loopable layers, Round 3b revealed the KV boundary is load-bearing, and
Round 4 revealed that perplexity stability does not imply reasoning
stability. Any of these findings, uncovered mid-training run, would have
wasted substantial compute.
