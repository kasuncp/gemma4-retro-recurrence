"""Wikitext input preparation and perplexity computation.

The perplexity loop disables the KV cache because our loop hooks re-enter
decoder layers; each re-entry otherwise appends to ``past_key_values`` and
the attention mask goes out of sync.
"""

import math

import torch
from datasets import load_dataset


def prepare_inputs(tokenizer, num_sequences, max_length):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t.strip()) > 500][:num_sequences]
    inputs = [
        tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to("cuda")
        for t in texts
    ]
    print(f"Prepared {len(inputs)} eval sequences (max_length={max_length}).")
    return inputs


def compute_perplexity(model, inputs):
    # use_cache=False: we are computing forward-pass perplexity, not generating.
    # When the loop hooks re-enter a decoder layer with caching on, the layer's
    # self_attn appends new K/V to past_key_values on each iteration, making
    # the K-length grow beyond the attention mask's shape (crash observed as
    # "expanded size of tensor (1023) must match existing size (512)" on
    # sliding-attention layers). Disabling cache makes the measurement pure.
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for inp in inputs:
            out = model(**inp, labels=inp["input_ids"], use_cache=False)
            n_tokens = inp["input_ids"].numel() - 1
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens
    mean_nll = total_nll / total_tokens
    return mean_nll, math.exp(mean_nll)
