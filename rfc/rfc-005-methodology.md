# RFC 005 — Methodology and Harness Decisions

| Field | Value |
|---|---|
| **RFC** | 005 |
| **Title** | Methodology and Harness Decisions |
| **Status** | Methodological (normative for all Path 1 experiments) |
| **Path** | 1 — More CoT tokens on baseline E2B |
| **Applies to** | All Path 1 plans (plan 1 through plan 5) |

## Abstract

This RFC records the harness, prompting protocol, evaluation conventions, pinned dependencies, and sanity-check discipline that govern every Path 1 experiment. It also preserves two specific harness-bug stories from early runs, so a future investigator does not rediscover them the expensive way. Any future Path 1 plan must either follow this specification or explicitly and visibly document its deviations.

## Model pinning

All Path 1 experiments use exactly these two model checkpoints:

| Role | HuggingFace ID | Commit SHA |
|---|---|---|
| Primary (IT) | `google/gemma-4-E2B-it` | `b4a601102c3d45e2b7b50e2057a6d5ec8ed4adcf` |
| Secondary (base) | `google/gemma-4-E2B` | `9d53598892698e981fc42f78b0f8c005cecd63ca` |

Both variants. Both SHAs. Always record both in the manifest even when only one is used.

## Dependency pinning

| Package | Version |
|---|---|
| torch | `2.4.1+cu124` |
| transformers | `5.6.1` |
| datasets | `4.8.4` |
| accelerate | (bundled with transformers install chain) |

Do **not** upgrade `transformers` mid-investigation. Generation determinism under sampling is stable within a version but not across versions. A seed that produces identical output in 5.6.1 may diverge in 5.7.0.

## Benchmark

### GSM8K

- Source: `openai/gsm8k`, config `main`, split `test`.
- Slice: first `n = 500` problems, deterministic (`dataset.select(range(n))`). Plan 1 used the first 100; plan 2 and later extend to 500 on the same head slice.
- Gold extraction: `####` regex on the `answer` field (all 1319 GSM8K test answers match this regex; sanity-checked).

### ARC-Easy

- Source: `allenai/ai2_arc`, config `ARC-Easy`, split `test`.
- Slice: first `n = 500` problems.
- Gold extraction: `answerKey` field ∈ `{A, B, C, D}`. Historical numeric answer keys must be remapped to letters.

Do not mix slices across plans. The shared "first 500" slice is a cross-plan invariant; using different problems in different plans makes cross-plan paired tests impossible.

## Prompting

### 8-shot CoT exemplars (canonical Path 1 set)

- Source: Wei et al. 2022 GSM8K CoT paper.
- Format adjustment: the terminal `"The answer is X."` in each exemplar is replaced with `"#### X"` to unify answer extraction across direct and CoT cells.
- **Exemplar hash: `a33e6d90c6844317`**. Every plan's manifest must record this hash and the prompt-builder must be unit-tested against it.

If this hash ever fails to match in a future run, **stop**. The gate reference and the plan 2 plateau both depend on this exact exemplar set. A different exemplar set is a different experiment and must be labeled as such.

### Prompt template (IT model)

For the IT model, use `tokenizer.apply_chat_template` with a single user turn. The message content is:

- **CoT, 8-shot:** `<8-shot exemplars rendered as "Question: ...\nAnswer: ..."><blank line>Question: {q}\nAnswer:`
- **Direct, 8-shot:** same structure but exemplar answers stripped of reasoning, terminal `#### X` only.
- **CoT, 0-shot:** `{q}` or `{q}\n\nLet's think step by step.` — **both as a single user turn**, NOT raw text. The IT model expects the chat template.

Failure to apply the chat template on the IT model caused the plan 1 IT-direct cell to pattern-match exemplars instead of answering. See the harness-bug story below.

### Prompt template (base model)

Base model: raw-text prompt, no chat template. The base model does not have turn markers in its training.

## Decoding

### Greedy (axis A, gate cells)

```python
model.generate(
    **ids,
    max_new_tokens=<see per-plan>,
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id,
)
```

### Sampled (axis B, self-consistency)

```python
model.generate(
    **ids,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    num_return_sequences=k,
    pad_token_id=tokenizer.eos_token_id,
)
```

**Do not** loop `k` times over `generate(num_return_sequences=1)`. `num_return_sequences=k` batches the k chains through a shared KV cache and is 2–3× faster. This has already been caught once in the codebase and should remain caught.

## Seeding

At script start:

```python
import torch, random, numpy
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
numpy.random.seed(0)
```

Greedy decoding is bit-identical across runs with or without seeding (deterministic given the model weights and the prompt). Sampling is bit-identical across runs only with the seeds set and the transformers version pinned.

## Answer extraction

### GSM8K

1. Primary: `re.search(r'####\s*(-?\d+)', completion)`.
2. Fallback: last integer in the completion via `re.findall(r'(-?\d+)', completion)`.

Report both `hash_hit_rate` (fraction where the primary regex hit) and the overall accuracy. A cell where `hash_hit_rate < 0.85` warrants inspection — the prompt format or the model's adherence to it may have drifted.

### ARC-Easy

1. Primary: `re.search(r'[Tt]he answer is ([A-D])', completion)`.
2. Fallback: last standalone `[A-D]` in the completion.

Report `answer_hit_rate` analogously.

## Statistical convention

### Wilson 95% CI for accuracy

```python
import math
def wilson(k, n, z=1.96):
    p = k/n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return (p, centre - half, centre + half)
```

Reported alongside every accuracy. Wilson is used (not Wald) because it handles k = 0 and k = n without collapsing to a zero-width CI.

### McNemar's test for paired comparisons

All Path 1 comparisons are paired (same problems, different configurations). Use the exact binomial form on the discordant pairs:

```python
def mcnemar_exact(only_a, only_b):
    d = only_a + only_b
    if d == 0: return 1.0
    kmin = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(d, i) * 0.5**d for i in range(kmin+1)))
```

Significance threshold for conclusions of "a difference": p < 0.05. Threshold for "strong effect": p < 10⁻³. Gate findings (RFC 002) cleared p < 10⁻⁹.

## Data persistence conventions

### Per-problem JSONL (required output for every cell)

One line per problem (or per problem for axis B, with a `chains` sub-array):

```json
{"idx": int, "gold": int, "pred": int | null, "correct": int,
 "hash_hit": int, "completion": str,
 "n_gen_tokens": int, "prompt_tokens": int, "gen_secs": float}
```

Axis B per-problem format:

```json
{"idx": int, "gold": int, "k": int, "chains": [...],
 "voted_pred": int | null, "voted_count": int,
 "vote_degenerate": bool, "correct": int,
 "prompt_tokens": int, "gen_secs": float}
```

### Manifest (required with every results JSON)

Must include:
- Plan identifier
- Model IDs and commits (for every model used in the run)
- Dtype
- Every per-cell hyperparameter
- Exemplar hash
- Library versions (`torch`, `transformers`, `datasets`)
- Seed

### Resume-on-restart

All plans from plan 2 onward must support resume. Append-JSONL per cell in a checkpoint dir; on start, read existing JSONL, skip already-completed idxs, refuse to resume if the new run's manifest differs from the existing checkpoint's manifest.

## Mandatory sanity checks before trusting any run

Every plan's "sanity checks" section must include at least:

1. **Exemplar hash check** — verify the prompt builder produces the byte-identical exemplar text as previous plans.
2. **Cross-plan paired reproduction** — if the current plan shares a cell with a previous plan (e.g., plan 2 A3 vs plan 1 IT-CoT), verify completions match byte-for-byte on the shared problems. A mismatch > 0.5% means drift; stop and diagnose.
3. **Hash-hit rate floor** — any cell with `hash_hit_rate < 0.85` needs inspection before accuracy is trusted.
4. **Seed determinism** — restart any sampled cell on the first 10 problems; re-runs must match byte-for-byte.
5. **Plausibility check against published numbers** — if an accuracy is much higher or lower than the known community range for this model class, treat the entire run as suspect until the harness is verified.

## Harness-bug history (preserve, do not re-learn)

### Bug 1: IT-direct pattern-matching (plan 1, early run)

**Symptom:** IT-direct cell scored 0% on GSM8K at n = 20; every completion emitted a number that was the answer to a nearby *exemplar* question rather than the asked question. Generations looked like `" #### 8 Question: Janet's ducks..."`.

**Diagnosis:** The prompt was being fed as raw text to the IT model. Without the chat template, the model was in document-continuation mode and simply continued the 8-shot pattern by emitting the next exemplar's question.

**Fix:** Use `tokenizer.apply_chat_template` with a single user turn for the IT model.

**Lesson:** IT models are not drop-in replacements for base models in harnesses that assume raw-text prompting. Always apply the chat template for IT variants. Always inspect a full (prompt, completion) pair before trusting a batch run.

### Bug 2: GSM8K baseline at 4.8% (plan 1, rerun stage)

**Symptom:** The GSM8K baseline during an intermediate run scored 4.8% instead of the expected 40–55% range for Gemma 4 E2B at 8-shot CoT. 155 of 238 wrong answers hit `max_new_tokens = 256` mid-calculation.

**Diagnosis:** Three stacked issues — `max_new_tokens` too small (cutting off reasoning mid-arithmetic), the base model being fed with "Question:/Answer:" exemplars that triggered a document-completion mode, and no stop sequence on the exemplar boundary.

**Fix:** `max_new_tokens ≥ 512` for CoT, use the IT model with chat template for GSM8K, and add `"\nQuestion:"` as a stop sequence when using the base model.

**Lesson:** A low baseline accuracy is not a model measurement — it is a harness signal. Any baseline below the plausible range for the model class is a bug until proven otherwise. Do not run sweeps on a broken baseline; the sweep will be uninterpretable noise on top of a fixable-but-unfixed error.

### Both bugs were caught by sanity-check 5 (plausibility against published numbers)

Both would have taken weeks to notice if the only check had been "does the code run." The plausibility check — "is this accuracy in the expected range for Gemma 4 E2B?" — caught both within one run each. Treat the plausibility check as a blocking gate for every new plan, not an afterthought.

## Reporting conventions

### Table format for cell results

```
cell         acc    CI          hash_hit   mean_tok   mean_sec
<name>      0.xxx  (lo, hi)    0.xx        xxx        xx.x
```

One row per cell. All plans should produce output in this shape, converted from the per-plan JSON.

### Pre-registered decision rules

Every plan **must** include a "Pre-registered interpretation" section that commits to at least 3 qualitative outcomes (e.g., A/B/C, GREEN/YELLOW/RED) **before** running. No choosing the interpretation after seeing the data.

### Null results are reportable

If a pre-registered follow-up shows no effect, that is a completed experiment, not a failure. Write it up, put it in the RFC, and move on. The four-paths investigation's credibility depends on symmetric treatment of positive and null results.

## References

- [RFC 002](./rfc-002-cot-gate-finding.md) — first full application of this methodology.
- [RFC 003](./rfc-003-inference-compute-plateau.md) — cross-plan paired reproduction was the key sanity check for plan 2.
- Wei, J. et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* arXiv:2201.11903. Source of the canonical 8-shot exemplars.

---

*This RFC is normative. Any deviation from its specification must be explicitly justified in the plan document that deviates.*
