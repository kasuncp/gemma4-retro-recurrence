# Path 2 v2 — Phase 0: Infrastructure (no GPU)

**Project:** Retrofitted recurrence on Gemma 4 E2B
**Status:** Ready for implementation
**Depends on:** Path 1 results (plans 5, 7, 8, 9), original Path 2 results (rounds 1, 3b, 3c, 5, 5c)
**Blocks:** Phase 1 (`phase1_hook_sanity.md`), and every later phase

---

## Why Phase 0 exists

Original Path 2 burned ~30 GPU-hours mapping perplexity tolerance on the **base** model and only discovered after the fact that:

- The map doesn't transfer to the IT model (round 5c: IT raw-text ppl is 4068× base — the metric breaks)
- The reasoning eval harness was misconfigured for ~5 rounds (round 4 baseline was 4.8 % GSM8K vs the model's true ceiling of ~71 %)
- The 8-shot CoT prompt the harness used was actively harmful on the IT model (Path 1 plan 5: 30.0 % vs 71.6 %)
- Recurrent configs at r=8 collapse into 97 % truncation under that broken prompt; we don't know whether collapse is structural or prompt-induced

Phase 0 ships the harness *Path 1 already validated* and the **measurement strategy** for everything that follows, **before any GPU time is spent**. No model loading, no dataset downloads bigger than a tokenizer, no pod spin-ups. The bar is: a developer can run `make test-phase0` on a CPU laptop and confirm every diagnostic, prompt builder, and extractor works against fixtures.

---

## Scope

**In scope** (in priority order):

1. Port Path 1's `smart_v2` GSM8K extractor + 12 unit-test cases.
2. Port Path 1's repetition-loop detector regex + truncation flag.
3. Port Path 1's `C2` prompt builder (chat-templated zero-shot, no CoT).
4. Port Path 1's `8-shot CoT` prompt builder (Wei et al. exemplars) as a *control* (it underperforms on IT, but it's the cross-round bridge to Path 2 round 5).
5. Loaders for the three benchmarks Phase 3 will use:
   - GSM8K (test split, deterministic shuffle, first N)
   - ARC-Challenge (test split, MC, deterministic shuffle, first N)
   - BBH-lite (a ~5-task subset, the "where prompt format inverts" axis)
6. A single new mode `--mode=path2-v2-eval` on `ple_sanity_check.py` that takes a config name + benchmark + N and produces a JSONL stream of per-problem rows + a JSON summary.
7. New registry entry `path2-v2` in `scripts/run.sh` with `result_root=results/path_2_depth_recurrence_v2/<phase>` and `recursive` depth.
8. New experiment yaml `experiment_path2_v2.yaml` parameterised by `phase` (the phase yaml lives under `experiments/path_2_v2/<phase>.yaml` and the runner picks one via env var).
9. Test harness on a CPU laptop:
   - `tests/path2_v2/test_extractors.py` — 12 GSM8K extractor cases + 6 loop-detector cases
   - `tests/path2_v2/test_prompts.py` — chat template round-trip
   - `tests/path2_v2/test_runtime_smoke.py` — load 1 GSM8K problem, build prompt, run extractor on a fixture completion, assert the row schema
   - `tests/run/test_dry_run.sh` — register `path2-v2` and assert dispatch

**Explicitly out of scope** (do *not* let this PR sprawl):

- No new architectural probes. No perplexity sweeps. No model loading at all in Phase 0.
- No new metrics beyond the four we name below (accuracy, loop_rate, truncation_rate, parse_rate).
- No retrofit-training prep. No LoRA. No on-device measurement.
- No re-derivation of `smart_v2`. Pull verbatim from `experiments/path1_c2_vs_a3_inspection.py`. We *know* it lifts C2 from 0.716 → 0.780 on N=500 GSM8K; that's the contract.
- No new benchmark beyond GSM8K / ARC-C / BBH-lite. MATH stays out (Path 1 confirmed E2B caps at 0 %); ARC-Easy stays out (saturated); HumanEval / GPQA / MMLU stay out (separate question, separate plan).

If Phase 0 grows beyond ~6 source files + 3 test files + 1 plan doc, something is wrong — stop and re-scope.

---

## Deliverables (file-by-file)

### `probes/extractors.py` — answer extraction + pathology detection

```python
"""Path-1-validated extractors and pathology detectors for path 2 v2.

Borrowed verbatim from experiments/path1_c2_vs_a3_inspection.py
(smart_v2) and experiments/path1_rep_loop_analysis.py (repetition
detector). Both have shipped on N=500 GSM8K runs and a multi-thousand-
completion rep-loop study; do not modify the regexes without re-running
those test suites.
"""

import re

# --- GSM8K integer extractor (smart_v2) ---
# Path 1 plan 9 finding: lifts N=500 C2 accuracy from 0.716 -> 0.780,
# i.e. recovers ~6.4 pp of correct-but-buried answers.

_BOLD_RE   = re.compile(r"\*\*[^*]*?\$?(-?\d+)(?:\.\d+)?[^*]*?\*\*")
_ANSWER_RE = re.compile(r"(?:[Aa]nswer|[Tt]otal|[Ff]inal).{0,80}?\$?(-?\d+)(?:\.\d+)?")
_EQ_END_RE = re.compile(r"=\s*\$?(-?\d+)(?:\.\d+)?\s*(?:[A-Za-z%]+)?\s*\.?\s*$", re.MULTILINE)
_INT_RE    = re.compile(r"-?\d+")

def smart_v2(text: str) -> int | None:
    """Path 1 plan 9 extractor. See plans/path_1_cot_tokens/plan9.md.
    Apply rules in order, first match wins, normalise to int.
    """
    if not text:
        return None
    norm = text.replace("\\$", "$").replace(",", "")
    bolds = _BOLD_RE.findall(norm)
    if bolds:
        try: return int(bolds[-1])
        except ValueError: pass
    m = _ANSWER_RE.search(norm[-400:])
    if m:
        try: return int(m.group(1))
        except ValueError: pass
    matches = list(_EQ_END_RE.finditer(norm))
    if matches:
        try: return int(matches[-1].group(1))
        except ValueError: pass
    nums = _INT_RE.findall(norm)
    return int(nums[-1]) if nums else None


def flexible_extract_legacy(text: str) -> int | None:
    """Last-integer fallback. Identical to round 4/5's extractor.
    Kept so we can report a 'legacy accuracy' column for cross-round
    bridging without re-running anything.
    """
    if not text:
        return None
    nums = _INT_RE.findall(text.replace(",", ""))
    return int(nums[-1]) if nums else None


# --- ARC letter extractor ---
# Greedy regex for "Answer: A" / "(B)" / final standalone letter.
_ARC_RE = re.compile(
    r"(?:answer\s*[:\-]?\s*|\bthe answer is\s+|^\s*)\(?([A-D])\)?",
    re.IGNORECASE | re.MULTILINE,
)

def extract_arc(text: str) -> str | None:
    if not text:
        return None
    m = _ARC_RE.search(text[-200:])  # bias toward final letter
    if m:
        return m.group(1).upper()
    # Fallback: last standalone A-D in the string.
    last = re.findall(r"\b([A-D])\b", text)
    return last[-1].upper() if last else None


# --- Pathology detectors ---

# Path 1 plan 3 / rep_loop_analysis canonical regex. Detected at
# 9-10% on the IT baseline and 97-100% on round 5 recurrent configs.
_LOOP_RE = re.compile(r"(.{10,60})\1{2,}", re.DOTALL)

def has_repetition_loop(text: str) -> bool:
    return bool(_LOOP_RE.search(text or ""))

def is_truncated(n_gen_tokens: int, max_new_tokens: int = 512) -> bool:
    """Truncation = generation hit the max_new_tokens cap with no EOS.
    Round 4 hit 155/238 truncations at 256-token cap; the 510-token
    cutoff in Path 1 plan 9 (vs the configured 512) is the conservative
    threshold that survives off-by-one in tokenizer post-processing.
    """
    return n_gen_tokens >= (max_new_tokens - 2)
```

### `probes/prompts.py` — prompt builders

```python
"""Prompt builders shared across phases.

C2 = chat-templated zero-shot. Path 1's headline finding: this beats
8-shot CoT on Gemma-4-E2B-it (71.6% vs 30.0% on N=500 GSM8K).

ALWAYS use apply_chat_template — feeding the prompt as raw text drops
performance ~30pp because the IT model expects turn markers.
"""

def build_c2_gsm8k(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def build_c2_arc(tokenizer, question: str, choices: list[str]) -> str:
    """ARC-C question + lettered MC choices, zero-shot via chat template."""
    options = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
    user = f"{question}\n\n{options}\n\nAnswer with a single letter."
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def build_c2_bbh(tokenizer, question: str) -> str:
    """BBH-lite: chat-templated, no exemplars. Path 1 plan 7 found
    that on BBH the 'direct' raw-text prompt actually beats C2 (49.8%
    vs 20.6% on N=500). We still use C2 here so all three benchmarks
    share a single decoding configuration; if BBH lands far below
    expectation, swap in raw-text prompts as a follow-up — don't
    confound it with recurrence in the headline run.
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# 8-shot CoT exemplars — Path 1 plan 5 sourced from Wei et al. 2022.
# Kept as a CONTROL only. Reproducing 54.8% with this on N=500 is the
# cross-round bridge to Path 2 round 5's headline number.
WEI_8SHOT_EXEMPLARS = [
    # ... 8 (Q, CoT, A) tuples; copy verbatim from experiments/path1_zero_shot.py.
]

def build_8shot_cot_gsm8k(tokenizer, question: str) -> str:
    pre = "\n\n".join(
        f"Q: {q}\nA: {cot} The answer is {a}."
        for (q, cot, a) in WEI_8SHOT_EXEMPLARS
    )
    user = f"{pre}\n\nQ: {question}\nA:"
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
```

### `probes/datasets.py` — benchmark loaders with deterministic shuffle

```python
"""Path 2 v2 benchmark loaders.

Determinism contract: every loader takes (split, n, seed=42) and
returns the same N problems for a given (split, seed, dataset version).
Shuffling is essential — first-N in dataset order is non-iid in subtle
ways (GSM8K's first 50 over-represent particular templates).

Each row dict has at least: idx (int), question (str), gold (any).
ARC-C also has choices and gold_letter; BBH-lite has task and gold.
"""

import random
from datasets import load_dataset

def load_gsm8k(n: int, seed: int = 42) -> list[dict]:
    ds = load_dataset("gsm8k", "main", split="test")
    idxs = list(range(len(ds))); random.Random(seed).shuffle(idxs)
    return [
        {"idx": i, "question": ds[i]["question"], "gold": ds[i]["answer"]}
        for i in idxs[:n]
    ]

def load_arc_challenge(n: int, seed: int = 42) -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    idxs = list(range(len(ds))); random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs[:n]:
        row = ds[i]
        choices = row["choices"]["text"]
        labels = row["choices"]["label"]
        gold = row["answerKey"]
        # Map label to letter A-D (some labels are "1","2","3","4")
        gold_letter = chr(65 + labels.index(gold)) if gold in labels else None
        out.append({
            "idx": i,
            "question": row["question"],
            "choices": choices,
            "gold": gold,
            "gold_letter": gold_letter,
        })
    return out

def load_bbh_lite(n: int, seed: int = 42, tasks: list[str] | None = None) -> list[dict]:
    """Sample uniformly across N tasks. Default 5: object_counting,
    boolean_expressions, navigate, web_of_lies, formal_fallacies.
    Path 1 plan 7 used these in its 'easy/hard' BBH split.
    """
    if tasks is None:
        tasks = [
            "object_counting", "boolean_expressions", "navigate",
            "web_of_lies", "formal_fallacies",
        ]
    rng = random.Random(seed)
    per_task = max(1, n // len(tasks))
    out = []
    for t in tasks:
        ds = load_dataset("lukaemon/bbh", t, split="test")
        idxs = list(range(len(ds))); rng.shuffle(idxs)
        for i in idxs[:per_task]:
            out.append({
                "idx": f"{t}/{i}",
                "task": t,
                "question": ds[i]["input"],
                "gold": ds[i]["target"],
            })
    return out[:n]
```

### `probes/eval_v3.py` — unified eval loop

```python
"""Path 2 v2 unified eval loop.

Generation contract:
  - max_new_tokens = 512 (Path 1 plan 5: 256 truncated 65% of GSM8K)
  - do_sample = False, temperature = None, top_p = None (greedy)
  - use_cache = False (REQUIRED for the block-loop hook)
  - pad_token_id = tokenizer.eos_token_id

Per-row schema (one JSONL line per problem):
  {idx, question, gold, completion, n_gen_tokens, pred_smart_v2,
   pred_legacy, correct_smart_v2, correct_legacy, truncated,
   loop_flag, t_gen_seconds}

Per-config summary fields:
  accuracy_smart_v2, accuracy_legacy, truncation_rate, loop_rate,
  parse_rate, n_problems, mean_gen_tokens, mean_t_gen_seconds.
"""

# ... see implementation in experiments/path2_v2_eval.py
```

### `experiments/path2_v2_eval.py` — entry point

A new mode on the existing `ple_sanity_check.py` would balloon that file. Keep this
script standalone, import shared utilities from `probes/`. Argparse:

- `--phase {sanity, layer-map, block-sweep, r-sweep}` — the high-level role
- `--config <name>` — config dict from a registry (baseline-C2, W5-r8, etc.)
- `--benchmark {gsm8k, arc-c, bbh-lite}`
- `--n <int>` — number of problems
- `--seed 42`
- `--max-new-tokens 512`
- `--model-id google/gemma-4-E2B-it` (default; path 1's anchor)
- `--output-dir results/path_2_depth_recurrence_v2/<phase>/`
- `--resume` (default true) — append to existing JSONL if present
- `--summarize-only` — read JSONL, emit summary JSON + table, no GPU

### `scripts/run.sh` registry update

```bash
EXPERIMENT_KEYS=(... probe-plan5c path2-v2)
EXPERIMENT_SCRIPTS=(... experiments/ple_sanity_check.py experiments/path2_v2_eval.py)
EXPERIMENT_DEFAULTS=(... "--mode it-perplexity-bridge" "--phase sanity --config baseline-C2 --benchmark gsm8k --n 50")
EXPERIMENT_ROOTS=(... results/path_2_depth_recurrence/plan5c results/path_2_depth_recurrence_v2)
EXPERIMENT_DEPTHS=(... recursive recursive)
```

### `experiment_path2_v2.yaml`

```yaml
# Default to phase 1 sanity. Each phase has its own yaml under
# experiments/path_2_v2/<phase>.yaml; override flags via CLI.
run:
  flags: "--script path2-v2 --phase sanity --benchmark gsm8k --n 50"
  result_dir: "results/path_2_depth_recurrence_v2/sanity"

pod:
  gpu_types: "NVIDIA GeForce RTX 4090"
  gpu_count: 1
  cloud_type: SECURE
  container_disk_gb: 30
  volume_gb: 50

git:
  url: "https://github.com/kasuncp/gemma4-retro-recurrence.git"
  ref: "path_2_v2"

budget:
  cap_usd: 4
  emergency_usd: 0.50
  max_hours: 3

watch:
  tick_seconds: 300
```

---

## Tests (CPU only, run via `make test-phase0`)

### `tests/path2_v2/test_extractors.py`

12 GSM8K cases borrowed from Path 1 plan 9, plus 4 ARC cases:

```python
GSM8K_CASES = [
    ("The answer is **$16.00**.", 16),
    ("So the customer saved $50 - $34 = $16.", 16),
    ("**16**", 16),
    ("Final answer: 16", 16),
    ("After 32% off, the price is $34. With tax: $34 × 1.06 = $36.04. The savings: $50 - $34 = **$16.00**.", 16),
    ("The total cost is 1,250 dollars.", 1250),
    ("Negative two: -2", -2),
    ("$3.50 and $4.50, total $8.", 8),
    ("Step 1: 5+3=8. Step 2: 8*2=16.", 16),
    ("", None),
    ("There is no number here.", None),
    # Loop case — extractor still returns last int; loop_flag is the gate
    ("Step 1: 5+3=8. " * 10, 8),
]

LOOP_CASES = [
    ("normal completion no loop here", False),
    ("aaa bbb ccc " * 5, True),                # repeated 11-char block
    ("Step 1: 5+3=8. " * 4, True),
    ("short", False),
    ("a" * 100, False),                        # single char doesn't trigger
    ("ab" * 50, False),                        # 2-char block below 10-char minimum
]
```

### `tests/path2_v2/test_prompts.py`

- Round-trip a GSM8K question through `build_c2_gsm8k` with a Gemma tokenizer (downloaded once on CI), assert the output contains `<start_of_turn>user`, the question text, and `<start_of_turn>model` exactly once.
- Same for `build_c2_arc` with 4 choices, assert lettered options.
- Sanity: `apply_chat_template(... add_generation_prompt=True)` returns a prompt that ends in the `model` turn header, not the `user` turn — Path 1 plan 5 spent its first hour on this exact bug.

### `tests/path2_v2/test_runtime_smoke.py`

- Load 1 GSM8K problem (cached, no network).
- Build a fixture completion text + n_gen_tokens.
- Run `score_one_row(...)` and assert the returned dict has the per-row schema.
- Confirm `accuracy_smart_v2` and `accuracy_legacy` agree on a 'normal' row and disagree on a `**$16.00**` row.

### `tests/run/test_dry_run.sh`

Add `path2-v2|experiments/path2_v2_eval.py|--phase sanity --config baseline-C2 --benchmark gsm8k --n 50|results/path_2_depth_recurrence_v2|recursive` to EXPECTED.

---

## Phase 0 acceptance criteria

All must pass before Phase 1 spins up a pod:

1. `pytest tests/path2_v2/` → 100% pass on a CPU laptop in <30 s.
2. `bash tests/run/test_dry_run.sh` → all assertions pass including the new `path2-v2` row.
3. `python experiments/path2_v2_eval.py --summarize-only --output-dir tests/fixtures/path2_v2_smoke/` produces a valid summary JSON without loading a model.
4. The 12 GSM8K extractor cases match the Path 1 plan 9 numbers verbatim — no derivation drift.
5. The repetition detector matches Path 1 plan 3's canonical 9–10 % rate when run on `results/path_1_cot_tokens/plan5/cells/C2_zeroshot_plain__0000_0500.jsonl` (sanity check that the regex hasn't changed under the hood).
6. README in `plans/path_2_depth_recurrence_v2/` lists the four metrics + their pass/fail bands so any later phase can quote them by name without re-deriving.

---

# Design decisions (the why behind every test)

This section is load-bearing. The tests below depend on these choices,
and any later plan that wants to deviate must explicitly justify the
deviation against this list.

### D1. Reasoning accuracy is the primary metric, not perplexity.

**Decision:** From Phase 1 onward, every cell reports four headline
numbers — `accuracy_smart_v2`, `loop_rate`, `truncation_rate`,
`parse_rate` — and *no* perplexity number above the fold.

**Why:** Round 5c established that raw-text Wikitext perplexity on
the IT model is ~4068× the base model — IT chat post-training breaks
raw next-token prediction so completely that block-looping then
*reduces* perplexity by acting as an inadvertent denoiser. Comparing
ratios across base and IT crosses two different distributions. The
target deployment is a chat-IT model answering reasoning questions;
that's what we measure.

**Consequence:** rounds 1, 2a, 2b, 2c, 3a, 3b, 3c, 5c are all
diagnostic on the wrong axis. Their hooks are valuable; their
metrics aren't.

### D2. Single-config-name registry; no in-script branching on flags.

**Decision:** Each cell is a named dict in a single `CONFIGS` list,
identical to round 5's pattern. Phase 1's `baseline-C2` and Phase 4's
`W5-r3` are both first-class entries; the entry decides the prompt,
block, r, and PLE strategy.

**Why:** Round 5 added `--ple-strategy iter1-only` as a flag. The
flag interacts with `--block` and `--r` and `--prompt`, producing a
combinatorial branch every reader has to chase. A name → dict makes
every config a primary key the JSONL rows can be filtered on.

### D3. C2 prompt is the only primary prompt; 8-shot CoT is a control.

**Decision:** Every reasoning cell uses C2 (chat-templated zero-shot,
no CoT suffix). One exception: the `baseline-8shot-control` cell on
GSM8K, which exists only to reproduce round 5's 54.8 % within ±2 pp
as a cross-round bridge.

**Why:** Path 1 plan 5: 71.6 % C2 vs 30.0 % 8-shot CoT on N=500. McNemar p ≈ 5e-43.
8-shot CoT actively degrades E2B-it on math reasoning. Round 5's
"recurrent configs collapse" finding was *partly* prompt-induced — we
can't tell how much without re-running on a competitive baseline.

**Consequence:** Phase 3 can't fairly compare to round 5's recurrent
numbers. It compares to Path 1's 71.6 % anchor and the new
`baseline-8shot-control` (which serves as a bridge to round 5).

### D4. Three orthogonal benchmarks, not one.

**Decision:** GSM8K + ARC-Challenge + BBH-lite. ARC-Easy and MATH stay out.

**Why:**
  - **GSM8K** — math reasoning, free-form answer, the canonical
    bench for 2 B-class models. Path 1 anchor: 71.6 %.
  - **ARC-Challenge** — multiple-choice reasoning, no free-form
    parsing risk. Path 1 plan 7 anchor: 75.6 % C2. The 62-pp
    separation between C2 and 8-shot proves it discriminates prompts.
  - **BBH-lite** — *prompt format inverts* on BBH (raw text 49.8 %
    beats C2 20.6 %). That makes BBH a useful canary for "is the
    recurrent block damaging *general* reasoning, or specifically
    chat-format reasoning?" If recurrence helps BBH but hurts GSM8K
    on C2, the failure mode is distribution-specific, not capability-
    specific.
  - **ARC-Easy** is excluded — saturated at 83 % across every
    prompt, no signal.
  - **MATH** is excluded — capability ceiling at 0 %, no signal.

### D5. Smart_v2 + legacy extractors reported side by side.

**Decision:** Every GSM8K row records `pred_smart_v2`,
`pred_legacy`, `correct_smart_v2`, `correct_legacy`. Headline
accuracy is `accuracy_smart_v2`. Legacy column lets us re-derive
round-5-style numbers without re-running anything.

**Why:** Path 1 plan 9 lifted C2 from 0.716 → 0.780 just by using a
better extractor — 6.4 pp of "accuracy" was extraction noise. We can
afford to compute both for free.

### D6. Loop rate and truncation rate are FIRST-CLASS, not diagnostic.

**Decision:** Any cell with `loop_rate > 0.5` is reported as
"generation-pathology" *before* its accuracy is interpreted. Same
for `truncation_rate > 0.5`.

**Why:** Round 5 partials had `truncation_rate ≈ 1.0` on every
recurrent config and reported 1–3 % accuracy — that's not a
"recurrence hurts reasoning" finding, it's a "model never gets to
answer" finding. Conflating them led to a months-long design
detour. The 30 % flag rate is a Path 1 calibration point.

### D7. Path 1's 50-problem in-band gate before any 200-problem sweep.

**Decision:** Phase 1 runs `baseline-C2` on N=50 GSM8K and asserts
`accuracy ∈ [65 %, 82 %]` before any other Phase-1+ cell starts.

**Why:** Round 4's 4.8 % GSM8K baseline would have failed this gate
in 5 minutes. We spent ~15 hours of compute past that point. The
gate's CI is roomy (path 1 anchor 71.6 %, ±10 pp at N=50), so
legitimate variation passes; only structural breakage trips it.

### D8. r=1 must be a token-for-token no-op on every model and every block.

**Decision:** Phase 1 includes a per-block r=1 generation check
against the no-hook baseline (10 problems, generated text bitwise
equal). Any drift halts.

**Why:** The hook's correctness depends on `r=1` reducing to the
identity. We've checked this on perplexity (rounds 2c, 3b, 5c), but
generation is a different code path — it goes through `model.generate()`,
which calls forward many times and stitches token-by-token output.
Bitwise generation match is the load-bearing test, not perplexity drift.

### D9. `use_cache=False` during generation; this is structural.

**Decision:** Hard-coded; no flag.

**Why:** The block-loop hook re-enters decoder layers; with caching
on, each re-entry appends to `past_key_values` and the K-length
grows beyond the attention mask shape. We saw this exact crash in
round 1 on sliding-attention layers. Round 4/5 already pinned this;
keeping it pinned in Phase 0 prevents regression.

**Consequence:** Generation is ~5–10× slower than cached. Plan
sample sizes (N=50 sanity, N=200 sweep) are scaled to that
constraint. Bigger N requires bigger compute.

### D10. Deterministic shuffle, fixed seed, before truncation to N.

**Decision:** Every benchmark loader does `random.Random(42).shuffle(...)`
before slicing to N.

**Why:** First-N in dataset order is non-iid. GSM8K's first 50 over-
represent particular templates; ARC-C's first 200 over-represent
specific Bloom's-taxonomy categories. Shuffling makes "N=50 ⊂ N=200 ⊂
N=500" so smaller subsets are statistically valid prefixes of larger
ones. This is what Path 1 plan 5 did and what round 5 did NOT do.

### D11. JSONL append-and-fsync; no in-memory accumulation.

**Decision:** Every per-problem row is JSONL-appended and fsync'd
before the next problem runs.

**Why:** Round 5's job ran for hours, hit OOM in the middle of
config 6 of 9, and lost the partial accuracy of configs 4–5. JSONL
append survives crashes; `runpod.sh sync-down` rsyncs partial JSONL
naturally. This is round 4/5's pattern; we're not changing it.

### D12. Single-layer probe before block probe before r-sweep.

**Decision:** Phase 2 probes one layer at a time. Phase 3 probes
contiguous blocks. Phase 4 sweeps r on the winning block.

**Why:** Original path 2 inverted this — went straight to blocks
in round 3 because round 2c's perplexity map looked clean enough.
That assumption was right for base perplexity and wrong for
IT reasoning. A single-layer reasoning map on IT directly tells us
"which layers, when looped at r=8, preserve generation accuracy?"
The block sweep is then guided by that map; the r-sweep narrows in.

### D13. Architectural metadata is per-cell, not a sweep dimension.

**Decision:** Each cell records `attention_type` and `is_kv_consumer`
for every involved layer, but we don't *sweep* on these. We use them
*after the run* to ask "did all the loopable layers cluster in the
KV-consumer region?" — a correlational analysis, not a predictive one.

**Why:** Round 2c's correlation analysis treated PLE importance and
KV role as predictors of looping tolerance and got reasonable r²
values on perplexity. Those values don't transfer to IT reasoning;
we don't yet know what does. Don't pre-bet on the dimensions.

### D14. CPU laptop is the reference for the harness; pods only execute it.

**Decision:** The full extractor + prompt + scoring path runs on a
CPU laptop against fixtures. Pod jobs only add the model and the
loop hook on top.

**Why:** Round 4's harness bugs were extractor + prompt bugs that
required GPU spinup to debug. With Phase 0's structure, every
non-model bug is found in <30 s on a laptop.

---

# Future directions

These are the post-Phase-5 plans we're committing to *prior* to running
Phase 1, so the bucket-to-plan mapping in Phase 5 already has names to
hand off to.

### F1. Retrofit training (Phase 6 candidate, lands if Phase 5 = bucket A or B)

If any pretrained-only config preserves reasoning within 3 pp of
`baseline-C2`, Phase 6 is full retrofit-training as in McLeish et al.
2025:

- Same block geometry as Phase 4's winner.
- LoRA-only update, frozen base weights, applied to the looping block.
- Training data: open-source instruction-tuning mix + a small math
  CoT slice. ~24 GPU-hours on a single H100, per the paper's recipe.
- Eval: same 4 metrics, same 3 benchmarks, plus on-device latency
  (next bullet).
- Pre-commit to a "no improvement = ship the pretrained config" gate.
  If 24 GPU-hours of LoRA produces less than +2 pp on GSM8K, the
  retrofit thesis is uneconomical at this model scale.

### F2. Healing-training stage (Phase 6 alt, lands if bucket C)

If degradation is in the -2 to -10 pp range, the healing-training
recipe from the paper applies:

- Step 1: a few hundred steps of straight LM healing on the post-
  hook activations, before any reasoning data.
- Step 2: standard retrofit-training on top.
- Two ablations worth running: healing length (100 / 500 / 2000
  steps) and healing data mix (raw text only vs raw + CoT).

### F3. Anti-loop decoding (Phase 6 alt, lands if bucket E)

If loop-rate collapse is the dominant pathology even at r=2:

- Implement r-annealing during decoding: r(t) = r_max during the
  prompt + first ~50 tokens, then linearly decay to 1 over the next
  100 tokens, then 1 thereafter.
  Hypothesis: recurrence helps build the answer prefix, then sticks
  the model in a fixed point during natural-language continuation.
  Annealing un-sticks it.
- Implement a runtime loop detector that fires once at any
  generation step and halts: same regex as Phase 0's `has_repetition_loop`,
  applied to a sliding window over the last 200 tokens. Aborts the
  sample, marks `aborted_loop=True`. Cheap and quantifies how much
  of the truncation we're seeing is rep-loop vs other.

### F4. FFN-only / attention-only loop pivots (Phase 6 alt, lands if bucket D)

If all whole-block recurrent configs collapse, the next narrower
mechanisms to test:

- **FFN-only loop:** loop just the MLP sub-block of layers L_start..L_end.
  Skips the self-attention recompute and its KV interaction. Faster
  to run, fewer moving parts, isolates "is recurrence tolerable
  if we don't redo attention?"
- **Attention-only loop:** the inverse. Loop just the attention
  sub-block; FFN runs once per outer pass. Tests whether the
  recurrent dynamic that helps reasoning lives in attention or
  feed-forward.
- Both reuse the existing block-loop hook but with module-level
  rather than layer-level granularity. Two new probes; ~1 day
  each to implement.

### F5. Cross-model generalisation

Once Phase 6 lands a working retrofit on E2B, replicate on:

- **Gemma-4-E4B** (the 4B variant, 47 layers) — does the valley
  position scale linearly with depth?
- **Phi-3-mini** (38 layers) — does the technique transfer outside
  the Gemma family?

These are independent plans (each ~2 days) but they're the obvious
externalisation. Don't run them until Phase 6 has produced a working
recipe; replicating a non-result wastes compute.

### F6. On-device measurement (Phase 7 candidate)

The path-2 thesis is a phone-class win: more compute per token at
no parameter cost. Phase 7 measures the *actual* phone-class metric,
not desktop-GPU perplexity:

- Run Phase 6's winning config through Path 1 plan 8's pipeline
  (Snapdragon-class measurement on a 4090 with throughput cap).
- Headline numbers: tokens/sec at r=1 vs r=4; energy per answered
  GSM8K problem; memory peak.
- Pre-commit to a 2-pp / 2× decoding-speedup tradeoff curve;
  anywhere on the curve is publishable, anywhere off it is not.

### F7. Cross-method comparison with Path 1's CoT-token approach

Path 1 ended with a working CoT-token recipe that lifts E2B-it from
a baseline to 71.6 % on GSM8K with no parameter changes. Path 2's
retrofit thesis only beats that if it produces an *additive* gain
on top of (or independent of) Path 1's prompt-side gains. Phase 7
or 8 should test:

- Does the retrofit-trained recurrent E2B exceed 71.6 % on GSM8K
  with the C2 prompt? (the only meaningful comparison)
- Does the retrofit-trained recurrent E2B exceed Path 1 numbers
  on ARC-Challenge and BBH-lite?
- If Path 1 + retrofit are *not* additive, retrofit's value is
  inference-compute rebalancing, not capability lift — and that
  argument needs F6's numbers to land.

### F8. Failed-direction documentation

Whichever bucket Phase 5 lands in, write a 1-page "what we tried
and why it failed" doc covering:

- The original perplexity-as-primary-metric mistake (fully
  characterised here, future readers should not repeat it).
- The 8-shot CoT prompt mistake (Path 1 already documented).
- Whatever Phase 1–5 *itself* fails to find, with concrete numbers.

This is the artifact that prevents Path 2 v3 from re-running these
loops.

---

## Open questions resolved before Phase 1

| Question | Decision (and why) |
|---|---|
| Use base or IT for Phase 2's per-layer map? | **IT.** That's the deployment target; Phase 5c proved base->IT doesn't transfer. |
| Sample size for the per-layer map? | **N=50 GSM8K per layer × 35 layers = 1,750 generations.** ~3.5 h on a 4090. |
| Include 8-shot CoT control on every benchmark? | **GSM8K only.** ARC-C's prompt-format finding (62 pp gap) is already published in Path 1; no need to re-measure. |
| Re-derive smart_v2? | **No.** Port verbatim. Re-derivation drift would invalidate the Path 1 anchor. |
| Compute chat-wrapped ppl on IT for cross-check? | **As a per-cell secondary number, not a headline.** It's free during the same forward pass; it adds a sanity dimension. But it doesn't gate anything. |
| Add a Wikitext perplexity sanity in Phase 1? | **Yes — base only, single number, hook-on vs hook-off at r=1.** Confirms the hook hasn't regressed bitwise in code refactoring. ~30 s of compute. |
| Where does retrofit training compute come from? | **Out of scope until Phase 5 lands in bucket A/B.** Don't pre-allocate. |
| Should we keep round 5 partials? | **Yes — as the cross-round bridge target for `baseline-8shot-control`.** Don't delete. |

---

## Exit criteria for Phase 0

- All Phase 0 acceptance criteria pass.
- Plan doc for Phase 1 (`phase1_hook_sanity.md`) is drafted, referencing this
  document for design decisions; not yet executed.
- A 1-page README at `plans/path_2_depth_recurrence_v2/README.md` lists
  all 5 phases with their gate metrics so a new collaborator can see the
  whole route at a glance.

Phase 1 spins up a pod only after these three artefacts exist on the branch.
