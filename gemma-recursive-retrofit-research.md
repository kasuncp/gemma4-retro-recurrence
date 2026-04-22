# Evidence Dossier: Gemma 4 E2B Retrofitted Recurrence

**Prepared for:** peer review of a blog-post series in progress  
**Scope:** evidence gathering only — reviewer writes the critique  
**Date of evidence collection:** 2026-04-22  
**CWD:** `/Users/kasun/Documents/projects/gemma-4-retrofitted-recurrence`

---

## 1. Project Summary

**What was built:** A series of probe-only experiments applying forward-pass hooks to pretrained `google/gemma-4-E2B` (35 decoder layers, bfloat16, base model — not `-it`) to characterise whether its layers can support the depth-recurrence loop from McLeish et al. 2511.07384. No weight updates of any kind have been performed. The hook mechanism replaces or wraps `Gemma4TextDecoderLayer.forward` at runtime to loop selected layer(s) `r` times, then restores the original forward. Results are perplexity on 50 Wikitext-2 sequences (max 512 tokens each) and zero-shot accuracy on GSM8K (n=250) and ARC-Easy (n=200) using greedy decode.

**Status:** Rounds 1 through 4 complete (confirmed by matching JSON files). Round 5 is written (`plans/plan5.md`, `probes/mode_round5.py`) but not executed — there is no `results_round5_*.json`. The top-line finding is that perplexity stability and reasoning stability are *inversely* related across block widths, which Round 5 is designed to follow up on.

**Key architectural constraints discovered (Gemma 4 E2B-specific):**
- Per-Layer Embeddings (PLE): each layer has its own embedding table passed positionally by the outer forward.
- Hybrid attention: every 5th layer is full/global; others are sliding-window (512-token local).
- Shared KV states: layers 15–34 are KV consumers; 0–14 are producers. The boundary at L14→L15 acts as a hard wall for block looping.
- The above properties are absent in all three of the paper's validation models (TinyLlama, OLMo-2-1B, Llama-3.2-1B).

---

## 2. Per-Round Summary Table

| Round | File(s) | Question | Method | Headline Result | Author Caveats |
|-------|---------|----------|--------|-----------------|----------------|
| **1** | `docs/01_round1_sanity_check.md` / `results/results_round1_fixed.json` | Does naive looping instantly break E2B? | Single layer 17, r∈{1,2,4,8}, vanilla PLE, 50×512-token Wikitext-2 sequences | Log-linear degradation: r=1→12.5 ppl, r=8→42.6 ppl (3.40× baseline). Hook drift 0.0. No NaN. | r=8 is slightly over the hoped-for 3× target. PLE policy not yet varied. Layer 17 not known to be representative. |
| **2a** | `docs/02_round2a_ple_policies.md` / `results/results_round2a_fixed.json` (authoritative); broken: `results_round2a.json`, `results_round2a_run2.json`, `results_round2a_addendum.json` | Which PLE injection policy tolerates looping best at layer 17? | 3 PLE modes (vanilla/scaled/once) × r∈{1,4,8} at layer 17. Had critical hook bug (PLE passed positionally, not as kwarg — see §4). Fixed in addendum v2. | `once` is 14% better than vanilla at r=8 (36.6 vs 42.6 ppl). `scaled` ≡ vanilla. | Fix invalidated first two runs. Old JSON files retained in repo. `once` advantage may not generalise (tested at one layer only). |
| **2b** | `docs/03_round2b_ple_and_location.md` / `results/results_round2b_importance.json` + `results_round2b_location.json` | Per-layer PLE importance + does `once>vanilla` generalise? | PLE importance: vanilla vs zero at r=1, all 35 layers (70 cells). Location sweep: layers {5,17,28} × {vanilla,once} × r∈{4,8} (18 cells). | PLE signal spiky (L3=+1.83, L33=+0.74, L8=+0.65 NLL diff). KV consumer zone (15–32) nearly PLE-inert. `once` advantage is layer-dependent: hurts 48% at L5, helps 14% at L17, hurts 8% at L28. | Three locations insufficient to generalise. PLE importance does not directly predict loopability. |
| **2c** | `docs/04_round2c_full_loop_map.md` / `results/results_round2c_full_map.json` | Full 35-layer tolerance map | 35 × {vanilla,once} × {r=2,4,8} = 210 cells. Per-layer r=1 regression (all 35 pass, max drift 0.0). Correlations computed for 4 factors. | Valley at L15–L19 (only 5 layers <110 ppl at r=8 vanilla). KV boundary L14→L15 is a sharp edge (L14=1873, L15=35). `once` has no systematic advantage — helps 16 layers, hurts 19. No single factor has |ρ|>0.40. | No causal explanation for valley. Pair-looping still untested. |
| **3a** | `docs/05_round3a_pair_looping.md` / `results/results_round3a_pair_looping.json` | Does looping adjacent layer pairs rescue fragile layers? | 34 pairs (L,L+1), r∈{2,4,8}, vanilla only. 102 cells. Regression at 3 pairs (all drift 0.0). | Pairs systematically *worse* than single layers in most of the model. Only pair (15,16) clearly beats both its singles (31.9 vs 35.3). Flagged as suspected hook bug. | Author explicitly treats data as uninformative for retrofit design. Bug not subsequently debugged. |
| **3b** | `docs/06_round3b_blocks.md` / `results/results_round3b_blocks.json` | Do candidate multi-layer blocks work? | 6 blocks × {r=2,4,8} = 18 cells, vanilla only. Regression at B and E (drift 0.0). | KV boundary is a hard wall (C 12-19: 209k ppl; E 13-22: 846k ppl even at r=2). D (15-22, w=8): 33.3 ppl. A (15-19, w=5): 30.1. F (25-32): fine at r=2, collapses by r=4 (slow drift). | Non-symmetry of extension (D works, C doesn't) labelled "unexpected" in JSON. Maximum viable width still unknown. |
| **3c** | `docs/07_round3c_extended_blocks.md` / `results/results_round3c_extended_blocks.json` | Maximum viable width? Anchor at L15 required? | 3 new blocks (G 15-24, H 15-23, I 20-27) + refs A/B/D. 18 cells. Regression at B and G (drift 0.0). | G (15-24, w=10): 36.0 ppl ✓. H (15-23, w=9): 36.7 ppl ✓. I (20-27, w=8): 970 ppl ✗ — catastrophic even at r=2. Anchor at L15 is load-bearing. | L25+ not tested inside valley-anchored block. Mechanistic reason for L15 anchor not established. |
| **4** | `docs/08_round4_reasoning_eval.md` / `results/results_round4_reasoning.json` (3.9 MB) | Does recurrence preserve reasoning zero-shot? | 7 configs × GSM8K (n=250) + ARC-Easy (n=200), greedy decode, `use_cache=False`, `max_gen_tokens=256`. D-r1 bitwise sanity check included. | ARC-Easy baseline 83.5%, A-r8 40%, D/G blocks ~24–26%. GSM8K baseline 4.8% = harness bug (172/238 failures are textbook continuations). D-r1 matches baseline 250/250 on GSM8K. | GSM8K column is explicitly flagged as uninformative. Perplexity-stability ≠ reasoning-stability. Round 5 planned to fix. |
| **5** | `docs/09_round5_plan.md` / `probes/mode_round5.py` | Fix GSM8K; sweep narrow block widths; PLE ablation | Planned: 9 configs, E2B-it + chat template + 8-shot CoT for GSM8K, ARC-Easy (base). W5-r1 and W5-r8 cross-round sanity checks defined. | **Not executed. No results file.** | N/A |

---

## 3. Claim→Data Provenance Check

All numbers verified directly from JSON files via Python. No discrepancies found between the docs and the underlying data.

| Claim (from docs) | File | JSON path / query | Cited value | Found value | Match? |
|-------------------|------|-------------------|-------------|-------------|--------|
| Layer 17, r=8 → "3.4× baseline ppl" | `results/results_round1_fixed.json` | `summary[3].ratio` | 3.40× | **3.397×** (ppl=42.589) | ✅ Rounds correctly |
| Baseline ppl (r=1) | `results/results_round1_fixed.json` | `unmodified.ppl` | 12.537 | **12.5366** | ✅ |
| Hook drift r=1 | `results/results_round1_fixed.json` | `hook_drift` | 0.0 | **0.0** | ✅ |
| Once-PLE "14% better at r=8" on L17 | `results/results_round2a_fixed.json` | cells where ple_mode=once/vanilla, r=8 | 14% | **vanilla=42.589, once=36.627 → 0.860× = 14.0% improvement** | ✅ |
| PLE location = "positional" | `results/results_round2a_addendum_v2.json` | `ple_location` | positional | **"positional"** | ✅ |
| Patch status after fix = WORKING | `results/results_round2a_addendum_v2.json` | `patch_status` | WORKING | **"WORKING"** | ✅ |
| Zero-PLE r=1 NLL diff ~0.4% | `results/results_round2a_addendum_v2.json` | ppl_zero=12.587 vs ppl_vanilla=12.537 | ~0.4% | **(12.587-12.537)/12.537 = 0.40% ppl difference** | ✅ |
| Round 2c valley at L15-L19 | `results/results_round2c_full_map.json` | cells layer∈{14,15,16,17,18,19,20}, vanilla, r=8 | L15≈35, L16≈39, L17≈43; L14≈1873; L20≈417 | **L14=1872.7, L15=35.3, L16=38.8, L17=42.6, L18=108.1, L19=103.7, L20=417.0** | ✅ |
| KV boundary at L14/L15 | `results/results_round2c_full_map.json` | layer_metadata[14].is_kv_consumer | L14=False, L15=True | **Confirmed in layer_metadata** | ✅ |
| Regression: all 35 layers r=1 pass | `results/results_round2c_full_map.json` | `regression_checks.all_r1_match_baseline`, `max_rel_drift` | all pass, drift 0.0 | **True, 0.0** | ✅ |
| Correlations (ple_importance Spearman +0.392) | `results/results_round2c_full_map.json` | `analysis.correlations_r8_vanilla.ple_importance.spearman` | +0.392 | **+0.3919** | ✅ |
| Round 3b block C (12-19) ppl at r=8 | `results/results_round3b_blocks.json` | block_cells name=C, r=8 | 209,050 | **209,050.3** | ✅ |
| Round 3b block D (15-22) ppl at r=8 | `results/results_round3b_blocks.json` | block_cells name=D, r=8 | 33.3 | **33.287** | ✅ |
| Round 3b block F (25-32) ppl r=2/r=4 | `results/results_round3b_blocks.json` | block_cells name=F | r=2: 37.1, r=4: 13,850 | **r=2: 37.08, r=4: 13,849.6** | ✅ |
| Round 3c block G (15-24) ppl at r=8 | `results/results_round3c_extended_blocks.json` | block_cells name=G, r=8 | 36.0 | **36.011** | ✅ |
| Round 3c block I (20-27) catastrophic | `results/results_round3c_extended_blocks.json` | block_cells name=I, r=8 | 970 | **970.256** | ✅ |
| Round 4 ARC-Easy baseline 83.5% | `results/results_round4_reasoning.json` | `arc_easy.summary.rows[config=baseline].accuracy` | 83.5% | **0.8350 (167/200)** | ✅ |
| Round 4 A-r8 ARC-Easy 40% | `results/results_round4_reasoning.json` | `arc_easy.summary.rows[config=A-r8].accuracy` | 40.0% | **0.4000 (80/200)** | ✅ |
| Round 4 D-r4 ARC-Easy ~25% | `results/results_round4_reasoning.json` | `arc_easy.summary.rows[config=D-r4].accuracy` | ~25% | **0.2450 (49/200)** | ✅ |
| Round 4 G-r8 ARC-Easy ~25% | `results/results_round4_reasoning.json` | `arc_easy.summary.rows[config=G-r8].accuracy` | ~24.5% | **0.2450 (49/200)** | ✅ |
| Round 4 GSM8K baseline 4.8% | `results/results_round4_reasoning.json` | `gsm8k.summary.rows[config=baseline].accuracy` | 4.8% | **0.0480 (12/250)** | ✅ |
| D-r1 bitwise matches baseline (GSM8K) | `results/results_round4_reasoning.json` | `sanity_checks.gsm8k_d_r1_vs_baseline` | 250/250 | **{matches: 250, total: 250}** | ✅ |
| Round 3c regression drift 0.0 | `results/results_round3c_extended_blocks.json` | `regression_checks.max_rel_drift` | 0.0 | **0.0** (blocks B and G verified) | ✅ |

**Summary:** All 23 load-bearing headline numbers verified exactly against JSON. No fabricated or misquoted numbers found.

**Minor rounding note:** The doc claims r=8 → "3.4×" and the actual ratio is 3.397×. This is correct rounding, not an error.

---

## 4. Code Correctness Notes

### 4.1 `probes/hooks.py` — Core hook mechanism

**`make_looped_forward` (Round 1, lines ~12-20)**
Simple and correct. Wraps `orig_forward`, calls it `r` times, threads `hidden_states` between calls. Returns final `out` in original tuple format. No PLE manipulation — PLE is applied naturally by each `orig_forward` call (vanilla semantics). ✅

**`make_looped_forward_ple` (Round 2a+, lines ~24-99)**
Post-addendum-2 fix handles both positional and kwarg PLE:
- Checks `ple_kwarg in kwargs` first; falls back to `args[0]` (positional) if not found. Correct for Gemma 4 E2B's `per_layer_input` being the 2nd positional parameter.
- Scale logic (vanilla=1.0, scaled=1/r, once=1.0 at i=0 else 0.0, zero=0.0) is correct.
- At `scale == 1.0`, args/kwargs are passed through unchanged → preserves bitwise identity at r=1. ✅
- `first_call` flag logs detected `ple_location` once. ✅
- **Red flag:** If `args[0]` is genuinely `None` (PLE not provided for this layer), `original_ple is None` → scale branch skipped → correct. But this path is never exercised in the valley blocks where PLE is inert (L15-L24 PLE values are near-zero but not None).

**`install_pair_loop_hooks` (Round 3a, lines ~101-170)**
Uses PyTorch `register_forward_pre_hook(with_kwargs=True)` on layers L and L+1, and `register_forward_hook` on L+1. The post-hook fires after the outer model's first natural (L, L+1) application, then runs r-1 additional iterations using `module.forward(...)` (bypasses `__call__` to avoid retriggering hooks). Hidden states are correctly threaded via `x`.

**Suspected bug source (supporting the 3a interpretation):** `captured_L1["args"]` is populated by the pre-hook on L+1 during the outer model's first pass. At that moment, the outer model has already called L (producing updated hidden states) and is now calling L+1. So `captured_L1["args"][0]` = correct hidden states entering L+1 on iter 1, and `captured_L1["args"][1]` = L+1's per_layer_input PLE. On iterations 2+, we use `(x,) + captured_L1["args"][1:]` — this correctly replaces hidden_states with the updated `x` and keeps L+1's PLE. This looks *correct* in principle.

**Alternative bug hypothesis:** The pre-hook on L+1 may fire during the first outer pass *before* the pair-loop hook on L has run, or the captured args may include stale references to tensors modified in-place by L's forward. The docs don't conclusively identify the bug; the data is "flagged as likely contaminated" but the specific failure mechanism isn't diagnosed in the code or logs. **The reviewer should note this is unresolved.**

**`install_block_loop_hooks` (Rounds 3b/3c/4, lines ~173-280)**
Generalises the pair hook to arbitrary block width. Pre-hook on every layer in [L_start..L_end]; post-hook on L_end. `_replay_args_for` handles `ple_strategy` ("every-iter" vs "iter1-only"). `new_args = (x,) + a_i[1:]` correctly threads updated hidden states. ✅

**Critical semantic issue not addressed in code:**
The paper's recurrent block `R` takes a *concatenation* of `e` (prelude output) and `s_{i-1}` (previous block output) through a linear adapter (2h→h). This project's block-loop hook replays `(x, original_PLE, ...)` where x is the previous block output, but there is **no linear adapter and no injection of the prelude embedding** into each iteration. This is a fundamental departure from the paper's architecture. The code is internally consistent (correct for what it claims to test: pretrained-only probing with looped pretrained weights), but it is not implementing the paper's recurrent block. The docs acknowledge this but a blog reader needs to understand the delta.

### 4.2 PLE handling correctness

- The fix in `make_looped_forward_ple` is correct and verified by the zero-PLE diagnostic (addendum v2: NLL diff = 0.004035 > 1e-6, `patch_status: WORKING`).
- All subsequent rounds use `install_block_loop_hooks` which captures PLE via the pre-hook from whatever the outer model passes. This is correct.
- PLE injection mode in block-loop hooks: all rounds 3b/3c/4 use `ple_strategy="every-iter"` (default), meaning each of the r iterations re-applies each layer's PLE. This corresponds to the "vanilla" mode from round 2. The round 2c finding that vanilla and once have no systematic pattern in the valley means this choice is defensible but not optimal.

### 4.3 `probes/mode_round2a.py` — PLE variants probe

- Correctly loads the fixed `make_looped_forward_ple` hook.
- Runs regression checks (checks 1-4: vanilla≈baseline, scaled==vanilla at r=1, once==vanilla at r=1, zero≠vanilla at r=1) before main sweep. ✅
- Regression check 4 is the inline zero-PLE diagnostic (replaces the separate addendum 1 approach). ✅
- Aborts with `sys.exit(3)` if any check fails. ✅

### 4.4 `probes/mode_round2c.py` — Full looping map

- Correctly iterates all 35 layers, restores `orig_forward` in `finally` block. ✅
- r=1 regression uses same-run unmodified ppl (not the round-1 constant), which is stricter and more meaningful. ✅
- Imports PLE importance from `results_round2b_importance.json` — this creates a dependency on prior results. If the importance file is missing, it warns but continues. ✅

### 4.5 `probes/mode_round3a.py` — Pair-looping

- Loads round-2c single-layer results for comparison. Falls back gracefully if missing. ✅
- Regression at 3 representative pairs before the sweep. ✅
- Calls `_interpret_pair_map` which correctly identifies Bucket 4 ("WORSE — investigate") when pair rate < single rate - 0.05. The JSON `interpretation_hint` matches. ✅

### 4.6 `probes/mode_round4.py` — Reasoning eval harness

**GSM8K format:** `"Question: {question}\nAnswer:"` with greedy decode, `max_new_tokens=256` (at time of round 4 run — the code's current default is 512 after the plan5 change). Zero-shot, no CoT. This matches the identified harness bug: the base model produces textbook continuations rather than solving the asked question. ✅ (bug correctly identified)

**ARC-Easy format:** Multiple-choice, zero-shot, parse first valid label character. `_parse_mc_answer` iterates characters; handles alphabetic (A/B/C/D) and numeric (1/2/3/4) labels since `valid_labels` comes from `problem["choices"]["label"]`. ✅

**ARC-Easy is zero-shot** (confirmed in code). This is lower than typical published few-shot ARC-Easy numbers, so the 83.5% baseline is actually reasonable for zero-shot.

**D-r1 sanity check:** Uses `_d_r1_bitwise_match` which compares generated strings (not just accuracy) — strictly correct test for hook no-op at r=1. Result: 250/250 match. ✅

**`use_cache=False` globally during generation:** Avoids subtle cache corruption from the loop hook re-entering layers. Correct. Performance cost acceptable for n=200-250 problems. ✅

**JSONL checkpoint/resume:** Append + fsync per problem. Manifest guards against incompatible reruns. Well-implemented for long-running evals. ✅

### 4.7 `ple_sanity_check.py` — Entry point

- Correctly routes 10 modes to their respective functions.
- `mode_round5.py` is imported but round 5 has not been run.
- `--max-gen-tokens` default is now 512 (raised from round 4's 256). This means if round 4 is re-run with the current script defaults, results would differ from the archived `results_round4_reasoning.json`. **Reproducibility concern.**

### 4.8 Sample sizes and seeds

| Experiment | n | Seed |
|------------|---|------|
| Perplexity probes (Rounds 1–3c) | 50 Wikitext-2 sequences, ≤512 tokens each | No explicit seed set. Dataset loading is deterministic by index. |
| GSM8K (Round 4) | 250 problems (first 250 of test split) | No sampling — greedy decode is deterministic |
| ARC-Easy (Round 4) | 200 problems (first 200 of test split) | No sampling — greedy decode is deterministic |

**No explicit `torch.manual_seed` or `random.seed` anywhere in the codebase.** For perplexity, this is fine (no randomness in evaluation). For generation with greedy decode, this is also fine. **However**, no seed means bfloat16 non-determinism across GPU models is unchecked. The round 2b run2 (`results_round2b_importance_run2.json`) confirms reproducibility of perplexity probes in practice (same values), but this is not formally verified for reasoning evals.

---

## 5. Paper Reference Verification

**Title confirmed:** "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence" — McLeish, Li, Kirchenbauer, Kalra, Bartoldson, Kailkhura, Schwarzschild, Geiping, Goldstein, Goldblum. arXiv:2511.07384v1. Published at NeurIPS 2025 ER Workshop (Spotlight). ✅

### What the paper actually prescribes

**Training IS required and is the paper's central contribution.** The abstract states: "we study how to convert existing pretrained non-recurrent language models into depth-recurrent models through *continued pretraining*." The paper's efficiency claims are all about training FLOP budgets. Pretrained-only recurrence is *not tested* in the paper.

**The specific recipe (from §3 "Experimental Setup"):**

1. **Model surgery — layers are DROPPED:** The paper takes a 22-layer TinyLlama and creates a (4,8,4) recurrent model by selecting layers [0-3] for prelude, [10-17] for recurrent block, [18-21] for coda. Layers 4-9 are **removed**. This project never drops any of Gemma 4 E2B's 35 layers.

2. **Linear adapter added:** The recurrent block R begins with a **linear adapter (2h→h)** that takes the concatenation of `s_{i-1}` (previous block output) and `e` (prelude embedding). This project has **no such adapter**.

3. **Initial state injection:** `s_0 ~ N(0, σ²)` random noise. This project has **no initial state** — each iteration just re-feeds the previous block output.

4. **Training with Muon optimizer + recurrence curriculum:** Mean of Poisson-Lognormal distribution over r is scheduled from low→high during training. This project does **no training**.

5. **Healing period:** After surgery, a healing phase on minimal-distribution-shift data recovers language modeling performance before task-specific training. The paper shows ARC-Challenge and HellaSwag are preserved after healing. **This project measures the pre-healing state only.**

6. **Block selection heuristic:** The paper finds (Appendix C.1) that "selecting early layers for the prelude and later layers for the recurrent block and coda performs best." No mention of KV consumers, PLE, or hybrid attention (their models have none of these features). The paper also compares against ShortGPT pruning for layer selection.

**The paper's baselines:**
- TinyLlama-1.1B-step-1431k-3T (22 layers, standard Llama-style pre-norm decoder)
- OLMo-2-0425-1B (16 layers, standard decoder with QK-norm)
- Llama-3.2-1B (16 layers, standard Llama decoder)
- **All are standard architectures without PLE, shared KV states, or hybrid local/global attention.**

### Gap between paper and project

| Paper prescribes | Project does |
|-----------------|--------------|
| Surgery: drop layers between prelude and recurrent block | Keep all 35 layers; loop a subset |
| Linear adapter (2h→h) at start of recurrent block | No adapter |
| Initial state injection (random noise) | No initial state; re-feeds directly |
| Continued pretraining with Muon optimizer | No training whatsoever |
| Recurrence curriculum (Poisson-Lognormal scheduler) | Fixed r values (probe-only) |
| Healing phase before task-specific training | No healing |
| Models without PLE/hybrid-attn/shared-KV | E2B has all three |

**Implication for the blog series:** The project is correctly framed in `docs/00_paper_summary.md` as a "pretrained-only probe" designed to inform *whether* to attempt training, not to replicate the paper's results. The ARC-Easy collapse (83.5% → 24-40%) is consistent with what the paper would predict for the pre-healing state (before recurrence-aware training). The blog posts need to be clear that they are not testing the paper's method; they are testing a necessary precondition for applying the paper's method to a non-standard architecture.

---

## 6. Related Work Gaps

The docs cite McLeish et al. as the primary reference and mention Geiping et al. (Huginn/latent reasoning) incidentally. The following clearly relevant works are not cited in any doc:

1. **Dehghani et al., "Universal Transformers," ICLR 2019** — The foundational work showing that Transformer-style models with shared weights and adaptive computation are Turing-complete. Directly relevant: the valley-anchored loop is a special case of a Universal Transformer with fixed r. The paper itself cites this (Dehghani et al., 2018). Blog posts should cite it for historical context.

2. **Giannou et al., "Looped Transformers as Programmable Computers," ICML 2023** — Shows that fixed-weight looped transformers can implement arbitrary algorithms if given appropriate input formatting. Directly relevant to the theoretical question of what pretrained-only looping can achieve. Cited in McLeish et al.

3. **Raposo et al., "Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models," 2024** (arXiv:2404.02258) — Routes tokens through or around transformer blocks adaptively. Alternative approach to test-time compute scaling. Cited in McLeish et al. The blog posts should distinguish MoD (routing per token) from depth-recurrence (looping the same block per position).

4. **Geiping et al., "Scaling LLM Test-Time Compute Globally and Locally via Thinking," 2025** — Trains a depth-recurrent transformer (Huginn-0125) from scratch on 800B tokens. The McLeish paper positions itself as more efficient than this approach. Understanding the prior work helps readers contextualise what "retrofitting" adds.

**Also potentially relevant (not verified against paper):**
- Bae et al. 2024 (cited in McLeish, converts pretrained transformers with LoRA and fixed r=2/3 — closer analogue to this project's setup)
- Li et al. 2025 (cited in McLeish, finetunes looped GPT-2/OPT on multiple-choice — small gains)

**Recommendation for blog posts:** At minimum, Universal Transformers (Dehghani 2019) and Giannou et al. (ICML 2023) should appear in the introduction to ground the approach theoretically. Mixture-of-Depths (Raposo 2024) should appear as the main architectural alternative being compared against.

---

## 7. Figure Spot-Check Notes

Three figures inspected:

### fig01_round1_layer17.png
- **Axes:** Y-axis = "Perplexity (Wikitext-2)", X-axis = "Loop iterations r". Both labeled. ✅
- **Scale:** Log-scale Y-axis (using `10^n` notation). Appropriate for the data range (12.5–42.6). ✅
- **Values:** Annotated directly on data points: 12.5, 13.5, 20.5, 42.6. Consistent with JSON (12.537, 13.454, 20.480, 42.589 — correct rounding). ✅
- **Error bars:** None. ❌ n=50 sequences is not annotated.
- **Sample size:** Not shown in figure or title.
- **Appearance:** Clean, not smoothed. Four discrete points only (r∈{1,2,4,8}). No visual overclaiming.

### fig08_round4_accuracy.png
- **Axes:** Y-axis = "Accuracy (%)", X-axis = config names. Both labeled. ✅
- **Scale:** Linear 0–85%. Appropriate. ✅
- **Values:** Annotated on bars (4.8, 1.2, 2.0, etc. for GSM8K; 83.5, 40.0, 24.5, 26.5 for ARC-Easy). All consistent with JSON. ✅
- **Error bars:** None. ❌
- **Sample size:** Not shown. n=250 GSM8K, n=200 ARC-Easy not annotated.
- **Concern:** The GSM8K 4.8% baseline bar (blue) appears as a 5% bar alongside an 83.5% ARC-Easy bar. A casual reader who misses the docs' explicit harness-bug disclosure could interpret GSM8K as a real model measurement. The figure title ("pretrained-only, no training") is honest about the setup but does not flag the GSM8K harness issue. The legend labels both series as "0-shot" without any warning annotation on the GSM8K series.
- **Legend:** Present, distinguishes GSM8K vs ARC-Easy. ✅

### fig04_round2c_loop_tolerance.png
- **Axes:** Y-axis = "Perplexity (log)", X-axis = "Layer index (looped)". Both labeled. ✅
- **Scale:** Log-scale Y-axis, spanning 10^-1 to 10^21 (accommodates L34's 2.9×10^20). ✅
- **Annotations:** Valley region (15-19) highlighted in yellow shading. KV boundary annotated with a vertical dashed line and label. ✅
- **Three series:** r=2 (green circles), r=4 (orange squares), r=8 (red triangles). Legend present. ✅
- **Error bars:** None. ❌
- **Baseline dashed line:** "unmodified ppl=12.54" marked as horizontal dashed line. ✅
- **Concern:** The r=2 line for layers 15-19 dips *below* the baseline dashed line visually (because log scale compresses the 13-20 ppl range). This is accurate data but could confuse a reader who hasn't looked at the raw numbers.
- **Overall:** This is the most informative figure in the project. Structural features (valley, KV boundary spike, late-layer catastrophe) are all visible and annotated.

**General figure issues across the project:**
- No error bars or confidence intervals anywhere (n=50 Wikitext sequences is modest for perplexity; n=200/250 for reasoning is enough for ±3-4% resolution at 83.5% baseline but narrower at 25% accuracy).
- No sample size annotations in any figure.
- No figure shows which run was used when multiple re-runs exist (e.g., round 2b has `importance_run2.json`).
- Values appear to be plotted directly from JSON without smoothing. ✅

---

## 8. Reproducibility Surface

### `run.sh`
- Self-contained shell script with tmux wrapper. Sources `.env` for `HF_TOKEN`. ✅
- Installs/upgrades `transformers datasets accelerate` with `pip install -U` — **no pinned versions**. ❌ A `transformers` update could silently break PLE positional-arg detection or change `Gemma4TextDecoderLayer.forward` signature.
- `.deps-installed` marker skips reinstall on subsequent runs unless `FORCE_INSTALL=1`. Minor risk: old deps stay installed across version bumps.
- Defaults to `--mode ple-variants` (round 2a), not `--mode original` (round 1). Backward compatibility maintained via argparse.
- Auto-commits results JSONs and pushes to origin after each run. Convenient but means partial/buggy results could be pushed mid-investigation.

### Dependency pinning
- **No `requirements.txt`, `setup.py`, or `pyproject.toml`** anywhere in the repo. ❌
- `.gitignore` only ignores `__pycache__/` and `.env` — no other exclusions.
- The PLE positional-arg fix in `hooks.py` is tightly coupled to the current `Gemma4TextDecoderLayer.forward` signature. The `probes/introspect.py` (inspects the live class) provides some resilience, but a version change could silently shift to Strategy B or C and the hook would break without a clear error.

### Seeds
- No `torch.manual_seed`, `random.seed`, or `numpy.seed` set anywhere. For greedy decode and deterministic perplexity, this is acceptable. Cross-GPU reproducibility untested but round 2b run2 confirms same-GPU reproducibility. ✅ (with caveat)

### Model weights
- `google/gemma-4-E2B` loaded via HuggingFace Hub with `HF_TOKEN`. License-gated (Gemma license must be accepted). Weights are not pinned to a specific commit hash. ❌ If Google updates the weights (e.g., a bug fix), the experiments would not be reproducible exactly.

### Result files
- Authoritative result files are committed to git: all `results/results_round*.json` files are staged (seen from run.sh `git add` logic).
- **Broken result files retained:** `results_round2a.json`, `results_round2a_run2.json`, and `results_round2a_addendum.json` (broken, pre-fix) coexist with `results_round2a_fixed.json` (authoritative). This is good for showing the debugging process but could confuse someone who finds the repo and uses the wrong file.
- `results_round4_reasoning.json` is 3.9 MB — large but committed, enabling full downstream analysis. ✅
- No `results_round5_*.json` exists. ✅ (round 5 not run)

### Compute environment
- Designed for RunPod (single GPU ≥16 GB VRAM). Hardware model not recorded in result JSONs (only `dtype: bf16` and `model_id`). ❌ bfloat16 results can differ between GPU generations.

---

## 9. Open Questions / Red Flags for the Reviewer

These are the issues most likely to require explicit treatment in the blog post series, roughly ordered by importance.

### 🔴 Critical

**RF-1: The project is not testing the paper's method.**
The paper's method requires training (Muon optimizer, recurrence curriculum, healing phase, linear adapter). The project runs zero training. The ARC-Easy collapse (83.5% → 25-40%) is the *expected* state before healing, not evidence that the method fails. The blog posts need to make this crystal clear, and the "pretrained-only probing" framing in the docs needs to be prominent in every post, not just the paper summary.

**RF-2: The paper removes layers; this project doesn't.**
When the paper creates a (4,8,4) recurrent model from 22-layer TinyLlama, it drops 6 layers. The project loops blocks within the original 35-layer stack with no layer removal. This means the "recurrent block" in this project is 35 layers wide total (with a 5-10 layer loop); the paper's recurrent block is 8 layers from a 16-layer effective model. The compute ratio is very different.

**RF-3: No linear adapter = no prelude embedding injection.**
Each iteration of the paper's recurrent block receives a fresh injection of the prelude embedding `e` via the linear adapter. This project's block re-runs itself with only its own previous output. This is a structurally different computation and may be why wider blocks collapse reasoning (they lack the anchoring signal from the prelude).

### 🟡 Significant

**RF-4: Round 3a hook bug is unresolved.**
The pair-looping data is flagged as "likely contaminated" but the bug was never diagnosed or fixed. The blog post on round 3a should acknowledge this explicitly — it's intellectually honest but means an entire round's data is essentially void.

**RF-5: GSM8K figure (fig08) shows a harness bug as a data point.**
The figure is visually misleading without reading the accompanying text. For a blog post audience, the figure caption (not the doc text) needs to flag that the GSM8K column is not a model measurement.

**RF-6: `once`-PLE advantage was never tested for reasoning.**
The round 2c conclusion was "drop `once` from future testing" based on perplexity data. But round 4 found that perplexity-stability ≠ reasoning-stability. It's possible `once` mode on reasoning evals could be different from vanilla. This is the round 5 PLE ablation, but phrased as a pivot rather than a correction.

**RF-7: No pinned dependencies — reproducibility is fragile.**
A `requirements.txt` with pinned `transformers` version is missing. Given that the entire hook mechanism depends on the exact forward signature of `Gemma4TextDecoderLayer`, a version bump could silently break every round.

### 🟢 Noteworthy

**RF-8: The KV boundary discovery is genuine and novel.**
The finding that L14→L15 (the KV producer→consumer boundary) acts as a hard wall for block looping is the project's most architecturally interesting finding. It's not predicted by the paper (which uses uniform architectures) and provides a concrete design constraint. The blog post should state this clearly as an original empirical finding, not as something expected from the paper.

**RF-9: Perplexity stability ≠ reasoning stability is the pivot finding.**
The inversion (narrower block → better reasoning despite similar or worse perplexity) is the finding that most changes the story. It's presented well in the docs but the blog posts should make the implication explicit: perplexity is not a valid proxy for reasoning capability under loop surgery.

**RF-10: The valley (L15–L19) co-incides with the KV consumer boundary but also with the first global-attention layer (L19) inside the consumer region.**
The docs note this but don't fully explore whether the global attention layer at L19 being *inside* vs *outside* the block matters. Block A (15-19) includes L19 (global); block B (15-18) excludes it. Both have similar ppl (~30). This suggests global attention inside the block doesn't hurt ppl — but the interaction with reasoning is untested.

**RF-11: E2B base vs -it is a key unresolved choice for Round 5.**
All probes use base `google/gemma-4-E2B`. Round 5's GSM8K fix requires switching to `-it` (or adding stop sequences to base). If `-it` is used for round 5, the prior perplexity probes on base model may not transfer, since the instruction-tuned weights are different. This transition needs to be flagged explicitly when blog post 5 is written.

**RF-12: The "improvement over worst single" metric in block analysis is presented but potentially misleading.**
Block D (15-22) "rescues" layer L22 (single ppl 976 → block ppl 33, improvement 29×). This is correct but the improvement metric compares apples to oranges: single-layer looping has 33 downstream layers processing the perturbed state; block looping has 13. The block "improvement" partly reflects within-block absorption and partly reflects reducing the downstream cascade. The docs implicitly acknowledge this in the "cascade hypothesis" framing but don't explicitly separate the two effects.

---

## 10. Appendix: File Inventory for the Reviewer

### Primary evidence files
| File | Status | Role |
|------|--------|------|
| `results/results_round1_fixed.json` | ✅ Authoritative | Round 1 results |
| `results/results.json` | ⚠️ Earlier run, near-identical | Superseded by _fixed |
| `results/results_round2a.json` | ❌ Hook bug | Broken — do not use for variant comparison |
| `results/results_round2a_run2.json` | ❌ Hook bug | Same bug, second run |
| `results/results_round2a_addendum.json` | ❌ Hook bug | Addendum 1 — also broken |
| `results/results_round2a_addendum_v2.json` | ✅ Fixed | Zero-PLE diagnostic after fix |
| `results/results_round2a_fixed.json` | ✅ Authoritative | 9-cell grid with working hook |
| `results/results_round2b_importance.json` | ✅ Authoritative | 35-layer PLE importance scan |
| `results/results_round2b_importance_run2.json` | ✅ Confirms reproducibility | Same values |
| `results/results_round2b_location.json` | ✅ Authoritative | Location sweep |
| `results/results_round2c_full_map.json` | ✅ Authoritative | 210-cell full map |
| `results/results_round3a_pair_looping.json` | ⚠️ Suspected hook bug | Treat as uninformative |
| `results/results_round3b_blocks.json` | ✅ Authoritative | 6-block results |
| `results/results_round3c_extended_blocks.json` | ✅ Authoritative | G/H/I block results |
| `results/results_round4_reasoning.json` | ✅ Authoritative (3.9 MB) | ARC-Easy valid; GSM8K harness-bugged |

### Code files checked
| File | Role | Issues found |
|------|------|------|
| `probes/hooks.py` | Core hook mechanism | No bugs found; pair hook suspected but undiagnosed |
| `probes/mode_round2a.py` | PLE variants + zero-diagnostic | Correct post-fix |
| `probes/mode_round2c.py` | Full looping map | Correct |
| `probes/mode_round3a.py` | Pair-looping | Correct code; data quality suspect |
| `probes/mode_round4.py` | Reasoning eval | GSM8K format causes harness bug; rest correct |
| `ple_sanity_check.py` | Entry point | `--max-gen-tokens` default changed post-round-4 (256→512) |
| `run.sh` | Automation | No pinned deps; auto-commits results |

### Missing files
| Expected | Status | Impact |
|----------|--------|--------|
| `requirements.txt` | ❌ Absent | No reproducibility pinning |
| `results/results_round5_*.json` | ❌ Absent | Round 5 not run |
| `docs/fig*/` (figs not in `figs/` subdirectory) | ⚠️ Figures are in `docs/` not `docs/figs/` as README states | Minor discrepancy (README says "Figures are in `figs/`") |

---

*Evidence gathered from direct file reads, Python JSON parsing, and web search. Paper content from arxiv.org/html/2511.07384v1. All JSON verifications run locally on the project directory. No model inference was run during this evidence collection.*
