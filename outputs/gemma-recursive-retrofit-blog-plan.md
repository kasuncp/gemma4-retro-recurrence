# Blog Series Plan — Option C: On-Device Test-Time Compute

**Date:** 2026-04-22 (revised same day — consolidated from 14 posts to 8)
**Status:** Planning document — not yet executed
**Supersedes:** direction framing in `outputs/gemma-recursive-retrofit-review.md` §8 sequencing

---

## 1. Series Title and Positioning

**Primary title:** *Scaling Test-Time Compute on a Phone: Can Small Models Think Harder?*

**Subtitle variants (pick one; I recommend #1):**
1. *An empirical investigation of four paths on Gemma 4 E2B.*
2. *Four ways to trade compute for reasoning on-device — and which actually work.*
3. *A measurement-first comparison of recurrence, CoT, quantization, and routing.*

**Tagline (for post headers and social):**
> How much harder can a phone-class model think, and at what cost in watts, milliseconds, and accuracy?

---

## 2. Thesis Paragraph (drop-in ready for post 0)

> Large models reason better by being larger. Phones can't run large models. The open question is whether small models can reason *harder* — scaling the compute they spend per query instead of the parameters they hold in memory. Four candidate approaches exist: (1) generating more tokens of chain-of-thought on a baseline model, (2) retrofitting depth-recurrence to loop a subset of layers at inference, (3) running a larger sibling model under aggressive quantization, and (4) routing tokens through variable-depth paths via Mixture-of-Depths. Each trades memory, latency, power, and accuracy differently. This series benchmarks all four on Google's Gemma 4 E2B and measures them head-to-head on a phone-class budget. We start with the least obvious and most architecturally invasive path — depth-recurrence — because that's the one that told us the most about what the model is actually doing, and because its failure modes constrain how we should evaluate the others.

---

## 3. Why Option C Changes the Rules

The review's three "must-fix before publishing" items (framing gap, structural departures, fig08 caption) all stemmed from a mismatch between what the docs claimed to test and what they actually tested. Option C resolves these structurally:

| Review issue | Resolution under Option C |
|---|---|
| "You're not testing the paper's method" | Correct — you're testing one of four test-time compute scaling paths. The paper's method is a stretch goal within that path, not the whole series' thesis. |
| "No linear adapter / no layer drop / no training" | Correct — Leg 1 is deliberately pretrained-only because training-free intervention is itself a design criterion for phone deployment (no on-device finetuning budget for most users). |
| "GSM8K figure is misleading" | Still must be fixed, but no longer load-bearing. Under Option C, Leg 1's reasoning numbers are diagnostic signals, not final measurements. The final comparison post has its own cleanly-measured benchmark. |

**Additional wins under Option C:**
- The "pivot" from Round 4 (perplexity ≠ reasoning) becomes a *design constraint* for the whole series — you now know to evaluate every leg on downstream reasoning, not just perplexity.
- The KV-boundary and valley findings become *evidence about Gemma 4 E2B's architecture* that informs how quantization (Leg 3) and MoD (Leg 4) should be implemented, not orphaned observations.

---

## 4. Full Post Outline (consolidated — 8 posts)

Consolidated from an earlier 14-post draft after concluding that several findings were being padded into standalone posts when they belonged together. Merge rationale in §4.1.

**Bold = exists as draft in current `docs/`.** *Italic = requires new experiments.*

| # | Title | Status | Primary source |
|---|---|---|---|
| 0 | **The On-Device Reasoning Problem: Four Paths, One Phone** | New | This plan doc §2, §11 |
| 1 | **Leg 1a: Probing Gemma 4 E2B with Loops** — sanity check + PLE importance + the hook bug | Draft exists | `docs/01_round1_sanity_check.md` + `docs/02_round2a_ple_policies.md` + `docs/03_round2b_ple_and_location.md` (merged) |
| 2 | **Leg 1b: The Valley and the Wall** ⭐ — 210-cell tolerance map + KV boundary + block extension | Draft exists | `docs/04_round2c_full_loop_map.md` + `docs/06_round3b_blocks.md` + `docs/07_round3c_extended_blocks.md` (merged; `docs/05_round3a_pair_looping.md` demoted to appendix) |
| 3 | **Leg 1c: The Inversion — When Perplexity Stops Predicting Reasoning** | Draft exists (fix `fig08` first) | `docs/08_round4_reasoning_eval.md` |
| 4 | *Leg 1d: Closing the Probe — Width Sweep, PLE Ablation, and the Adapter Dry-Run* | **Requires Round 5 + training dry-run** | `docs/09_round5_plan.md` + E6 in §6 |
| 5 | *Leg 2: More Tokens, Same Model — CoT Scaling on Baseline E2B* | **Requires new experiments** (E3 in §6) | New |
| 6 | *Leg 3: The Quantized Sibling — Gemma 4 E4B at 4-Bit* | **Requires new experiments** (E4 in §6) | New |
| 7 | *Head-to-Head + What We Learned* — matched-budget comparison **and** synthesis combined | **Requires all legs** (E5 in §6) | The payoff post |

**Total: 8 posts.** At one post every ~2 weeks, this is a ~4-month series. At one per week, ~2 months.

### 4.1 Merge rationale

- **Post 1 (Leg 1a)** merges Rounds 1, 2a, 2b. Round 1 alone is a 30-minute sanity check with one thin finding (log-linear degradation, no catastrophic collapse). It reads as prelude, not a post. The hook-drift regression discipline it establishes is what catches the Round 2a PLE bug later, so the narrative thread justifies the merge.
- **Post 2 (Leg 1b, hero)** merges Rounds 2c, 3b, 3c. All three answer the same underlying question — *which regions of the model tolerate looping?* — at escalating granularity. The valley finding (2c) motivates block probing (3b), which surfaces the KV wall and the extension rule (3c). One coherent architectural map with `fig04` and `fig07` as co-heroes.
- **Post 4 (Leg 1d)** merges Round 5 with the training dry-run (E6 in §6). Both are follow-ups to the inversion (Post 3). Round 5 asks whether the inversion survives better measurement; the dry-run asks whether adding the paper's missing components (linear adapter, prelude injection) changes it. One post, one narrative, one decision point for whether to commit to full training.
- **Post 7 (Head-to-Head + Synthesis)** merges the comparison post and the synthesis post. In practice readers want to read them together — the comparison table *is* the synthesis. Separating them creates an artificial wait between evidence and interpretation.
- **Measurement rig folded into Post 0** as a "how we'll measure" section rather than a standalone post. It's short enough to fit inside framing.
- **MoD (Leg 4) dropped from the main series.** Mentioned as future work in Post 7. Always the descopable leg; see §8.

### 4.2 What stays standalone and why

- **Post 0** — the public commitment moment. Cannot share space with anything else.
- **Post 3 (Inversion)** — pivot finding. Merging dilutes the punch of "perplexity stability and reasoning stability are inversely related."
- **Posts 5 and 6 (Legs 2 and 3)** — different methods, different setup, different failure modes. Merging them hides the per-method reasoning.
- **Post 7 (Head-to-Head)** — the payoff. The reason Option C exists.

---

## 5. The "Commitment" Problem

Option C makes an implicit promise that Posts 5, 6, and 7 will exist and will contain real measurements. Three things can go wrong:

1. **You run out of time / interest before Legs 2 and 3 are done.** Mitigation: front-load the cheapest leg. Leg 2 (more CoT tokens) needs zero new code — run the harness with more budget. Get it drafted within 2 weeks of starting the series.
2. **Leg 3 (quantized E4B) destroys the recurrence story.** If E4B-Q4 beats retrofitted-E2B-recurrence at matched VRAM and latency by 10+ points on ARC, the series thesis shifts to "recurrence is interesting but not competitive — here's what is." Be willing to let this happen. The review's §7 direction critique warned this is plausible.
3. **The adapter dry-run (inside Post 4) is too ambitious to bundle.** If it slips, Post 4 becomes a Round-5-only post and the dry-run moves to a short follow-up between Posts 4 and 5.

**Recommendation:** Pre-commit publicly to Posts 0–3 (Leg 1 content you already have) and Posts 5 + 6 + 7 (Legs 2, 3, and the head-to-head). Treat MoD as a possible future post mentioned in Post 7's "what's next" section. This keeps your options open without undercutting the central comparison.

---

## 6. Experiments Required Before the Series Concludes

Everything needed to deliver on Option C's implicit promise. Ordered by dependency + cost.

### E1 — Round 5 (Leg 1f)
**What:** Execute the already-planned Round 5 per `docs/09_round5_plan.md`.
- Width sweep: blocks of width {2, 3, 4, 5} at r=8 on ARC-Easy.
- GSM8K harness fix: E2B-it + chat template + 8-shot CoT.
- PLE ablation: W5-r8-noPLE.
**Cost:** Already budgeted. Single GPU-day. ~$30 on RunPod.
**Blocks:** Posts 7, 12.
**Why first:** Already planned, already scoped, closes Leg 1.

### E2 — Measurement rig (Post 1)
**What:** Implement and document the benchmark harness that every leg will share:
- ARC-Easy (n=500, 0-shot and 5-shot) as primary accuracy metric.
- GSM8K (n=500, 8-shot CoT) as secondary — once the harness is fixed.
- Wall-clock time per output token on a reference GPU (pick one: A6000 bfloat16 or a quantization-sympathetic config).
- FLOP count per generated token (analytical, using model config).
- Peak VRAM measured with `torch.cuda.max_memory_allocated`.
- **Deliberately postpone on-device phone measurements to a stretch goal** — they add complexity without being necessary for a matched-GPU comparison.
**Cost:** Engineering time. No new experiments yet. 1–2 days of coding.
**Blocks:** Posts 9, 10, 11, 12 — and every leg from here on.
**Why second:** Without a shared harness, the four legs produce apples-to-oranges numbers and the head-to-head post collapses.

### E3 — Leg 2: CoT token scaling (Post 9)
**What:** Measure ARC-Easy and GSM8K on baseline Gemma 4 E2B-it across generation budgets ∈ {64, 128, 256, 512, 1024, 2048} tokens with 0-shot, 4-shot, and 8-shot prompting.
**Cost:** Small. Uses existing model, existing harness. ~$20 of compute for the full sweep.
**Blocks:** Post 9, post 12.
**Why third:** Cheapest, gives you an unambiguous baseline for everything else. Answers the question "can we just generate more tokens?" before spending on alternatives.

### E4 — Leg 3: Quantized E4B (Post 10)
**What:** Run Gemma 4 E4B under three quantization recipes on the same benchmarks:
- E4B bfloat16 (upper bound, only for reference — doesn't fit phone budget).
- E4B INT8 (bitsandbytes or AWQ).
- E4B INT4 (GPTQ or AWQ — the phone-relevant config).
Record: accuracy, tokens/sec, VRAM, FLOPs/token.
**Cost:** Medium. Quantization implementation is mature but you need to pick a stack (bitsandbytes fastest; AWQ / GPTQ more accurate). ~$50 compute + 2 days engineering.
**Blocks:** Post 10, post 12.
**Why fourth:** This is the scariest comparison — it has the best chance of invalidating the whole recurrence bet. Run it *before* the training dry-run (E6) so you know whether recurrence is worth training for at all.

### E5 — Head-to-head (Post 12)
**What:** Matched-budget comparison table. Pick 2–3 target budgets (e.g., 6GB VRAM / 200ms TTFT / 10W average power) and report accuracy for each leg's best configuration at each budget. Pareto frontier plot.
**Cost:** Data aggregation from E1, E3, E4 plus any gap-filling runs. ~$30.
**Blocks:** Post 12, post 13.
**Why fifth:** Requires E1+E3+E4 complete.

### E6 — Training dry-run (Leg 1g, Post 8) — conditionally required
**What:** The minimal LoRA training prototype from the review's §12. Width-3 block (L15–L17), linear adapter `nn.Linear(2h→h)` at block entry, prelude embedding injection each iteration, LoRA r=2, 1–2k steps on Wikitext or similar, Muon vs AdamW.
**Cost:** ~$20, <1 GPU-day.
**Blocks:** Post 8.
**Why conditional:** Only run if E4 (quantized E4B) doesn't obviously dominate. If E4B-Q4 wins on every metric, a training dry-run for Leg 1 becomes a sunk-cost investigation — publish post 8 as "we chose not to pursue training because here's what the comparison showed" instead.

### E7 — Leg 4: MoD (Post 11) — optional / descopable
**What:** Implement Mixture-of-Depths routing on E2B. This is a substantial engineering lift — the router has to be trained even if the rest of the model is frozen.
**Cost:** High. Days to weeks of engineering. ~$100+ compute.
**Blocks:** Post 11 only.
**Why optional:** If E1–E6 take 5 months, E7 becomes the difference between shipping in 6 months and 8 months. Good candidate to descope: see §8.

---

## 7. Reframing the Existing Docs

Current `docs/00_*` through `docs/10_*` were written for the old framing. Under Option C, they need surface-level reframing, not rewriting. Key changes:

### Universal changes (every post)
- **New above-the-fold callout (replaces the review's "pretrained-only" one):**
  > This post is part of the series *Scaling Test-Time Compute on a Phone*. It covers Leg 1 (depth-recurrence retrofit). For the full comparison against CoT scaling, quantized E4B, and MoD, see [head-to-head post link].
- **Drop** any framing that suggests this work is evaluating McLeish et al.'s method end-to-end. Keep McLeish as the specific paper motivating Leg 1; it is no longer the frame of the series.
- **Update the related work paragraph:** Universal Transformers, Looped Transformers, Mixture-of-Depths, and Huginn all fit naturally now because Leg 1 is positioned inside a broader compute-scaling landscape.

### Doc-by-doc reframing (under the 8-post structure)

| Doc | Reframe action | Destination |
|---|---|---|
| `00_paper_summary.md` | Shorten. "Four architectural frictions" section moves into Post 1's opening. Paper-vs-project gap becomes an inline callout in Post 1, not a full doc. | Absorbed into Post 1 |
| `01_round1_sanity_check.md` | Becomes the **opening section** of Post 1. ~400 words. The hook-drift regression discipline introduced here is presented as the methodology the rest of Leg 1 relies on. | Post 1 §1 |
| `02_round2a_ple_policies.md` | Becomes the **hook-bug narrative** middle section of Post 1. Keep the debugging story — it's the transparency strength. | Post 1 §2 |
| `03_round2b_ple_and_location.md` | Becomes the **closing section** of Post 1 (PLE importance scan + location sweep). 35-row importance table goes to an appendix; show only top-5 / bottom-5 in main text. | Post 1 §3 |
| `04_round2c_full_loop_map.md` | **Hero post of Leg 1.** Becomes the opening of Post 2 — the valley discovery. `fig04` is the lead image. | Post 2 §1 |
| `05_round3a_pair_looping.md` | **Appendix of Post 2.** Short, flagged as inconclusive. Retain for transparency; do not cite elsewhere. | Post 2 appendix |
| `06_round3b_blocks.md` | Becomes the **KV-wall middle section** of Post 2. The "blocks can't cross L14→L15" finding. | Post 2 §2 |
| `07_round3c_extended_blocks.md` | Becomes the **block extension closing section** of Post 2 (blocks G/H/I, anchor-at-L15 rule). | Post 2 §3 |
| `08_round4_reasoning_eval.md` | Becomes Post 3 (the Inversion) largely as-is. **Must fix `fig08`** (regenerate with GSM8K bars hatched/grayed/labeled as harness artifact directly in the image). Add the "this is why every subsequent leg is evaluated on reasoning, not perplexity" bridge at the end. | Post 3 |
| `09_round5_plan.md` | Becomes half of Post 4 once Round 5 runs. The other half is the adapter dry-run (E6). | Post 4 §1 |
| `10_synthesis_and_open_questions.md` | **Largest change.** Repurposed as the skeleton for Post 7. "Open questions" become "questions we answered in Legs 2 and 3." Leg 1 synthesis becomes a single section inside Post 7, not a standalone doc. | Post 7 (skeleton) |

---

## 8. Descoping Plan

The 8-post structure is already the consolidated version. Two of the original cuts (MoD as standalone post; measurement rig as standalone post) are now baked in. Further descoping options, in priority order:

1. **Cut the adapter dry-run from Post 4** if Leg 3 (Post 6) shows E4B-Q4 obviously dominates. Post 4 becomes a Round-5-only post. Rationale: training for a method already outperformed by quantization is a sunk-cost bet.
2. **Merge Posts 5 and 6** into a single "Legs 2 and 3 together" post. Only if time is very tight. Costs clarity — the two methods have different failure modes and should be discussed separately.
3. **Do not cut Posts 5, 6, or 7.** These are the series' reason for existence under Option C. If you can't commit to them, you should have picked Option B instead.

**Hard floor: 6 posts.** Posts 0, 1, 2, 3 (Leg 1), Post 6 (Leg 3 — the most important comparison), Post 7 (head-to-head + synthesis). This still delivers on Option C's central question ("does recurrence beat quantization at matched budget?") even if Leg 2 and the dry-run get dropped.

---

## 9. What to Do This Week

1. **Run Round 5 (E1).** You already have the plan. Budget and compute are known. This unblocks post 7 and gives you clean Leg 1 numbers for the head-to-head.
2. **Draft the measurement rig (E2).** Even as a bare `eval_harness.py` that takes a model + config and spits out accuracy/tokens/sec/VRAM. Its existence will constrain the rest of your engineering.
3. **Draft post 0** using the thesis paragraph in §2 above. Publish it as a "series launch" — this locks you into the commitment publicly, which is a feature not a bug.
4. **Apply the review's three mandatory fixes** (framing callouts, structural-departure disclosures, fig08 caption) to the existing Leg 1 drafts. Under Option C these are still required; the framing is just different.

**Do not start Leg 2 or 3 experiments until the measurement rig (E2) is ready.** Running legs with ad-hoc harnesses produces numbers you can't legitimately compare in post 12.

---

## 10. Risks Specific to Option C

Flagged here so you can't say nobody warned you.

**R1 — Leg 3 eats the series.** E4B-Q4 is a strong baseline. If it dominates, the natural conclusion is "just use E4B-Q4." This is a valid scientific outcome but a weaker narrative. Contingency: if it happens, pivot post 13 to "why quantization wins on phones today, and what would have to change for the other approaches to compete." Honest and interesting, just not the story you started with.

**R2 — Series fatigue.** 8 posts over 2–4 months is tractable but still requires discipline. Series commonly stall around post 4–5, which in this structure is exactly Post 4 (closing Leg 1) and Post 5 (Leg 2) — the handoff from "writing up existing work" to "executing new experiments." Contingency: write Post 7 (head-to-head + synthesis) as a *short provisional draft* now with the data you have, updating as Legs 2 and 3 complete. Easier to update than to generate from scratch at the end.

**R3 — E4B licensing / availability.** Gemma 4 E4B may be under the same license as E2B (Gemma Terms of Use) but could have different access gating. Verify before announcing Leg 3 — a gated or unavailable E4B would be catastrophic for the series plan. **Action item:** Confirm E4B access on Hugging Face Hub and document the commit SHA before publishing post 0.

**R4 — The benchmarks you picked (ARC, GSM8K) aren't enough.** ARC-Easy tops out around 90% on small models; headroom compresses. GSM8K for small models is hit-or-miss and depends heavily on prompting. Contingency: add one harder benchmark (MMLU-Lite or a reasoning-focused subset like BBH-lite) at the measurement rig stage. Cheap insurance.

**R5 — "Phone-class" is hand-wavy.** The series title says "phone" but all your experiments run on a datacenter GPU. If someone rightly asks "did you measure this on an actual phone?" you need an answer. Options: (a) include a "phone compute budget" definition (e.g., 6GB RAM / 4–8W sustained) and map datacenter measurements to that budget via FLOPs; (b) actually run one recipe on a phone as a demo in post 13. Option (a) is safer and lower cost.

---

## 11. Concrete "Post 0" Opening

Drop-in ready. You'll want to edit for voice.

---

> # The On-Device Reasoning Problem: Four Paths, One Phone
>
> Large models reason better by being larger. Phones can't run large models. The question this series asks is whether small models can reason *harder* — scaling the compute they spend per query instead of the parameters they hold in memory.
>
> There are at least four ways to do this, and they don't have the same trade-offs:
>
> 1. **Generate more tokens.** Run the baseline model for longer — more chain-of-thought, more deliberation, more self-consistency samples. Zero architectural change. Pays in latency and energy per answer.
> 2. **Retrofit depth-recurrence.** Re-run a block of layers `r` times per forward pass so the model thinks "deeper" without growing wider. Requires architectural surgery and, per the method we're following (McLeish et al., 2024), training to recover from that surgery.
> 3. **Run a bigger sibling model, quantized.** Keep the architecture stock but run the 4B-parameter E4B at 4-bit quantization instead of the 2B E2B at bfloat16. Same VRAM; more parameters; different compute profile.
> 4. **Route tokens adaptively (Mixture-of-Depths).** Each token picks its own compute budget at inference — easy tokens skip layers, hard tokens get more. No recurrence; spatial rather than temporal compute scaling.
>
> All four can plausibly scale test-time compute on a phone. Only measurements will tell us which actually *does*, and at what cost in milliseconds, watts, and accuracy points.
>
> This series runs each of these legs on Google's Gemma 4 E2B and compares them head-to-head on a shared measurement rig. We begin with Leg 1 — depth-recurrence retrofit — because it was the most architecturally invasive and taught us the most about what the model is actually doing. Its findings (a loopable "valley" at layers 15–19, a hard wall at the shared-KV boundary, and a surprising inversion where perplexity stability anti-correlates with reasoning preservation) constrain how the other legs should be evaluated.
>
> **What this series is NOT:** a replication of any specific paper. The McLeish et al. method motivates Leg 1 but isn't the thesis. The goal is not to validate a single technique; it's to measure which technique wins on the actual constraint — a phone.
>
> **Series roadmap and progress tracker:** [link]

---

## 12. Bottom Line

- Option C is the right call if you commit to Posts 5, 6, and 7. It's the wrong call if you'll publish Leg 1 posts and stall.
- The three gating experiments are: **Round 5 (closes Leg 1, feeds Post 4), measurement rig (unblocks Posts 5 and 6), and Leg 3 quantized E4B (the comparison in Post 6 that matters most).**
- The review's three mandatory fixes still apply. Option C reframes them but doesn't remove them. The `fig08` regeneration (GSM8K bars hatched/grayed directly in the image) is the single most important pre-publication fix — it blocks Post 3.
- Publish Post 0 as a public commitment once you've verified E4B access on HF Hub and have the measurement rig scoped. Don't launch without those two safety checks.

### 12.1 Revision history

- **2026-04-22 (initial):** 14-post structure across 4 legs.
- **2026-04-22 (revised):** Consolidated to 8 posts. Merged Rounds 1/2a/2b into Post 1; Rounds 2c/3b/3c into Post 2; Round 5 + training dry-run into Post 4; head-to-head + synthesis into Post 7. Measurement rig folded into Post 0. MoD dropped from main series to "future work" mention in Post 7.

---

## Sources

- This plan is derivative of:
  - `outputs/gemma-recursive-retrofit-review.md` (peer review, 2026-04-22)
  - `gemma-recursive-retrofit-research.md` (evidence dossier, 2026-04-22)
- Primary paper for Leg 1: McLeish et al., "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence," arXiv:2511.07384 — https://arxiv.org/abs/2511.07384
- Related work to cite in post 0: Dehghani et al. (Universal Transformers, arXiv:1807.03819), Raposo et al. (Mixture-of-Depths, arXiv:2404.02258), Giannou et al. (Looped Transformers, arXiv:2301.13196), Geiping et al. (Huginn, test-time compute scaling).
