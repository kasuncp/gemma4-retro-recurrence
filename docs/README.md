# Retrofitting Recurrence onto Gemma 4 E2B — Findings

**Project:** Apply the method from McLeish et al., *"Teaching Pretrained
Language Models to Think Deeper with Retrofitted Recurrence"* (arXiv:2511.07384,
Nov 2025) to Google's Gemma 4 E2B.

**Status as of the most recent result:** Five rounds of pretrained-only probes
complete. No training runs have been executed. Round 5 is planned but not
implemented.

**Top-line finding:** Gemma 4 E2B has a narrow "valley" of loopable layers
(15–19) anchored at the KV producer→consumer boundary, and a wider stable
block (15–24) exists for perplexity. But when moved to downstream reasoning
(ARC-Easy), the wider blocks drop accuracy from 83.5% → ~25% while the narrow
5-layer block retains 40%. The relationship between perplexity stability and
reasoning capability under recurrence is *inverse* in our data, not aligned.
This is the finding Round 5 follows up on.

---

## Documents

The findings are split into per-round documents so each result can be read
standalone. All numbers come directly from the result JSONs in the project
directory; no values were invented. Where multiple re-runs exist, the
post-fix version is used and earlier broken versions are flagged.

| File | Covers |
|------|--------|
| [00_paper_summary.md](00_paper_summary.md) | What the paper proposes and why E2B is a non-trivial target for it |
| [01_round1_sanity_check.md](01_round1_sanity_check.md) | 30-minute probe: loop one middle layer, measure perplexity |
| [02_round2a_ple_policies.md](02_round2a_ple_policies.md) | vanilla / scaled / once PLE injection; includes the hook bug + fix |
| [03_round2b_ple_and_location.md](03_round2b_ple_and_location.md) | Full 35-layer PLE importance scan + 3-point layer-location sweep |
| [04_round2c_full_loop_map.md](04_round2c_full_loop_map.md) | 210-cell loop-tolerance map across all 35 layers × {r=2,4,8} × {vanilla, once} |
| [05_round3a_pair_looping.md](05_round3a_pair_looping.md) | Looping pairs of contiguous layers; inconclusive with suspected hook bug |
| [06_round3b_blocks.md](06_round3b_blocks.md) | Six candidate blocks; the KV boundary is a hard wall |
| [07_round3c_extended_blocks.md](07_round3c_extended_blocks.md) | Three more blocks pinning down the stable region (15–24) |
| [08_round4_reasoning_eval.md](08_round4_reasoning_eval.md) | First downstream eval (GSM8K + ARC-Easy); the finding that changed the story |
| [09_round5_plan.md](09_round5_plan.md) | Planned (not executed): harness fix + narrow-block width sweep + PLE ablation |
| [10_synthesis_and_open_questions.md](10_synthesis_and_open_questions.md) | What 5 rounds actually taught us; what's still load-bearing going forward |

Figures are in [`figs/`](figs/) and are referenced from the individual
documents.

---

## Quick chronology

| Round | Purpose | Headline result |
|-------|---------|-----------------|
| 1 | Does naive looping break the model instantly? | No — log-linear degradation: layer 17 r=8 → 3.4× baseline ppl |
| 2a | Does PLE injection policy matter? | Once-PLE beats vanilla at layer 17 (14% better at r=8). Scaled ≡ vanilla. First run had a hook bug; fixed in addendum v2. |
| 2b | Per-layer PLE importance + 3-point location sweep | PLE signal is spiky (L3/L8/L13/L33 carry most); once-vs-vanilla is layer-dependent |
| 2c | Full 35-layer tolerance map | Valley at 15–19. KV boundary at 14/15. Once-vs-vanilla has no systematic pattern. No single factor correlates strongly with tolerance. |
| 3a | Does pair-looping rescue fragile layers? | No — pair-looping is mostly worse than singles. Flagged as suspected hook bug. |
| 3b | Do candidate blocks work? | Yes for valley-anchored blocks (A, D); no for boundary-crossing blocks (C, E, F) |
| 3c | How wide can a stable block get? | Up to 10 layers (block G, 15–24). Must anchor at L15; shifted-late block I (20–27) catastrophic. |
| 4 | Does recurrence preserve reasoning? | Wider valley blocks collapse ARC-Easy to ~25%; narrow A-block holds 40%. GSM8K baseline is a harness bug (4.8%). |
| 5 (planned) | Fix harness; sweep narrow widths; PLE ablation | Not yet run |

---

## File provenance

All numbers cited in these documents were extracted directly from:

- `/mnt/project/results.json` and `results_round1_fixed.json` (Round 1)
- `/mnt/project/results_round2a*.json` (5 files — Round 2a, a buggy run, and
  two versions of a diagnostic addendum)
- `/mnt/project/results_round2b_importance.json` +
  `results_round2b_location.json`
- `/mnt/project/results_round2c_full_map.json`
- `/mnt/project/results_round3a_pair_looping.json`
- `/mnt/project/results_round3b_blocks.json` +
  `results_round3c_extended_blocks.json`
- `/mnt/project/results_round4_reasoning.json` (3.9 MB, full per-problem
  records)

Plans come from `/mnt/project/plan1.md` through `plan5.md` including the two
plan2a addenda.

The paper is in `/mnt/project/2511_07384v1.pdf`.
