# Outline — Blog Post 0: Four Paths, One Phone

**Slug:** `four-paths-one-phone`
**Source:** `outputs/gemma-recursive-retrofit-blog-plan.md` §2, §4, §11 (drop-in opening); plus `docs/00_paper_summary.md`, `docs/10_synthesis_and_open_questions.md` for Leg 1 teasers.

## Proposed title
The On-Device Reasoning Problem: Four Paths, One Phone

## Audience & voice
Technical blog readers (ML practitioners). Measurement-first, skeptical. First post in an 8-post series — its job is to frame the question and lock in commitments, not to report results.

## Section plan
1. **Abstract / TL;DR** — one paragraph, restates thesis and what will/won't be measured.
2. **Problem statement** — why "reason harder" on a phone is a real question; what "phone-class" means operationally.
3. **Related work** (short) — Universal Transformers, Looped Transformers, MoD, Huginn, and the specific paper motivating Leg 1 (McLeish et al. 2025, arXiv:2511.07384).
4. **The four paths** — one subsection per leg: CoT token scaling, depth-recurrence retrofit, quantized sibling, MoD. Each ends with the trade-off it pays. Includes a Mermaid "landscape" diagram and a Mermaid diagram for depth-recurrence (from paper) and MoD routing.
5. **How we'll measure** — the shared rig (ARC-Easy, GSM8K, VRAM, tokens/sec, FLOPs/token). Mermaid pipeline diagram. Explicit "not measuring on-device phone yet" caveat.
6. **Why we start with Leg 1 (depth-recurrence)** — because its findings constrain the other legs' evaluation. Tease three findings from Leg 1 so far (valley at L15–19, KV wall at L14→L15, perplexity↛reasoning inversion) **qualitatively only** — no numbers in Post 0.
7. **What this series is NOT** — not a replication of any single paper; not an on-device deployment study yet.
8. **Limitations** — benchmark ceiling (ARC), phone-class is hand-wavy, E4B access risk, series-fatigue commitment risk.
9. **Roadmap** — table of the 8 posts.
10. **Conclusion** — restates commitment.
11. **Sources** — URLs.

## Key claims and their support
| Claim | Source | Type |
|---|---|---|
| Large models reason better by being larger; phones can't run them | General ML knowledge; framing | Framing, no numeric |
| Four paths to scale test-time compute | Blog plan §2, §11 | Framing |
| McLeish et al. retrofits recurrence by looping middle layers | `docs/00_paper_summary.md`; arXiv:2511.07384 | Paper-backed |
| Universal Transformers: Dehghani et al. 2018 | arXiv:1807.03819 | Paper-backed |
| Mixture-of-Depths: Raposo et al. 2024 | arXiv:2404.02258 | Paper-backed |
| Looped Transformers: Giannou et al. 2023 | arXiv:2301.13196 | Paper-backed |
| Leg 1 found a "valley" at L15–19, a KV wall at L14→L15, and a perplexity↛reasoning inversion | `docs/10_synthesis_and_open_questions.md` §4,§5,§6 | Internal finding (qualitative only in Post 0) |
| Shared measurement rig uses ARC-Easy / GSM8K / tokens-per-sec / VRAM / analytical FLOPs | Blog plan §6 E2 | Planning doc |

## Figures planned (all Mermaid — no fabricated benchmark charts)
- **Figure 1:** Landscape diagram — the four paths branching off baseline E2B, each labeled with its primary trade-off. Mermaid flowchart. Source: blog plan §2,§11.
- **Figure 2:** Depth-recurrence architecture (Prelude / Recurrent / Coda with the loop edge). Mermaid. Source: `docs/00_paper_summary.md` "method in one picture"; arXiv:2511.07384.
- **Figure 3:** MoD token-level routing. Mermaid. Source: arXiv:2404.02258.
- **Figure 4:** Shared measurement-rig pipeline. Mermaid. Source: blog plan §6 E2.

**No pi-charts / quantitative figures in Post 0** — this is a framing post with zero new measurements. Any number would be either decorative or a teaser with no provenance. Forbidden by system prompt.

## Verification log (what to double-check before delivery)
- [ ] arXiv IDs: 2511.07384, 1807.03819, 2404.02258, 2301.13196 — verify exact and live (local docs confirm 2511.07384 and 2404.02258; 1807.03819 and 2301.13196 cross-check via existing blog plan §Sources).
- [ ] No fabricated accuracy / latency / VRAM numbers anywhere in Post 0.
- [ ] Leg 1 findings presented qualitatively, with forward-reference to later posts that will carry the numbers.
- [ ] McLeish et al. described as motivating Leg 1 only, not as the series thesis.
- [ ] "Phone-class" defined operationally (6 GB / 4–8 W) per blog plan §10 R5 — flagged as a mapping, not a phone measurement.
- [ ] 8-post roadmap matches blog plan §4 table exactly.
- [ ] Sources section includes direct URLs.

## Known uncertainties / things to label tentative
- Leg 1 findings are internal observations not yet external-peer-reviewed → phrase as "we observed" with pointer to full analysis in Post 2–3.
- MoD is explicitly descoped (may become a future post); must not promise Leg 4 in Post 0's roadmap.
- E4B access not yet confirmed on HF Hub (blog plan §10 R3) → mention as a series risk in Limitations.
