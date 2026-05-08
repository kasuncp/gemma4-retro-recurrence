"""Prompt builders shared across path 2 v2 phases.

Two prompt families:

  - C2 (zero-shot, chat-templated) is the primary prompt for every
    headline cell. Path 1 plan 5 / 8 measured 71.6% on N=500 GSM8K
    with this on Gemma-4-E2B-it; 8-shot CoT measured 30.0% on the
    same 500 problems (McNemar p ~= 5e-43). C2 is a strict win.

  - 8-shot CoT (Wei et al. 2022 exemplars) is used ONLY as a
    cross-round control to reproduce Path 2 round 5's 54.8% baseline.
    Anywhere else it's a known-bad prompt on this model.

Always go through ``tokenizer.apply_chat_template`` --- feeding
the prompt as raw text drops accuracy ~30 pp on the IT model
because the user/assistant turn markers are missing. Path 1 plan 5
spent its first hour debugging this exact bug.

Generation contract (Phase 1+):
  max_new_tokens=512, do_sample=False, temperature=None, top_p=None,
  use_cache=False, pad_token_id=tokenizer.eos_token_id.
"""

from typing import Sequence, Tuple


# ---------------------------------------------------------------------------
# C2 zero-shot prompts (primary)
# ---------------------------------------------------------------------------

def build_c2_gsm8k(tokenizer, question: str) -> str:
    """Chat-templated zero-shot for GSM8K. No CoT suffix.

    Path 1 anchor at N=500, smart_v2 extractor: 71.6% (95% CI
    [0.674, 0.755]).
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def build_c2_arc(tokenizer, question: str, choices: Sequence[str]) -> str:
    """Chat-templated zero-shot for ARC-Challenge.

    Question + lettered MC options. Always 4 choices in ARC. The
    "Answer with a single letter" instruction nudges the IT model
    away from open-ended re-justification of its choice; Path 1
    plan 7 tested this against bare options and saw a small but
    real lift.
    """
    if not choices:
        raise ValueError("ARC question has no choices")
    options = "\n".join(
        f"({chr(65 + i)}) {c}" for i, c in enumerate(choices)
    )
    user = f"{question}\n\n{options}\n\nAnswer with a single letter."
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


def build_c2_bbh(tokenizer, question: str, task: str | None = None) -> str:
    """Chat-templated zero-shot for BBH-lite.

    Path 1 plan 7 found that on BBH the *raw-text* prompt actually
    beats C2 (49.8% vs 20.6% on N=500). We still use C2 here so all
    three benchmarks share a single decoding configuration: BBH
    serves as the canary that catches "is recurrence damaging
    *general* reasoning, or specifically chat-format reasoning?".
    If a recurrent config recovers ground on BBH but not GSM8K
    under C2, the failure mode is distribution-specific.

    The ``task`` argument is unused in C2 mode but kept in the
    signature for parity with ``build_c2_arc``.
    """
    del task  # kept for signature parity
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# 8-shot CoT control (cross-round bridge ONLY)
# ---------------------------------------------------------------------------

# Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in
# Large Language Models", Appendix G. Same 8 exemplars Path 1 plan 1 +
# Path 2 round 5 used. The "#### N" suffix is the published Gemma
# evaluation protocol's answer marker; Path 1 verified it's the only
# deviation from Wei verbatim.
WEI_8SHOT_EXEMPLARS: Tuple[Tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. How "
        "many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted. So there must have been 21 - 15 = 6.\n#### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, "
        "how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how "
        "many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8.\n#### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and "
        "dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9",
    ),
    (
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. How many "
        "computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 = 29.\n#### 29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the "
        "end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, "
        "he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 "
        "golf balls.\n#### 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much "
        "money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be "
        "5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 "
        "is 8.\n#### 8",
    ),
)
EXEMPLAR_SET_ID = "wei-et-al-2022-hashformat"  # recorded in result JSON


def build_8shot_cot_gsm8k(tokenizer, question: str) -> str:
    """Wei et al. 8-shot CoT, chat-templated.

    Use as the ``baseline-8shot-control`` cell ONLY --- the bridge
    to Path 2 round 5's 54.8% headline. Reproducing 54.8 +/- 2 pp on
    N=500 confirms harness equivalence with round 5.
    """
    pre = "\n\n".join(
        f"Q: {q}\nA: {a}" for (q, a) in WEI_8SHOT_EXEMPLARS
    )
    user = f"{pre}\n\nQ: {question}\nA:"
    messages = [{"role": "user", "content": user}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# Prompt registry (used by eval_v3 / path2_v2_eval)
# ---------------------------------------------------------------------------

PROMPT_BUILDERS = {
    ("gsm8k", "C2"): lambda tok, row: build_c2_gsm8k(tok, row["question"]),
    ("gsm8k", "8shot-CoT"): lambda tok, row: build_8shot_cot_gsm8k(tok, row["question"]),
    ("arc-c", "C2"): lambda tok, row: build_c2_arc(tok, row["question"], row["choices"]),
    ("bbh-lite", "C2"): lambda tok, row: build_c2_bbh(
        tok, row["question"], row.get("task"),
    ),
}


def build_prompt(tokenizer, *, benchmark: str, prompt_name: str, row: dict) -> str:
    """Dispatch helper --- looks up (benchmark, prompt_name) in
    PROMPT_BUILDERS and calls the bound builder. Raises if the pair
    isn't registered (so a typo in a config name fails immediately,
    not silently)."""
    key = (benchmark, prompt_name)
    if key not in PROMPT_BUILDERS:
        raise ValueError(
            f"no prompt builder for ({benchmark!r}, {prompt_name!r}); "
            f"registered: {sorted(PROMPT_BUILDERS)}"
        )
    return PROMPT_BUILDERS[key](tokenizer, row)
