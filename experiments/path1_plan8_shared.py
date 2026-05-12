"""Path 1 plan 8 shared primitives - torch/transformers-free.

The legacy experiments/path1_plan8.py runs on RunPod with torch+transformers,
so importing it requires those heavy deps. The on-device Mac script
(experiments/path1_phone_mac.py) only needs the pure-python pieces:
exemplars, GSM8K loader, regex, answer extraction, and the chat-template
prompt builders (which work with any HF tokenizer, including the one
mlx-lm returns).

This module factors those out so the Mac script can import them without
pulling torch into its venv. The legacy script is unchanged - it now
re-imports these constants from here, preserving its public symbols
unchanged.
"""

from __future__ import annotations

import re

MODEL_ID = "google/gemma-4-E2B-it"

# Eight Wei et al. CoT exemplars. The original wording matches the legacy
# experiments/path1_plan8.py byte-for-byte so byte-equivalence sanity
# check 5 has teeth.
EXEMPLARS_COT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6"),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5"),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39"),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 = 29.\n#### 29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8"),
]
EXEMPLARS_DIRECT = [
    (q, "#### " + a.split("####")[-1].strip())
    for q, a in EXEMPLARS_COT
]

GOLD_RE = re.compile(r"####\s*(-?\d+)")
FALLBACK_RE = re.compile(r"(-?\d+)")


def load_problems(start, end, n_total):
    """Deterministic head of GSM8K test split. Lazy-imports `datasets` so
    callers that don't need GSM8K (e.g. byte-equivalence-only tools) don't
    pay the import cost."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    golds = []
    for i, row in enumerate(ds):
        if i >= n_total:
            break
        if start <= i < end:
            q = row["question"]
            a = row["answer"]
            m = GOLD_RE.search(a)
            gold = int(m.group(1)) if m else None
            problems.append({"question": q, "idx": i})
            golds.append(gold)
    return problems, golds


def build_c2_prompt(tokenizer, question):
    """C2: zero-shot plain prompt - deployment target cell.

    enable_thinking=False is required for Gemma 4's chat template, which
    otherwise injects a <|think|> system token and routes the model into a
    `<|channel>thought ...` extended reasoning mode that easily exhausts
    max_new_tokens=512 before emitting the final user-facing answer (this
    was the root cause of the plan-8 sanity gate 1 false failure).
    "Plain" in the cell definition implies no thinking channel.
    """
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def build_a3_prompt(tokenizer, question):
    """A3: 8-shot Wei et al. CoT prompt.

    enable_thinking=False: the CoT exemplars in the prompt are the chain
    of thought we want the model to imitate; stacking the chat template's
    own thinking channel on top is redundant and would compete for
    max_new_tokens budget.
    """
    messages = []
    for q, a in EXEMPLARS_COT:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_answer(text):
    m = GOLD_RE.search(text)
    if m:
        return int(m.group(1)), True
    nums = FALLBACK_RE.findall(text)
    if nums:
        return int(nums[-1]), False
    return None, False
