"""Path 2 v2 benchmark loaders with deterministic shuffle.

Three loaders, identical contract:
    load_<bench>(n: int, seed: int = 42) -> list[dict]
Each row dict has at least ``idx`` (str|int), ``question`` (str),
``gold`` (any). ARC adds ``choices`` and ``gold_letter``; BBH adds
``task``.

Why deterministic shuffle (Phase 0 design decision D10):
  First-N in dataset order is non-iid. GSM8K's first 50 over-
  represent particular templates; ARC-C's first 200 over-represent
  particular Bloom's-taxonomy categories. Shuffling with a fixed seed
  makes "N=50 ⊂ N=200 ⊂ N=500" so smaller subsets are statistically
  valid prefixes of larger ones --- a sanity gate at N=50 actually
  predicts the headline at N=200/500.

Network access required at first call; after that the HF cache
under HF_HOME serves subsequent runs offline.
"""

import random
from typing import List, Dict


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def load_gsm8k(n: int, seed: int = 42) -> List[Dict]:
    """Load the first ``n`` GSM8K test problems after a seed=``seed`` shuffle.

    Path 1 anchor: same dataset/split, identical seed, lifted from
    experiments/path1_zero_shot.py. Round-trip determinism is part
    of the contract.
    """
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs[:n]:
        out.append({
            "idx": int(i),
            "question": ds[i]["question"],
            "gold": ds[i]["answer"],
        })
    return out


# ---------------------------------------------------------------------------
# ARC-Challenge
# ---------------------------------------------------------------------------

def load_arc_challenge(n: int, seed: int = 42) -> List[Dict]:
    """ARC-Challenge test split, deterministic shuffle.

    HuggingFace stores ``answerKey`` as one of {"A".."D", "1".."5"};
    older snapshots use the numeric form. We always emit a
    ``gold_letter`` ∈ {"A","B","C","D"} for the eval harness.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    out = []
    for i in idxs[:n]:
        row = ds[i]
        choices = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        gold = row["answerKey"]
        # Map labels (which can be A-D or 1-4) to a canonical letter.
        if gold in labels:
            gold_letter = chr(65 + labels.index(gold))
        else:
            gold_letter = None
        out.append({
            "idx": int(i),
            "question": row["question"],
            "choices": choices,
            "gold": gold,
            "gold_letter": gold_letter,
        })
    return out


# ---------------------------------------------------------------------------
# BBH-lite
# ---------------------------------------------------------------------------

# Five tasks chosen to span BBH's question shapes:
#   - object_counting        --- numeric answer
#   - boolean_expressions    --- True/False
#   - navigate               --- Yes/No
#   - web_of_lies            --- Yes/No (multi-step)
#   - formal_fallacies       --- valid/invalid
# Path 1 plan 7 used a 26-task superset; the 5-task lite version
# is enough to discriminate prompt shape × recurrence interactions
# without ballooning the eval cost.
BBH_LITE_TASKS = (
    "object_counting",
    "boolean_expressions",
    "navigate",
    "web_of_lies",
    "formal_fallacies",
)


def load_bbh_lite(
    n: int,
    seed: int = 42,
    tasks=BBH_LITE_TASKS,
) -> List[Dict]:
    """Sample roughly uniformly across ``tasks``.

    Path 1 plan 7 found ``dawidmt/bigbenchhard`` is the most
    reliably-cached HF mirror of BBH's per-task splits. We use the
    "train" split because that's the only one published per-task on
    that mirror; BBH's official paper uses these as its test set in
    practice.
    """
    from datasets import load_dataset
    rng = random.Random(seed)
    per_task = max(1, n // len(tasks))
    out = []
    for t in tasks:
        ds = load_dataset("dawidmt/bigbenchhard", t, split="train")
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        for i in idxs[:per_task]:
            out.append({
                "idx": f"{t}/{i}",
                "task": t,
                "question": ds[i]["input"],
                "gold": ds[i]["target"],
            })
    return out[:n]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

LOADERS = {
    "gsm8k": load_gsm8k,
    "arc-c": load_arc_challenge,
    "bbh-lite": load_bbh_lite,
}


def load_benchmark(name: str, n: int, seed: int = 42) -> List[Dict]:
    if name not in LOADERS:
        raise ValueError(
            f"unknown benchmark {name!r}; registered: {sorted(LOADERS)}"
        )
    return LOADERS[name](n=n, seed=seed)
