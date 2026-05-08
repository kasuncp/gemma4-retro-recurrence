"""Compare per-problem completions across two cells (Phase 1, Cell 1C).

Given two JSONL paths produced by ``path2_v2_eval.py``, verify that
for every shared ``idx`` (up to ``--limit``) the ``completion`` field
is byte-equal.

This is THE structural correctness gate for "r=1 hook is a no-op".
Phase 0 confirmed the hook is correct in code and at the perplexity
level (rounds 2c, 3b, 5c). What this tool catches is generation-time
drift that perplexity doesn't surface --- a hook side-effect that
only manifests after several decode steps, e.g.:

  - The hook mutates a captured-args dict between generation calls
    (stale ``position_ids`` reused).
  - ``model.generate`` advancing cache state inconsistently when the
    hook re-enters layers.
  - bf16 numerical non-associativity differing depending on whether
    the layer is wrapped vs. naked.

CPU only --- no torch, no model load.

Exit codes:
  0  every shared idx within --limit byte-equal
  1  one or more mismatches (first 3 dumped)
  2  no shared idxs (likely a path or cell-name typo)

Usage:
    python experiments/path2_v2_token_match.py \\
        --left  results/path_2_depth_recurrence_v2/phase1/gsm8k__baseline-C2.jsonl \\
        --right results/path_2_depth_recurrence_v2/phase1/gsm8k__W5-r1.jsonl \\
        --limit 20
"""

import argparse
import json
import sys
from pathlib import Path


def _load_by_idx(path: Path) -> dict:
    if not path.is_file():
        print(f"FAIL: file not found: {path}")
        sys.exit(2)
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["idx"]] = row
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--left", required=True, type=Path,
                   help="JSONL for the reference cell (e.g. baseline-C2).")
    p.add_argument("--right", required=True, type=Path,
                   help="JSONL for the cell under test (e.g. W5-r1).")
    p.add_argument("--limit", type=int, default=20,
                   help="Compare only the first N shared idxs (sorted).")
    p.add_argument("--quiet", action="store_true",
                   help="On match, print only the one-line summary.")
    return p.parse_args()


def compare(left: Path, right: Path, limit: int) -> tuple[int, dict]:
    L = _load_by_idx(left)
    R = _load_by_idx(right)
    shared = sorted(set(L) & set(R))[:limit]
    if not shared:
        return 2, {"reason": "no shared idxs",
                   "left_count": len(L), "right_count": len(R)}
    mismatches = []
    for i in shared:
        if L[i].get("completion") != R[i].get("completion"):
            mismatches.append(i)
    if not mismatches:
        return 0, {"shared": len(shared), "matches": len(shared)}
    return 1, {"shared": len(shared), "mismatches": mismatches}


def main():
    args = parse_args()
    code, info = compare(args.left, args.right, args.limit)

    if code == 0:
        if args.quiet:
            print(f"MATCH {info['matches']}/{info['shared']}")
        else:
            print(f"MATCH {info['matches']}/{info['shared']}: "
                  f"r=1 is bitwise no-op.")
        return 0

    if code == 2:
        print(f"FAIL: no shared idxs between cells "
              f"(left has {info['left_count']}, right has {info['right_count']}).")
        return 2

    # code == 1
    print(f"FAIL: {len(info['mismatches'])}/{info['shared']} idxs differ.")
    L = _load_by_idx(args.left)
    R = _load_by_idx(args.right)
    for i in info["mismatches"][:3]:
        l_comp = (L[i].get("completion") or "")[:200]
        r_comp = (R[i].get("completion") or "")[:200]
        print(f"  idx={i}")
        print(f"    left:  {l_comp!r}")
        print(f"    right: {r_comp!r}")
    if len(info["mismatches"]) > 3:
        print(f"  ... and {len(info['mismatches']) - 3} more.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
