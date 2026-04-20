"""Pure-Python statistics helpers used by analysis passes.

Kept minimal and dependency-free so correlation code runs in-process
without pulling in numpy/scipy.
"""


def _pearson(xs, ys):
    """Pearson correlation coefficient. NaN if insufficient variance."""
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx2 = sum((x - mx) ** 2 for x in xs)
    sy2 = sum((y - my) ** 2 for y in ys)
    denom = (sx2 * sy2) ** 0.5
    if denom == 0.0:
        return float("nan")
    return sxy / denom


def _average_ranks(values):
    """Return average ranks (1-indexed, mid-ranks for ties), same order as input."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # mid-rank, 1-indexed
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs, ys):
    """Spearman rank correlation, tie-safe."""
    return _pearson(_average_ranks(xs), _average_ranks(ys))
