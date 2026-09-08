from __future__ import annotations
from itertools import combinations
import random


def parity_truth(n: int):
    return tuple((sum((x >> i) & 1 for i in range(n)) & 1) for x in range(2**n))


def balanced_random_truth(n: int, seed: int):
    rng = random.Random(seed)
    idx = list(range(2**n))
    rng.shuffle(idx)
    ones = set(idx[:2**(n-1)])
    return tuple(1 if i in ones else 0 for i in range(2**n))


def projection_patterns(truth, n: int, coords):
    coords = tuple(coords)
    pats = set()
    for x,b in enumerate(truth):
        if not b:
            continue
        pats.add(tuple((x >> i) & 1 for i in coords))
    return pats


def full_projections_up_to(truth, n: int, k: int):
    for r in range(1, k+1):
        target = 2**r
        for coords in combinations(range(n), r):
            if len(projection_patterns(truth,n,coords)) != target:
                return False
    return True


def find_balanced_projection_complete(n: int, k: int, seed0: int=20260908, tries: int=10000):
    for t in range(tries):
        truth = balanced_random_truth(n, seed0+t)
        if full_projections_up_to(truth,n,k):
            return truth, seed0+t
    raise RuntimeError("no projection-complete balanced truth table found")


def robdd_node_count(truth, n: int):
    """Exact reduced ordered BDD nonterminal node count for order x_0,...,x_(n-1)."""
    unique = {}
    def restrict(level, mask, value):
        vals=[b for x,b in enumerate(truth) if (x & mask) == value]
        if all(v==vals[0] for v in vals):
            return ("T", vals[0])
        if level>=n:
            return ("T", vals[0])
        bit=1<<level
        lo=restrict(level+1, mask|bit, value)
        hi=restrict(level+1, mask|bit, value|bit)
        if lo==hi:
            return lo
        key=(level,lo,hi)
        if key not in unique:
            unique[key]=len(unique)+1
        return ("N", unique[key])
    restrict(0,0,0)
    return len(unique)
