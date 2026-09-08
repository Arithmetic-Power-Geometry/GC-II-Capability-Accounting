from __future__ import annotations
from itertools import combinations
from typing import Dict, FrozenSet, Hashable, Mapping

Key = FrozenSet[Hashable]


def powerset(items):
    items = tuple(items)
    for r in range(len(items) + 1):
        for c in combinations(items, r):
            yield frozenset(c)


def mobius_decomposition(values: Mapping[Key, float], channels) -> Dict[Key, float]:
    """Exact Möbius coefficients δ_J = sum_{K⊆J} (-1)^(|J|-|K|) v(K)."""
    out: Dict[Key, float] = {}
    for J in powerset(channels):
        total = 0.0
        for K in powerset(J):
            total += ((-1.0) ** (len(J) - len(K))) * float(values[K])
        out[J] = total
    return out


def reconstruct_from_mobius(coeffs: Mapping[Key, float], S: Key) -> float:
    return sum(float(v) for J, v in coeffs.items() if J.issubset(S))


def additive_prediction(values: Mapping[Key, float], channels) -> float:
    base = float(values[frozenset()])
    return base + sum(float(values[frozenset([c])]) - base for c in channels)


def interaction_order_mass(coeffs: Mapping[Key, float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for J, value in coeffs.items():
        out[len(J)] = out.get(len(J), 0.0) + abs(float(value))
    return out
