from __future__ import annotations
from math import factorial
from typing import Mapping

from .mobius import powerset


def exact_shapley(values: Mapping[frozenset, float], channels):
    """Exact Shapley allocation of v(N)-v(empty) for a finite set function."""
    channels = tuple(channels)
    n = len(channels)
    out = {}
    for i in channels:
        others = [j for j in channels if j != i]
        total = 0.0
        for S in powerset(others):
            w = factorial(len(S)) * factorial(n - len(S) - 1) / factorial(n)
            total += w * (float(values[frozenset(set(S) | {i})]) - float(values[S]))
        out[i] = total
    return out


def walsh_anova(values: Mapping[frozenset, float], channels):
    """Orthogonal 2-level factorial/Walsh coefficients under uniform weighting."""
    channels = tuple(channels)
    n = len(channels)
    coeff = {}
    all_sets = list(powerset(channels))
    for T in all_sets:
        s = 0.0
        for S in all_sets:
            sign = 1.0
            for i in T:
                sign *= 1.0 if i in S else -1.0
            s += float(values[S]) * sign
        coeff[T] = s / (2 ** n)
    return coeff


def sobol_from_walsh(values: Mapping[frozenset, float], channels):
    """Variance shares for uniform independent binary interventions."""
    beta = walsh_anova(values, channels)
    denom = sum(v*v for T, v in beta.items() if T)
    if denom <= 0:
        return {T: 0.0 for T in beta if T}
    return {T: (v*v)/denom for T, v in beta.items() if T}


def truncated_mobius_prediction(values: Mapping[frozenset, float], channels, target, max_order: int):
    """Predict target from Mobius terms of order <= max_order."""
    target = frozenset(target)
    total = 0.0
    for J in powerset(target):
        if len(J) > max_order:
            continue
        c = 0.0
        for K in powerset(J):
            c += ((-1.0) ** (len(J)-len(K))) * float(values[K])
        total += c
    return total
