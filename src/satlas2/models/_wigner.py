"""
Wigner 3j and 6j symbols, calculated with the Racah formulas.

The sums are evaluated exactly with integers and fractions; only the final
square root is taken in floating point. Spins are given as integers or
half-integers.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""

from __future__ import annotations

from fractions import Fraction
from functools import cache
from math import factorial, sqrt

__all__ = ["wigner_3j", "wigner_6j"]


def _doubled(value: float) -> int:
    """Twice a (half-)integer spin, as an exact integer."""
    doubled = round(2 * value)
    if abs(doubled - 2 * value) > 1e-9:
        raise ValueError(f"{value} is not an integer or half-integer")
    return doubled


def _is_triad(a: int, b: int, c: int) -> bool:
    """Triangle condition for doubled spins, with an integer sum."""
    return abs(a - b) <= c <= a + b and (a + b + c) % 2 == 0


def _triangle_squared(a: int, b: int, c: int) -> Fraction:
    """Square of the triangle coefficient Delta(abc), for doubled spins."""
    return Fraction(
        factorial((a + b - c) // 2)
        * factorial((a - b + c) // 2)
        * factorial((-a + b + c) // 2),
        factorial((a + b + c) // 2 + 1),
    )


def _signed_sqrt(squared_prefactor: Fraction, total: Fraction) -> float:
    """``sqrt(squared_prefactor) * total``, rounded only once."""
    if total == 0:
        return 0.0
    sign = 1 if total > 0 else -1
    return sign * sqrt(squared_prefactor * total * total)


@cache
def _wigner_6j(j1: int, j2: int, j3: int, j4: int, j5: int, j6: int) -> float:
    triads = ((j1, j2, j3), (j1, j5, j6), (j4, j2, j6), (j4, j5, j3))
    if not all(_is_triad(*t) for t in triads):
        return 0.0
    prefactor = Fraction(1)
    for triad in triads:
        prefactor *= _triangle_squared(*triad)
    # all sums below are even, since the spins are doubled
    lower = [sum(t) // 2 for t in triads]
    upper = [(j1 + j2 + j4 + j5) // 2, (j2 + j3 + j5 + j6) // 2, (j3 + j1 + j6 + j4) // 2]
    total = Fraction(0)
    for t in range(max(lower), min(upper) + 1):
        denominator = 1
        for value in lower:
            denominator *= factorial(t - value)
        for value in upper:
            denominator *= factorial(value - t)
        total += Fraction((-1) ** t * factorial(t + 1), denominator)
    return _signed_sqrt(prefactor, total)


@cache
def _wigner_3j(j1: int, j2: int, j3: int, m1: int, m2: int, m3: int) -> float:
    if m1 + m2 + m3 != 0 or not _is_triad(j1, j2, j3):
        return 0.0
    if any(abs(m) > j or (j - m) % 2 for j, m in ((j1, m1), (j2, m2), (j3, m3))):
        return 0.0
    prefactor = _triangle_squared(j1, j2, j3)
    for j, m in ((j1, m1), (j2, m2), (j3, m3)):
        prefactor *= factorial((j + m) // 2) * factorial((j - m) // 2)
    # arguments of the factorials in the denominator, as functions of k
    fixed = [(j1 + j2 - j3) // 2, (j1 - m1) // 2, (j2 + m2) // 2]
    shifted = [(j3 - j2 + m1) // 2, (j3 - j1 - m2) // 2]
    total = Fraction(0)
    for k in range(max(0, *(-s for s in shifted)), min(fixed) + 1):
        denominator = factorial(k)
        for value in fixed:
            denominator *= factorial(value - k)
        for value in shifted:
            denominator *= factorial(value + k)
        total += Fraction((-1) ** k, denominator)
    phase = (-1) ** ((j1 - j2 - m3) // 2)
    return phase * _signed_sqrt(prefactor, total)


def wigner_6j(
    j1: float, j2: float, j3: float, j4: float, j5: float, j6: float
) -> float:
    """Wigner 6j symbol {j1 j2 j3; j4 j5 j6}."""
    return _wigner_6j(*map(_doubled, (j1, j2, j3, j4, j5, j6)))


def wigner_3j(
    j1: float, j2: float, j3: float, m1: float, m2: float, m3: float
) -> float:
    """Wigner 3j symbol (j1 j2 j3; m1 m2 m3)."""
    return _wigner_3j(*map(_doubled, (j1, j2, j3, m1, m2, m3)))
