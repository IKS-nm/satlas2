"""Tests for the numeric Wigner symbols, against known values and sympy."""

import itertools

import numpy as np
import pytest
from sympy.physics.wigner import wigner_3j as sympy_3j
from sympy.physics.wigner import wigner_6j as sympy_6j

from satlas2.models import hfsModel
from satlas2.models._wigner import wigner_3j, wigner_6j
from satlas2.models.hfsModel import HFS


def reference(symbol, *spins):
    """sympy raises for spins that do not couple; the symbol is zero then."""
    try:
        return float(symbol(*spins))
    except ValueError:
        return 0.0


@pytest.mark.parametrize(
    "spins, expected",
    [
        ((1, 1, 1, 1, 1, 1), 1 / 6),
        # {a b c; b a 0} = (-1)^(a+b+c) / sqrt((2a+1)(2b+1))
        ((0.5, 0.5, 1, 0.5, 0.5, 0), 1 / 2),
        ((1, 1, 0, 1, 1, 0), 1 / 3),
        ((1.5, 1, 1.5, 1, 1.5, 0), 1 / np.sqrt(12)),
        ((1, 1, 3, 1, 1, 1), 0.0),  # triangle condition not met
    ],
)
def test_6j_known_values(spins, expected):
    assert wigner_6j(*spins) == pytest.approx(expected)


@pytest.mark.parametrize(
    "spins, expected",
    [
        ((1, 1, 0, 0, 0, 0), -1 / np.sqrt(3)),
        ((0.5, 0.5, 1, 0.5, -0.5, 0), 1 / np.sqrt(6)),
        ((1, 1, 1, 0, 0, 0), 0.0),  # odd sum with all m = 0
        ((1, 1, 1, 1, 1, 0), 0.0),  # m values do not sum to zero
    ],
)
def test_3j_known_values(spins, expected):
    assert wigner_3j(*spins) == pytest.approx(expected)


def test_6j_matches_sympy():
    # a broader grid (spins up to 4, 400 000 symbols) was checked once; this
    # smaller one keeps the test fast
    spins = [x / 2 for x in range(6)]
    for js in itertools.product(spins, repeat=6):
        if sum(js) > 8:
            continue
        assert wigner_6j(*js) == pytest.approx(reference(sympy_6j, *js), abs=1e-15)


def test_3j_matches_sympy():
    spins = [x / 2 for x in range(9)]
    for j1, j2, j3 in itertools.product(spins, repeat=3):
        for m1 in np.arange(-j1, j1 + 1):
            for m2 in np.arange(-j2, j2 + 1):
                m3 = -m1 - m2
                assert wigner_3j(j1, j2, j3, m1, m2, m3) == pytest.approx(
                    reference(sympy_3j, j1, j2, j3, m1, m2, m3), abs=1e-15
                )


def test_rejects_spins_that_are_not_half_integers():
    with pytest.raises(ValueError):
        wigner_6j(0.3, 1, 1, 1, 1, 1)


def test_chunked_evaluation_is_identical(monkeypatch):
    model = HFS(4.5, [3.5, 4.5], A=[500, 300], B=[50, 30], fwhmg=40, fwhml=20)
    x = np.linspace(-2000, 2000, 1001)
    whole = model.f(x)
    monkeypatch.setattr(hfsModel, "_MAX_CHUNK_ELEMENTS", 100)
    # only the order of the floating point sums differs
    assert model.f(x) == pytest.approx(whole, rel=1e-13, abs=1e-15)
