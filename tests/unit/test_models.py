"""Tests for the lineshapes and the general models, against closed forms."""

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import erf, voigt_profile

from satlas2 import lineshapes
from satlas2.models import (
    ExponentialDecay,
    PiecewiseConstant,
    Polynomial,
    SkewedVoigt,
    Voigt,
)

X = np.linspace(-60, 60, 25)
SIGMA_PER_FWHM = 1 / (2 * np.sqrt(2 * np.log(2)))


def half_width(profile):
    """Numerical half width at half maximum of a peak with height 1 at 0."""
    return brentq(lambda x: profile(x) - 0.5, 0, 1e4)


# ----------------------------------------------------------------------------
# Lineshapes
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, kwargs",
    [
        (lineshapes.gaussian, {"fwhm": 12.0}),
        (lineshapes.lorentzian, {"fwhm": 12.0}),
        (lineshapes.voigt, {"fwhmg": 12.0, "fwhml": 5.0}),
    ],
)
def test_peaks_have_height_one(shape, kwargs):
    assert shape(np.array([0.0]), **kwargs) == pytest.approx([1.0])
    assert np.all(shape(X, **kwargs) <= 1.0)


@pytest.mark.parametrize("shape", [lineshapes.gaussian, lineshapes.lorentzian])
def test_half_maximum_at_half_fwhm(shape):
    assert half_width(lambda x: shape(x, 12.0)) == pytest.approx(6.0)


def test_voigt_is_the_scipy_profile():
    sigma, gamma = 12 * SIGMA_PER_FWHM, 2.5
    expected = voigt_profile(X, sigma, gamma) / voigt_profile(0, sigma, gamma)
    assert lineshapes.voigt(X, 12.0, 5.0) == pytest.approx(expected)


def test_voigt_limits():
    assert lineshapes.voigt(X, 12.0, 1e-9) == pytest.approx(
        lineshapes.gaussian(X, 12.0), abs=1e-7
    )
    assert lineshapes.voigt(X, 1e-9, 12.0) == pytest.approx(
        lineshapes.lorentzian(X, 12.0), abs=1e-7
    )


def test_skew():
    assert lineshapes.skew(X, 0.0) == pytest.approx(np.ones_like(X))
    assert lineshapes.skew(X, 0.3) == pytest.approx(1 + erf(0.3 * X))


@pytest.mark.parametrize("G, L", [(10.0, 0.0), (0.0, 10.0), (10.0, 4.0), (3.0, 9.0)])
def test_voigt_fwhm_approximation(G, L):
    value, uncertainty = lineshapes.voigtFWHM(G, L)
    exact = 2 * half_width(lambda x: lineshapes.voigt(x, max(G, 1e-12), max(L, 1e-12)))
    # Olivero and Longbothum quote 0.02% accuracy
    assert value == pytest.approx(exact, rel=3e-4)
    assert uncertainty == 0


def test_voigt_fwhm_uncertainty_propagation():
    G, L, dG, dL, rho = 10.0, 4.0, 0.5, 0.3, 0.4

    def fwhm(g, l):
        return 0.5346 * l + np.sqrt(0.2166 * l**2 + g**2)

    h = 1e-6
    jac = np.array(
        [
            (fwhm(G + h, L) - fwhm(G - h, L)) / (2 * h),
            (fwhm(G, L + h) - fwhm(G, L - h)) / (2 * h),
        ]
    )
    cov = np.array([[dG**2, rho * dG * dL], [rho * dG * dL, dL**2]])
    _, uncertainty = lineshapes.voigtFWHM(G, L, dG, dL, rho)
    assert uncertainty == pytest.approx(np.sqrt(jac @ cov @ jac), rel=1e-6)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def test_polynomial_matches_polyval():
    coefficients = [0.5, -2.0, 3.0]
    model = Polynomial(coefficients)
    assert model.f(X) == pytest.approx(np.polyval(coefficients, X))
    assert model.params["p0"].value == 3.0
    assert model.params["p2"].value == 0.5


def test_piecewise_constant():
    model = PiecewiseConstant([1.0, 5.0, 2.0], bounds=[0.0, 10.0])
    x = np.array([-5.0, 0.0, 3.0, 10.0, 20.0])
    assert model.f(x) == pytest.approx([1.0, 5.0, 5.0, 2.0, 2.0])
    assert [model.params[f"value{i}"].value for i in range(3)] == [1.0, 5.0, 2.0]


def test_exponential_decay_halves_after_a_halflife():
    model = ExponentialDecay(8.0, 3.0)
    assert model.f(np.array([0.0, 3.0, 6.0])) == pytest.approx([8.0, 4.0, 2.0])


def test_voigt_model():
    model = Voigt(A=3.0, mu=5.0, FWHMG=12.0, FWHML=5.0)
    assert model.f(X) == pytest.approx(3.0 * lineshapes.voigt(X - 5.0, 12.0, 5.0))


def test_voigt_model_fwhm():
    model = Voigt(A=3.0, mu=5.0, FWHMG=12.0, FWHML=5.0)
    model.params["FWHMG"].unc, model.params["FWHML"].unc = 0.5, 0.3
    model.params["FWHMG"].correl = {"FWHML": 0.4}
    assert model.calculateFWHM() == pytest.approx(
        lineshapes.voigtFWHM(12.0, 5.0, 0.5, 0.3, 0.4)
    )


def test_skewed_voigt_definition():
    model = SkewedVoigt(A=3.0, mu=5.0, FWHMG=12.0, FWHML=5.0, skew=0.7)
    sigma = 12.0 * SIGMA_PER_FWHM
    expected = (
        3.0
        * lineshapes.voigt(X - 5.0, 12.0, 5.0)
        * (1 + erf(0.7 * (X - 5.0) / (sigma * np.sqrt(2))))
    )
    assert model.f(X) == pytest.approx(expected)


def test_skewed_voigt_without_skew_is_voigt():
    skewed = SkewedVoigt(A=3.0, mu=5.0, FWHMG=12.0, FWHML=5.0, skew=0.0)
    assert skewed.f(X) == pytest.approx(Voigt(3.0, 5.0, 12.0, 5.0).f(X))


@pytest.mark.parametrize(
    "model",
    [
        lambda prefunc: Polynomial([0.5, -2.0, 3.0], prefunc=prefunc),
        lambda prefunc: Voigt(3.0, 5.0, 12.0, 5.0, prefunc=prefunc),
        lambda prefunc: SkewedVoigt(3.0, 5.0, 12.0, 5.0, 0.7, prefunc=prefunc),
        lambda prefunc: ExponentialDecay(8.0, 3.0, prefunc=prefunc),
    ],
)
def test_prefunc_is_applied_to_the_whole_model(model):
    def prefunc(x):
        return 0.5 * x + 20

    assert model(prefunc).f(X) == pytest.approx(model(None).f(prefunc(X)))
