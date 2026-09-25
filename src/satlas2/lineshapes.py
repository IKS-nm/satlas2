"""
Peak shapes shared by the models. All peaks are normalised to a height of 1
at their centre, and take the full widths at half maximum (FWHM) as input.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""

from __future__ import annotations

import numpy as np
import uncertainties as unc
from numpy.typing import ArrayLike
from scipy.special import erf, voigt_profile

__all__ = [
    "gaussian",
    "lorentzian",
    "voigt",
    "skew",
    "fwhmToSigma",
    "voigtFWHM",
]

# FWHM = 2 sqrt(2 ln 2) sigma for a Gaussian
FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def fwhmToSigma(fwhm: float) -> float:
    """Standard deviation of a Gaussian with the given FWHM."""
    return fwhm / FWHM_PER_SIGMA


def gaussian(x: ArrayLike, fwhm: float) -> ArrayLike:
    """Gaussian peak with height 1."""
    return np.exp(-0.5 * (x / fwhmToSigma(fwhm)) ** 2)


def lorentzian(x: ArrayLike, fwhm: float) -> ArrayLike:
    """Lorentzian peak with height 1."""
    return 1 / (1 + (2 * x / fwhm) ** 2)


def voigt(x: ArrayLike, fwhmg: float, fwhml: float) -> ArrayLike:
    """Voigt peak (convolution of a Gaussian and a Lorentzian) with height 1."""
    sigma, gamma = fwhmToSigma(fwhmg), fwhml / 2
    return voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma)


def skew(x: ArrayLike, skew: float, fwhmg: float) -> ArrayLike:
    r"""Skewing factor of a peak centred at 0 with Gaussian FWHM `fwhmg`:

    .. math::
        1 + \mathrm{erf}\left(\frac{\alpha x}{\sigma\sqrt{2}}\right)

    with :math:`\alpha` the skew and :math:`\sigma` the standard deviation of
    the Gaussian component. This is twice the normal cumulative distribution
    function of the skew normal distribution, and the definition used by
    ``SkewedGaussianModel`` and ``SkewedVoigtModel`` in lmfit. A positive skew
    moves intensity to the right, a negative skew to the left."""
    return 1 + erf(skew * x / (fwhmToSigma(fwhmg) * np.sqrt(2)))


def voigtFWHM(
    fwhmg: float,
    fwhml: float,
    fwhmg_unc: float = 0,
    fwhml_unc: float = 0,
    correlation: float = 0,
) -> tuple[float, float]:
    """Total FWHM of a Voigt peak, with its uncertainty.

    Uses the approximation of Olivero and Longbothum (1977), accurate to
    0.02%: ``0.5346 L + sqrt(0.2166 L^2 + G^2)``.

    Returns
    -------
    tuple[float, float]
        Tuple of the form (value, uncertainty)
    """
    G, L = unc.correlated_values_norm(
        [(fwhmg, fwhmg_unc), (fwhml, fwhml_unc)],
        np.array([[1, correlation], [correlation, 1]]),
    )
    fwhm = 0.5346 * L + (0.2166 * L * L + G * G) ** 0.5
    return fwhm.nominal_value, fwhm.std_dev


def voigtFWHMFromParameters(params: dict) -> tuple[float, float]:
    """:meta private:
    Total FWHM of a Voigt peak from the ``FWHMG`` and ``FWHML`` parameters of
    a model, taking their uncertainties and correlation into account."""
    G, L = params["FWHMG"], params["FWHML"]
    return voigtFWHM(G.value, L.value, G.unc, L.unc, G.correl.get("FWHML", 0))
