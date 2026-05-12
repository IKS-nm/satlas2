"""
Implementation of various functions that ease the work, but do not belong in one of the other modules.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import chi2, norm, poisson

from .core import Model

__all__ = ["weightedAverage", "poissonInterval", "generateSpectrum"]


def weightedAverage(
    x: ArrayLike, sigma: ArrayLike, axis: Optional[int] = None
) -> Tuple[float, float]:
    r"""Takes the weighted average of an array of values and the associated
    errors. Calculates the scatter and statistical error, and returns
    the greater of these two values.

    Parameters
    ----------
    x: ArrayLike
        Array-like assortment of measured values, is transformed into a
        1D-array.
    sigma: ArrayLike
        Array-like assortment of errors on the measured values, is transformed
        into a 1D-array.
    axis: Optional[int]
        Axis over which the weighted average should be calculated

    Returns
    -------
    Tuple[float, float]
        Returns a tuple (weighted average, uncertainty), with the uncertainty
        being the greater of the uncertainty calculated from the statistical
        uncertainty and the scattering uncertainty.

    Note
    ----
    The formulas used are

    .. math::

        \left\langle x\right\rangle_{weighted} &= \frac{\sum_{i=1}^N \frac{x_i}
                                                                 {\sigma_i^2}}
                                                      {\sum_{i=1}^N \frac{1}
                                                                {\sigma_i^2}}

        \sigma_{stat}^2 &= \frac{1}{\sum_{i=1}^N \frac{1}{\sigma_i^2}}

        \sigma_{scatter}^2 &= \frac{\sum_{i=1}^N \left(\frac{x_i-\left\langle
                                                    x\right\rangle_{weighted}}
                                                      {\sigma_i}\right)^2}
               {\left(N-1\right)\sum_{i=1}^N \frac{1}{\sigma_i^2}}"""
    x = np.array(x)
    sigma = np.array(sigma)
    Xstat = (1 / sigma**2).sum(axis=axis)
    Xm = (x / sigma**2).sum(axis=axis) / Xstat
    Xscatt = (((x - Xm) / sigma) ** 2).sum(axis=axis) / ((len(x) - 1) * Xstat)
    Xstat = 1 / Xstat
    return Xm, np.maximum.reduce([Xstat, Xscatt], axis=axis) ** 0.5


def poissonInterval(
    counts: ArrayLike,
    sigma: float = 1,
    alpha: Optional[float] = None,
    is_mean: bool = False,
) -> Tuple[float, float]:
    """Calculates the confidence interval for a Poisson distribution.

    Two modes are supported:

    - ``is_mean=False`` (default): *counts* are observed Poisson counts. Returns
      the interval of means λ consistent with each count at the given confidence
      level, using the exact chi-squared method (Garwood, 1936).
    - ``is_mean=True``: *counts* are the exact Poisson mean λ. Returns the
      interval of counts expected with the given probability, using the exact
      Poisson CDF.

    Parameters
    ----------
    counts: ArrayLike
        Observed Poisson counts (``is_mean=False``) or exact Poisson means λ
        (``is_mean=True``).
    sigma: float
        Confidence level expressed as equivalent Gaussian sigma. Defaults to 1.
    alpha: Optional[float]
        Significance level (two-sided). If given, *sigma* is ignored.
    is_mean: bool
        If True, *counts* is interpreted as the exact Poisson mean λ.
        Default is False.

    Returns
    -------
    low, high: Tuple[float, float]
        Lower and upper limits of the interval.

    References
    ----------
    Garwood, F. (1936). Fiducial limits for the Poisson distribution.
    *Biometrika*, 28(3-4), 437-442. https://doi.org/10.2307/2333958"""
    if alpha is None:
        alpha = (1 - norm.cdf(np.abs(sigma))) * 2
    if is_mean:
        low, high = poisson.interval(1 - alpha, counts)
    else:
        low, high = (
            chi2.ppf(alpha / 2, 2 * counts) / 2,
            chi2.ppf(1 - alpha / 2, 2 * counts + 2) / 2,
        )
    low = np.nan_to_num(low)
    return low, high


def generateSpectrum(
    models: Union[Model, list],
    x: ArrayLike,
    generator: Optional[callable] = np.random.default_rng().poisson,
) -> ArrayLike:
    """Generates a dataset based on the models and x-values provided.

    Parameters
    ----------
    models : Union[Model, list]
        A single Model or list of models. In case of a list, all models are summed together.
    x : ArrayLike
        The x values for which a y value has to be generated.
    generator : callable, optional
        A callable with one parameter that returns a random value based on this.
        The default is a Poisson generator.

    Returns
    -------
    ArrayLike
        A same-sized array as x with values given by feeding the Model.f(x) value
        to the generator.
    """

    def evaluate(x):
        try:
            for model in models:
                try:
                    f += model.f(x)
                except UnboundLocalError:
                    f = model.f(x)
        except TypeError:
            f = models.f(x)
        return f

    y = evaluate(x)
    y = generator(y)
    return y
