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
    data: ArrayLike,
    sigma: float = 1,
    alpha: Optional[float] = None,
    mean: bool = False,
) -> Tuple[float, float]:
    """Calculates the confidence interval
    for the mean of a Poisson distribution.

    Parameters
    ----------
    data: ArrayLike
        Samples of separate Poisson distributions.
    sigma: float
        The significance level given in equivalent sigma.
        Defaults to 1-sigma.
    alpha: Optional[float]
        Significance level of interval. If given, *sigma* is ignored.
    mean: bool
        Set to True if the exact mean is given, by default False

    Returns
    -------
    low, high: Tuple[float, float]
        Lower and higher limits for the interval."""
    if alpha is None:
        a = (1 - norm.cdf(np.abs(sigma))) * 2
    else:
        a = alpha
    if mean:
        a = 1 - a
        low, high = poisson.interval(a, data)
    else:
        low, high = (
            chi2.ppf(a / 2, 2 * data) / 2,
            chi2.ppf(1 - a / 2, 2 * data + 2) / 2,
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
