"""
Implementation of various functions that ease the work, but do not belong in one of the other modules.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
import numpy as np
from scipy.stats import chi2, poisson
from numpy.typing import ArrayLike
from typing import Tuple, Optional, Union
from .core import Model
from scipy.optimize import minimize

__all__ = ['weightedAverage', 'generateSpectrum', 'poissonInterval']


def weightedAverage(x: ArrayLike,
                    sigma: ArrayLike,
                    axis: Optional[int] = None) -> Tuple[float, float]:
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

    Returns
    -------
    tuple
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
    Xstat = (1 / sigma**2).sum(axis=axis)
    Xm = (x / sigma**2).sum(axis=axis) / Xstat
    Xscatt = (((x - Xm) / sigma)**2).sum(axis=axis) / ((len(x) - 1) * Xstat)
    Xstat = 1 / Xstat
    return Xm, np.maximum.reduce([Xstat, Xscatt], axis=axis)**0.5


def poissonInterval(data: ArrayLike,
                    alpha: float = 0.32) -> Tuple[float, float]:
    """Calculates the confidence interval
    for the mean of a Poisson distribution.

    Parameters
    ----------
    data: ArrayLike
        Data giving the mean of the Poisson distributions.
    alpha: float
        Significance level of interval. Defaults to
        one sigma (0.32).

    Returns
    -------
    low, high: ArrayLike
        Lower and higher limits for the interval."""
    a = alpha
    low, high = (chi2.ppf(a / 2, 2 * data) / 2,
                 chi2.ppf(1 - a / 2, 2 * data + 2) / 2)
    low = np.nan_to_num(low)
    return low, high


def generateSpectrum(
        models: Union[Model, list],
        x: ArrayLike,
        generator: callable = np.random.default_rng().poisson) -> ArrayLike:
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