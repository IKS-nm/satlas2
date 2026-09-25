"""
Implementation of the various common Models.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
.. moduleauthor:: Bram van den Borne <bram.vandenborne@kuleuven.be>
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .. import lineshapes
from ..core import Model, Parameter

__all__ = [
    "ExponentialDecay",
    "Polynomial",
    "PiecewiseConstant",
    "Voigt",
    "SkewedVoigt"
]

log2 = np.log(2)


class Polynomial(Model):
    """Model class for a polynomial response

    Parameters
    ----------
    p : ArrayLike
        Polynomial coefficients, from the highest order down to the constant
        (as in :func:`numpy.polyval`). The parameters are named after their
        order: ``p0`` is the constant, ``p1`` the linear coefficient, ...
    name : str
        Name of the model
    prefunc : callable, optional
        Transform function for the input, by default None
    """

    def __init__(
        self, p: ArrayLike, name: str = "Polynomial", prefunc: callable = None
    ):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            "p"
            + str(len(p) - (i + 1)): Parameter(
                value=P, min=-np.inf, max=np.inf, vary=True
            )
            for i, P in enumerate(p)
        }

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        x = self.transform(x)
        p = [self.params[paramkey].value for paramkey in self.params.keys()]
        return np.polyval(p, x)


class PiecewiseConstant(Model):
    """Model class for a PiecewiseConstant response

    Parameters
    ----------
    values : ArrayLike
        Background values between bounds, starting at -inf
    bounds: ArrayLike
        Bounds for background values
    name : str, optional
        Name of the model
    prefunc : callable, optional
        Transform function for the input, by default None
    """

    def __init__(
        self,
        values: ArrayLike,
        bounds: ArrayLike,
        name: str = "PiecewiseConstant",
        prefunc: callable = None,
    ):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            f"value{i}": Parameter(value=value, min=0, max=np.inf, vary=True)
            for i, value in enumerate(values)
        }
        self.bounds = np.hstack([-np.inf, bounds, np.inf])

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        x = self.transform(x)
        values = np.array([p.value for p in self.params.values()])
        return values[np.digitize(x, self.bounds) - 1]


class ExponentialDecay(Model):
    """Model for an exponential decay

    Parameters
    ----------
    a : float
        Amplitude of the exponential
    tau : float
        Half-life of the exponential
    name : str, optional
        Name of the model, by default 'ExponentialDecay'
    prefunc : callable, optional
        Transform function for the input, by default None
    """

    def __init__(
        self,
        a: float,
        tau: float,
        name: str = "ExponentialDecay",
        prefunc: callable = None,
    ):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            "amplitude": Parameter(
                value=a, min=-np.inf, max=np.inf, vary=True
            ),
            "halflife": Parameter(
                value=tau, min=-np.inf, max=np.inf, vary=True
            ),
        }

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        x = self.transform(x)
        a = self.params["amplitude"].value
        b = self.params["halflife"].value
        return a * np.exp(-log2 * x / b)


class Voigt(Model):
    """Model for a Voigt lineshape

    Parameters
    ----------
    A : float
        Amplitude of the profile
    mu : float
        Position of the peak
    FWHMG : float
        Gaussian FWHM of the peak
    FWHML : float
        Lorentzian FWHM of the peak
    name : str, optional
        Name of the model, by default 'Voigt'
    prefunc : callable, optional
        Transform function of the input, by default None
    """

    def __init__(
        self,
        A: float,
        mu: float,
        FWHMG: float,
        FWHML: float,
        name: str = "Voigt",
        prefunc: callable = None,
    ):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            "A": Parameter(value=A, min=0, max=np.inf, vary=True),
            "mu": Parameter(value=mu, min=-np.inf, max=np.inf, vary=True),
            "FWHMG": Parameter(value=FWHMG, min=0, max=np.inf, vary=True),
            "FWHML": Parameter(value=FWHML, min=0, max=np.inf, vary=True),
        }

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        return self.params["A"].value * self._profile(self.transform(x))

    def _profile(self, x: ArrayLike) -> ArrayLike:
        """Peak with height 1 in the (transformed) points x."""
        return lineshapes.voigt(
            x - self.params["mu"].value,
            self.params["FWHMG"].value,
            self.params["FWHML"].value,
        )

    def calculateFWHM(self) -> Tuple[float, float]:
        """Calculate the total FWHM of the profiles, with uncertainty,
        taking the correlations into account.

        Returns
        -------
        Tuple[float, float]
            Tuple of the form (value, uncertainty)
        """
        return lineshapes.voigtFWHMFromParameters(self.params)


class SkewedVoigt(Voigt):
    """Model for a skewed Voigt peak by the error function. Negative skew value is left-skewed, positive skew value is right-skewed.

    Parameters
    ----------
    A : float
        Amplitude of the peak
    mu : float
        Position of the peak
    FWHMG : float
        Gaussian FWHM
    FWHML : float
        Lorentzian FWHM
    skew : float
        Skew of the peak, as defined in :func:`satlas2.lineshapes.skew`
    name : str, optional
        Name of the model, by default 'SkewedVoigt'
    prefunc : callable, optional
        Transform of the input, by default None
    """

    def __init__(
        self,
        A: float,
        mu: float,
        FWHMG: float,
        FWHML: float,
        skew: float,
        name: str = "SkewedVoigt",
        prefunc: callable = None,
    ):
        super().__init__(A, mu, FWHMG, FWHML, name=name, prefunc=prefunc)
        self.params["Skew"] = Parameter(
            value=skew, min=-np.inf, max=np.inf, vary=True
        )

    def _profile(self, x: ArrayLike) -> ArrayLike:
        """Skewed peak in the (transformed) points x."""
        return super()._profile(x) * lineshapes.skew(
            x - self.params["mu"].value,
            self.params["Skew"].value,
            self.params["FWHMG"].value,
        )
