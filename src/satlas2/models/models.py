"""
Implementation of the various common Models.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import uncertainties as unc
from numpy.typing import ArrayLike
from scipy.special import erf, voigt_profile

from satlas2.core import Model, Parameter

__all__ = ['ExponentialDecay', 'Polynomial', 'Step', 'Voigt', 'SkewedVoigt']

sqrt2 = 2**0.5
sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
log2 = np.log(2)


class Polynomial(Model):
    """Model class for a polynomial response

    Parameters
    ----------
    p : ArrayLike
        Polynomial coefficients, sorted in increasing order
    name : str
        Name of the model
    prefunc : callable, optional
        Transform function for the input, by default None
    """
    def __init__(self,
                 p: ArrayLike,
                 name: str = 'Polynomial',
                 prefunc: callable = None):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            'p' + str(len(p) - (i + 1)): Parameter(value=P,
                                                   min=-np.inf,
                                                   max=np.inf,
                                                   vary=True)
            for i, P in enumerate(p)
        }

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        x = self.transform(x)
        p = [self.params[paramkey].value for paramkey in self.params.keys()]
        return np.polyval(p, x)

class Step(Model):
    """Model class for a Step response

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
    def __init__(self, values, bounds, name=None, prefunc=None):
        super().__init__(name=name, prefunc=prefunc)
        self.params = {'bkg_val_'+str(len(values)-(i+1)): Parameter(value=P, min=0, max=np.inf, vary=True) for i, P in enumerate(values)}
        self.bounds = bounds

    def f(self, x):
        x = self.transform(x)
        bkg = np.zeros(len(x))
        bkg_sections = len(self.bounds)+1
        if bkg_sections > 1:
            bkg[x<self.bounds[0]] = self.params['bkg_val_0'].value
        else:
            return self.params['bkg_val_0'].value

        for i in range(1,bkg_sections-1):
            section = np.logical_and(x>self.bounds[i-1], x<self.bounds[i])
            bkg[section] = self.params['bkg_val_'+str(i)].value
        bkg[x>=self.bounds[-1]] = self.params['bkg_val_'+str(bkg_sections-1)].value
        return bkg

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
    def __init__(self,
                 a: float,
                 tau: float,
                 name: str = 'ExponentialDecay',
                 prefunc: callable = None):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            'amplitude': Parameter(value=a, min=-np.inf, max=np.inf,
                                   vary=True),
            'halflife': Parameter(value=tau,
                                  min=-np.inf,
                                  max=np.inf,
                                  vary=True),
        }

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        x = self.transform(x)
        a = self.params['amplitude'].value
        b = self.params['halflife'].value
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
    def __init__(self,
                 A: float,
                 mu: float,
                 FWHMG: float,
                 FWHML: float,
                 name: str = 'Voigt',
                 prefunc: callable = None):
        super().__init__(name, prefunc=prefunc)
        self.params = {
            'A': Parameter(value=A, min=0, max=np.inf, vary=True),
            'mu': Parameter(value=mu, min=-np.inf, max=np.inf, vary=True),
            'FWHMG': Parameter(value=FWHMG, min=0, max=np.inf, vary=True),
            'FWHML': Parameter(value=FWHML, min=0, max=np.inf, vary=True)
        }

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        x = self.transform(x)
        A = self.params['A'].value
        mu = self.params['mu'].value
        x = x - mu
        FWHMG = self.params['FWHMG'].value
        FWHML = self.params['FWHML'].value
        sigma, gamma = FWHMG / sqrt2log2t2, FWHML / 2
        ret = voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma)
        return A * ret

    def calculateFWHM(self) -> Tuple[float, float]:
        """Calculate the total FWHM of the profiles, with uncertainty,
        taking the correlations into account.

        Returns
        -------
        Tuple[float, float]
            Tuple of the form (value, uncertainty)
        """
        G, Gu = self.params['FWHMG'].value, self.params['FWHMG'].unc
        L, Lu = self.params['FWHML'].value, self.params['FWHML'].unc
        try:
            correl = self.params['FWHMG'].correl['FWHML']
        except KeyError:
            correl = 0
        G, L = unc.correlated_values_norm([(G, Gu), (L, Lu)],
                                          np.array([[1, correl], [correl, 1]]))
        fwhm = 0.5346 * L + (0.2166 * L * L + G * G)**0.5
        return fwhm.nominal_value, fwhm.std_dev


class SkewedVoigt(Voigt):
    """Model for a Voigt peak skewed by the error function.

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
        Skew of the peak
    name : str, optional
        Name of the model, by default 'SkewedVoigt'
    prefunc : callable, optional
        Transform of the input, by default None
    """
    def __init__(self,
                 A: float,
                 mu: float,
                 FWHMG: float,
                 FWHML: float,
                 skew: float,
                 name: str = 'SkewedVoigt',
                 prefunc: callable = None):
        super().__init__(A, mu, FWHMG, FWHML, name=name, prefunc=prefunc)
        self.params['Skew'] = Parameter(value=skew,
                                        min=-np.inf,
                                        max=np.inf,
                                        vary=True)

    def f(self, x: ArrayLike) -> ArrayLike:
        """:meta private:"""
        ret = super().f(x)
        mu = self.params['mu'].value
        FWHMG = self.params['FWHMG'].value
        sigma = FWHMG / sqrt2log2t2
        skew = self.params['Skew'].value
        beta = skew / (sigma * sqrt2)
        asym = 1 + erf(beta * (x - mu))
        return ret * asym
