"""
Implementation of the HFSModel class, currently only supplied with a Voigt profile.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import uncertainties as unc
from numpy.typing import ArrayLike
from scipy.special import voigt_profile, erf
from sympy.physics.wigner import wigner_3j, wigner_6j

from ..core import Model, Parameter

__all__ = ["HFS"]

sqrt2 = 2**0.5
sqrt2log2t2 = 2 * np.sqrt(2 * np.log(2))
log2 = np.log(2)


class HFS(Model):
    """Initializes a hyperfine spectrum Model with the given hyperfine parameters.

    Parameters
    ----------
    I : float
        Integer or half-integer value of the nuclear spin
    J : ArrayLike
        A sequence of 2 spins, respectively the J value of the lower state
        and the J value of the higher state
    A : ArrayLike, optional
        A sequence of 2 A values, respectively for the lower and the higher state, by default [0, 0]
    B : ArrayLike, optional
        A sequence of 2 B values, respectively for the lower and the higher state, by default [0, 0]
    C : ArrayLike, optional
        A sequence of 2 C values, respectively for the lower and the higher state, by default [0, 0]
    df : float, optional
        The centroid of the spectrum, by default 0
    fwhmg : float, optional
        The Gaussian FWHM of the Voigt profile, by default 50
    fwhml : float, optional
        The Lorentzian FWHM of the Voigt profile, by default 50
    name : str, optional
        Name of the model, by default 'HFS'
    peak: str, optional
        peak function to use, by default 'voigt'
    peak_kwargs: dict, optional
        additional fitting parameters for skewl, skewr and custom peaks
    N : int, optional
        Number of sidepeaks to be generated, by default None
    offset : float, optional
        Offset in units of x for the sidepeak, by default 0
    poisson : float, optional
        The poisson factor for the sidepeaks, by default 0
    scale : float, optional
        The amplitude of the entire spectrum, by default 1.0
    racah : bool, optional
        Use individual amplitudes are setting the Racah intensities, by default True
    prefunc : callable, optional
        Transformation to be applied on the input before evaluation, by default None
        """
    def __init__(self,
                 I: float,
                 J: ArrayLike,
                 A: ArrayLike = [0, 0],
                 B: ArrayLike = [0, 0],
                 C: ArrayLike = [0, 0],
                 df: float = 0,
                 fwhmg: float = 50,
                 fwhml: float = 50,
                 name: str = 'HFS',
                 peak: str = 'voigt',
                 peak_kwargs: dict = None,
                 N: int = None,
                 offset: float = 0,
                 poisson: float = 0,
                 scale: float = 1.0,
                 racah: bool = True,
                 prefunc: callable = None):
        super().__init__(name, prefunc=prefunc)
        J1, J2 = J
        lower_F = np.arange(abs(I - J1), I + J1 + 1, 1)
        upper_F = np.arange(abs(I - J2), I + J2 + 1, 1)

        self.peakfunc = {
            'voigt': self.voigtPeak,
            'gaussian': self.gaussPeak,
            'lorentzian': self.lorentzPeak,
            'skewleft': self.skewLPeak,
            'skewright': self.skewRPeak,
            'custom': self.customPeak
        }[peak.lower()]

        self.lines = []
        self.intensities = {}
        self.scaling_Al = {}
        self.scaling_Bl = {}
        self.scaling_Cl = {}
        self.scaling_Au = {}
        self.scaling_Bu = {}
        self.scaling_Cu = {}

        for i, F1 in enumerate(lower_F):
            for j, F2 in enumerate(upper_F):
                if abs(F2 - F1) <= 1 and not F2 == F1 == 0.0:
                    if F1 % 1 == 0:
                        F1_str = "{:.0f}".format(F1)
                    else:
                        F1_str = "{:.0f}_2".format(2 * F1)

                    if F2 % 1 == 0:
                        F2_str = "{:.0f}".format(F2)
                    else:
                        F2_str = "{:.0f}_2".format(2 * F2)

                    line = "{}to{}".format(F1_str, F2_str)
                    self.lines.append(line)

                    C1, D1, E1 = self.calcShift(I, J1, F1)
                    C2, D2, E2 = self.calcShift(I, J2, F2)

                    self.scaling_Al[line] = C1
                    self.scaling_Bl[line] = D1
                    self.scaling_Cl[line] = E1
                    self.scaling_Au[line] = C2
                    self.scaling_Bu[line] = D2
                    self.scaling_Cu[line] = E2

                    intens = float(
                        (2 * F1 + 1)
                        * (2 * F2 + 1)
                        * wigner_6j(J2, F2, I, F1, J1, 1.0) ** 2
                    )  # DO NOT REMOVE CAST TO FLOAT!!!
                    self.intensities["Amp" + line] = Parameter(
                        value=intens, min=0, vary=not racah
                    )

        norm = max([p.value for p in self.intensities.values()])
        for n, v in self.intensities.items():
            v.value /= norm

        pars = {
            "centroid": Parameter(value=df),
            "Al": Parameter(value=A[0]),
            "Au": Parameter(value=A[1]),
            "Bl": Parameter(value=B[0]),
            "Bu": Parameter(value=B[1]),
            "Cl": Parameter(value=C[0]),
            "Cu": Parameter(value=C[1]),
            "FWHMG": Parameter(value=fwhmg, min=0.01),
            "FWHML": Parameter(value=fwhml, min=0.01),
            "scale": Parameter(value=scale, min=0, vary=racah),
        }

        if peak.lower() == 'lorentzian':
            pars['FWHMG'].value,pars['FWHMG'].vary, pars['FWHMG'].min=0,False,0
        if peak.lower() == 'gaussian':
            pars['FWHML'].value,pars['FWHML'].vary,pars['FWHML'].min=0,False,0
        if peak_kwargs is not None:
            for peak_arg in peak_kwargs:
                pars[peak_arg] = Parameter(value=peak_kwargs[peak_arg]['value'], 
                                           min=peak_kwargs[peak_arg].get('min', -np.inf), 
                                           max=peak_kwargs[peak_arg].get('max', np.inf), 
                                           vary=peak_kwargs[peak_arg].get('vary', True),
                                           expr=peak_kwargs[peak_arg].get('expr', None))
        if N is not None:
            pars["N"] = Parameter(value=N, vary=False)
            pars["Offset"] = Parameter(value=offset)
            pars["Poisson"] = Parameter(value=poisson, min=0, max=1)
            self.f = self.fShifted
        else:
            self.f = self.fUnshifted
        pars = {**pars, **self.intensities}

        self.params = pars

        if I < 1.5 or J1 < 1.5:
            self.params["Cl"].vary = False
        if I < 1.5 or J2 < 1.5:
            self.params["Cu"].vary = False
        if I < 1 or J1 < 1:
            self.params["Bl"].vary = False
        if I < 1 or J2 < 1:
            self.params["Bu"].vary = False
        if I == 0 or J1 == 0:
            self.params["Al"].vary = False
        if I == 0 or J2 == 0:
            self.params["Au"].vary = False

    def fUnshifted(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculate the response for an unshifted spectrum

        Parameters
        ----------
        x : ArrayLike

        Returns
        -------
        ArrayLike
        """
        centroid = self.params['centroid'].value
        Al = self.params['Al'].value
        Au = self.params['Au'].value
        Bl = self.params['Bl'].value
        Bu = self.params['Bu'].value
        Cl = self.params['Cl'].value
        Cu = self.params['Cu'].value
        scale = self.params['scale'].value

        try:
            result = np.zeros(len(x))
        except TypeError:
            x = np.array([x])
            result = np.zeros(len(x))
        x = self.transform(x)
        for line in self.lines:
            pos = centroid + Au * self.scaling_Au[line] + Bu * self.scaling_Bu[
                line] + Cu * self.scaling_Cu[line] - Al * self.scaling_Al[
                    line] - Bl * self.scaling_Bl[line] - Cl * self.scaling_Cl[
                        line]
            result += scale * self.params['Amp' + line].value * self.peak(
                x - pos)

        return result

    def fShifted(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculate the response with :attr:`N` sidepeaks with an offset
        of :attr:`offset`

        Parameters
        ----------
        x : ArrayLike

        Returns
        -------
        ArrayLike
        """
        centroid = self.params['centroid'].value
        Al = self.params['Al'].value
        Au = self.params['Au'].value
        Bl = self.params['Bl'].value
        Bu = self.params['Bu'].value
        Cl = self.params['Cl'].value
        Cu = self.params['Cu'].value
        scale = self.params['scale'].value
        N = self.params['N'].value
        offset = self.params['Offset'].value
        poisson = self.params['Poisson'].value

        result = np.zeros(len(x))
        x = self.transform(x)
        for line in self.lines:
            pos = (
                centroid
                + Au * self.scaling_Au[line]
                + Bu * self.scaling_Bu[line]
                + Cu * self.scaling_Cu[line]
                - Al * self.scaling_Al[line]
                - Bl * self.scaling_Bl[line]
                - Cl * self.scaling_Cl[line]
            )
            for i in range(N + 1):
                result += self.params['Amp' + line].value * self.peak(
                    self.transform(x - i * offset) - pos) * (poisson**i) / np.math.factorial(i)
            result *= scale

        return result

    def peak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates the profile given the peak_func method

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        returnvalue = self.peakfunc(x)
        return returnvalue

    def voigtPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates the Voigt profile with the Gaussian
        and Lorentzian FWHM

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        sigma, gamma = self.params['FWHMG'].value / sqrt2log2t2, self.params['FWHML'].value / 2
        return voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma)

    def lorentzPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates the lorentzian profile 

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        gamma = self.params['FWHML'].value / 2
        return voigt_profile(x, 0, gamma) / voigt_profile(0, 0, gamma)

    def gaussPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates the Gaussian profile

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        sigma = self.params['FWHMG'].value / sqrt2log2t2
        return voigt_profile(x, sigma, 0) / voigt_profile(0, sigma, 0)

    def skewRPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates a right skewed voigt profile with gaussian and lorentzian FWHM and a skewness parameter

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        sigma, gamma = self.params['FWHMG'].value / 2 * np.sqrt(2 * np.log(2)), self.params['FWHML'].value/2
        erf_x = self.params['skew'].value*x/self.params['FWHMG'].value
        return (voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma))*(1+erf(erf_x/np.sqrt(2)))

    def skewLPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates a left skewed voigt profile with gaussian and lorentzian FWHM and a skewness parameter

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        sigma, gamma = self.params['FWHMG'].value / 2 * np.sqrt(2 * np.log(2)), self.params['FWHML'].value/2
        erf_x = self.params['skew'].value*x/self.params['FWHMG'].value
        return (voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma))*(-erf(erf_x/np.sqrt(2))+1)

    def customPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculate a custom peak

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        raise NotImplementedError

    def calcShift(self, I: float, J: float, F: int) -> ArrayLike:
        """:meta private:
        Calculate the coefficients for the energy shift due to the hyperfine
        interaction up to the octupole moment. A general equation is used
        so extending to higher orders is possible.

        Parameters
        ----------
        I : float
            Nuclear spin
        J : float
            Electronic spin
        F : int
            Hyperfine level spin

        Returns
        -------
        ArrayLike
            Individual coefficients, in ascending order
        """
        phase = (-1) ** (I + J + F)
        contrib = []
        for k in range(1, 4):
            n = float(wigner_6j(I, J, F, J, I, k))
            d = float(
                wigner_3j(I, k, I, -I, 0, I) * wigner_3j(J, k, J, -J, 0, J)
            )
            shift = phase * n / d
            if not np.isfinite(shift):
                contrib.append(0)
            else:
                if k == 1:
                    shift = shift * (I * J)
                elif k == 2:
                    shift = shift / 4
                contrib.append(shift)
        return contrib

    def pos(self) -> ArrayLike:
        """Returns the positions of the peaks in MHz in the hyperfine spectrum

        Returns
        -------
        ArrayLike
        """
        centroid = self.params["centroid"].value
        Al = self.params["Al"].value
        Au = self.params["Au"].value
        Bl = self.params["Bl"].value
        Bu = self.params["Bu"].value
        Cl = self.params["Cl"].value
        Cu = self.params["Cu"].value
        pos = []
        for line in self.lines:
            p = (
                centroid
                + Au * self.scaling_Au[line]
                + Bu * self.scaling_Bu[line]
                + Cu * self.scaling_Cu[line]
                - Al * self.scaling_Al[line]
                - Bl * self.scaling_Bl[line]
                - Cl * self.scaling_Cl[line]
            )
            pos.append(p)
        return pos

    def calculateFWHM(self) -> Tuple[float, float]:
        """Calculate the total FWHM of the profiles, with uncertainty,
        taking the correlations into account.

        Returns
        -------
        Tuple[float, float]
            Tuple of the form (value, uncertainty)
        """
        G, Gu = self.params["FWHMG"].value, self.params["FWHMG"].unc
        L, Lu = self.params["FWHML"].value, self.params["FWHML"].unc
        try:
            correl = self.params["FWHMG"].correl["FWHML"]
        except KeyError:
            correl = 0
        G, L = unc.correlated_values_norm(
            [(G, Gu), (L, Lu)], np.array([[1, correl], [correl, 1]])
        )
        fwhm = 0.5346 * L + (0.2166 * L * L + G * G) ** 0.5
        return fwhm.nominal_value, fwhm.std_dev
