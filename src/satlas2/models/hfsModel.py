"""
Implementation of the HFSModel class, currently only supplied with a Voigt profile.

.. moduleauthor:: Wouter Gins <wouter.gins@kuleuven.be>
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from math import factorial

import numpy as np
from numpy.typing import ArrayLike

from .. import lineshapes
from ..core import Model, Parameter
from ._wigner import wigner_3j, wigner_6j

__all__ = ["HFS"]

# Amplitudes below this (before normalisation) are considered forbidden
_MIN_STRENGTH = 1e-12
# Largest number of elements of the (peaks x points) array evaluated at once,
# which limits the memory use for large spectra (2**20 floats is 8 MB)
_MAX_CHUNK_ELEMENTS = 2**20


def triangle_condition(spin_1: float, spin_2: float, order: float) -> bool:
    """Check if three angular momenta can be coupled to a total of zero."""
    return (abs(spin_1 - spin_2) <= order <= spin_1 + spin_2) and (
        spin_1 + spin_2 + order
    ) % 1 == 0


def _level_label(F: float) -> str:
    """Label of a hyperfine level, half-integers as '<2F>_2'."""
    return f"{F:.0f}" if F % 1 == 0 else f"{2 * F:.0f}_2"


# The Wigner symbols only depend on the spins, so the derived coefficients are
# cached across all HFS instances.
@cache
def _quadrupole_norm(I: float, J: float, k: int) -> float:
    """Normalisation of the rank-k interaction, independent of F."""
    return wigner_3j(I, k, I, -I, 0, I) * wigner_3j(J, k, J, -J, 0, J)


@cache
def _shift_coefficients(I: float, J: float, F: float) -> tuple[float, float, float]:
    """Energy shift of the level F per unit of the A, B and C constants.

    The shift of the level is ``A * a + B * b + C * c``, with (a, b, c) the
    returned coefficients. For example, a = K/2 with K = F(F+1) - I(I+1) - J(J+1).
    If the interaction does not exist for the given spins (e.g. the quadrupole
    interaction for J < 1), its coefficient is 0.
    """
    phase = (-1) ** (I + J + F)
    coefficients = []
    for k in (1, 2, 3):  # dipole, quadrupole, octupole
        six_j = wigner_6j(I, J, F, J, I, k)
        norm = _quadrupole_norm(I, J, k)
        shift = phase * six_j / norm if norm != 0 else 0
        if not np.isfinite(shift):
            shift = 0
        # conventions of the A and B constants
        if k == 1:
            shift *= I * J
        elif k == 2:
            shift /= 4
        coefficients.append(shift)
    return tuple(coefficients)


@cache
def _line_strength(
    I: float, J1: float, J2: float, F1: float, F2: float, order: int
) -> float:
    """Squared 6j symbol determining the strength of a line."""
    return wigner_6j(J2, F2, I, F1, J1, order) ** 2


class _SaturationParameter(Parameter):
    """Saturation of an :class:`HFS` model.

    The amplitudes of the lines depend on the saturation, so the model
    recalculates them every time the value is set (by the user or by a fit).
    This is deliberately not done with lmfit expressions on the amplitudes,
    see the notes of :class:`HFS`.
    """

    def __init__(self, model: HFS, value: float):
        self._model = model
        super().__init__(value=value, min=0)

    @property
    def value(self) -> float:
        return self._value

    @value.setter
    def value(self, value: float) -> None:
        self._value = value
        self._model._updateSaturatedAmplitudes(value)


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
        additional fitting parameters for skew and custom peaks. The
        ``"skewvoigt"`` peak needs a ``"skew"`` parameter, defined as in
        :func:`satlas2.lineshapes.skew`
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
    order : int, optional
        Define the order of the transition, used to calculate the allowed transitions
        and initial intensities, by default 1
    use_saturation : bool, optional
        If True, apply the saturation model to transition amplitudes. Defaults to False.
        Cannot be combined with `racah`. Adds a `Saturation` parameter to the model;
        the (fixed) amplitudes are recalculated whenever its value changes.
    saturation : float, optional
        Saturation parameter (>= 0). Only used when `use_saturation` is True;
        otherwise the value is ignored. The value controls the exponential
        mapping between Racah intensities and saturated amplitudes.
    prefunc : callable, optional
        Transformation to be applied on the input before evaluation, by default None

    Notes
    -----
    **Saturation.** With ``use_saturation=True`` the amplitudes of the lines are
    not free parameters but follow from the ``Saturation`` parameter. The
    ``Amp<line>`` parameters are fixed and are recalculated every time the value
    of ``Saturation`` is set, so they always correspond to the current saturation:
    after a fit or a random walk they, and the results dataframe, show the
    amplitudes belonging to the fitted saturation.

    This is done in the model, and not by giving each ``Amp<line>`` parameter an
    lmfit expression of ``Saturation``, for two reasons:

    * The amplitudes are normalised to the strongest line, so the expression of
      every line has to contain the terms of all lines. Evaluating these
      expressions is slow: for a spectrum with 24 lines it takes about 6 ms per
      change of the saturation, more than ten times the cost of evaluating the
      spectrum itself, which would dominate every fit and random walk.
    * The model must give the correct spectrum without a
      :class:`~satlas2.core.Fitter`, e.g. when plotting. Expressions only exist
      inside lmfit, so the calculation would have to be duplicated in the model.

    Expressions remain the right tool for relations chosen by the user, such as
    :meth:`~satlas2.core.Fitter.setExpr`.
    """

    def __init__(
        self,
        I: float,
        J: ArrayLike,
        A: ArrayLike | None = None,
        B: ArrayLike | None = None,
        C: ArrayLike | None = None,
        df: float = 0,
        fwhmg: float = 50,
        fwhml: float = 50,
        name: str = "HFS",
        peak: str = "voigt",
        peak_kwargs: dict | None = None,
        N: int | None = None,
        offset: float = 0,
        poisson: float = 0,
        scale: float = 1.0,
        racah: bool = True,
        order: int = 1,
        use_saturation: bool = False,
        saturation: float = 0.0,
        prefunc: Callable | None = None,
    ):
        super().__init__(name, prefunc=prefunc)
        if racah and use_saturation:
            raise ValueError(
                "Parameters 'racah' and 'use_saturation' cannot both be True"
            )
        self.use_racah = racah
        self.use_saturation = use_saturation

        self.peakfunc = {
            "voigt": self.voigtPeak,
            "gaussian": self.gaussPeak,
            "lorentzian": self.lorentzPeak,
            "skewvoigt": self.skewPeak,
            "custom": self.customPeak,
        }[peak.lower()]

        J1, J2 = J
        self._build_lines(I, J1, J2, order)
        self.params = self._build_parameters(
            A=[0, 0] if A is None else A,
            B=[0, 0] if B is None else B,
            C=[0, 0] if C is None else C,
            df=df,
            fwhmg=fwhmg,
            fwhml=fwhml,
            peak=peak.lower(),
            peak_kwargs=peak_kwargs,
            N=N,
            offset=offset,
            poisson=poisson,
            scale=scale,
            saturation=saturation,
        )
        self._amplitude_keys = ["Amp" + line for line in self.lines]
        self.f = self.fUnshifted if N is None else self.fShifted
        self._fix_unused_couplings(I, J1, J2)

    def _build_lines(self, I: float, J1: float, J2: float, order: int) -> None:
        """Determine the allowed transitions, their shifts and their intensities.

        Sets :attr:`lines`, the shift matrix and the normalised Racah and
        saturated amplitudes.
        """
        if not triangle_condition(J1, J2, order):
            raise ValueError(
                f"Triangle condition not satisfied for J1={J1}, J2={J2} and order={order}."
            )
        lower_F = np.arange(abs(I - J1), I + J1 + 1, 1)
        upper_F = np.arange(abs(I - J2), I + J2 + 1, 1)

        lines, lower_shifts, upper_shifts, racah, saturated = [], [], [], [], []
        for F1 in map(float, lower_F):
            for F2 in map(float, upper_F):
                if not triangle_condition(F1, F2, order):
                    continue
                wigner = _line_strength(I, J1, J2, F1, F2, order)
                if wigner < _MIN_STRENGTH:
                    continue
                lines.append(f"{_level_label(F1)}to{_level_label(F2)}")
                lower_shifts.append(self.calcShift(I, J1, F1))
                upper_shifts.append(self.calcShift(I, J2, F2))
                racah.append((2 * F1 + 1) * (2 * F2 + 1) * wigner)
                saturated.append(2 * F1 + 1)

        self.lines = lines
        # (n_lines, 3) coefficients of A, B and C of the lower and upper level
        self._lower_shifts = np.array(lower_shifts, dtype=float).reshape(-1, 3)
        self._upper_shifts = np.array(upper_shifts, dtype=float).reshape(-1, 3)
        racah = np.array(racah, dtype=float)
        saturated = np.array(saturated, dtype=float)
        # normalise to the strongest line
        self.racah_amplitudes = racah / racah.max() if len(racah) else racah
        self.saturated_amplitudes = (
            saturated / saturated.max() if len(saturated) else saturated
        )

    def _build_parameters(
        self,
        A,
        B,
        C,
        df,
        fwhmg,
        fwhml,
        peak,
        peak_kwargs,
        N,
        offset,
        poisson,
        scale,
        saturation,
    ) -> dict:
        """Create the Parameter objects of the model."""
        # Amplitudes are only free if neither Racah nor saturation fixes them
        vary_amplitudes = not (self.use_racah or self.use_saturation)
        initial = self._calculate_transitional_intensities(
            saturation if self.use_saturation else 0.0
        )
        self.intensities = {
            "Amp" + line: Parameter(value=float(amp), min=0, vary=vary_amplitudes)
            for line, amp in zip(self.lines, initial)
        }

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
            "scale": Parameter(value=scale, min=0, vary=not vary_amplitudes),
        }
        if self.use_saturation:
            pars["Saturation"] = _SaturationParameter(self, saturation)

        if peak == "lorentzian":
            pars["FWHMG"].value, pars["FWHMG"].vary, pars["FWHMG"].min = 0, False, 0
        if peak == "gaussian":
            pars["FWHML"].value, pars["FWHML"].vary, pars["FWHML"].min = 0, False, 0
        for peak_arg, spec in (peak_kwargs or {}).items():
            pars[peak_arg] = Parameter(
                value=spec["value"],
                min=spec.get("min", -np.inf),
                max=spec.get("max", np.inf),
                vary=spec.get("vary", True),
                expr=spec.get("expr", None),
            )
        if N is not None:
            pars["N"] = Parameter(value=N, vary=False)
            pars["Offset"] = Parameter(value=offset)
            pars["Poisson"] = Parameter(value=poisson, min=0, max=1)
        return {**pars, **self.intensities}

    def _fix_unused_couplings(self, I: float, J1: float, J2: float) -> None:
        """Fix the coupling constants that have no effect for the given spins."""
        for level, J in (("l", J1), ("u", J2)):
            if I < 1.5 or J < 1.5:
                self.params["C" + level].vary = False
            if I < 1 or J < 1:
                self.params["B" + level].vary = False
            if I == 0 or J == 0:
                self.params["A" + level].vary = False

    def _calculate_transitional_intensities(self, s: float) -> np.ndarray:
        """Calculate transitional amplitudes between Racah and saturated.

        Uses the exponential mapping: transitional = -sat * expm1(-rac*s/sat)
        and returns normalized array.
        """
        if len(self.racah_amplitudes) == 0:
            return np.array([])
        if s is None or s <= 0:
            return self.racah_amplitudes.copy()
        sat = self.saturated_amplitudes
        rac = self.racah_amplitudes
        transitional = -sat * np.expm1(-rac * s / sat)
        return transitional / transitional.max()

    def _updateSaturatedAmplitudes(self, saturation: float) -> None:
        """Set the amplitudes of the lines for the given saturation."""
        amplitudes = self._calculate_transitional_intensities(saturation)
        for parameter, amplitude in zip(self.intensities.values(), amplitudes):
            parameter.value = float(amplitude)

    def _amplitudes(self) -> np.ndarray:
        """Current relative amplitude of each line."""
        return np.array([self.params[key].value for key in self._amplitude_keys])

    def _couplings(self, level: str) -> np.ndarray:
        """Current values of A, B and C of the lower ("l") or upper ("u") level."""
        return np.array([self.params[c + level].value for c in "ABC"])

    def _positions(self) -> np.ndarray:
        """Current position of each line: the centroid plus the shift of the
        upper level minus the shift of the lower level."""
        upper = self._upper_shifts @ self._couplings("u")
        lower = self._lower_shifts @ self._couplings("l")
        return self.params["centroid"].value + upper - lower

    def _prepare(self, x: ArrayLike) -> np.ndarray:
        """Turn the input into a transformed 1D array."""
        return self.transform(np.atleast_1d(x))

    def _sum_peaks(
        self, x: np.ndarray, positions: np.ndarray, amplitudes: np.ndarray
    ) -> np.ndarray:
        """Sum the peaks at the given positions and amplitudes in the (already
        transformed) points x, evaluating a limited number of points at once."""
        chunk = max(1, _MAX_CHUNK_ELEMENTS // max(len(positions), 1))
        return np.concatenate(
            [
                amplitudes
                @ self.peak(x[np.newaxis, start : start + chunk] - positions[:, np.newaxis])
                for start in range(0, len(x), chunk)
            ]
            or [np.zeros(0)]
        )

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
        x = self._prepare(x)
        scale = self.params["scale"].value
        return scale * self._sum_peaks(x, self._positions(), self._amplitudes())

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
        x = self._prepare(x)
        scale = self.params["scale"].value
        N = int(self.params["N"].value)
        offset = self.params["Offset"].value
        poisson = self.params["Poisson"].value

        # every line has N+1 copies, shifted by i*offset and weighted by a
        # Poisson-like factor
        order = np.arange(N + 1)
        weights = poisson**order / np.array([factorial(int(i)) for i in order])
        positions = self._positions()[:, np.newaxis] + order * offset
        amplitudes = self._amplitudes()[:, np.newaxis] * weights
        return scale * self._sum_peaks(x, positions.ravel(), amplitudes.ravel())

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
        return lineshapes.voigt(
            x, self.params["FWHMG"].value, self.params["FWHML"].value
        )

    def lorentzPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates the lorentzian profile, normalised to a height of 1

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        return lineshapes.lorentzian(x, self.params["FWHML"].value)

    def gaussPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates the Gaussian profile, normalised to a height of 1

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        return lineshapes.gaussian(x, self.params["FWHMG"].value)

    def skewPeak(self, x: ArrayLike) -> ArrayLike:
        """:meta private:
        Calculates a skewed voigt profile with gaussian and lorentzian FWHM and a skewness parameter

        Parameters
        ----------
        x : ArrayLike
            Evaluation points

        Returns
        -------
        ArrayLike
        """
        fwhmg = self.params["FWHMG"].value
        return lineshapes.voigt(x, fwhmg, self.params["FWHML"].value) * lineshapes.skew(
            x, self.params["skew"].value, fwhmg
        )

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

    def calcShift(self, I: float, J: float, F: int) -> list[float]:
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
        list[float]
            Individual coefficients, in ascending order
        """
        return list(_shift_coefficients(float(I), float(J), float(F)))

    def pos(self) -> ArrayLike:
        """Returns the positions of the peaks in MHz in the hyperfine spectrum

        Returns
        -------
        ArrayLike
        """
        return self._positions()

    def calculateFWHM(self) -> tuple[float, float]:
        """Calculate the total FWHM of the profiles, with uncertainty,
        taking the correlations into account.

        Returns
        -------
        tuple[float, float]
            Tuple of the form (value, uncertainty)
        """
        return lineshapes.voigtFWHMFromParameters(self.params)
