"""Tests for the HFS model.

Three kinds of tests are used:

* Golden values (``data/hfs_golden.json``) were captured from the implementation
  before the performance/clarity refactor. They guard against accidental changes
  of the calculated positions, amplitudes and spectra.
* Analytic references (hyperfine energy formulas, intensity sum rule, closed-form
  peak shapes) check the physics independently of the implementation.
* Behavioural tests for the saturation and sidepeak options.
"""

import copy
import json
import pickle
from math import factorial
from pathlib import Path

import numpy as np
import pytest
from scipy.special import erf, voigt_profile

import satlas2
from satlas2.models.hfsModel import HFS, triangle_condition

GOLDEN_FILE = Path(__file__).parent / "data" / "hfs_golden.json"
XS = np.array([-1500.0, -700.0, -100.0, 0.0, 50.0, 300.0, 900.0, 2000.0])

BASE = dict(
    I=4.5,
    J=[3.5, 4.5],
    A=[500, 300],
    B=[50, 30],
    C=[1, 2],
    df=10,
    fwhmg=40,
    fwhml=20,
    scale=1.5,
)
SMALL = dict(I=1.5, J=[0.5, 1.5], A=[100, 50], B=[0, 20], fwhmg=30, fwhml=10)
CASES = {
    "voigt": BASE,
    "gaussian": {**BASE, "peak": "gaussian"},
    "lorentzian": {**BASE, "peak": "lorentzian"},
    "small": SMALL,
    "integer_spin": dict(I=1, J=[1, 2], A=[80, 40], B=[5, 10], fwhmg=30, fwhml=10),
    "order2": dict(I=3.5, J=[1.5, 1.5], A=[10, 10], B=[1, 2], order=2, fwhmg=5, fwhml=3),
    "saturation": {**BASE, "racah": False, "use_saturation": True, "saturation": 2.0},
}


def build(name):
    return HFS(**CASES[name])


def parse_F(label):
    """Convert a line label part such as '3' or '7_2' to the value of F."""
    return float(label.split("_")[0]) / 2 if "_" in label else float(label)


def split_line(line):
    lower, upper = line.split("to")
    return parse_F(lower), parse_F(upper)


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN_FILE.read_text())


# ----------------------------------------------------------------------------
# Golden values
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES)
def test_golden_lines(case, golden):
    model = build(case)
    assert model.lines == golden[case]["lines"]


@pytest.mark.parametrize("case", CASES)
def test_golden_positions(case, golden):
    model = build(case)
    assert model.pos() == pytest.approx(golden[case]["pos"], rel=1e-10, abs=1e-9)


@pytest.mark.parametrize("case", CASES)
def test_golden_amplitudes(case, golden):
    model = build(case)
    amps = [model.params["Amp" + line].value for line in model.lines]
    assert amps == pytest.approx(golden[case]["amps"], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_golden_spectrum(case, golden):
    model = build(case)
    assert model.f(XS) == pytest.approx(golden[case]["spectrum"], rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("case", ["voigt", "small", "order2"])
def test_golden_parameter_names(case, golden):
    assert sorted(build(case).params) == golden[case]["params"]


# ----------------------------------------------------------------------------
# Analytic references
# ----------------------------------------------------------------------------
def coupling_K(F, I, J):
    return F * (F + 1) - I * (I + 1) - J * (J + 1)


def analytic_shift(F, I, J, A, B):
    """Energy shift of a hyperfine level due to the A and B constants."""
    K = coupling_K(F, I, J)
    shift = A * K / 2
    if I >= 1 and J >= 1:
        shift += B * (0.75 * K * (K + 1) - I * (I + 1) * J * (J + 1)) / (
            2 * I * (2 * I - 1) * J * (2 * J - 1)
        )
    return shift


@pytest.mark.parametrize(
    "I, J",
    [(1.5, [0.5, 1.5]), (2.5, [2.5, 3.5]), (1, [1, 2]), (4.5, [3.5, 4.5])],
)
def test_positions_match_analytic_energies(I, J):
    A, B, df = [120.0, 45.0], [12.0, 7.0], 33.0
    model = HFS(I, J, A=A, B=B, df=df)
    for line, position in zip(model.lines, model.pos()):
        F1, F2 = split_line(line)
        expected = (
            df
            + analytic_shift(F2, I, J[1], A[1], B[1])
            - analytic_shift(F1, I, J[0], A[0], B[0])
        )
        assert position == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("order", [1, 2])
def test_intensity_sum_rule(order):
    """The summed strength out of each lower level is proportional to 2F+1."""
    J = [1.5, 1.5] if order == 2 else [1.5, 2.5]
    model = HFS(3.5, J, order=order)
    strength = {}
    for line in model.lines:
        F1, _ = split_line(line)
        strength[F1] = strength.get(F1, 0.0) + model.params["Amp" + line].value
    # Amp = (2F1+1)(2F2+1)6j^2 (normalised); sum_F2 (2F2+1)6j^2 is constant
    ratios = [total / (2 * F1 + 1) for F1, total in strength.items()]
    assert ratios == pytest.approx([ratios[0]] * len(ratios), rel=1e-10)


def test_racah_amplitudes_normalised_and_fixed():
    model = HFS(**SMALL)
    amps = [model.params["Amp" + line].value for line in model.lines]
    assert max(amps) == pytest.approx(1.0)
    assert not any(model.params["Amp" + line].vary for line in model.lines)
    assert model.params["scale"].vary


def test_amplitudes_free_without_racah():
    model = HFS(**{**SMALL, "racah": False})
    assert all(model.params["Amp" + line].vary for line in model.lines)
    assert not model.params["scale"].vary


def test_order_two_lines_are_allowed_transitions():
    model = HFS(3.5, [1.5, 1.5], order=2)
    for line in model.lines:
        F1, F2 = split_line(line)
        assert abs(F1 - F2) <= 2 <= F1 + F2


@pytest.mark.parametrize(
    "s1, s2, order, expected",
    [(1, 1, 1, True), (1, 1, 3, False), (0.5, 0.5, 1, True), (0.5, 1, 1, False)],
)
def test_triangle_condition(s1, s2, order, expected):
    assert triangle_condition(s1, s2, order) is expected


def test_forbidden_order_raises():
    with pytest.raises(ValueError):
        HFS(1.5, [0.5, 0.5], order=2)


def test_prefunc_is_applied_once():
    plain = HFS(**SMALL)
    shifted = HFS(**SMALL, prefunc=lambda x: 2 * x + 5)
    assert shifted.f(XS) == pytest.approx(plain.f(2 * XS + 5))


def test_scalar_input():
    model = HFS(**SMALL)
    assert model.f(25.0) == pytest.approx(model.f(np.array([25.0])))


# ----------------------------------------------------------------------------
# Peak shapes
# ----------------------------------------------------------------------------
def single_peak_model(peak, **peak_kwargs):
    """A two-level system without hyperfine structure has exactly one line."""
    return HFS(0, [0.5, 0.5], df=0, fwhmg=30, fwhml=12, peak=peak, **peak_kwargs)


def test_gaussian_half_maximum_at_half_fwhm():
    model = single_peak_model("gaussian")
    assert model.f(np.array([15.0]))[0] == pytest.approx(0.5)
    assert model.f(np.array([0.0]))[0] == pytest.approx(1.0)


def test_lorentzian_half_maximum_at_half_fwhm():
    model = single_peak_model("lorentzian")
    assert model.f(np.array([6.0]))[0] == pytest.approx(0.5)
    assert model.f(np.array([0.0]))[0] == pytest.approx(1.0)


def test_voigt_matches_definition():
    model = single_peak_model("voigt")
    sigma, gamma = 30 / (2 * np.sqrt(2 * np.log(2))), 12 / 2
    x = np.linspace(-80, 80, 21)
    expected = voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma)
    assert model.f(x) == pytest.approx(expected)


def test_skewvoigt_matches_definition():
    skew = 0.4
    model = single_peak_model("skewvoigt", peak_kwargs={"skew": {"value": skew}})
    sigma, gamma = 30 / (2 * np.sqrt(2 * np.log(2))), 12 / 2
    x = np.linspace(-80, 80, 21)
    expected = (
        voigt_profile(x, sigma, gamma)
        / voigt_profile(0, sigma, gamma)
        * (1 + erf(skew * x / (sigma * np.sqrt(2))))
    )
    assert model.f(x) == pytest.approx(expected)


def test_skewvoigt_matches_skewed_voigt_model():
    from satlas2.models import SkewedVoigt

    model = single_peak_model("skewvoigt", peak_kwargs={"skew": {"value": 0.4}})
    reference = SkewedVoigt(A=1.0, mu=0.0, FWHMG=30, FWHML=12, skew=0.4)
    x = np.linspace(-80, 80, 21)
    assert model.f(x) == pytest.approx(reference.f(x))


def test_skewvoigt_without_skew_is_voigt():
    skewed = single_peak_model("skewvoigt", peak_kwargs={"skew": {"value": 0.0}})
    voigt = single_peak_model("voigt")
    x = np.linspace(-80, 80, 21)
    assert skewed.f(x) == pytest.approx(voigt.f(x))


def test_custom_peak_not_implemented():
    model = single_peak_model("custom")
    with pytest.raises(NotImplementedError):
        model.f(XS)


def test_calculate_fwhm_propagates_uncertainty():
    model = HFS(**SMALL)
    G, L = model.params["FWHMG"].value, model.params["FWHML"].value
    model.params["FWHMG"].unc = 1.0
    model.params["FWHML"].unc = 2.0

    def fwhm(g, l):
        return 0.5346 * l + np.sqrt(0.2166 * l**2 + g**2)

    h = 1e-6
    dG = (fwhm(G + h, L) - fwhm(G - h, L)) / (2 * h)
    dL = (fwhm(G, L + h) - fwhm(G, L - h)) / (2 * h)
    value, uncertainty = model.calculateFWHM()
    assert value == pytest.approx(fwhm(G, L))
    assert uncertainty == pytest.approx(np.hypot(dG * 1.0, dL * 2.0), rel=1e-6)


# ----------------------------------------------------------------------------
# Saturation
# ----------------------------------------------------------------------------
def saturation_model(saturation, **kwargs):
    return HFS(
        **{**SMALL, "racah": False, "use_saturation": True, "saturation": saturation},
        **kwargs,
    )


def test_saturation_parameter_only_present_when_used():
    assert "Saturation" not in HFS(**SMALL).params
    assert "Saturation" not in HFS(**{**SMALL, "racah": False}).params
    model = saturation_model(1.5)
    assert model.params["Saturation"].value == 1.5
    assert model.params["Saturation"].vary
    assert model.params["Saturation"].min == 0


def test_saturation_fixes_amplitudes_and_frees_scale():
    model = saturation_model(1.5)
    assert model.params["scale"].vary
    assert not any(model.params["Amp" + line].vary for line in model.lines)


def test_racah_and_saturation_are_exclusive():
    with pytest.raises(ValueError):
        HFS(**SMALL, use_saturation=True)


def test_zero_saturation_is_racah():
    racah = HFS(**SMALL)
    assert saturation_model(0.0).f(XS) == pytest.approx(racah.f(XS))


def test_strong_saturation_gives_statistical_weights():
    model = saturation_model(1e6)
    weights = np.array([2 * split_line(line)[0] + 1 for line in model.lines])
    amps = np.array([model.params["Amp" + line].value for line in model.lines])
    assert amps == pytest.approx(weights / weights.max(), rel=1e-4)


def test_intermediate_saturation_mapping():
    s = 2.5
    model = saturation_model(s)
    racah = HFS(**SMALL)
    rac = np.array([racah.params["Amp" + line].value for line in model.lines])
    sat = np.array([2 * split_line(line)[0] + 1 for line in model.lines], dtype=float)
    sat /= sat.max()
    expected = -sat * np.expm1(-rac * s / sat)
    expected /= expected.max()
    amps = [model.params["Amp" + line].value for line in model.lines]
    assert amps == pytest.approx(expected)


def saturation_amplitudes(model):
    return np.array([model.params["Amp" + line].value for line in model.lines])


def test_spectrum_follows_saturation_parameter():
    model = saturation_model(0.5)
    model.params["Saturation"].value = 3.0
    assert model.f(XS) == pytest.approx(saturation_model(3.0).f(XS))


def test_amplitudes_are_recalculated_when_saturation_changes():
    model = saturation_model(0.5)
    before = saturation_amplitudes(model)
    model.params["Saturation"].value = 3.0
    assert not np.allclose(before, saturation_amplitudes(model))
    assert saturation_amplitudes(model) == pytest.approx(
        saturation_amplitudes(saturation_model(3.0))
    )
    model.params["Saturation"].value = 0.0
    assert saturation_amplitudes(model) == pytest.approx(
        saturation_amplitudes(HFS(**SMALL))
    )


@pytest.mark.parametrize("copier", [copy.deepcopy, lambda m: pickle.loads(pickle.dumps(m))])
def test_saturation_survives_copying(copier):
    """Models are copied when fitting in parallel."""
    model = copier(saturation_model(0.5))
    model.params["Saturation"].value = 3.0
    assert saturation_amplitudes(model) == pytest.approx(
        saturation_amplitudes(saturation_model(3.0))
    )
    assert model.f(XS) == pytest.approx(saturation_model(3.0).f(XS))


def test_fit_reports_amplitudes_at_fitted_saturation():
    truth = saturation_model(5.0)
    x = np.linspace(-400, 400, 300)
    model = saturation_model(0.5)
    source = satlas2.Source(x, truth.f(x), yerr=np.full_like(x, 0.01), name="s")
    source.addModel(model)
    fitter = satlas2.Fitter()
    fitter.addSource(source)
    fitter.fit()

    fitted = model.params["Saturation"].value
    assert fitted == pytest.approx(5.0, rel=1e-3)
    expected = saturation_amplitudes(saturation_model(fitted))
    assert saturation_amplitudes(model) == pytest.approx(expected)

    frame = fitter.createResultDataframe()
    reported = [
        frame.loc[frame["Parameter"] == "Amp" + line, "Value"].item()
        for line in model.lines
    ]
    assert reported == pytest.approx(expected)
    assert not frame.loc[frame["Parameter"].str.startswith("Amp"), "Vary"].any()


# ----------------------------------------------------------------------------
# Sidepeaks
# ----------------------------------------------------------------------------
def sidepeak_model(**kwargs):
    return HFS(**{**SMALL, "scale": 2.0, "N": 2, "offset": -40.0, "poisson": 0.3}, **kwargs)


def reference_sidepeak_spectrum(model, x):
    """Sum over lines and sidepeaks, evaluated with explicit loops."""
    scale = model.params["scale"].value
    N = int(model.params["N"].value)
    offset = model.params["Offset"].value
    poisson = model.params["Poisson"].value
    plain = HFS(**{**SMALL})
    plain.params["FWHMG"].value = model.params["FWHMG"].value
    plain.params["FWHML"].value = model.params["FWHML"].value
    result = np.zeros_like(x)
    for line, pos in zip(model.lines, model.pos()):
        amp = model.params["Amp" + line].value
        for i in range(N + 1):
            result += amp * plain.peak(x - i * offset - pos) * poisson**i / factorial(i)
    return scale * result


def test_sidepeaks_without_poisson_equal_unshifted():
    model = sidepeak_model()
    model.params["Poisson"].value = 0.0
    unshifted = HFS(**{**SMALL, "scale": 2.0})
    assert model.f(XS) == pytest.approx(unshifted.f(XS))


def test_sidepeak_spectrum_matches_reference():
    model = sidepeak_model()
    assert model.f(XS) == pytest.approx(reference_sidepeak_spectrum(model, XS))


def test_sidepeak_spectrum_with_prefunc():
    prefunc = lambda x: 0.5 * x + 20  # noqa: E731
    model = sidepeak_model(prefunc=prefunc)
    assert model.f(XS) == pytest.approx(reference_sidepeak_spectrum(model, prefunc(XS)))


def test_sidepeak_parameters_present():
    model = sidepeak_model()
    assert {"N", "Offset", "Poisson"} <= set(model.params)
    assert not model.params["N"].vary
