import numpy as np
import pytest

import satlas2
from satlas2.utilities import generateSpectrum, poissonInterval, weightedAverage


def reference_weighted_average(x, sigma):
    """Weighted average of a 1D array, written out term by term."""
    w = 1 / sigma**2
    mean = np.sum(w * x) / np.sum(w)
    stat = 1 / np.sum(w)
    scatter = np.sum(((x - mean) / sigma) ** 2) / ((len(x) - 1) * np.sum(w))
    return mean, np.sqrt(max(stat, scatter))


def test_weighted_average_statistical_uncertainty():
    mean, uncertainty = weightedAverage([1.0, 1.1, 0.9], [1.0, 1.0, 1.0])
    assert mean == pytest.approx(1.0)
    assert uncertainty == pytest.approx(1 / np.sqrt(3))


def test_weighted_average_scatter_uncertainty():
    x, sigma = np.array([1.0, 5.0, 2.0, 8.0]), np.array([0.1, 0.2, 0.1, 0.3])
    assert weightedAverage(x, sigma) == pytest.approx(reference_weighted_average(x, sigma))


@pytest.mark.parametrize("axis", [0, 1])
def test_weighted_average_along_an_axis(axis):
    x = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 9.0]])
    sigma = np.array([[1.0, 0.5, 1.0, 2.0], [0.2, 0.3, 0.2, 1.0]])
    mean, uncertainty = weightedAverage(x, sigma, axis=axis)
    rows = zip(np.moveaxis(x, axis, -1), np.moveaxis(sigma, axis, -1))
    expected = np.array([reference_weighted_average(xi, si) for xi, si in rows])
    assert mean == pytest.approx(expected[:, 0])
    assert uncertainty == pytest.approx(expected[:, 1])


def test_poisson_interval_known_values():
    """Garwood intervals, 1 sigma: closed forms for 0 and 1 counts, and the
    tabulated values of Gehrels (1986) for 10 counts."""
    low, high = poissonInterval(np.array([0, 1, 10]))
    tail = 1 - 0.8413447460685429
    assert low[:2] == pytest.approx([0.0, -np.log(1 - tail)])
    assert high[0] == pytest.approx(-np.log(tail))
    assert high[1] == pytest.approx(3.300, abs=1e-3)
    assert (low[2], high[2]) == pytest.approx((6.891, 14.27), abs=5e-3)


def test_poisson_interval_alpha_overrides_sigma():
    assert poissonInterval(5, sigma=3, alpha=0.3173105) == pytest.approx(
        poissonInterval(5, sigma=1), rel=1e-6
    )


def test_poisson_interval_for_a_known_mean():
    low, high = poissonInterval(np.array([4.0, 100.0]), is_mean=True)
    assert low == pytest.approx([2.0, 90.0])
    assert high == pytest.approx([6.0, 110.0])


def test_generate_spectrum_is_reproducible_with_a_seeded_generator():
    models = [satlas2.Polynomial([0.0, 50.0], name="a"), satlas2.Polynomial([10.0], name="b")]
    x = np.linspace(0, 1, 5)
    first = generateSpectrum(models, x, np.random.default_rng(1).poisson)
    second = generateSpectrum(models, x, np.random.default_rng(1).poisson)
    assert np.array_equal(first, second)


def test_generate_spectrum_of_a_single_model():
    model = satlas2.Polynomial([2.0, 1.0], name="line")
    x = np.linspace(0, 1, 5)
    assert generateSpectrum(model, x, generator=lambda f: f) == pytest.approx(2 * x + 1)
    assert generateSpectrum([model, model], x, generator=lambda f: f) == pytest.approx(
        4 * x + 2
    )
