"""Statistical tests of the reported uncertainties.

Every fitting method is applied to many simulated datasets with known true
parameters. The reported uncertainty must match the scatter of the best fits
over the datasets, and about 68% of the 1-sigma intervals must contain the
true value. These tests catch errors that give plausible numbers, such as a
covariance off by a constant factor, which no single fit reveals.

They take about a minute and run in the weekly ``Statistical tests``
workflow, or with ``pytest -m slow``.
"""

import numpy as np
import pytest

from satlas2 import Fitter, Polynomial, Source

pytestmark = pytest.mark.slow

REPLICAS = 200
X = np.linspace(-5, 5, 40)
INTERCEPT, SLOPE, SIGMA = 1.5, 0.0, 0.2
NAMES = ["s___bg___p0", "s___bg___p1"]


def line_fitter(y, yerr):
    source = Source(X, y, yerr=yerr, name="s")
    # the slope starts at exactly 0, as the truth
    source.addModel(Polynomial([0.0, 1.0], name="bg"))
    fitter = Fitter()
    fitter.addSource(source)
    return fitter


def gaussian_data(rng):
    return INTERCEPT + SLOPE * X + rng.normal(0, SIGMA, X.size)


def poisson_data(rng):
    return rng.poisson(20 + 2 * X).astype(float)


def collect(fit, make_data, replicas=REPLICAS, seed=0):
    """Best values and reported uncertainties over simulated datasets."""
    rng = np.random.default_rng(seed)
    values, errors = [], []
    for _ in range(replicas):
        fitter = fit(make_data(rng))
        values.append([fitter.result.params[n].value for n in NAMES])
        errors.append([fitter.result.params[n].stderr for n in NAMES])
    return np.array(values), np.array(errors, dtype=float)


def check(values, errors, truth, tolerance):
    """The mean reported uncertainty matches the scatter of the values, and
    the 1-sigma intervals cover the truth about 68% of the time."""
    scatter = values.std(axis=0, ddof=1)
    ratio = errors.mean(axis=0) / scatter
    assert ratio == pytest.approx(np.ones_like(ratio), abs=tolerance), ratio
    covered = np.mean(np.abs(values - truth) < errors, axis=0)
    assert covered == pytest.approx(np.full_like(covered, 0.683), abs=0.1), covered


def chisquare(y):
    fitter = line_fitter(y, np.full(X.size, SIGMA))
    fitter.fit(scale_covar=False)
    return fitter


def chisquare_scaled(y):
    fitter = line_fitter(y, np.full(X.size, SIGMA))
    fitter.fit()
    return fitter


def gaussian_likelihood(y):
    fitter = line_fitter(y, np.full(X.size, SIGMA))
    fitter.fit(llh=True, llh_method="gaussian")
    return fitter


def poisson_likelihood(y):
    fitter = line_fitter(y, np.sqrt)
    fitter.fit(llh=True, llh_method="poisson")
    return fitter


def gaussian_likelihood_model_uncertainty(y):
    fitter = line_fitter(y, np.sqrt)
    fitter.fit(llh=True, llh_method="gaussian")
    return fitter


@pytest.mark.parametrize(
    "fit", [chisquare, chisquare_scaled, gaussian_likelihood], ids=lambda f: f.__name__
)
def test_gaussian_data(fit):
    values, errors = collect(fit, gaussian_data)
    check(values, errors, [INTERCEPT, SLOPE], tolerance=0.12)


@pytest.mark.parametrize(
    "fit",
    [poisson_likelihood, gaussian_likelihood_model_uncertainty],
    ids=lambda f: f.__name__,
)
def test_poisson_data(fit):
    values, errors = collect(fit, poisson_data)
    check(values, errors, [20, 2], tolerance=0.12)
    # the estimates are unbiased: within 4 standard errors of the mean
    bias = values.mean(axis=0) - [20, 2]
    assert np.all(np.abs(bias) < 4 * values.std(axis=0) / np.sqrt(len(values)))


def random_walk(y):
    """A walk started at the truth, without a previous fit, so the slope
    starts at exactly 0."""
    fitter = line_fitter(y, np.full(X.size, SIGMA))
    fitter.sources[0][1].models[0][1].params["p0"].value = INTERCEPT
    fitter.fit(method="emcee", nwalkers=16, steps=400, progress=False)
    return fitter


def test_random_walk():
    values, errors = collect(random_walk, gaussian_data, replicas=40)
    check(values, errors, [INTERCEPT, SLOPE], tolerance=0.3)
