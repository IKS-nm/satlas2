"""Tests for the Fitter, Source and Model classes.

Fitted values are compared to closed-form solutions (weighted linear least
squares, weighted means), and likelihoods to scipy.stats. Behaviour that is
known to be wrong is marked ``xfail(strict=True)`` and names the fix.
"""

import copy
import inspect

import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import poisson

import satlas2
from satlas2 import Fitter, Model, Polynomial, Source


def line_source(name, intercept, slope, sigma=0.2, seed=0, n=40):
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, n)
    y = intercept + slope * x + rng.normal(0, sigma, n)
    source = Source(x, y, yerr=np.full(n, sigma), name=name)
    # Polynomial takes the coefficients from the highest order down: p1*x + p0
    source.addModel(Polynomial([1.0, 0.0], name="bg"))
    return source


def fitter_with(*sources):
    fitter = Fitter()
    for source in sources:
        fitter.addSource(source)
    return fitter


def weighted_lstsq(design, y, sigma):
    """Closed-form weighted linear least squares: values, covariance, chisquare."""
    w = 1 / sigma**2
    normal = design.T @ (design * w[:, None])
    cov = np.linalg.inv(normal)
    beta = cov @ design.T @ (w * y)
    chisqr = np.sum(w * (y - design @ beta) ** 2)
    return beta, cov, chisqr


def result_value(fitter, name):
    return fitter.result.params[name].value


def result_stderr(fitter, name):
    return fitter.result.params[name].stderr


# ----------------------------------------------------------------------------
# Chisquare fitting
# ----------------------------------------------------------------------------
def test_chisquare_fit_matches_linear_least_squares():
    source = line_source("s", intercept=1.5, slope=-0.7)
    fitter = fitter_with(source)
    fitter.fit()

    design = np.column_stack([np.ones_like(source.x), source.x])
    beta, cov, chisqr = weighted_lstsq(design, source.y, source.yerr_data)
    redchi = chisqr / (len(source.x) - 2)

    assert result_value(fitter, "s___bg___p0") == pytest.approx(beta[0], rel=1e-6)
    assert result_value(fitter, "s___bg___p1") == pytest.approx(beta[1], rel=1e-6)
    assert fitter.result.chisqr == pytest.approx(chisqr, rel=1e-6)
    # uncertainties are scaled by the root of the reduced chisquare by default
    assert result_stderr(fitter, "s___bg___p0") == pytest.approx(
        np.sqrt(cov[0, 0] * redchi), rel=1e-4
    )
    assert result_stderr(fitter, "s___bg___p1") == pytest.approx(
        np.sqrt(cov[1, 1] * redchi), rel=1e-4
    )


def test_unscaled_uncertainties():
    source = line_source("s", intercept=1.5, slope=-0.7)
    fitter = fitter_with(source)
    fitter.fit(scale_covar=False)
    design = np.column_stack([np.ones_like(source.x), source.x])
    _, cov, _ = weighted_lstsq(design, source.y, source.yerr_data)
    assert result_stderr(fitter, "s___bg___p1") == pytest.approx(
        np.sqrt(cov[1, 1]), rel=1e-4
    )


def test_fit_updates_model_parameters_and_uncertainties():
    source = line_source("s", intercept=1.5, slope=-0.7)
    fitter = fitter_with(source)
    fitter.fit()
    model = source.models[0][1]
    assert model.params["p1"].value == result_value(fitter, "s___bg___p1")
    assert model.params["p1"].unc == result_stderr(fitter, "s___bg___p1")
    assert "p0" in model.params["p1"].correl


def test_fixed_parameter_is_not_fitted():
    source = line_source("s", intercept=1.5, slope=-0.7)
    source.models[0][1].params["p0"].vary = False
    source.models[0][1].params["p0"].value = 1.5
    fitter = fitter_with(source)
    fitter.fit()
    assert result_value(fitter, "s___bg___p0") == 1.5
    beta, _, _ = weighted_lstsq(
        source.x[:, None], source.y - 1.5, source.yerr_data
    )
    assert result_value(fitter, "s___bg___p1") == pytest.approx(beta[0], rel=1e-6)


def test_revert_fit_restores_initial_values():
    source = line_source("s", intercept=1.5, slope=-0.7)
    fitter = fitter_with(source)
    fitter.fit()
    fitter.revertFit()
    model = source.models[0][1]
    assert model.params["p1"].value == 1.0
    assert model.params["p0"].value == 0.0


def test_result_dataframe():
    fitter = fitter_with(line_source("s", intercept=1.5, slope=-0.7))
    fitter.fit()
    frame = fitter.createResultDataframe()
    assert list(frame.columns) == [
        "Source",
        "Model",
        "Parameter",
        "Value",
        "Stderr",
        "Minimum",
        "Maximum",
        "Expression",
        "Vary",
    ]
    assert set(frame["Parameter"]) == {"p0", "p1"}
    assert set(frame["Source"]) == {"s"}


# ----------------------------------------------------------------------------
# Linking parameters
# ----------------------------------------------------------------------------
def joint_intercept_solution(sources):
    """Least squares with a common intercept and a slope per source."""
    x = np.concatenate([s.x for s in sources])
    y = np.concatenate([s.y for s in sources])
    sigma = np.concatenate([s.yerr_data for s in sources])
    columns = [np.ones_like(x)]
    for i, s in enumerate(sources):
        mask = np.concatenate(
            [np.full_like(o.x, i == j, dtype=bool) for j, o in enumerate(sources)]
        )
        columns.append(np.where(mask, x, 0))
    beta, _, _ = weighted_lstsq(np.column_stack(columns), y, sigma)
    return beta


def two_sources():
    return (
        line_source("s1", intercept=2.0, slope=0.5, seed=1),
        line_source("s2", intercept=2.2, slope=-0.3, seed=2),
    )


@pytest.mark.parametrize("share", ["shareParams", "shareModelParams"])
def test_shared_intercept_matches_joint_least_squares(share):
    sources = two_sources()
    fitter = fitter_with(*sources)
    getattr(fitter, share)(["p0"])
    fitter.fit()
    beta = joint_intercept_solution(sources)
    assert result_value(fitter, "s1___bg___p0") == pytest.approx(beta[0], rel=1e-6)
    assert result_value(fitter, "s2___bg___p0") == pytest.approx(beta[0], rel=1e-6)
    assert result_value(fitter, "s1___bg___p1") == pytest.approx(beta[1], rel=1e-6)
    assert result_value(fitter, "s2___bg___p1") == pytest.approx(beta[2], rel=1e-6)
    assert fitter.result.nvarys == 3


def test_share_model_params_only_links_models_with_the_same_name():
    s1, s2 = two_sources()
    s2.models[0][1].name = "other"
    s2.models = [("other", s2.models[0][1])]
    fitter = fitter_with(s1, s2)
    fitter.shareModelParams(["p0"])
    fitter.fit()
    assert fitter.result.params["s2___other___p0"].expr is None
    assert fitter.result.nvarys == 4


@pytest.mark.parametrize("share", ["shareParams", "shareModelParams"])
def test_share_accepts_a_single_name(share):
    fitter = Fitter()
    getattr(fitter, share)("p0")
    assert fitter.share + fitter.shareModel == ["p0"]
    getattr(fitter, "remove" + share[0].upper() + share[1:])("p0")
    assert fitter.share + fitter.shareModel == []


def test_correlations_are_kept_per_model():
    """Models whose names start with the same text must not mix correlations."""
    rng = np.random.default_rng(0)
    x = np.linspace(-5, 5, 40)
    y = 1 + 0.5 * x + 0.1 * x**2 + rng.normal(0, 0.1, 40)
    source = Source(x, y, np.full(40, 0.1), name="s")
    source.addModel(Polynomial([0.4, 0.9], name="bg"))
    source.addModel(Polynomial([0.05, 0, 0], name="bg2"))
    fitter = fitter_with(source)
    fitter.fit()
    correl = fitter.result.params["s___bg___p1"].correl
    stored = source.models[0][1].params["p1"].correl
    assert stored == pytest.approx({"p0": correl["s___bg___p0"]})


def test_remove_shared_parameters():
    sources = two_sources()
    fitter = fitter_with(*sources)
    fitter.shareParams(["p0"])
    fitter.removeShareParams("p0")
    fitter.fit()
    assert fitter.result.nvarys == 4


def test_expression_links_parameters():
    sources = two_sources()
    fitter = fitter_with(*sources)
    fitter.setExpr("s2___bg___p1", "-0.6 * s1___bg___p1")
    fitter.fit()
    assert result_value(fitter, "s2___bg___p1") == pytest.approx(
        -0.6 * result_value(fitter, "s1___bg___p1")
    )
    assert fitter.result.nvarys == 3


def test_remove_expression():
    fitter = fitter_with(*two_sources())
    fitter.setExpr(["s2___bg___p1"], ["-0.6 * s1___bg___p1"])
    fitter.removeExpr("s2___bg___p1")
    fitter.removeExpr("does_not_exist")
    fitter.fit()
    assert fitter.result.nvarys == 4


def test_expression_takes_priority_over_sharing():
    fitter = fitter_with(*two_sources())
    fitter.shareParams(["p0"])
    fitter.setExpr("s2___bg___p0", "s1___bg___p0 + 1")
    fitter.fit()
    assert result_value(fitter, "s2___bg___p0") == pytest.approx(
        result_value(fitter, "s1___bg___p0") + 1
    )


# ----------------------------------------------------------------------------
# Priors
# ----------------------------------------------------------------------------
def constant_source(values, sigma, name="s"):
    x = np.arange(len(values), dtype=float)
    source = Source(x, np.asarray(values, float), yerr=np.full(len(values), sigma), name=name)
    source.addModel(Polynomial([0.0], name="c"))
    return source


def test_gaussian_prior_is_a_weighted_mean():
    values, sigma = [1.0, 2.0, 3.0, 2.5], 0.5
    fitter = fitter_with(constant_source(values, sigma))
    fitter.setParamPrior("s", "c", "p0", 4.0, 0.25)
    fitter.fit()
    weights = np.array([1 / sigma**2] * len(values) + [1 / 0.25**2])
    expected = np.sum(weights * np.array(values + [4.0])) / weights.sum()
    assert result_value(fitter, "s___c___p0") == pytest.approx(expected, rel=1e-6)


def test_remove_priors():
    values = [1.0, 2.0, 3.0]
    fitter = fitter_with(constant_source(values, 0.5))
    fitter.setParamPrior("s", "c", "p0", 4.0, 0.25)
    fitter.removeParamPrior("s", "c", "p0")
    fitter.fit()
    assert result_value(fitter, "s___c___p0") == pytest.approx(2.0, rel=1e-6)
    fitter.setParamPrior("s", "c", "p0", 4.0, 0.25)
    fitter.removeAllPriors()
    assert fitter.priors == {}


# ----------------------------------------------------------------------------
# Residuals and likelihoods
# ----------------------------------------------------------------------------
def prepared(fitter):
    """Set up the fitter as fit() does, without fitting."""
    fitter._prepareFit()
    return fitter


def counting_source(name="s"):
    x = np.linspace(0, 1, 6)
    y = np.array([3.0, 5.0, 4.0, 7.0, 6.0, 9.0])
    source = Source(x, y, yerr=np.sqrt, name=name)
    source.addModel(Polynomial([4.0, 3.0], name="bg"))
    return source


def test_callable_yerr_is_applied_to_the_model():
    source = counting_source()
    fitter = prepared(fitter_with(source))
    f = source.f()
    assert fitter.resid() == pytest.approx((f - source.y) / np.sqrt(f))


def test_combined_mode_residuals():
    source = counting_source()
    fitter = prepared(fitter_with(source))
    fitter.mode = "combined"
    f, y = source.f(), source.y
    assert fitter.resid() == pytest.approx((f - y) / np.sqrt(3 / (1 / y + 2 / f)))


def test_xerr_is_propagated_through_the_derivative():
    x = np.linspace(0, 2, 5)
    source = Source(x, 3 * x, yerr=np.full(5, 0.3), name="s", xerr=np.full(5, 0.1))
    source.addModel(Polynomial([3.0, 0.0], name="bg"))
    assert source.yerr() == pytest.approx(np.full(5, np.hypot(0.3, 3 * 0.1)))


def test_gaussian_loglikelihood():
    source = counting_source()
    fitter = prepared(fitter_with(source))
    llh = fitter.llh(fitter.lmpars, method="gaussian", emcee=True)
    f = source.f()
    assert llh == pytest.approx(-0.5 * np.sum((f - source.y) ** 2 / f))


def test_poisson_loglikelihood_matches_scipy():
    source = counting_source()
    fitter = prepared(fitter_with(source))
    llh = fitter.llh(fitter.lmpars, method="poisson", emcee=True)
    # the constant log(y!) is left out of the likelihood
    expected = poisson.logpmf(source.y, source.f()).sum() + gammaln(source.y + 1).sum()
    assert llh == pytest.approx(expected)


def test_negative_loglikelihood_for_minimisers():
    source = counting_source()
    fitter = prepared(fitter_with(source))
    per_point = fitter.llh(fitter.lmpars, method="poisson")
    assert np.sum(per_point) == pytest.approx(
        -fitter.llh(fitter.lmpars, method="poisson", emcee=True)
    )


def poisson_llh_with_priors(names):
    source = counting_source()
    fitter = prepared(fitter_with(source))
    for name in names:
        fitter.setParamPrior("s", "bg", name, 1.0, 0.5)
    return fitter.poissonLlh(), len(source.x)


def test_poisson_loglikelihood_includes_priors():
    llh, n_data = poisson_llh_with_priors(["p0", "p1"])
    assert len(llh) == n_data + 2


def test_poisson_loglikelihood_includes_a_single_prior():
    llh, n_data = poisson_llh_with_priors(["p0"])
    assert len(llh) == n_data + 1


def test_poisson_likelihood_fit_recovers_mean():
    rng = np.random.default_rng(3)
    y = rng.poisson(50, 200).astype(float)
    source = Source(np.arange(200.0), y, yerr=np.sqrt, name="s")
    source.addModel(Polynomial([10.0], name="c"))
    fitter = fitter_with(source)
    fitter.fit(llh=True, llh_method="poisson")
    # the maximum likelihood estimate of a Poisson mean is the sample mean
    assert result_value(fitter, "s___c___p0") == pytest.approx(y.mean(), rel=1e-4)


# ----------------------------------------------------------------------------
# Sources and models
# ----------------------------------------------------------------------------
class Offset(Model):
    def __init__(self, value, name="offset", prefunc=None):
        super().__init__(name, prefunc=prefunc)
        self.params = {"value": satlas2.Parameter(value=value)}

    def f(self, x):
        return np.full_like(self.transform(x), self.params["value"].value, dtype=float)


def test_source_sums_its_models():
    x = np.linspace(0, 1, 4)
    source = Source(x, x, yerr=np.ones(4), name="s")
    source.addModel(Polynomial([2.0, 0.0], name="line"))
    source.addModel(Offset(5.0))
    assert source.f() == pytest.approx(2 * x + 5)
    assert source.evaluate(np.array([3.0])) == pytest.approx([11.0])


def test_prefunc_transforms_the_input():
    model = Polynomial([1.0, 0.0], name="line", prefunc=lambda x: 10 * x)
    assert model.f(np.array([1.0, 2.0])) == pytest.approx([10.0, 20.0])


def test_set_transform_replaces_the_transformation():
    model = Polynomial([1.0, 0.0], name="line", prefunc=lambda x: 10 * x)
    model.f(np.array([1.0, 2.0]))
    model.setTransform(lambda x: x - 1)
    assert model.f(np.array([1.0, 2.0])) == pytest.approx([0.0, 1.0])
    model.prefunc = lambda x: x + 1
    assert model.f(np.array([1.0, 2.0])) == pytest.approx([2.0, 3.0])


def test_transform_distinguishes_inputs_with_equal_bytes():
    model = Polynomial([1.0, 0.0], name="line", prefunc=lambda x: 2 * x)
    x = np.arange(4.0)
    assert model.f(x) == pytest.approx(2 * x)
    assert model.f(x.reshape(2, 2)).shape == (2, 2)
    assert model.f(3.0) == pytest.approx(6.0)


def test_base_model_is_abstract():
    with pytest.raises(NotImplementedError):
        Model("base").f(np.zeros(2))


# ----------------------------------------------------------------------------
# Random walks
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def walk(tmp_path_factory):
    """A short random walk on two sources."""
    filename = str(tmp_path_factory.mktemp("walk") / "chain.h5")
    sources = two_sources()
    fitter = fitter_with(*sources)
    fitter.fit()
    np.random.seed(0)
    fitter.fit(
        method="emcee",
        llh_method="gaussian",
        nwalkers=12,
        steps=60,
        filename=filename,
        sampler_kwargs={},
        mcmc_kwargs={},
    )
    return fitter, sources, filename


def test_walk_result_is_near_least_squares(walk):
    fitter, sources, _ = walk
    for source in sources:
        design = np.column_stack([np.ones_like(source.x), source.x])
        beta, cov, _ = weighted_lstsq(design, source.y, source.yerr_data)
        value = result_value(fitter, f"{source.name}___bg___p1")
        assert abs(value - beta[1]) < 5 * np.sqrt(cov[1, 1])


def test_read_walk_uses_chain_percentiles(walk):
    fitter, _, filename = walk
    fitter.readWalk(filename, burnin=20)
    reader = satlas2.SATLASHDFBackend(filename)
    chain = reader.get_chain(flat=True, discard=20)
    labels = list(reader.labels)
    column = labels.index("s1___bg___p1")
    low, median, high = np.percentile(chain[:, column], [15.87, 50, 84.13])
    assert result_value(fitter, "s1___bg___p1") == pytest.approx(median)
    assert result_stderr(fitter, "s1___bg___p1") == pytest.approx((high - low) / 2)


def test_evaluate_over_walk_band_shape(walk):
    fitter, sources, filename = walk
    x = np.linspace(-5, 5, 7)
    X, bands = fitter.evaluateOverWalk(filename, burnin=20, x=x, evals=30)
    assert len(bands) == len(sources)
    for band in bands:
        assert band.shape == (3, len(x))
        # lower bound <= median evaluation <= upper bound
        assert np.all(band[0] <= band[1] + 1e-12)
        assert np.all(band[1] <= band[2] + 1e-12)


def test_evaluate_over_walk_returns_one_x_per_source(walk):
    fitter, sources, filename = walk
    X, _ = fitter.evaluateOverWalk(filename, burnin=20, evals=30)
    assert len(X) == len(sources)


def test_fit_does_not_change_its_defaults(tmp_path):
    before = copy.deepcopy(
        {k: v.default for k, v in inspect.signature(Fitter.fit).parameters.items()}
    )
    fitter = fitter_with(line_source("s", intercept=1.0, slope=0.5))
    fitter.fit(method="emcee", nwalkers=8, steps=5, filename=str(tmp_path / "c.h5"))
    after = {k: v.default for k, v in inspect.signature(Fitter.fit).parameters.items()}
    assert after == before


@pytest.mark.parametrize("method", ["emcee", "EMCEE"])
def test_walk_without_file(method):
    fitter = fitter_with(line_source("s", intercept=1.0, slope=0.5))
    fitter.fit(method=method, nwalkers=8, steps=5)
    assert fitter.result.method == "emcee"
    assert fitter.result.chain.shape == (5, 8, 2)


def test_unknown_mode_raises():
    fitter = prepared(fitter_with(counting_source()))
    fitter.mode = "typo"
    with pytest.raises(ValueError, match="typo"):
        fitter.resid()


def test_callable_yerr_evaluates_models_once():
    calls = []

    class Counting(Polynomial):
        def f(self, x):
            calls.append(1)
            return super().f(x)

    source = Source(np.linspace(1, 2, 5), np.full(5, 4.0), yerr=np.sqrt, name="s")
    source.addModel(Counting([4.0], name="c"))
    fitter = prepared(fitter_with(source))
    calls.clear()
    fitter.resid()
    assert len(calls) == 1


def test_likelihood_can_be_evaluated_after_a_fit():
    fitter = fitter_with(counting_source())
    fitter.fit()
    assert np.isfinite(fitter.llh(fitter.lmpars, method="poisson", emcee=True))


def small_walk(tmp_path, **kwargs):
    fitter = fitter_with(line_source("s", intercept=1.0, slope=0.5))
    fitter.fit()
    filename = str(tmp_path / "walk.h5")
    np.random.seed(5)
    fitter.fit(method="emcee", nwalkers=8, filename=filename, **kwargs)
    return fitter, filename


def test_resuming_a_walk_extends_the_chain(tmp_path):
    fitter, filename = small_walk(tmp_path, steps=20)
    first = satlas2.SATLASHDFBackend(filename).get_chain()
    fitter.fit(method="emcee", nwalkers=8, steps=15, filename=filename, overwrite=False)
    chain = satlas2.SATLASHDFBackend(filename).get_chain()
    assert chain.shape == (35, 8, 2)
    assert np.array_equal(chain[:20], first)


def test_walk_stops_at_convergence(tmp_path):
    fitter, _ = small_walk(
        tmp_path, steps=5000, convergence=True, convergence_iter=5, convergence_tau=0.5
    )
    assert fitter.result.chain.shape[0] < 5000
    assert "converged" in fitter.result.message


def test_short_walk_warns_about_autocorrelation(tmp_path):
    with pytest.warns(RuntimeWarning, match="autocorrelation"):
        small_walk(tmp_path, steps=10)


def test_walk_uses_a_given_pool(tmp_path):
    class Pool:
        calls = 0

        def map(self, func, iterable):
            Pool.calls += 1
            return list(map(func, iterable))

    small_walk(tmp_path, steps=5, sampler_kwargs={"pool": Pool()})
    assert Pool.calls > 0


def test_walk_statistics_match_the_chain(tmp_path):
    fitter, _ = small_walk(tmp_path, steps=40)
    flat = fitter.result.chain.reshape(-1, 2)
    names = fitter.result.var_names
    low, median, high = np.percentile(flat, [15.87, 50, 84.13], axis=0)
    for i, name in enumerate(names):
        assert result_value(fitter, name) == pytest.approx(median[i])
        assert result_stderr(fitter, name) == pytest.approx((high[i] - low[i]) / 2)
    correl = np.corrcoef(flat.T)[0, 1]
    assert fitter.result.params[names[0]].correl[names[1]] == pytest.approx(correl)
