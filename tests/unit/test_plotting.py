"""Tests for the plotting functions: the numerical helpers against known
values, and the figures as smoke tests with the Agg backend."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import satlas2  # noqa: E402
from satlas2 import plotting  # noqa: E402

NAMES = ["s1___bg___p0", "s1___bg___p1", "s2___peak___mu"]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def walk(tmp_path_factory):
    rng = np.random.default_rng(0)
    fitter = satlas2.Fitter()
    x = np.linspace(-5, 5, 40)
    source = satlas2.Source(x, 2 + 0.5 * x + rng.normal(0, 0.2, 40), np.full(40, 0.2), "s")
    source.addModel(satlas2.Polynomial([1.0, 0.0], name="bg"))
    fitter.addSource(source)
    fitter.fit()
    filename = str(tmp_path_factory.mktemp("walk") / "walk.h5")
    np.random.seed(1)
    fitter.fit(method="emcee", nwalkers=10, steps=200, filename=filename)
    return fitter, filename


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "source, model, expected",
    [
        (False, False, ["p0", "p1", "mu"]),
        (False, True, ["bg p0", "bg p1", "peak mu"]),
        (True, True, ["s1 bg p0", "s1 bg p1", "s2 peak mu"]),
    ],
)
def test_parameter_labels(source, model, expected):
    assert plotting._parameterLabels(NAMES, source, model, " ") == expected


def test_select_parameters_follows_the_filter_order():
    labels = ["bg p0", "bg p1", "peak mu"]
    assert plotting._selectParameters(["mu", "p0"], labels, NAMES) == [
        ("peak mu", NAMES[2]),
        ("bg p0", NAMES[0]),
    ]
    assert plotting._selectParameters(None, labels, NAMES) == list(zip(labels, NAMES))


def test_scott_bins():
    x = np.random.default_rng(0).normal(0, 1, 10_000)
    expected = int(np.ptp(x) / (3.5 * np.std(x) / x.size ** (1 / 3)))
    assert plotting._scottBins(x) == expected
    assert plotting._scottBins(x, reduction=2) == expected // 2


def test_scott_bins_for_a_constant_parameter():
    assert plotting._scottBins(np.full(100, 3.0)) == plotting._FALLBACK_BINS


def test_credible_levels_enclose_the_gaussian_masses():
    samples = np.random.default_rng(1).normal(size=(400_000, 2))
    H, _, _ = np.histogram2d(*samples.T, bins=80, range=[[-5, 5], [-5, 5]])
    levels = plotting._credibleLevels(H)
    enclosed = [H[H >= level].sum() / H.sum() for level in levels]
    assert enclosed == pytest.approx(plotting._MASSES_2D, abs=0.01)
    assert list(plotting._MASSES_2D) == pytest.approx([0.3935, 0.8647, 0.9889], abs=1e-4)


def test_interval_title():
    assert plotting._intervalTitle("a", 1.2341, 0.012, 0.021) == "a\n$1.234_{-0.012}^{+0.021}$"
    assert plotting._intervalTitle("b", 1.2341e-7, 1.2e-9, 2.1e-9) == (
        "b\n$(1.234_{-0.012}^{+0.021})e-07$"
    )


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def test_correlation_plot(walk):
    fitter, filename = walk
    fig, axes, cbar = plotting.generateCorrelationPlot(filename)
    names = fitter.result.var_names
    assert axes.shape == (len(names), len(names))
    assert sum(ax is not None for ax in axes.ravel()) == 3
    for i, name in enumerate(names):
        median = np.median(fitter.result.chain.reshape(-1, len(names))[:, i])
        assert axes[i, i].get_title().startswith(name.replace("___", "\n"))
        assert len(axes[i, i].get_lines()) == 3  # the median and the 1 sigma limits
        assert axes[i, i].get_lines()[1].get_xdata()[0] == pytest.approx(median)
    assert [t.get_text() for t in cbar.ax.get_yticklabels()] == [
        r"3$\sigma$",
        r"2$\sigma$",
        r"1$\sigma$",
    ]


def test_correlation_plot_with_filter(walk):
    _, filename = walk
    fig, axes, _ = plotting.generateCorrelationPlot(
        filename, filter=["p1"], source=False, model=False, bins=12
    )
    assert axes.shape == (1, 1)
    assert axes[0, 0].get_title().startswith("p1\n")


def test_walk_plot(walk):
    _, filename = walk
    fig, axes = plotting.generateWalkPlot(filename, burnin=50, thin=5)
    assert len(axes) == 2
    lines = axes[0].get_lines()
    assert len(lines) == 10 + 1  # one per walker, plus the median
    assert lines[0].get_xdata()[0] == 50
    assert axes[-1].get_xlabel() == "Step"

