"""
Custom HFS lineshape: extending the built-in peak profile.

This example demonstrates how to replace the standard Voigt peak shape
inside an HFS model with an arbitrary custom profile.

Background: in collinear laser spectroscopy, the observed lineshape can deviate
from a symmetric Voigt profile due to effects such as:
  - non-linear Doppler compression at low energies (asymmetric wings)
  - charge-exchange cross-section energy dependence
  - laser power / saturation effects

Here we define a peak shape that is a Voigt *minus* a narrower Voigt, creating
a profile with a dip at the centre — a simple toy model for such effects.

The custom peak is injected via `peak='custom'` and `peak_kwargs` which
declares any extra parameters the custom method requires.

Three fitting strategies are compared:
  - Chi-square (least squares)
  - Custom Poisson log-likelihood
  - MCMC posterior sampling
"""

import functools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import voigt_profile

import satlas2


# ---------------------------------------------------------------------------
# Custom Fitter: Poisson log-likelihood for bunched-beam rate data
# (same as in extraattributes.py — extracted here for self-containedness)
# ---------------------------------------------------------------------------
class BunchedBeamFitter(satlas2.Fitter):
    def customLlh(self):
        if not hasattr(self, "_bunches"):
            self._bunches = self.getSourceAttr("bunches")
        if not hasattr(self, "_data_counts"):
            self._data_counts = self.temp_y * self._bunches

        model_rates  = self.f()
        model_counts = model_rates * self._bunches

        llh = self._data_counts * np.log(model_counts) - model_counts
        llh[model_counts <= 0] = -np.inf

        priors = self.gaussianPriorResid()
        if len(priors) > 1:
            llh = np.append(llh, -0.5 * priors**2)
        return llh


def rate_uncertainty(y, bunches=None):
    yerr = np.sqrt(y * bunches) / bunches
    yerr[y <= 0] = 1 / bunches[y <= 0]
    return yerr


# ---------------------------------------------------------------------------
# Custom HFS subclass: replace the peak shape with a Voigt-minus-narrower-Voigt
# ---------------------------------------------------------------------------
class DipVoigtHFS(satlas2.HFS):
    """
    HFS model with a lineshape that dips at the centre.

    The peak function is:
        f(x) = V(x; σ, γ) - V(x; σ/w, γ/w) / amp

    where V is a normalised Voigt profile, w = neg_width narrows the
    subtracted component, and neg_amp controls its depth.

    Both extra parameters are declared via peak_kwargs when constructing.
    """

    def customPeak(self, x):
        sigma = self.params["FWHMG"].value / 2 * np.sqrt(2 * np.log(2))
        gamma = self.params["FWHML"].value / 2
        w     = self.params["neg_width"].value
        amp   = self.params["neg_amp"].value

        main = voigt_profile(x, sigma, gamma) / voigt_profile(0, sigma, gamma)
        dip  = voigt_profile(x, sigma / w, gamma / w) / voigt_profile(0, sigma / w, gamma / w)
        return main - dip / amp


# ---------------------------------------------------------------------------
# Simulate bunched-beam data with the custom lineshape
# ---------------------------------------------------------------------------
spin = 1
J    = [0, 1]
A    = [0, 50]
B    = [0, 0]
C    = [0, 0]
FWHMG    = 20
FWHML    = 20
centroid = 0
bkg      = 0.0005
scale    = 0.1

true_hfs = DipVoigtHFS(
    spin, J, A, B, C,
    df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale,
    peak="custom",
    peak_kwargs={"neg_width": {"value": 4}, "neg_amp": {"value": 2}},
)
background = satlas2.Polynomial([bkg])

x = np.arange(-150, 150, 1)
rng = np.random.default_rng(10)

n_bunches_nominal = 100_000
noise = min(10, n_bunches_nominal / 5)
bunches = np.full(
    x.shape,
    satlas2.generateSpectrum(
        satlas2.Polynomial([n_bunches_nominal]), x,
        lambda mu: rng.normal(mu, noise),
    ).max(),
    dtype=int,
)
y = np.array([
    satlas2.generateSpectrum([true_hfs, background],
                              np.array([X] * B), rng.poisson).sum() / B
    for X, B in zip(x, bunches)
])

# ---------------------------------------------------------------------------
# Fit with a perturbed starting model
# ---------------------------------------------------------------------------
fit_hfs = DipVoigtHFS(
    spin, J, [0, 51], B, C,
    df=9, fwhmg=FWHMG, fwhml=FWHML, scale=scale,
    peak="custom",
    peak_kwargs={"neg_width": {"value": 4}, "neg_amp": {"value": 4}},
    name="HFS",
)
fit_hfs.params["Bu"].vary = False

yerr_values = rate_uncertainty(y, bunches)
source = satlas2.Source(x, y, yerr=yerr_values, name="Data", bunches=bunches)
source.addModel(fit_hfs)
source.addModel(background)

fitter = BunchedBeamFitter()
fitter.addSource(source)

# ---------------------------------------------------------------------------
# Plot layout
# ---------------------------------------------------------------------------
size = 1.5
fig = plt.figure(constrained_layout=True, figsize=(16 / 2 * size, 9 / 2 * size))
gs  = gridspec.GridSpec(nrows=3, ncols=1, figure=fig)
ax_bunches = fig.add_subplot(gs[0, :])
ax_counts  = fig.add_subplot(gs[1, :])
ax_rate    = fig.add_subplot(gs[2, :])

ax_bunches.plot(x, bunches,     drawstyle="steps-mid")
ax_counts.plot(x,  y * bunches, drawstyle="steps-mid", label="Data")
ax_rate.errorbar(x, y, yerr=source.yerr(), fmt="none", ecolor="tab:blue")
ax_rate.plot(x, y, drawstyle="steps-mid", label="Data", color="tab:blue")

plot_x = np.arange(x.min(), x.max() + 1)
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="Initial guess")

ax_bunches.set_ylabel("Bunches per step")
ax_counts.set_ylabel("Raw counts")
ax_rate.set_ylabel("Rate [cts/bunch]")
ax_rate.set_xlabel("Relative frequency [MHz]")
for ax in [ax_bunches, ax_counts]:
    ax.label_outer()

# ---------------------------------------------------------------------------
# Fit 1: chi-square
# ---------------------------------------------------------------------------
fitter.fit()
print("=== Chi-square fit ===")
print(fitter.reportFit())
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="χ² fit")

# ---------------------------------------------------------------------------
# Fit 2: Poisson log-likelihood
# ---------------------------------------------------------------------------
fitter.revertFit()
fitter.fit(llh=True, llh_method="custom")
print("=== Poisson likelihood fit ===")
print(fitter.reportFit())
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="LLH fit")

# ---------------------------------------------------------------------------
# Fit 3: MCMC
# ---------------------------------------------------------------------------
fitter.revertFit()
fitter.fit(
    llh=True, llh_method="custom",
    method="emcee", steps=2_000,
    filename="customHFS.h5",
)
print("=== MCMC fit ===")
print(fitter.reportFit())
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="MCMC fit")

ax_rate.legend(loc=0)
plt.show()

fig_walk, _ = satlas2.generateWalkPlot(filename="customHFS.h5")
plt.show()

fig_corner, _, _ = satlas2.generateCorrelationPlot(filename="customHFS.h5", burnin=500)
plt.show()
