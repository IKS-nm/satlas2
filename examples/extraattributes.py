"""
Bunched-beam laser spectroscopy: rate data, custom likelihood, and MCMC.

This example demonstrates a workflow typical of collinear laser spectroscopy
with a bunched ion beam, where the observable is a *rate* (counts per bunch)
rather than a raw count:

  1. Simulate bunched-beam data:
       - A fixed number of ion bunches is accumulated at each frequency step.
       - The detector records total counts per step; dividing by the number
         of bunches gives the detection rate.

  2. Propagate the uncertainty correctly for rate data:
       σ_rate = sqrt(total_counts) / n_bunches  (or 1/n_bunches when counts = 0)

  3. Compare three fitting strategies:
       a. Least-squares (chi-square) fit — fast but assumes Gaussian residuals
       b. Poisson log-likelihood fit    — correct for low-count data
       c. MCMC (emcee) sampling         — full posterior distribution

  4. Inspect the MCMC posterior with walk and corner plots.

The model includes a skewed Voigt lineshape to handle asymmetric peaks that
arise from, e.g., non-linear charge-exchange processes or laser power broadening.
"""

import functools

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import satlas2


# ---------------------------------------------------------------------------
# Custom Fitter: Poisson log-likelihood for rate (counts/bunch) data
# ---------------------------------------------------------------------------
class BunchedBeamFitter(satlas2.Fitter):
    """
    Log-likelihood for rate data from a bunched ion beam.

    Each source must carry a `bunches` attribute (array of bunch counts per
    frequency step).  The log-likelihood is:

        L = Σ  n_i * log(μ_i * B_i) - μ_i * B_i

    where  n_i = y_i * B_i  are the observed total counts and  μ_i  is the
    model rate at step i.
    """

    def customLlh(self):
        # Cache source attributes to avoid repeated attribute lookups
        if not hasattr(self, "_bunches"):
            self._bunches = self.getSourceAttr("bunches")
        if not hasattr(self, "_data_counts"):
            self._data_counts = self.temp_y * self._bunches

        model_rates  = self.f()
        model_counts = model_rates * self._bunches

        llh = self._data_counts * np.log(model_counts) - model_counts
        llh[model_counts <= 0] = -np.inf

        # Add Gaussian priors if any have been set
        priors = self.gaussianPriorResid()
        if len(priors) > 1:
            llh = np.append(llh, -0.5 * priors**2)

        return llh


# ---------------------------------------------------------------------------
# Uncertainty for rate data: σ = sqrt(total counts) / n_bunches
# ---------------------------------------------------------------------------
def rate_uncertainty(y, bunches=None):
    yerr = np.sqrt(y * bunches) / bunches
    yerr[y <= 0] = 1 / bunches[y <= 0]
    return yerr


# ---------------------------------------------------------------------------
# Simulate bunched-beam data
# ---------------------------------------------------------------------------
# "True" HFS model (spin-1, J: 0→1 transition)
spin = 1
J    = [0, 1]
A    = [0, 50]         # hyperfine A parameters [MHz]
B    = [0, 0]
C    = [0, 0]
FWHMG    = 135 / 4     # [MHz]
FWHML    = 101 / 4     # [MHz]
centroid = 0           # centre of gravity [MHz]
bkg      = 0.0005      # background rate [cts/bunch]
scale    = 0.001       # peak amplitude  [cts/bunch]

# Skewed Voigt to model an asymmetric lineshape
true_hfs = satlas2.HFS(
    spin, J, A, B, C,
    df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale,
    peak="skewvoigt", peak_kwargs={"skew": {"value": 11, "min": 1}},
)
background = satlas2.Polynomial([bkg])

# Simulate bunch counts: ~100 000 bunches per step with small Gaussian noise
x = np.arange(-150, 150, 5)
rng = np.random.default_rng(10)

n_bunches_nominal = 100_000
noise             = min(10, n_bunches_nominal / 5)
bunches_float = satlas2.generateSpectrum(
    satlas2.Polynomial([n_bunches_nominal]), x,
    lambda mu: rng.normal(mu, noise),
)
# Use the maximum bunch count uniformly for simplicity
bunches = np.full_like(bunches_float, bunches_float.max(), dtype=int)

# Draw total counts at each step from a Poisson distribution,
# accumulating over all bunches
y = np.array([
    satlas2.generateSpectrum([true_hfs, background],
                              np.array([X] * B), rng.poisson).sum() / B
    for X, B in zip(x, bunches)
])

# ---------------------------------------------------------------------------
# Set up the fit with a perturbed starting model
# ---------------------------------------------------------------------------
fit_hfs = satlas2.HFS(
    spin, J, [0, 51], B, C,
    df=9, fwhmg=120 / 4, fwhml=120 / 3, scale=scale,
    peak="skewvoigt", peak_kwargs={"skew": {"value": 11, "min": 1}},
    name="HFS",
)
fit_hfs.params["Bu"].vary = False   # B-upper is fixed (spin-0 upper state)

yerr_values = rate_uncertainty(y, bunches)
source = satlas2.Source(x, y, yerr=yerr_values, name="Data", bunches=bunches)
source.addModel(fit_hfs)
source.addModel(background)

fitter = BunchedBeamFitter()
fitter.addSource(source)

# ---------------------------------------------------------------------------
# Plot layout: bunches | raw counts | rate with successive fits overlaid
# ---------------------------------------------------------------------------
size = 1.5
fig = plt.figure(constrained_layout=True, figsize=(16 / 2 * size, 9 / 2 * size))
gs  = gridspec.GridSpec(nrows=3, ncols=1, figure=fig)
ax_bunches = fig.add_subplot(gs[0, :])
ax_counts  = fig.add_subplot(gs[1, :])
ax_rate    = fig.add_subplot(gs[2, :])

ax_bunches.plot(x, bunches,   drawstyle="steps-mid")
ax_counts.plot(x,  y * bunches, drawstyle="steps-mid", label="Data")
ax_rate.errorbar(x, y, yerr=source.yerr(), fmt="none", ecolor="tab:blue")
ax_rate.plot(x, y, drawstyle="steps-mid", label="Data", color="tab:blue")

plot_x = np.arange(x.min(), x.max() + 1, 1)
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="Initial guess")

ax_bunches.set_ylabel("Bunches per step")
ax_counts.set_ylabel("Raw counts")
ax_rate.set_ylabel("Rate [cts/bunch]")
ax_rate.set_xlabel("Relative frequency [MHz]")

for ax in [ax_bunches, ax_counts]:
    ax.label_outer()

# ---------------------------------------------------------------------------
# Fit 1: least-squares chi-square
# ---------------------------------------------------------------------------
fitter.fit()
print("=== Chi-square fit ===")
print(fitter.reportFit())
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="χ² fit")

# ---------------------------------------------------------------------------
# Fit 2: custom Poisson log-likelihood
# ---------------------------------------------------------------------------
fitter.revertFit()
fitter.fit(llh=True, llh_method="custom")
print("=== Poisson likelihood fit ===")
print(fitter.reportFit())
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="LLH fit")

# ---------------------------------------------------------------------------
# Fit 3: MCMC with emcee — full posterior sampling
# ---------------------------------------------------------------------------
fitter.revertFit()
fitter.fit(
    llh=True, llh_method="custom",
    method="emcee", steps=2_000,
    filename="extraattributes.h5",
)
print("=== MCMC fit ===")
print(fitter.reportFit())
ax_rate.plot(plot_x, fit_hfs.f(plot_x) + background.f(plot_x), label="MCMC fit")

ax_rate.legend(loc=0)
plt.show()

# ---------------------------------------------------------------------------
# Visualise the MCMC chains and the parameter correlation (corner) plot
# ---------------------------------------------------------------------------
fig_walk, _ = satlas2.generateWalkPlot(filename="extraattributes.h5")
plt.show()

fig_corner, _, _ = satlas2.generateCorrelationPlot(
    filename="extraattributes.h5", burnin=500
)
plt.show()
