"""
Generating a synthetic HFS spectrum with Poisson statistics.

This example demonstrates:
  - Building a hyperfine structure (HFS) model
  - Generating Poisson-distributed synthetic data
  - Computing and visualising Poisson confidence intervals
    on both the data and the underlying model mean

The two kinds of interval are conceptually different:
  - "data interval"   : given observed counts, what range of means is plausible?
  - "model interval"  : given a known mean, what range of counts should we expect?
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

import satlas2

# ---------------------------------------------------------------------------
# Define the physics model
# ---------------------------------------------------------------------------
spin = 0.5
J = [0.5, 1.5]          # lower / upper state angular momenta
A = [100, 175]           # magnetic hyperfine parameters [MHz]
B = [0, 0]               # electric quadrupole parameters [MHz]
C = [0, 0]               # octupole parameters [MHz]
FWHMG = 135 / 4          # Gaussian linewidth [MHz]
FWHML = 101 / 25         # Lorentzian linewidth [MHz]
centroid = 0             # centre of gravity [MHz]
bkg = 6                  # flat background level [counts]
scale = 10               # overall peak amplitude [counts]

hfs = satlas2.HFS(spin, J, A, B, C, df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale)
background = satlas2.Polynomial([bkg])

# ---------------------------------------------------------------------------
# Generate synthetic data: sample x values, draw Poisson-distributed counts
# ---------------------------------------------------------------------------
x = np.arange(-400, 300, 30)
y = satlas2.generateSpectrum([hfs, background], x)   # Poisson-sampled counts

# Poisson confidence intervals *on the data* (what mean is consistent with y?)
lower_d1, upper_d1 = satlas2.poissonInterval(y, sigma=1)
lower_d2, upper_d2 = satlas2.poissonInterval(y, sigma=2)
lower_d3, upper_d3 = satlas2.poissonInterval(y, sigma=3)

# Poisson confidence intervals *on the mean* (what data should we expect?)
plot_x = np.arange(-400, 300)
plot_y = hfs.f(plot_x) + background.f(plot_x)

lower_m1, upper_m1 = satlas2.poissonInterval(plot_y, sigma=1, is_mean=True)
lower_m2, upper_m2 = satlas2.poissonInterval(plot_y, sigma=2, is_mean=True)
lower_m3, upper_m3 = satlas2.poissonInterval(plot_y, sigma=3, is_mean=True)

# ---------------------------------------------------------------------------
# Plot: left = raw spectrum; top-right = data intervals; bottom-right = model intervals
# ---------------------------------------------------------------------------
size = 1.5
fig = plt.figure(constrained_layout=True, figsize=(16 / 2 * size, 9 / 2 * size))
gs = gridspec.GridSpec(nrows=2, ncols=2, figure=fig)
ax_base = fig.add_subplot(gs[:, 0])
ax_data = fig.add_subplot(gs[0, 1], sharex=ax_base, sharey=ax_base)
ax_model = fig.add_subplot(gs[1, 1], sharex=ax_base, sharey=ax_base)

# Left panel: data + model curve
ax_base.plot(x, y, drawstyle="steps-mid", label="Data")
ax_base.plot(plot_x, plot_y, label="Model")
ax_base.legend(loc=0)

# Top-right: 1/2/3-σ intervals around the data
(lc,) = ax_data.plot(x, y, drawstyle="steps-mid", label="Data")
for lo, hi in [(lower_d1, upper_d1), (lower_d2, upper_d2), (lower_d3, upper_d3)]:
    ax_data.fill_between(x, lo, hi, color=lc.get_color(), alpha=0.3, step="mid")
ax_data.fill_between([], [], [], color=lc.get_color(), alpha=0.7, label="1, 2, 3-σ")
ax_data.plot(plot_x, plot_y, label="Model")
ax_data.set_title("Confidence interval of mean\nbased on data")
ax_data.legend(loc=0)

# Bottom-right: 1/2/3-σ prediction bands around the model mean
ax_model.plot(x, y, drawstyle="steps-mid", label="Data")
(lc,) = ax_model.plot(plot_x, plot_y, label="Model")
for lo, hi in [(lower_m1, upper_m1), (lower_m2, upper_m2), (lower_m3, upper_m3)]:
    ax_model.fill_between(plot_x, lo, hi, color=lc.get_color(), alpha=0.3, step="mid")
ax_model.fill_between([], [], [], color=lc.get_color(), alpha=0.7, label="1, 2, 3-σ")
ax_model.set_title("Prediction band of data\nbased on model mean")
ax_model.legend(loc=0)

for ax in [ax_base, ax_data, ax_model]:
    ax.label_outer()
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Counts")

plt.show()
