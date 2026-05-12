"""
Piecewise-constant background model.

This example demonstrates satlas2.PiecewiseConstant, which is useful for
modelling step-like backgrounds that arise when, e.g.:
  - a detector threshold changes between frequency regions
  - different charge-exchange channels open at different energies
  - data is recorded in separate frequency windows with different beam tunings

A PiecewiseConstant model requires:
  - `values`  : the constant level in each segment
  - `bounds`  : the x-positions of the transitions between segments
                (len(bounds) == len(values) - 1)

The segments cover (-∞, bounds[0]), (bounds[0], bounds[1]), ..., (bounds[-1], +∞).
"""

import matplotlib.pyplot as plt
import numpy as np

import satlas2

# ---------------------------------------------------------------------------
# Define a three-segment piecewise-constant background
# ---------------------------------------------------------------------------
segment_levels = [3, 5, 2]   # count rates in each segment
breakpoints    = [300, 650]  # x-values where the level changes

step_bkg = satlas2.PiecewiseConstant(segment_levels, breakpoints)

x = np.linspace(0, 1000, 500)
y = step_bkg.f(x)

# ---------------------------------------------------------------------------
# Add a Voigt peak on top to show a realistic use case
# ---------------------------------------------------------------------------
peak = satlas2.Voigt(amplitude=15, mu=500, fwhmg=40, fwhml=20)
y_total = y + peak.f(x)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

axes[0].plot(x, y, drawstyle="steps-mid", color="tab:orange", label="Piecewise background")
for bp in breakpoints:
    axes[0].axvline(bp, color="gray", linestyle="--", linewidth=0.8)
axes[0].set_ylabel("Background level")
axes[0].legend()
axes[0].set_title("PiecewiseConstant background model")

axes[1].plot(x, y_total, drawstyle="steps-mid", label="Peak + background")
axes[1].plot(x, y,       drawstyle="steps-mid", linestyle="--",
             color="tab:orange", label="Background only")
for bp in breakpoints:
    axes[1].axvline(bp, color="gray", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("x")
axes[1].set_ylabel("Counts")
axes[1].legend()

plt.tight_layout()
plt.show()
