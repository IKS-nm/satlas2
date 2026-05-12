"""
Constraining parameters with algebraic expressions.

This example demonstrates how to link parameters across models using
symbolic expressions, which is useful when physical constraints relate
two or more free parameters.

Scenario: a spectrum contains contributions from two nuclear states —
a ground state and an isomer — with an *a priori* unknown population ratio.
Rather than fitting the isomer amplitude independently (which would be
poorly constrained), we express it as a fixed fraction of the ground-state
amplitude:

    scale_isomer = scale_ground * ratio

This enforces the physical constraint while keeping `ratio` (or equivalently
`scale_ground`) as the single free amplitude parameter.

The parameter naming convention used by setExpr follows the pattern:
    <SourceName>___<ModelName>___<ParamName>
"""

import matplotlib.pyplot as plt
import numpy as np

import satlas2


def modified_sqrt(y):
    out = np.sqrt(np.abs(y))
    out[y <= 0] = 1
    return out


# ---------------------------------------------------------------------------
# Build the two HFS models (ground state + isomer)
# ---------------------------------------------------------------------------
spin_gs  = 0.5          # ground-state nuclear spin
spin_iso = 0            # isomer nuclear spin (spin-0 → no hyperfine splitting)
J        = [0.5, 1.5]   # lower / upper electronic angular momenta
A        = [100, 175]   # hyperfine A parameters [MHz]
B        = [0, 0]
C        = [0, 0]
FWHMG    = 135 / 4
FWHML    = 101 / 25
centroid = 0
scale_gs  = 100

hfs_ground = satlas2.HFS(
    spin_gs, J, A, B, C,
    df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale_gs,
    name="GroundState",
)
hfs_isomer = satlas2.HFS(
    spin_iso, J, A, B, C,
    df=centroid, fwhmg=FWHMG, fwhml=FWHML, scale=scale_gs * 0.1,
    name="Isomer",
)

# ---------------------------------------------------------------------------
# Generate synthetic data (ground state + isomer, no background)
# ---------------------------------------------------------------------------
x = np.arange(-400, 300, 10)
y = satlas2.generateSpectrum([hfs_ground, hfs_isomer], x)

x_plot     = np.arange(-400, 300)
y_original = hfs_ground.f(x_plot) + hfs_isomer.f(x_plot)

# ---------------------------------------------------------------------------
# Set up the fit
# ---------------------------------------------------------------------------
source = satlas2.Source(x, y, yerr=modified_sqrt, name="Data")
source.addModel(hfs_ground)
source.addModel(hfs_isomer)

fitter = satlas2.Fitter()
fitter.addSource(source)

# ---------------------------------------------------------------------------
# Apply the expression constraint:
#   isomer scale = 25 % of ground-state scale
#
# The parameter name follows the <Source>___<Model>___<Param> convention.
# After calling setExpr the isomer scale is no longer a free parameter;
# it tracks the ground-state scale automatically during the fit.
# ---------------------------------------------------------------------------
fitter.setExpr(
    parameter_name="Data___Isomer___scale",
    parameter_expression="Data___GroundState___scale * 0.25",
)

fitter.fit()
print(fitter.reportFit())

# ---------------------------------------------------------------------------
# Plot: original model vs. fitted model
# ---------------------------------------------------------------------------
y_fitted = hfs_ground.f(x_plot) + hfs_isomer.f(x_plot)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y, drawstyle="steps-mid", label="Synthetic data")
ax.plot(x_plot, y_original, "--", label="True model")
ax.plot(x_plot, y_fitted,   "-",  label="Fitted model (constrained)")
ax.set_xlabel("Relative frequency [MHz]")
ax.set_ylabel("Counts")
ax.set_title("Ground state + isomer fit with expression constraint")
ax.legend()
plt.tight_layout()
plt.show()
