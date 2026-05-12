"""
Fitting multiple scans and aggregating results with pandas DataFrames.

This example demonstrates:
  - Fitting several independent datasets with different background levels
  - Collecting per-scan fit results into a pandas DataFrame via
    Fitter.createResultDataframe() and Fitter.createMetadataDataframe()
  - Computing a weighted average and its uncertainty across scans using
    the standard inverse-variance formula

The weighted average is computed two ways:
  - Statistical uncertainty  σ_stat = 1 / sqrt(Σ 1/σᵢ²)
  - Scatter uncertainty      σ_scat = sqrt(Σ ((xᵢ - x̄)/σᵢ)² / ((n-1) Σ 1/σᵢ²))
  The reported uncertainty is max(σ_stat, σ_scat), which accounts for
  extra scatter beyond the individual fit uncertainties.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import satlas2


# ---------------------------------------------------------------------------
# Helper: Poisson-safe square-root uncertainty
# ---------------------------------------------------------------------------
def modified_sqrt(y):
    """sqrt(y) but returns 1 where y ≤ 0 to avoid division by zero."""
    out = np.sqrt(np.abs(y))
    out[y <= 0] = 1
    return out


# ---------------------------------------------------------------------------
# Helper: build a Voigt + exponential-decay-background model for one scan
# ---------------------------------------------------------------------------
def build_models(background_level, decay_length, centre, fwhmg, fwhml, amplitude):
    peak = satlas2.Voigt(amplitude, centre, fwhmg, fwhml, name="Signal")
    bkg  = satlas2.ExponentialDecay(background_level, decay_length, name="Background")
    return peak, bkg


# ---------------------------------------------------------------------------
# "True" physics parameters (shared across scans)
# ---------------------------------------------------------------------------
centre    = 500
fwhmg     = 150
fwhml     = 150
amplitude = 200
decay     = 500

# Each scan has a different background level to simulate real-world variation
background_levels = [1000, 500, 700]
scan_names        = ["Scan1", "Scan2", "Scan3"]

rng = np.random.default_rng(0)
x   = np.linspace(0, 1000, 250)

# ---------------------------------------------------------------------------
# Fit each scan independently, collect DataFrames
# ---------------------------------------------------------------------------
metadata_frames = []
result_frames   = []

fig, axes = plt.subplots(1, len(scan_names), figsize=(14, 4), sharey=True)

for ax, bkg_level, name in zip(axes, background_levels, scan_names):
    peak_model, bkg_model = build_models(bkg_level, decay, centre, fwhmg, fwhml, amplitude)

    # Generate Poisson-distributed synthetic data
    y_true = peak_model.f(x) + bkg_model.f(x)
    y      = rng.poisson(y_true)

    source = satlas2.Source(x, y, yerr=modified_sqrt, name=name)
    source.addModel(peak_model)
    source.addModel(bkg_model)

    fitter = satlas2.Fitter()
    fitter.addSource(source)
    fitter.fit()

    ax.plot(x, y, ".", ms=3, label="Data")
    ax.plot(source.x, source.f(), label="Fit")
    ax.set_title(name)
    ax.set_xlabel("x")
    axes[0].set_ylabel("Counts")
    ax.legend(fontsize=8)

    metadata_frames.append(fitter.createMetadataDataframe())
    result_frames.append(fitter.createResultDataframe())

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Concatenate all scan results into a single DataFrame
# ---------------------------------------------------------------------------
metadata = pd.concat(metadata_frames, ignore_index=True)
results  = pd.concat(result_frames,   ignore_index=True)

print("Fit metadata (one row per scan):")
print(metadata.to_string())
print()
print("Fit results (one row per free parameter per scan):")
print(results.to_string())

# ---------------------------------------------------------------------------
# Weighted average of lineshape parameters across scans
# ---------------------------------------------------------------------------
def weighted_average(group):
    """Inverse-variance weighted mean."""
    x, sigma = group["Value"], group["Stderr"]
    w  = 1 / sigma**2
    xm = (x * w).sum() / w.sum()
    return xm


def weighted_uncertainty(group):
    """max(statistical, scatter) uncertainty."""
    x, sigma = group["Value"], group["Stderr"]
    n  = len(x)
    w  = 1 / sigma**2
    wt = w.sum()
    xm = (x * w).sum() / wt

    sigma_stat = np.sqrt(1 / wt)
    if n > 1:
        chi2_reduced = ((((x - xm) / sigma) ** 2).sum()) / ((n - 1) * wt)
        sigma_scat   = np.sqrt(max(chi2_reduced, 0))
    else:
        sigma_scat = 0.0
    return max(sigma_stat, sigma_scat)


lineshape_params = ["FWHMG", "FWHML"]
grouped = results[results["Parameter"].isin(lineshape_params)].groupby(["Model", "Parameter"])

summary = pd.DataFrame({
    "Weighted average": grouped.apply(weighted_average),
    "Uncertainty":      grouped.apply(weighted_uncertainty),
})

print()
print("Weighted average of lineshape parameters across all scans:")
print(summary.to_string())
