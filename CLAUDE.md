# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install all dependencies (including dev group)
uv sync --group dev

# Install with example dependencies too
uv sync --group dev --group examples

# Run all tests
uv run pytest -v -s

# Run a single test
uv run pytest tests/unit/test_fitting.py::test_name -v

# Build package
uv build

# Build documentation
uv run sphinx-build docs _build
```

## Architecture

satlas2 is a fitting framework for laser spectroscopy data, particularly hyperfine structure (HFS) spectra. The core abstraction is a three-layer hierarchy:

**`Fitter` → `Source` → `Model`**

- **`Fitter`** (`core.py`): Orchestrates fitting. Holds multiple `Source` objects, manages shared/constrained parameters via `shareParams`, `shareModelParams`, and symbolic `setExpr`. Supports chi-square minimisation (via lmfit), log-likelihood fitting, and MCMC random walks (via emcee). Parameter naming convention for expressions: `SourceName___ModelName___ParamName`.

- **`Source`** (`core.py`): Holds one dataset (x, y, yerr). Multiple `Model` objects added to a source are summed at evaluation time. `yerr` can be a fixed array or a callable applied to the summed model result (enabling Poisson-style weighting).

- **`Model`** (`core.py`): Abstract base. Subclass and implement `f(self, x)`. Holds a `params` dict of `Parameter` objects. An optional `prefunc` transforms x before evaluation (used for coordinate changes, e.g., voltage-to-frequency).

### Built-in models (`src/satlas2/models/`)

- **`HFS`** (`hfsModel.py`): The primary physics model. Computes a hyperfine spectrum from nuclear spin I, electronic spins J1/J2, A/B/C coupling constants, centroid df, and Voigt lineshape parameters (fwhmg, fwhml). Supports sidepeaks, custom peak shapes (pass `peak='custom'`), and Racah intensity constraints. Wigner 3j/6j symbols used for line intensity calculations.
- **`Polynomial`**, **`PiecewiseConstant`**, **`ExponentialDecay`**, **`Voigt`**, **`SkewedVoigt`** (`models.py`): Standard background/signal models.

### Compatibility layer (`interface.py`)

`HFSModel` and `SumModel` wrap the native `HFS`/`Source`/`Fitter` API to match satlas v1 calling conventions. These exist for migration purposes and are not fully benchmarked — prefer the native API for new code.

### Overwrite layer (`overwrite.py`)

Custom subclasses of emcee (`SATLASSampler`, `SATLASHDFBackend`) and lmfit (`SATLASMinimizer`, `minimize`) that add MCMC convergence checking and HDF5 walk persistence. These are internal and not part of the public API.

### Utilities and plotting

- `utilities.py`: `weightedAverage`, `poissonInterval` (Garwood exact Poisson CI), `generateSpectrum`.
- `plotting.py`: `generateChisquareMap`, `generateCorrelationPlot`, `generateWalkPlot` — post-fit visualisation helpers.

### Versioning

Version is derived from VCS tags via `hatch-vcs`. There is no hard-coded version string; `__version__` is resolved at import time from package metadata.
