# Changelog

All notable changes to satlas2 are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Changes are grouped under the following headings, in this order (headings without
entries are left out of a release):

- 🏆 Highlights
- 💥 Breaking changes
- ⚠️ Deprecations
- 🚀 Performance improvements
- ✨ Enhancements
- 🐞 Bug fixes
- 📖 Documentation
- 📦 Build system
- 🛠️ Other improvements

## [Unreleased]

### 🏆 Highlights

- `HFS` supports transitions of arbitrary order through the `order` argument; the
  allowed lines and their Racah intensities follow from the given order.
- `HFS` can model saturated transitions with `use_saturation=True`: a single
  `Saturation` parameter moves the line intensities from the Racah values towards
  the statistical (2F+1) weights.

### 💥 Breaking changes

- `Fitter.evaluateOverWalk` returns one array of x values per source, instead of
  one per source for every sample of the walk.
- A random walk that is too short for a reliable autocorrelation time emits a
  `RuntimeWarning` instead of printing the message; stopping at convergence is
  reported in the result message and logged instead of printed.
- `HFS` no longer has the `scaling_Al`, `scaling_Au`, `scaling_Bl`, `scaling_Bu`,
  `scaling_Cl` and `scaling_Cu` attributes; use `HFS.pos()` for the line positions.
- The `order`, `use_saturation` and `saturation` arguments of `HFS` are inserted
  before `prefunc`; pass `prefunc` by keyword.
- Spectra of `HFS` with `peak="skewvoigt"` change, since the width of the Gaussian
  component was wrong (see the bug fixes).

### 🚀 Performance improvements

- `Fitter.evaluateOverWalk` collects the evaluations in a list instead of
  stacking the arrays after every sample, which took quadratic time.
- The Wigner 3j and 6j symbols are calculated numerically (exactly, with the Racah
  formulas) instead of with sympy: creating the first `HFS` model takes about 2 ms
  instead of 100 ms, and importing satlas2 no longer imports sympy.
- `HFS` evaluates large spectra in chunks, limiting the memory use.
- Fits with a callable `yerr` (e.g. Poisson-style weighting) evaluate the models
  once per step instead of twice, making them about twice as fast.
- The Wigner symbols in `HFS` are cached across instances: creating a model is up to
  1000 times faster after the first model with the same spins.
- `HFS` evaluates all lines, and sidepeaks, in one vectorised call, and uses closed
  forms for the Gaussian and Lorentzian peaks.

### ✨ Enhancements

- New `satlas2.lineshapes` module with the Gaussian, Lorentzian and Voigt peaks,
  the skewing factor and the total Voigt FWHM, used by all models.
- `satlas2.sumModels(models, x)` sums the responses of several models.
- The metadata dataframe can be created after reading a random walk, and the
  results of a random walk have a message and success flag.

### 🐞 Bug fixes

- `Fitter.shareParams` and `Fitter.shareModelParams` with a single name shared
  each character of the name instead of the name.
- The correlations stored in a model included those of other models whose names
  start with the same text (e.g. `bg` and `bg2`), under the wrong names.
- A `pool` passed in `sampler_kwargs` was ignored by the random walk.
- A random walk without `filename` failed, as did `method="EMCEE"` in capitals.
- A single Gaussian prior was ignored in the Poisson likelihood.
- `Fitter.fit` changed its own default `mcmc_kwargs` and `sampler_kwargs`, so
  settings leaked into later fits.
- The likelihood can be calculated after a fit; an unknown `Fitter.mode` raises a
  clear `ValueError`.
- Changing the transformation of a model with `setTransform` (or `prefunc`) had
  no effect on inputs that had been evaluated before.
- `SkewedVoigt` with a `prefunc` applied the skew to the untransformed points.
- `weightedAverage` failed with `axis=1` and used the wrong number of values for
  multidimensional input.
- `Model.f` raises `NotImplementedError` instead of a `TypeError` when a model
  does not implement it.
- Fix the shape of the log-probabilities in the random walk.
- `HFS` with sidepeaks: the scale was applied once per line instead of once,
  `np.math.factorial` failed on NumPy 2, `prefunc` was applied twice and scalar
  input was not accepted.
- `HFS` with `peak="skewvoigt"` used a Gaussian width 1.39 times too large
  (`FWHM / 2 * sqrt(2 ln 2)` instead of `FWHM / (2 sqrt(2 ln 2))`).
- The `Amp` parameters of an `HFS` with saturation, and the results dataframe,
  now show the amplitudes belonging to the current (fitted) saturation.

### 📖 Documentation

- Correct the `Polynomial` documentation: the coefficients go from the highest
  order down, as in `numpy.polyval`.
- Explain in the `HFS` documentation why the saturated amplitudes are calculated
  by the model instead of with lmfit expressions.

### 📦 Build system

- sympy is no longer a dependency; it is only used in the tests.
- Run the tests with coverage on every push to master and every pull request,
  for Python 3.10 to 3.13 on Linux and Python 3.12 on Windows.

### 🛠️ Other improvements

- Add tests for `Fitter`, `Source` and `Model`, comparing fits to closed-form
  least squares solutions and likelihoods to `scipy.stats`.
- Add tests for the `HFS` model: golden values, analytic checks of the hyperfine
  energies, intensity sum rule and peak shapes, and the saturation and sidepeak
  options.
- The `Fitter` looks up parameters by their full name in one place, and builds the
  lmfit parameters in a single, documented loop.
- `HFS`, `Voigt` and `SkewedVoigt` share their peak shapes and FWHM calculation
  instead of each having a copy; add tests for all general models.
- `Source` and `generateSpectrum` share one way of summing models.
- The random walk is split into named steps (bounds, backend, sampling,
  summary), and summarising a walk is shared with `readWalk`; seeded walks
  give exactly the same chains as before.
- `Fitter.fit` builds the random walk options in a separate method, and the
  data is kept with the parameters instead of in a temporary attribute.
- Split the `HFS` constructor into smaller steps and calculate the line positions
  in one place.

## [0.3.0] - 2026-05-12

Changes up to and including this version are listed in the
[GitHub releases](https://github.com/IKS-nm/satlas2/releases).

[Unreleased]: https://github.com/IKS-nm/satlas2/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/IKS-nm/satlas2/releases/tag/v0.3.0
