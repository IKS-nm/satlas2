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

- `HFS` no longer has the `scaling_Al`, `scaling_Au`, `scaling_Bl`, `scaling_Bu`,
  `scaling_Cl` and `scaling_Cu` attributes; use `HFS.pos()` for the line positions.
- The `order`, `use_saturation` and `saturation` arguments of `HFS` are inserted
  before `prefunc`; pass `prefunc` by keyword.
- Spectra of `HFS` with `peak="skewvoigt"` change, since the width of the Gaussian
  component was wrong (see the bug fixes).

### 🚀 Performance improvements

- The Wigner symbols in `HFS` are cached across instances: creating a model is up to
  1000 times faster after the first model with the same spins.
- `HFS` evaluates all lines, and sidepeaks, in one vectorised call, and uses closed
  forms for the Gaussian and Lorentzian peaks.

### ✨ Enhancements

- The metadata dataframe can be created after reading a random walk, and the
  results of a random walk have a message and success flag.

### 🐞 Bug fixes

- Fix the shape of the log-probabilities in the random walk.
- `HFS` with sidepeaks: the scale was applied once per line instead of once,
  `np.math.factorial` failed on NumPy 2, `prefunc` was applied twice and scalar
  input was not accepted.
- `HFS` with `peak="skewvoigt"` used a Gaussian width 1.39 times too large
  (`FWHM / 2 * sqrt(2 ln 2)` instead of `FWHM / (2 sqrt(2 ln 2))`).
- The `Amp` parameters of an `HFS` with saturation, and the results dataframe,
  now show the amplitudes belonging to the current (fitted) saturation.

### 📖 Documentation

- Explain in the `HFS` documentation why the saturated amplitudes are calculated
  by the model instead of with lmfit expressions.

### 🛠️ Other improvements

- Add tests for the `HFS` model: golden values, analytic checks of the hyperfine
  energies, intensity sum rule and peak shapes, and the saturation and sidepeak
  options.
- Split the `HFS` constructor into smaller steps and calculate the line positions
  in one place.

## [0.3.0] - 2026-05-12

Changes up to and including this version are listed in the
[GitHub releases](https://github.com/IKS-nm/satlas2/releases).

[Unreleased]: https://github.com/IKS-nm/satlas2/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/IKS-nm/satlas2/releases/tag/v0.3.0
