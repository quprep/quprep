# Changelog

All notable changes to QuPrep will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
QuPrep uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Project scaffold and package structure
- `pyproject.toml` with optional framework dependencies (`qiskit`, `pennylane`, `cirq`, `tket`, `all`)
- GitHub Actions CI pipeline using `uv`
- MkDocs documentation setup
- **Ingest:** `CSVIngester` — CSV/TSV loading with auto delimiter detection and feature type inference (continuous, discrete, binary, categorical)
- **Ingest:** `NumpyIngester` — wraps NumPy arrays, Pandas DataFrames, and list-of-lists
- **Ingest:** `profiler.profile()` — dataset statistics (mean, std, min, max, missing counts per feature)
- **Clean:** `Imputer` — missing value handling with mean, median, mode, KNN, MICE, and drop strategies; column drop threshold
- **Clean:** `OutlierHandler` — IQR, Z-score, and Isolation Forest detection with clip or remove actions
- **Clean:** `CategoricalEncoder` — one-hot, label, and ordinal encoding; merges categorical columns into the numeric data matrix
- **Clean:** `FeatureSelector` — correlation, mutual information, and variance-based feature selection with optional qubit budget cap
- **Normalize:** `Scaler` — seven strategies: `l2`, `minmax`, `minmax_pi`, `minmax_pm_pi`, `zscore`, `binary`, `pm_one`
- **Normalize:** `auto_normalizer(encoding)` — returns the correct `Scaler` for a given encoding automatically
- **Encode:** `AngleEncoder` — maps features to single-qubit Ry/Rx/Rz rotation gates; depth O(1), NISQ-safe
- **Encode:** `AmplitudeEncoder` — embeds vectors as quantum state amplitudes; validates unit L2 norm; zero-pads to power of two
- **Encode:** `BasisEncoder` — binarizes features to computational basis states via X gates; depth O(1)
- **Export:** `QASMExporter` — OpenQASM 3.0 output for angle and basis encodings; no optional dependencies required
- **Export:** `QiskitExporter` — Qiskit `QuantumCircuit` output for angle, basis, and amplitude encodings (requires `pip install quprep[qiskit]`)
- **Pipeline:** `Pipeline.fit_transform()` — chains all stages with auto-normalization per encoding
- **Pipeline:** `prepare(source, encoding, framework)` — one-liner API; defaults to QASM output with no optional dependencies
- **CLI:** `quprep convert <file>` — converts CSV datasets to quantum circuits; supports `--encoding`, `--framework`, `--rotation`, `--output`, `--samples`

### Changed
- `Dataset` now carries `categorical_data` dict for non-numeric columns not yet encoded; `feature_names` and `feature_types` reflect numeric columns only
- `requires-python` bumped to `>=3.10` (pytket and modern quantum libs require it; Python 3.9 is EOL)

---

## [0.1.0] — _not yet released_

> First public alpha. Target: Phase 1 (Weeks 1–4).

### Planned
- PyPI release

---

[Unreleased]: https://github.com/quprep/quprep/compare/HEAD...HEAD
