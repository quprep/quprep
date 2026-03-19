# Changelog

All notable changes to QuPrep will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
QuPrep uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.1.0] — 2026-03-19

First public release. Covers the full ingest → clean → normalize → encode → export pipeline.

### Added

**Ingest**
- `CSVIngester` — CSV/TSV loading with auto delimiter detection and feature type inference (continuous, discrete, binary, categorical)
- `NumpyIngester` — wraps NumPy arrays, Pandas DataFrames, and list-of-lists
- `profiler.profile()` — dataset statistics: mean, std, min, max, missing counts per feature

**Clean**
- `Imputer` — missing value handling: mean, median, mode, KNN, MICE, and drop strategies; column drop threshold
- `OutlierHandler` — IQR, Z-score, and Isolation Forest detection with clip or remove actions
- `CategoricalEncoder` — one-hot, label, and ordinal encoding; merges categorical columns into the numeric matrix
- `FeatureSelector` — correlation, mutual information, and variance-based feature selection with optional qubit budget cap

**Normalize**
- `Scaler` — seven strategies: `l2`, `minmax`, `minmax_pi`, `minmax_pm_pi`, `zscore`, `binary`, `pm_one`
- `auto_normalizer(encoding)` — selects the mathematically correct scaler for a given encoding automatically

**Encode**
- `AngleEncoder` — maps features to single-qubit Ry/Rx/Rz rotation gates; depth O(1), NISQ-safe
- `AmplitudeEncoder` — embeds feature vectors as quantum state amplitudes; validates unit L2 norm; auto-pads to next power of two
- `BasisEncoder` — binarizes features to computational basis states via X gates; depth O(1)

**Export**
- `QASMExporter` — OpenQASM 3.0 output for angle and basis encodings; no optional dependencies required
- `QiskitExporter` — Qiskit `QuantumCircuit` output for angle, basis, and amplitude encodings (`pip install quprep[qiskit]`)

**Pipeline**
- `Pipeline.fit_transform()` — chains all stages with automatic normalization per encoding type
- `prepare(source, encoding, framework)` — one-liner API; defaults to QASM output with no optional dependencies

**CLI**
- `quprep convert <file>` — converts a CSV dataset to quantum circuits; supports `--encoding`, `--framework`, `--rotation`, `--output`, `--samples`

**Project**
- Apache 2.0 license
- GitHub Actions CI: Python 3.10, 3.11, 3.12 matrix
- MkDocs Material documentation with MathJax
- Read the Docs integration
- `examples/` — four worked examples as Python scripts and Jupyter notebooks

### Changed
- `requires-python` set to `>=3.10` (pytket and modern quantum libs require it)

---

[Unreleased]: https://github.com/quprep/quprep/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/quprep/quprep/releases/tag/v0.1.0
