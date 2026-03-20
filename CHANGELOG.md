# Changelog

All notable changes to QuPrep will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
QuPrep uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

**Reduce**
- `PCAReducer` — wraps sklearn PCA; supports integer or variance-fraction `n_components`; `explained_variance_ratio_` property after fit
- `LDAReducer` — wraps sklearn LDA; maximises class separability; labels passed at init or fit time
- `SpectralReducer` — row-wise FFT, keeps first n frequency magnitudes; outputs always ≥ 0
- `TSNEReducer` — wraps sklearn TSNE with `random_state=42` for reproducibility
- `UMAPReducer` — wraps umap-learn (optional: `pip install umap-learn`); raises `ImportError` with install hint if absent
- `HardwareAwareReducer` — auto-reduces to a backend's qubit budget via PCA; accepts backend name (e.g. `'ibm_brisbane'`) or integer qubit count

**Encode**
- `EntangledAngleEncoder` — rotation layer + CNOT entangling layer, repeated `layers` times; supports `linear`, `circular`, and `full` entanglement topologies
- `IQPEncoder` — Havlíček et al. 2019 feature map with pairwise ZZ interactions; `reps` parameter
- `ReUploadEncoder` — Pérez-Salinas et al. 2020 data re-uploading; `layers` and `rotation` parameters
- `HamiltonianEncoder` — Trotterized single-qubit Z Hamiltonian evolution; `evolution_time` and `trotter_steps` parameters

**Export**
- `PennyLaneExporter` — returns a callable `qml.QNode`; supports all encodings; `interface` and `device` parameters (`pip install quprep[pennylane]`)
- `CirqExporter` — returns a `cirq.Circuit`; supports angle, basis, IQP, re-upload, Hamiltonian encodings (`pip install quprep[cirq]`)
- `TKETExporter` — returns a `pytket.Circuit`; angles auto-converted to pytket half-turns (`pip install quprep[tket]`)

**Recommend**
- `recommend(source, task, qubits)` — scores all encodings against dataset profile and task; returns `EncodingRecommendation` with ranked alternatives
- `EncodingRecommendation.apply()` — directly applies the recommendation to data and returns a `PipelineResult`

**CLI**
- `quprep recommend <file> [--task classification|regression|qaoa|kernel|simulation] [--qubits N]` — prints encoding recommendation with reasoning and alternatives
- `quprep convert` now supports `--framework pennylane|cirq|tket`

### Changed
- `QASMExporter` now supports entangled angle, IQP, re-upload, and Hamiltonian encodings
- `Pipeline` auto-normalizes IQP/re-upload → `minmax_pm_pi`, Hamiltonian → `zscore`
- `prepare()` accepts `encoding='iqp'`, `'reupload'`, `'hamiltonian'` with matching kwargs

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
