# Changelog

All notable changes to QuPrep will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
QuPrep uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


---

## [Unreleased]

---

## [0.5.0] — 2026-04-01

### Added

**Encoding comparison** (`quprep.compare`)
- `compare_encodings(source, *, include, exclude, task, qubits)` — analytical side-by-side cost comparison of all (or selected) encoders; no circuits generated
- `ComparisonResult` — `.rows` (list of `CostEstimate`), `.best(prefer="nisq"|"depth"|"gates"|"qubits")`, `.to_dict()`, `__str__()` ASCII table with starred recommendation when `task=` is passed
- `quprep compare <file> [--task] [--qubits] [--include] [--exclude]` CLI subcommand
- Exported: `qd.compare_encodings`, `qd.ComparisonResult`

**Smarter encoding recommendation** (`quprep.core.recommender`)
- `entangled_angle` added to recommendation engine (was an encoder but previously invisible to `recommend()`)
- 4 new dataset profile signals: `missing_rate`, `sparsity`, `has_negatives`, `feature_collinear` (mean pairwise Pearson correlation)
- 9 new dataset-aware scoring rules: amplitude penalised for large sample counts and high missing rate; basis boosted for sparse data, penalised for negative values; IQP/entangled_angle boosted for correlated features; IQP penalised for wide data; reupload penalised for tiny datasets, boosted for large ones

**Auto qubit count suggestion** (`quprep.core.qubit_suggestion`)
- `suggest_qubits(source, *, task, max_qubits)` → `QubitSuggestion` — recommends a qubit budget based on dataset size and target task
- `QubitSuggestion` — `.n_qubits`, `.n_features`, `.nisq_safe`, `.encoding_hint`, `.reasoning`, `.warning` (set when reduction is needed)
- `quprep suggest <file> [--task] [--max-qubits]` CLI subcommand
- Exported: `qd.suggest_qubits`, `qd.QubitSuggestion`

**Pipeline serialization** (`quprep.core.pipeline`)
- `Pipeline.save(path)` — pickles the fitted pipeline; creates parent directories automatically
- `Pipeline.load(path)` — classmethod; restores a fitted pipeline ready for `transform()` without re-fitting; raises `TypeError` for non-Pipeline files

**Batch export** (`quprep.export.qasm_export`, `quprep.__init__`)
- `QASMExporter.save_batch(encoded_list, directory, stem)` — saves each sample as `{stem}_{i:04d}.qasm`; creates output directory automatically; returns list of `Path` objects
- `qd.batch_export(source, directory, *, encoding, stem)` — top-level one-liner: runs `prepare()` then `save_batch()`
- `quprep convert <file> --save-dir <dir> [--stem <stem>]` — CLI flag added to `convert` subcommand

**Data drift detection** (`quprep.core.drift`)
- `DriftDetector(mean_threshold=3.0, std_threshold=2.0, warn=True)` — detects statistical drift between training and new data
- `fit(dataset)` — records per-feature mean and std from training data (NaN-safe)
- `check(dataset)` → `DriftReport` — flags features where mean shifts > threshold σ or std ratio exceeds bounds; issues `QuPrepWarning` when drift found
- `DriftReport` — `.overall_drift`, `.drifted_features`, `.n_features_drifted`, `.feature_stats` (per-feature train/new mean, std, σ-shift, std_ratio)
- `Pipeline(drift_detector=DriftDetector())` — detector fitted post-reduction, checked on every `transform()` call
- `PipelineResult.drift_report` — `DriftReport | None`; preserved through `save()`/`load()`
- Exported: `qd.DriftDetector`, `qd.DriftReport`

### Changed
- `Pipeline.__init__` — new `drift_detector` parameter (default `None`; backwards compatible)
- `PipelineResult.__init__` — new `drift_report` attribute (default `None`; backwards compatible)
- `Pipeline.get_params()` / `set_params()` — `drift_detector` included in parameter dict

---

## [0.4.0] — 2026-03-28

### Added

**Validation & schema** (`quprep.validation`)
- `QuPrepWarning` — custom warning class; all pipeline warnings use this category so they can be filtered precisely
- `validate_dataset(dataset)` — structural checks at pipeline entry: shape, dtype, NaN detection with fractional coverage warning
- `warn_qubit_mismatch(n_features, n_qubits, encoding)` — warns when features exceed qubit budget
- `DataSchema` / `FeatureSpec` / `SchemaViolationError` — declare expected feature names, types, and value ranges; attach via `Pipeline(schema=...)` to enforce at entry; all violations collected and reported together
- `DataSchema.infer(dataset)` — auto-builds schema from a reference dataset
- `DataSchema.to_json()` / `from_json()` / `to_dict()` / `from_dict()` — full serialisation round-trip; terse output (omits `None` fields and `nullable=False`)

**Cost estimation** (`quprep.validation.cost`)
- `CostEstimate` — dataclass: `encoding`, `n_features`, `n_qubits`, `gate_count`, `circuit_depth`, `two_qubit_gates`, `nisq_safe`, `warning`
- `estimate_cost(encoder, n_features)` — formula-accurate gate counts for all 7 encoders; NISQ-safe flag (depth < 200, CNOTs < 50)

**Pipeline & PipelineResult** (`quprep.core.pipeline`)
- `PipelineResult.cost` — `CostEstimate | None`; populated at fit time whenever an encoder is configured; shown in `repr()`
- `PipelineResult.audit_log` — `list[dict] | None`; one entry per preprocessing stage with `{stage, n_samples_in, n_features_in, n_samples_out, n_features_out}`
- `PipelineResult.summary()` — prints audit log as an aligned table and cost breakdown
- `Pipeline.fit(source, y=None)` / `.transform(source)` — full sklearn-compatible split; `transform()` raises `RuntimeError` before `fit()`
- `Pipeline.get_params()` / `.set_params(**params)` — hyperparameter search ready
- `Pipeline(schema=...)` — validates dataset at entry before any stage runs
- `Pipeline.summary()` / `__str__` — human-readable snapshot: configured stages, fitted status, resolved normalizer, schema feature count, last cost estimate

**Sklearn-compatible fit/transform on all stateful stages**
- Every stage now has separate `fit(dataset)` and `transform(dataset)` methods; `fit_transform` delegates; `NotFittedError` raised on `transform()` before `fit()`
- Stages: `Scaler`, `Imputer`, `OutlierHandler`, `CategoricalEncoder`, `FeatureSelector`, `PCAReducer`, `LDAReducer`, `HardwareAwareReducer`, `SpectralReducer`, `TSNEReducer`, `UMAPReducer`
- `CategoricalEncoder` aligns one-hot columns between train and test sets at transform time
- `Dataset.copy()` — deep copy for safe fit/transform stage splitting

**`import quprep as qd`** — top-level namespace alias
- All public classes exported directly: all 7 encoders, all cleaners (`Imputer`, `OutlierHandler`, `CategoricalEncoder`, `FeatureSelector`), `Scaler`, all reducers, `QASMExporter`, `PipelineResult`, all validation classes
- No sub-imports needed: `qd.AngleEncoder()`, `qd.PCAReducer()`, `qd.DataSchema(...)`, etc.

**`quprep validate` CLI**
- `quprep validate dataset.csv` — shape, column names, NaN report per column (count + %), value ranges
- `quprep validate dataset.csv --schema schema.json` — validates against a JSON schema (array of `{name, dtype, min_value?, max_value?, nullable?}`); exits 1 on violation
- `quprep validate dataset.csv --infer-schema output.json` — infers schema from the CSV and writes it to a file; use `"-"` to print to stdout

**Type stubs** (`.pyi` files)
- Stubs added for: `Dataset`, `Pipeline` / `PipelineResult`, `Scaler`, `BaseEncoder` / `EncodedResult`, and the full `validation` public API

**Zenodo DOI**
- Placeholder badge and `doi` field added to README and BibTeX citation
- No custom GitHub Actions workflow needed — Zenodo's native GitHub integration archives each Release automatically

---

## [0.3.0] — 2026-03-21

### Added

**QUBO / Ising conversion** (`quprep.qubo`)
- `to_qubo(cost_matrix, constraints, penalty)` — converts any square cost matrix to upper-triangular QUBO form; supports equality and inequality constraints via Lagrangian penalty
- `QUBOResult` — holds Q matrix, offset, variable map, n_original; `.to_ising()`, `.evaluate(x)`, `.to_dwave()`, `.to_dict()` / `.from_dict()` methods
- `IsingResult` — holds h, J, offset; `.to_qubo()` round-trip conversion
- `qubo_to_ising(qubo)` — QUBO → Ising transformation (s = 2x − 1); energy-consistent for all binary inputs
- `ising_to_qubo(ising)` — Ising → QUBO inverse transformation; completes the bidirectional round-trip
- `equality_penalty(A, b, penalty)` — encodes Ax = b as a QUBO penalty matrix
- `inequality_penalty(A, b, penalty)` — encodes Ax ≤ b via binary slack variables; augments Q from (n,n) to (n+K,n+K)
- `add_qubo(q1, q2, weight)` — combines two same-size QUBOs; useful for multi-objective problems

**Problem library** (`quprep.qubo.problems`) — 7 NP-hard combinatorial problems
- `max_cut(adjacency)` — Max-Cut graph partitioning
- `knapsack(weights, values, capacity, penalty)` — 0/1 Knapsack
- `tsp(distance_matrix, penalty)` — Travelling Salesman Problem (n² binary variables)
- `portfolio(returns, covariance, budget, risk_penalty, budget_penalty)` — Markowitz portfolio optimization
- `graph_color(adjacency, n_colors, penalty)` — Graph Colouring (n×K binary variables)
- `scheduling(processing_times, n_machines, penalty)` — Job scheduling / load balancing
- `number_partition(values, penalty)` — Number Partitioning

**Solvers** (`quprep.qubo.solver`)
- `solve_brute(qubo, max_n=20)` — exact exhaustive solver; evaluates all 2^n states; practical up to n=20
- `solve_sa(qubo, n_steps, T_start, T_end, seed, restarts)` — simulated annealing heuristic; O(n) incremental energy update with geometric cooling; scales to n ~ 500+

**QAOA circuit generator** (`quprep.qubo.qaoa`)
- `qaoa_circuit(qubo, p, gamma, beta)` — generates a p-layer QAOA ansatz as OpenQASM 3.0; converts QUBO → Ising internally; compatible with Qiskit, Cirq, and any QASM backend

**Visualization** (`quprep.qubo.visualize`) — requires `pip install quprep[viz]`
- `draw_qubo(qubo, title, cmap, ax)` — heatmap of Q matrix with symmetric colour scale; annotates cells for n ≤ 10
- `draw_ising(ising, title, ax)` — circular graph layout; node colour = h_i bias; edge colour/width = J_ij coupling strength

**CLI** (`quprep qubo`)
- `quprep qubo maxcut --adjacency ... [--solve]`
- `quprep qubo knapsack --weights ... --values ... --capacity ... [--solve]`
- `quprep qubo tsp --distances ... [--solve]`
- `quprep qubo schedule --times ... --machines ... [--solve]`
- `quprep qubo partition --values ... [--solve]`
- `quprep qubo portfolio --returns ... --covariance ... --budget ... [--solve]`
- `quprep qubo graphcolor --adjacency ... --colors ... [--solve]`
- `quprep qubo qaoa <problem> ... [--p N] [--gamma ...] [--beta ...] [--output file]`
- `quprep qubo export <problem> ... [--format json|npy] [--output file]`
- `--solve` auto-switches from exact to simulated annealing for n > 20

---

## [0.2.0] — 2026-03-21

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
- `draw_ascii(encoded)` — no-dependency ASCII circuit diagram for any `EncodedResult`; returns a printable string
- `draw_matplotlib(encoded, filename=None)` — matplotlib circuit diagram; returns a `Figure` or saves to PNG/PDF/SVG (`pip install quprep[viz]`)

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

### Fixed
- `HamiltonianEncoder` via `prepare()` and `Pipeline` was broken — `_encoding_key()` returned `"zscore"` (a strategy name) instead of `"hamiltonian"`, causing `auto_normalizer()` to raise `ValueError`

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

[Unreleased]: https://github.com/quprep/quprep/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/quprep/quprep/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/quprep/quprep/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/quprep/quprep/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/quprep/quprep/releases/tag/v0.1.0
