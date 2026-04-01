# QuPrep — Quantum Data Preparation

**The missing preprocessing layer between classical datasets and quantum computing frameworks.**

[![PyPI version](https://img.shields.io/pypi/v/quprep.svg)](https://pypi.org/project/quprep/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/quprep.svg)](https://pypi.org/project/quprep/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1185611576.svg)](https://doi.org/10.5281/zenodo.19286258)
[![Documentation](https://readthedocs.org/projects/quprep/badge/?version=latest)](https://docs.quprep.org)
[![CI](https://github.com/quprep/quprep/actions/workflows/ci.yml/badge.svg)](https://github.com/quprep/quprep/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/quprep/quprep/graph/badge.svg?token=I26OBPRZ86)](https://codecov.io/github/quprep/quprep)
[![CodeQL](https://github.com/quprep/quprep/actions/workflows/codeql.yml/badge.svg)](https://github.com/quprep/quprep/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/quprep/quprep/badge)](https://scorecard.dev/viewer/?uri=github.com/quprep/quprep)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12341/badge)](https://www.bestpractices.dev/projects/12341)

---

QuPrep converts classical datasets into quantum-circuit-ready format. It is **not** a quantum computing framework, simulator, or training tool. It is the preprocessing step that feeds into [Qiskit](https://qiskit.org), [PennyLane](https://pennylane.ai), [Cirq](https://quantumai.google/cirq), [TKET](https://tket.quantinuum.com), and any other quantum workflow.

Think of QuPrep as the **pandas of quantum data preparation**: a focused, composable tool that does one thing exceptionally well.

```
CSV / DataFrame / NumPy  →  QuPrep  →  circuit-ready output for your framework
```

## What QuPrep does

- Ingest CSV, NumPy arrays, and Pandas DataFrames
- Clean missing values, outliers, and categorical features
- Reduce dimensionality to fit your hardware qubit budget (PCA, LDA, DFT, UMAP, hardware-aware)
- Normalize data correctly per encoding method — automatically
- Encode data using 7 encoding methods: Angle, Amplitude, Basis, IQP, Entangled Angle, Re-uploading, Hamiltonian
- Recommend the best encoding for your dataset and task
- Suggest a qubit budget based on dataset size and target task
- Compare encoders side-by-side on cost, depth, and NISQ safety
- Export circuits to OpenQASM 3.0, Qiskit, PennyLane, Cirq, and TKET
- Save entire batches of circuits as individual QASM files
- Visualize circuits as ASCII diagrams or matplotlib figures
- Save and reload fitted pipelines without re-fitting
- Detect data drift between training and new data automatically
- Formulate combinatorial optimization problems as QUBO / Ising models (Max-Cut, TSP, Knapsack, Portfolio, Graph Colouring, Scheduling, Number Partitioning)
- Solve with exact brute-force (n ≤ 20) or simulated annealing (any n)
- Generate QAOA circuits and export to D-Wave Ocean SDK format

## What QuPrep does NOT do

It does not train models, simulate circuits, run on quantum hardware, optimize variational parameters, or replace any existing framework.

---

## Installation

```bash
pip install quprep
```

With optional framework exports:

```bash
pip install quprep[qiskit]     # Qiskit QuantumCircuit
pip install quprep[pennylane]  # PennyLane QNode
pip install quprep[cirq]       # Cirq Circuit
pip install quprep[tket]       # TKET/pytket Circuit
pip install quprep[viz]        # matplotlib circuit diagrams
pip install quprep[all]        # everything above
```

**Requirements:** Python ≥ 3.10. Core dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`.

---

## Quickstart

### One-liner

```python
import quprep as qd

result = qd.prepare("data.csv", encoding="angle", framework="qasm")
print(result.circuit)
```

### Encoding recommendation

```python
import quprep as qd

rec = qd.recommend("data.csv", task="classification", qubits=8)
print(rec)                    # ranked table with reasoning
result = rec.apply("data.csv")
```

### Pipeline API

```python
import quprep as qd  # all public classes on the top-level namespace

pipeline = qd.Pipeline(
    reducer=qd.PCAReducer(n_components=8),
    encoder=qd.IQPEncoder(reps=2),
    exporter=qd.PennyLaneExporter(),   # pip install quprep[pennylane]
)
result = pipeline.fit_transform("data.csv")
qnode = result.circuit   # callable qml.QNode
```

### Circuit visualization

```python
import quprep as qd

# ASCII — no dependencies
print(qd.draw_ascii(result.encoded[0]))

# matplotlib — pip install quprep[viz]
qd.draw_matplotlib(result.encoded[0], filename="circuit.png")
```

### QUBO / combinatorial optimization

```python
from quprep.qubo import max_cut, knapsack, solve_brute, solve_sa, qaoa_circuit
import numpy as np

# Max-Cut on a weighted graph
adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)
print(q.evaluate(np.array([0., 1., 1.])))  # -2.0

# Brute-force (n ≤ 20) or simulated annealing (any n)
sol = solve_brute(q)        # exact
sol = solve_sa(q, seed=42)  # heuristic, scales to n ~ 500+

# Generate a QAOA circuit
qasm = qaoa_circuit(q, p=2)

# D-Wave Ocean SDK export
bqm_dict = q.to_dwave()   # {(i, j): coeff}
```

### Qubit suggestion

```python
import quprep as qd

s = qd.suggest_qubits("data.csv", task="classification")
print(s.n_qubits)        # recommended qubit count
print(s.encoding_hint)   # e.g. "angle"
print(s.warning)         # set if dataset exceeds NISQ ceiling
```

### Data drift detection

```python
import quprep as qd

det = qd.DriftDetector()
pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), drift_detector=det)
pipeline.fit(X_train)

result = pipeline.transform(X_new)
print(result.drift_report.overall_drift)      # True / False
print(result.drift_report.drifted_features)   # list of feature names
```

### Pipeline save / load

```python
import quprep as qd

pipeline = qd.Pipeline(reducer=qd.PCAReducer(n_components=8), encoder=qd.AngleEncoder())
pipeline.fit(X_train)
pipeline.save("pipeline.pkl")

loaded = qd.Pipeline.load("pipeline.pkl")
result = loaded.transform(X_new)   # no re-fitting needed
```

### Validation & cost estimation

```python
import quprep as qd

# Define expected schema and attach to pipeline
schema = qd.DataSchema([
    qd.FeatureSpec("age",    dtype="continuous", min_value=0, max_value=120),
    qd.FeatureSpec("income", dtype="continuous", min_value=0),
])
pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), schema=schema)
result = pipeline.fit_transform("data.csv")

# Cost estimate is computed automatically at fit time
print(result.cost.nisq_safe)    # True
print(result.cost.circuit_depth)
result.summary()                # audit table + cost breakdown
```

### CLI

```bash
quprep convert data.csv --encoding angle --framework qasm
quprep convert data.csv --encoding iqp --framework pennylane
quprep convert data.csv --encoding angle --save-dir circuits/  # save each sample as a file

quprep recommend data.csv --task classification --qubits 8
quprep suggest data.csv --task classification       # qubit budget recommendation
quprep compare data.csv --task classification       # side-by-side encoder comparison

quprep validate data.csv                              # shape, columns, NaN report
quprep validate data.csv --infer-schema schema.json  # infer schema and save
quprep validate data.csv --schema schema.json        # enforce schema (exit 1 on violation)

quprep qubo maxcut --adjacency "0,1,1;1,0,1;1,1,0" --solve
quprep qubo knapsack --weights "2,3,4" --values "3,4,5" --capacity 5
quprep qubo qaoa maxcut --adjacency "0,1,1;1,0,1;1,1,0" --p 2 --output circuit.qasm
```

---

## Supported encodings

| Encoding | Qubits | Depth | NISQ-safe | Best for |
|---|---|---|---|---|
| Angle (Ry/Rx/Rz) | n = d | O(1) | ✅ Excellent | Most QML tasks |
| Amplitude | ⌈log₂ d⌉ | O(2ⁿ) | ❌ Poor | Qubit-limited scenarios |
| Basis | n = d | O(1) | ✅ Excellent | Binary features / QAOA |
| Entangled Angle | n = d | O(d · layers) | ✅ Good | Feature correlations |
| IQP | n = d | O(d² · reps) | ⚠️ Medium | Kernel methods |
| Re-uploading | n = d | O(d · layers) | ✅ Good | High-expressivity QNNs |
| Hamiltonian | n = d | O(d · steps) | ⚠️ Medium | Physics simulation / VQE |

## Supported export frameworks

| Framework | Install | Output |
|---|---|---|
| OpenQASM 3.0 | _(included)_ | `str` |
| Qiskit | `pip install quprep[qiskit]` | `QuantumCircuit` |
| PennyLane | `pip install quprep[pennylane]` | `qml.QNode` |
| Cirq | `pip install quprep[cirq]` | `cirq.Circuit` |
| TKET | `pip install quprep[tket]` | `pytket.Circuit` |

---

## Documentation

Full documentation at **[docs.quprep.org](https://docs.quprep.org)**

- [Installation](https://docs.quprep.org/getting-started/installation)
- [Quickstart guide](https://docs.quprep.org/getting-started/quickstart)
- [Encoding guide](https://docs.quprep.org/guides/encodings)
- [API reference](https://docs.quprep.org/api)

---

## Examples

See the [`examples/`](examples/) directory:

| # | Topic |
|---|---|
| 01 | `prepare()` one-liner |
| 02 | Full pipeline |
| 03 | All 7 encoders compared |
| 04 | Framework export — QASM, Qiskit, PennyLane, Cirq, TKET |
| 05 | Encoding recommendation |
| 06 | Circuit visualization |
| 07 | QUBO / Ising — Max-Cut, Knapsack, solvers, D-Wave export, QAOA |
| 08 | Validation, schema & cost — `DataSchema`, `estimate_cost`, `result.summary()` |
| 09 | Data drift detection — `DriftDetector`, pipeline integration, serialization |
| 10 | Qubit suggestion — `suggest_qubits`, task hints, NISQ ceiling |

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

- [Open an issue](https://github.com/quprep/quprep/issues) for bugs or feature requests
- [Start a discussion](https://github.com/quprep/quprep/discussions) for questions or ideas

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use QuPrep in your research, please cite:

```bibtex
@software{quprep2026,
  author    = {Perera, Hasarindu},
  title     = {QuPrep: Quantum Data Preparation},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.5.0},
  doi       = {10.5281/zenodo.19286258},
  url       = {https://doi.org/10.5281/zenodo.19286258},
  license   = {Apache-2.0},
}
```
