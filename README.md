# QuPrep — Quantum Data Preparation

**The missing preprocessing layer between classical datasets and quantum computing frameworks.**

[![PyPI version](https://img.shields.io/pypi/v/quprep.svg)](https://pypi.org/project/quprep/)
[![Python 3.9+](https://img.shields.io/pypi/pyversions/quprep.svg)](https://pypi.org/project/quprep/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/quprep/quprep/actions/workflows/ci.yml/badge.svg)](https://github.com/quprep/quprep/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/quprep/badge/?version=latest)](https://quprep.readthedocs.io)

---

QuPrep converts classical datasets into quantum-circuit-ready format. It is **not** a quantum computing framework, simulator, or training tool. It is the preprocessing step that feeds into [Qiskit](https://qiskit.org), [PennyLane](https://pennylane.ai), [Cirq](https://quantumai.google/cirq), [TKET](https://tket.quantinuum.com), and any other quantum workflow.

Think of QuPrep as the **pandas of quantum data preparation**: a focused, composable tool that does one thing exceptionally well.

```
CSV / DataFrame / NumPy  →  QuPrep  →  circuit-ready output for your framework
```

## What QuPrep does

- Ingest CSV, JSON, Parquet, Excel, NumPy arrays, Pandas DataFrames
- Clean missing values, outliers, and categorical features
- Reduce dimensions with PCA, LDA, t-SNE, UMAP, DFT, and autoencoders
- Normalize data correctly per encoding method (automatically)
- Encode data using angle, amplitude, basis, IQP, Hamiltonian, and re-upload encodings
- Export circuit code for Qiskit, PennyLane, Cirq, TKET, and OpenQASM 3.0
- Convert optimization problems to QUBO / Ising format

## What QuPrep does NOT do

It does not train models, simulate circuits, run on quantum hardware, optimize variational parameters, or replace any existing framework. No hype, no overpromising.

---

## Installation

```bash
pip install quprep
```

With a specific framework export:

```bash
pip install quprep[qiskit]
pip install quprep[pennylane]
pip install quprep[cirq]
pip install quprep[tket]
pip install quprep[all]   # everything
```

**Requirements:** Python ≥ 3.9. Core dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`. Framework packages are optional.

---

## Quickstart

### One-liner

```python
import quprep

circuit = quprep.prepare("data.csv", encoding="angle", framework="qiskit")
circuit.draw()
```

### Pipeline API

```python
import quprep

pipeline = quprep.Pipeline(
    cleaner=quprep.Cleaner(impute="knn", outliers="clip"),
    reducer=quprep.LDAReducer(n_components=4),
    encoder=quprep.AngleEncoder(rotation="ry"),
    exporter=quprep.QiskitExporter(),
)
result = pipeline.fit_transform(df)
result.circuit.draw()
```

### Encoding recommendation

```python
rec = quprep.recommend(df, task="classification", qubits=8)
print(rec)
# EncodingRecommendation(method='angle_ry', qubits=8, depth=1, ...)

circuit = rec.apply(df, framework="pennylane")
```

### QUBO / Ising

```python
qubo = quprep.to_qubo(cost_matrix, constraints, penalty=10.0)
ising = qubo.to_ising()
```

### CLI

```bash
quprep convert data.csv --encoding angle --framework qiskit
quprep recommend data.csv --task classification --qubits 8
```

---

## Supported encodings

| Encoding | Qubits | Depth | NISQ-safe | Best for |
|---|---|---|---|---|
| Angle (Ry) | n = d | O(1) | Excellent | Most QML tasks |
| Amplitude | log₂(d) | O(2ⁿ) | Poor | Qubit-limited scenarios |
| Basis | n = d | O(1) | Excellent | Binary / QAOA |
| IQP | n = d | O(d²) | Medium | Kernel methods |
| Data Re-upload | n = d | O(d·L) | Medium | High-expressivity QNNs |
| Hamiltonian | n = d | O(d·T) | Poor | Physics simulations |

## Supported frameworks

Qiskit · PennyLane · Cirq · TKET/pytket · OpenQASM 3.0 · QUBO/Ising (D-Wave, QAOA)

---

## Documentation

Full documentation at **[quprep.readthedocs.io](https://quprep.readthedocs.io)**

- [Quickstart guide](https://quprep.readthedocs.io/quickstart)
- [API reference](https://quprep.readthedocs.io/api)
- [Encoding guide](https://quprep.readthedocs.io/encodings)
- [Tutorials](https://quprep.readthedocs.io/tutorials)

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

- [Open an issue](https://github.com/quprep/quprep/issues) for bugs or feature requests
- [Start a discussion](https://github.com/quprep/quprep/discussions) for questions or ideas
- See the [roadmap](https://github.com/quprep/quprep/discussions) for planned features

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use QuPrep in your research, please cite:

```bibtex
@software{quprep2026,
  author  = {Perera, Hasarindu},
  title   = {QuPrep: Quantum Data Preparation},
  year    = {2026},
  url     = {https://github.com/quprep/quprep},
  license = {Apache-2.0},
}
```
