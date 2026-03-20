# QuPrep — Quantum Data Preparation

**The missing preprocessing layer between classical datasets and quantum computing frameworks.**

[![PyPI version](https://img.shields.io/pypi/v/quprep.svg)](https://pypi.org/project/quprep/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/quprep.svg)](https://pypi.org/project/quprep/)
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

- Ingest CSV, NumPy arrays, and Pandas DataFrames
- Clean missing values, outliers, and categorical features
- Reduce dimensionality to fit your hardware qubit budget (PCA, LDA, DFT, UMAP, hardware-aware)
- Normalize data correctly per encoding method — automatically
- Encode data using 7 encoding methods: Angle, Amplitude, Basis, IQP, Entangled Angle, Re-uploading, Hamiltonian
- Recommend the best encoding for your dataset and task
- Export circuits to OpenQASM 3.0, Qiskit, PennyLane, Cirq, and TKET
- Visualize circuits as ASCII diagrams or matplotlib figures

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
import quprep

result = quprep.prepare("data.csv", encoding="angle", framework="qasm")
print(result.circuit)
```

### Encoding recommendation

```python
rec = quprep.recommend("data.csv", task="classification", qubits=8)
print(rec)                    # ranked table with reasoning
result = rec.apply("data.csv")
```

### Pipeline API

```python
from quprep import Pipeline
from quprep.reduce.pca import PCAReducer
from quprep.encode.iqp import IQPEncoder
from quprep.export.pennylane_export import PennyLaneExporter  # pip install quprep[pennylane]

pipeline = Pipeline(
    reducer=PCAReducer(n_components=8),
    encoder=IQPEncoder(reps=2),
    exporter=PennyLaneExporter(),
)
result = pipeline.fit_transform("data.csv")
qnode = result.circuit   # callable qml.QNode
```

### Circuit visualization

```python
# ASCII — no dependencies
print(quprep.draw_ascii(result.encoded[0]))

# matplotlib — pip install quprep[viz]
quprep.draw_matplotlib(result.encoded[0], filename="circuit.png")
```

### CLI

```bash
quprep convert data.csv --encoding angle --framework qasm
quprep convert data.csv --encoding iqp --framework pennylane
quprep recommend data.csv --task classification --qubits 8
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

Full documentation at **[quprep.readthedocs.io](https://quprep.readthedocs.io)**

- [Installation](https://quprep.readthedocs.io/getting-started/installation)
- [Quickstart guide](https://quprep.readthedocs.io/getting-started/quickstart)
- [Encoding guide](https://quprep.readthedocs.io/guides/encodings)
- [API reference](https://quprep.readthedocs.io/api)

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
  author  = {Perera, Hasarindu},
  title   = {QuPrep: Quantum Data Preparation},
  year    = {2026},
  url     = {https://github.com/quprep/quprep},
  license = {Apache-2.0},
}
```
