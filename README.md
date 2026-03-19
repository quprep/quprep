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
- Normalize data correctly per encoding method — automatically
- Encode data using angle, amplitude, and basis encodings
- Export circuit code for Qiskit and OpenQASM 3.0

## What QuPrep does NOT do

It does not train models, simulate circuits, run on quantum hardware, optimize variational parameters, or replace any existing framework.

---

## Installation

```bash
pip install quprep
```

With Qiskit export support:

```bash
pip install quprep[qiskit]
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

### Pipeline API

```python
from quprep import Pipeline
from quprep.clean.imputer import Imputer
from quprep.encode.angle import AngleEncoder
from quprep.export.qasm_export import QASMExporter

pipeline = Pipeline(
    cleaner=Imputer(strategy="knn"),
    encoder=AngleEncoder(rotation="ry"),
    exporter=QASMExporter(),
)
result = pipeline.fit_transform("data.csv")
print(result.circuit)        # first sample QASM string
print(result.circuits)       # all samples
```

### Qiskit export

```python
from quprep import Pipeline
from quprep.encode.angle import AngleEncoder
from quprep.export.qiskit_export import QiskitExporter  # pip install quprep[qiskit]

result = Pipeline(
    encoder=AngleEncoder(),
    exporter=QiskitExporter(),
).fit_transform("data.csv")

result.circuit.draw()   # Qiskit QuantumCircuit
```

### CLI

```bash
quprep convert data.csv --encoding angle --framework qasm
quprep convert data.csv --encoding basis --output circuits/
```

---

## Supported encodings (v0.1.0)

| Encoding | Qubits | Depth | NISQ-safe | Best for |
|---|---|---|---|---|
| Angle (Ry/Rx/Rz) | n = d | O(1) | Excellent | Most QML tasks |
| Amplitude | ⌈log₂ d⌉ | O(2ⁿ) | Poor | Qubit-limited scenarios |
| Basis | n = d | O(1) | Excellent | Binary features / QAOA |

## Supported export frameworks (v0.1.0)

| Framework | Install | Output |
|---|---|---|
| OpenQASM 3.0 | _(included)_ | `str` |
| Qiskit | `pip install quprep[qiskit]` | `QuantumCircuit` |

PennyLane, Cirq, and TKET exporters are planned for v0.2.0.

---

## Documentation

Full documentation at **[quprep.readthedocs.io](https://quprep.readthedocs.io)**

- [Installation](https://quprep.readthedocs.io/getting-started/installation)
- [Quickstart guide](https://quprep.readthedocs.io/getting-started/quickstart)
- [Encoding guide](https://quprep.readthedocs.io/guides/encodings)
- [API reference](https://quprep.readthedocs.io/api)

---

## Examples

See the [`examples/`](examples/) directory for worked examples as Python scripts and Jupyter notebooks.

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
