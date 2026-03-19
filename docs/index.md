# QuPrep

**Quantum data preparation — the missing preprocessing layer between classical datasets and quantum computing frameworks.**

```
CSV / DataFrame / NumPy  →  QuPrep  →  circuit-ready output for your framework
```

QuPrep converts classical datasets into quantum-circuit-ready format. It is not a quantum computing framework, simulator, or training tool. It is the preprocessing step that feeds into Qiskit, PennyLane, Cirq, TKET, and any other quantum workflow.

---

## Install

```bash
pip install quprep
pip install quprep[qiskit]   # with Qiskit export
pip install quprep[all]      # all framework exports
```

Requires Python ≥ 3.10.

---

## Quickstart

```python
import quprep

# One line — CSV to OpenQASM 3.0 (no extra dependencies)
result = quprep.prepare("data.csv", encoding="angle")
print(result.circuit)

# One line — CSV to Qiskit QuantumCircuit
result = quprep.prepare("data.csv", encoding="angle", framework="qiskit")
qc = result.circuit
qc.draw()
```

```python
# Build a pipeline manually
from quprep import Pipeline
from quprep.encode.angle import AngleEncoder
from quprep.export.qasm_export import QASMExporter

pipeline = Pipeline(
    encoder=AngleEncoder(rotation="ry"),
    exporter=QASMExporter(),
)
result = pipeline.fit_transform(df)
print(result.circuits[0])
```

---

## Pipeline stages

| Stage | Status | Description |
|---|---|---|
| **Ingest** | ✅ v0.1.0 | CSV, TSV, NumPy arrays, Pandas DataFrames |
| **Clean** | ✅ v0.1.0 | Missing values, outliers, categoricals, feature selection |
| **Normalize** | ✅ v0.1.0 | Auto-selected per encoding (L2, MinMax, Z-score, binary) |
| **Encode** | ✅ v0.1.0 | Angle, Amplitude, Basis |
| **Export** | ✅ v0.1.0 | OpenQASM 3.0, Qiskit |
| **Reduce** | 🔲 v0.2.0 | PCA, LDA, DFT, UMAP, hardware-aware |
| **Encode+** | 🔲 v0.2.0 | IQP, Data re-uploading, Hamiltonian |
| **Export+** | 🔲 v0.2.0 | PennyLane, Cirq, TKET |
| **QUBO** | 🔲 v0.3.0 | QUBO/Ising conversion, problem library |

---

## What QuPrep does NOT do

QuPrep is intentionally narrow in scope. It does not:

- Train quantum machine learning models
- Simulate quantum circuits
- Execute on quantum hardware
- Optimize variational parameters
- Replace Qiskit, PennyLane, Cirq, or any other framework

It prepares your data. Everything else is your framework's job.

---

## CLI

```bash
quprep convert data.csv --encoding angle --framework qasm
quprep convert data.csv --encoding angle --framework qasm --output circuit.qasm
quprep convert data.csv --encoding basis --samples 10
```
