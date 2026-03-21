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
pip install quprep[qiskit]     # Qiskit QuantumCircuit export
pip install quprep[pennylane]  # PennyLane QNode export
pip install quprep[cirq]       # Cirq Circuit export
pip install quprep[tket]       # TKET/pytket Circuit export
pip install quprep[viz]        # matplotlib circuit diagrams
pip install quprep[all]        # all framework exports + visualization
```

Requires Python ≥ 3.10.

---

## Quickstart

```python
import quprep

# One line — CSV to OpenQASM 3.0 (no extra dependencies)
result = quprep.prepare("data.csv", encoding="angle")
print(result.circuit)

# Get an encoding recommendation for your dataset and task
rec = quprep.recommend("data.csv", task="classification", qubits=8)
print(rec)                    # ranked recommendation with reasoning
result = rec.apply("data.csv")

# Visualize a circuit (no extra dependencies)
print(quprep.draw_ascii(result.encoded[0]))
```

```python
# Build a pipeline with reduction and Phase 2 encoding
from quprep import Pipeline
from quprep.reduce.pca import PCAReducer
from quprep.encode.iqp import IQPEncoder
from quprep.export.pennylane_export import PennyLaneExporter

pipeline = Pipeline(
    reducer=PCAReducer(n_components=8),
    encoder=IQPEncoder(reps=2),
    exporter=PennyLaneExporter(),
)
result = pipeline.fit_transform("data.csv")
qnode = result.circuit   # callable qml.QNode
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
| **Reduce** | ✅ v0.2.0 | PCA, LDA, DFT, t-SNE, UMAP, hardware-aware |
| **Encode+** | ✅ v0.2.0 | IQP, Entangled Angle, Data re-uploading, Hamiltonian |
| **Export+** | ✅ v0.2.0 | PennyLane, Cirq, TKET, ASCII + matplotlib visualization |
| **Recommend** | ✅ v0.2.0 | Automatic encoding selection for your dataset and task |
| **QUBO** | ✅ v0.3.0 | QUBO/Ising conversion, 7 problem formulations, solvers, QAOA, D-Wave export |

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
quprep convert data.csv --encoding iqp --framework pennylane
quprep convert data.csv --encoding basis --samples 10 --output circuits.qasm
quprep recommend data.csv --task classification --qubits 8

quprep qubo maxcut --adjacency "0,1,1;1,0,1;1,1,0" --solve
quprep qubo knapsack --weights "2,3,4" --values "3,4,5" --capacity 5
quprep qubo qaoa maxcut --adjacency "0,1,1;1,0,1;1,1,0" --p 2 --output circuit.qasm
```
