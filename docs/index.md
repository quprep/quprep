# QuPrep — Quantum Data Preparation

**The missing preprocessing layer between classical datasets and quantum computing frameworks.**

```
CSV / DataFrame / NumPy  →  QuPrep  →  circuit-ready output for your framework
```

QuPrep converts classical datasets into quantum-circuit-ready format. It is not a quantum computing framework, simulator, or training tool. It is the preprocessing step that feeds into Qiskit, PennyLane, Cirq, TKET, and any other quantum workflow.

## Install

```bash
pip install quprep
pip install quprep[qiskit]   # with Qiskit export
pip install quprep[all]      # all framework exports
```

## Quickstart

```python
import quprep

# One line
circuit = quprep.prepare("data.csv", encoding="angle", framework="qiskit")

# Or build a pipeline
pipeline = quprep.Pipeline(
    encoder=quprep.AngleEncoder(),
    exporter=quprep.QiskitExporter(),
)
result = pipeline.fit_transform(df)
```

## What QuPrep does

| Stage | Description |
|---|---|
| Ingest | CSV, JSON, Parquet, Excel, NumPy, Pandas |
| Clean | Missing values, outliers, categoricals |
| Reduce | PCA, LDA, t-SNE, UMAP, DFT |
| Normalize | Auto-selected per encoding |
| Encode | Angle, amplitude, basis, IQP, re-upload, Hamiltonian |
| Export | Qiskit, PennyLane, Cirq, TKET, OpenQASM 3.0 |

## What QuPrep does NOT do

It does not train models, simulate circuits, run on quantum hardware, optimize parameters, or replace any existing framework.
