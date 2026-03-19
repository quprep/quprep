# Framework Export

QuPrep encodes your data into circuit parameters. The exporter converts those parameters into a framework-specific circuit object.

---

## OpenQASM 3.0

No dependencies required. Universal interchange format accepted by all major frameworks and hardware platforms.

```python
from quprep.export.qasm_export import QASMExporter

exp = QASMExporter()
qasm_str = exp.export(encoded_result)
print(qasm_str)
```

```
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
ry(0.7853981633974483) q[0];
ry(1.5707963267948966) q[1];
ry(0.5235987755982988) q[2];
ry(1.0471975511965976) q[3];
```

Save to file:

```python
exp.save(encoded_result, "circuit.qasm")
```

Export a full batch:

```python
qasm_strings = exp.export_batch(encoded_list)
```

**Supported encodings:** `angle`, `basis`. Amplitude encoding requires exponential-depth state preparation — use Qiskit for that.

---

## Qiskit

```bash
pip install quprep[qiskit]
```

```python
from quprep.export.qiskit_export import QiskitExporter

exp = QiskitExporter()
qc = exp.export(encoded_result)   # qiskit.QuantumCircuit
qc.draw("mpl")
```

**Supported encodings:** `angle`, `basis`, `amplitude` (via `StatePreparation`).

```python
# Export a batch
circuits = exp.export_batch(encoded_list)
```

---

## Via the pipeline

The easiest way — pass an exporter to `Pipeline` or use `prepare()`:

```python
import quprep

# QASM (no deps)
result = quprep.prepare("data.csv", encoding="angle")

# Qiskit
result = quprep.prepare("data.csv", encoding="angle", framework="qiskit")

# Manual pipeline
from quprep import Pipeline
from quprep.encode.angle import AngleEncoder
from quprep.export.qasm_export import QASMExporter

pipeline = Pipeline(encoder=AngleEncoder(), exporter=QASMExporter())
result = pipeline.fit_transform("data.csv")

print(result.circuit)         # first circuit
print(result.circuits)        # all circuits
print(result.circuits[3])     # fourth sample
```

---

## Coming in v0.2.0

| Framework | Install | Notes |
|---|---|---|
| **PennyLane** | `pip install quprep[pennylane]` | torch/jax/tf autodiff interfaces |
| **Cirq** | `pip install quprep[cirq]` | Google QPU compatible |
| **TKET/pytket** | `pip install quprep[tket]` | Multi-vendor: IBM, IonQ, Quantinuum |
