# Framework Export

QuPrep encodes your data into circuit parameters. The exporter converts those parameters into a framework-specific circuit object.

---

## OpenQASM 3.0

No dependencies required. Universal interchange format accepted by all major frameworks and hardware platforms.

```python
import quprep as qd

exp = qd.QASMExporter()
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

Save an entire batch as individual files:

```python
paths = exp.save_batch(encoded_list, "circuits/")
# writes circuits/circuit_0000.qasm, circuit_0001.qasm, …

# custom stem
paths = exp.save_batch(encoded_list, "out/", stem="sample")
# writes out/sample_0000.qasm, sample_0001.qasm, …
```

The directory is created automatically if it does not exist. Returns a `list[Path]` of the written files.

Via the top-level helper (takes raw data, encodes internally):

```python
import quprep as qd

paths = qd.batch_export("data.csv", "circuits/", encoding="angle", stem="circuit")
# also works with np.ndarray or pd.DataFrame
```

**Supported encodings:** `angle`, `entangled_angle`, `basis`, `iqp`, `reupload`, `hamiltonian`. Amplitude encoding requires exponential-depth state preparation — use Qiskit for that.

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
import quprep as qd

# QASM (no deps)
result = qd.prepare("data.csv", encoding="angle")

# Qiskit
result = qd.prepare("data.csv", encoding="angle", framework="qiskit")

# Manual pipeline
pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
result = pipeline.fit_transform("data.csv")

print(result.circuit)         # first circuit
print(result.circuits)        # all circuits
print(result.circuits[3])     # fourth sample
```

---

## PennyLane

```bash
pip install quprep[pennylane]
```

Returns a callable `qml.QNode`. Supports `torch`, `jax`, and `tf` autodiff interfaces.

```python
from quprep.export.pennylane_export import PennyLaneExporter

exp = PennyLaneExporter(interface="torch", device="default.qubit")
qnode = exp.export(encoded_result)   # callable qml.QNode
output = qnode()                     # run the circuit
```

**Supported encodings:** all 7 encodings including amplitude (`qml.AmplitudeEmbedding`).

---

## Cirq

```bash
pip install quprep[cirq]
```

Returns a `cirq.Circuit` using `cirq.LineQubit`.

```python
from quprep.export.cirq_export import CirqExporter

exp = CirqExporter()
circuit = exp.export(encoded_result)   # cirq.Circuit
print(circuit)
```

**Supported encodings:** `angle`, `entangled_angle`, `basis`, `iqp`, `reupload`, `hamiltonian`. Amplitude raises `NotImplementedError` — use QiskitExporter.

---

## TKET / pytket

```bash
pip install quprep[tket]
```

Returns a `pytket.Circuit`. Angles are automatically converted from radians to pytket half-turns ($\text{angle}/\pi$).

```python
from quprep.export.tket_export import TKETExporter

exp = TKETExporter()
circuit = exp.export(encoded_result)   # pytket.Circuit
```

**Supported encodings:** same as Cirq.

---

## Visualization

Draw circuit diagrams without any additional dependencies:

```python
import quprep as qd

encoded = result.encoded[0]

# ASCII — always available
print(qd.draw_ascii(encoded))

# matplotlib — requires pip install quprep[viz]
fig = qd.draw_matplotlib(encoded)
fig.savefig("circuit.png")

# or save directly
qd.draw_matplotlib(encoded, filename="circuit.pdf")
```

Supports all 7 encodings. The matplotlib diagram shows qubit wires, gate boxes, and CNOT/ZZ connectors.
