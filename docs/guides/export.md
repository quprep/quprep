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

**Supported encodings:** `angle`, `entangled_angle`, `basis`, `iqp`, `zz_feature_map`, `pauli_feature_map`, `random_fourier`, `tensor_product`, `reupload`, `hamiltonian`. Amplitude encoding requires exponential-depth state preparation — use Qiskit for that.

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

## Amazon Braket

```bash
pip install quprep[braket]
```

Returns a `braket.circuits.Circuit`. Compatible with AWS managed simulators and IonQ, Rigetti, OQC, and other Braket hardware providers.

```python
from quprep.export.braket_export import BraketExporter

exp = BraketExporter()
circuit = exp.export(encoded_result)   # braket.circuits.Circuit
print(circuit)
```

**Supported encodings:** `angle`, `entangled_angle`, `basis`, `iqp`, `zz_feature_map`, `tensor_product`, `reupload`, `hamiltonian`.

```python
# Via prepare()
result = qd.prepare(data, encoding="angle", framework="braket")
```

---

## Q# (Microsoft Azure Quantum)

No extra dependencies required to generate Q# source strings. Install `quprep[qsharp]` only when submitting to Azure Quantum via the `qsharp` Python package.

```python
from quprep.export.qsharp_export import QSharpExporter

exp = QSharpExporter(namespace="MyExperiment", operation_name="FeatureMap")
qsharp_str = exp.export(encoded_result)   # str — Q# 1.0 source
print(qsharp_str)
```

```qsharp
namespace MyExperiment {
    open Microsoft.Quantum.Intrinsic;
    open Microsoft.Quantum.Math;

    operation FeatureMap() : Unit {
        use q = Qubit[3];
        Ry(0.7853981633974483, q[0]);
        Ry(1.5707963267948966, q[1]);
        Ry(0.5235987755982988, q[2]);
        ResetAll(q);
    }
}
```

**Supported encodings:** `angle`, `entangled_angle`, `basis`, `iqp`, `zz_feature_map`, `pauli_feature_map`, `random_fourier`, `tensor_product`, `reupload`, `hamiltonian`.

```python
# Via prepare()
result = qd.prepare(data, encoding="angle", framework="qsharp")
# result.circuits[0] is a Q# source string
```

---

## IQM

No extra dependencies required to generate the circuit dict. Install `quprep[iqm]` only when submitting to IQM hardware via `iqm-client`.

Uses IQM's native gate set: **PRX(angle_t, phase_t)** and **CZ**. Returns a plain Python `dict` matching the IQM circuit JSON schema — serializable with `json.dumps()`.

```python
import json
from quprep.export.iqm_export import IQMExporter

exp = IQMExporter(circuit_name="my_circuit", qubit_prefix="QB")
circuit_dict = exp.export(encoded_result)   # dict
print(json.dumps(circuit_dict, indent=2))
```

```json
{
  "name": "my_circuit",
  "instructions": [
    {"name": "prx", "qubits": ["QB1"], "args": {"angle_t": 0.125, "phase_t": 0.25}},
    {"name": "prx", "qubits": ["QB2"], "args": {"angle_t": 0.25,  "phase_t": 0.25}},
    {"name": "prx", "qubits": ["QB3"], "args": {"angle_t": 0.083, "phase_t": 0.25}}
  ]
}
```

Pass directly to `iqm_client.Circuit.from_dict()` when submitting to hardware.

**Gate mapping:** Ry → PRX(θ/2π, 0.25), Rx → PRX(θ/2π, 0), Rz → H·Rx·H (virtual decomposition), CZ → native.

**Supported encodings:** `angle`, `entangled_angle`, `basis`, `iqp`, `zz_feature_map`, `pauli_feature_map`, `random_fourier`, `tensor_product`, `reupload`, `hamiltonian`.

```python
# Via prepare()
result = qd.prepare(data, encoding="angle", framework="iqm")
# result.circuits[0] is a JSON-serializable dict
```

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
