# Exporters

Exporters convert `EncodedResult` objects into framework-specific circuit objects.

All exporters expose:

- `export(encoded)` — single sample
- `export_batch(encoded_list)` — full batch

---

## QASMExporter

No dependencies. Universal OpenQASM 3.0 output.

::: quprep.export.qasm_export.QASMExporter
    options:
      show_source: true

---

## QiskitExporter

Requires `pip install quprep[qiskit]`.

::: quprep.export.qiskit_export.QiskitExporter
    options:
      show_source: true

---

## Coming in v0.2.0

### PennyLaneExporter

```bash
pip install quprep[pennylane]
```

Produces PennyLane QNode templates. Supports `torch`, `jax`, and `tf` autodiff interfaces via the `interface` parameter.

### CirqExporter

```bash
pip install quprep[cirq]
```

Produces `cirq.Circuit` objects. Compatible with Google Quantum Computing Service.

### TKETExporter

```bash
pip install quprep[tket]
```

Produces `pytket.Circuit` objects. Multi-vendor: IBM, IonQ, Quantinuum, and more.
