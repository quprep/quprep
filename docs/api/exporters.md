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

## PennyLaneExporter

Requires `pip install quprep[pennylane]`.

::: quprep.export.pennylane_export.PennyLaneExporter
    options:
      show_source: true

---

## CirqExporter

Requires `pip install quprep[cirq]`.

::: quprep.export.cirq_export.CirqExporter
    options:
      show_source: true

---

## TKETExporter

Requires `pip install quprep[tket]`.

::: quprep.export.tket_export.TKETExporter
    options:
      show_source: true

---

## BraketExporter

Requires `pip install quprep[braket]`.

::: quprep.export.braket_export.BraketExporter
    options:
      show_source: true

---

## QSharpExporter

No dependencies for string generation. Requires `pip install quprep[qsharp]` for Azure Quantum submission.

::: quprep.export.qsharp_export.QSharpExporter
    options:
      show_source: true

---

## IQMExporter

No dependencies for dict generation. Requires `pip install quprep[iqm]` for hardware submission.

::: quprep.export.iqm_export.IQMExporter
    options:
      show_source: true

---

## Plugin registry

Register custom encoders and exporters for use with `prepare()`.

::: quprep.plugins.register_encoder

::: quprep.plugins.register_exporter

::: quprep.plugins.list_encoders

::: quprep.plugins.list_exporters

::: quprep.plugins.get_encoder_class

::: quprep.plugins.get_exporter_class

::: quprep.plugins.unregister_encoder

::: quprep.plugins.unregister_exporter

---

## Visualization

No dependencies for `draw_ascii`. Requires `pip install quprep[viz]` for `draw_matplotlib`.

::: quprep.export.visualize.draw_ascii

::: quprep.export.visualize.draw_matplotlib
