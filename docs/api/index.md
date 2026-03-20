# API Reference

QuPrep's public API has three top-level entry points:

| Name | Description |
|---|---|
| `quprep.prepare()` | One-liner: source → circuits |
| `quprep.Pipeline` | Composable pipeline with full stage control |
| `quprep.recommend()` | Encoding recommendation engine |

Everything else is accessed via submodules.

---

## Top-level

```python
import quprep

quprep.__version__   # "0.2.0"
quprep.prepare(...)
quprep.Pipeline(...)
quprep.recommend(...)
quprep.draw_ascii(...)
quprep.draw_matplotlib(...)
```

---

## Submodules

| Module | Contents |
|---|---|
| `quprep.encode.angle` | `AngleEncoder` |
| `quprep.encode.amplitude` | `AmplitudeEncoder` |
| `quprep.encode.basis` | `BasisEncoder` |
| `quprep.encode.entangled_angle` | `EntangledAngleEncoder` |
| `quprep.encode.iqp` | `IQPEncoder` |
| `quprep.encode.reupload` | `ReUploadEncoder` |
| `quprep.encode.hamiltonian` | `HamiltonianEncoder` |
| `quprep.encode.base` | `BaseEncoder`, `EncodedResult` |
| `quprep.export.qasm_export` | `QASMExporter` |
| `quprep.export.qiskit_export` | `QiskitExporter` |
| `quprep.export.pennylane_export` | `PennyLaneExporter` |
| `quprep.export.cirq_export` | `CirqExporter` |
| `quprep.export.tket_export` | `TKETExporter` |
| `quprep.export.visualize` | `draw_ascii`, `draw_matplotlib` |
| `quprep.normalize.scalers` | `Scaler`, `auto_normalizer` |
| `quprep.clean.imputer` | `Imputer` |
| `quprep.clean.outlier` | `OutlierHandler` |
| `quprep.clean.categorical` | `CategoricalEncoder` |
| `quprep.clean.selector` | `FeatureSelector` |
| `quprep.ingest.csv_ingester` | `CSVIngester` |
| `quprep.ingest.numpy_ingester` | `NumpyIngester` |
| `quprep.ingest.profiler` | `profile`, `DatasetProfile` |
| `quprep.core.dataset` | `Dataset` |

---

::: quprep
    options:
      show_root_heading: true
      show_source: false
      members:
        - prepare
        - Pipeline
        - recommend
        - draw_ascii
        - draw_matplotlib
