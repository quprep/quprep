# API Reference

QuPrep's public API has three top-level entry points:

| Name | Description |
|---|---|
| `quprep.prepare()` | One-liner: source → circuits |
| `quprep.Pipeline` | Composable pipeline with full stage control |
| `quprep.recommend()` | Encoding recommendation engine _(v0.2.0)_ |

Everything else is accessed via submodules.

---

## Top-level

```python
import quprep

quprep.__version__   # "0.1.0"
quprep.prepare(...)
quprep.Pipeline(...)
quprep.recommend(...)  # v0.2.0
```

---

## Submodules

| Module | Contents |
|---|---|
| `quprep.encode.angle` | `AngleEncoder` |
| `quprep.encode.amplitude` | `AmplitudeEncoder` |
| `quprep.encode.basis` | `BasisEncoder` |
| `quprep.encode.base` | `BaseEncoder`, `EncodedResult` |
| `quprep.export.qasm_export` | `QASMExporter` |
| `quprep.export.qiskit_export` | `QiskitExporter` |
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
