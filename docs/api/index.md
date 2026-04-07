# API Reference

QuPrep's public API has four top-level entry points (use `import quprep as qd`):

| Name | Description |
|---|---|
| `qd.prepare()` | One-liner: source → circuits |
| `qd.Pipeline` | Composable pipeline with full stage control |
| `qd.recommend()` | Encoding recommendation engine |
| `qd.compare_encodings()` | Side-by-side cost comparison of all encoders |

Everything else is accessed via submodules.

---

## Top-level

```python
import quprep as qd

qd.__version__         # "0.6.0"
qd.prepare(...)
qd.Pipeline(...)
qd.recommend(...)
qd.draw_ascii(...)
qd.draw_matplotlib(...)

# All classes are on the top-level namespace — no sub-imports needed:
qd.AngleEncoder()
qd.Imputer()
qd.PCAReducer()
qd.DataSchema(...)
qd.estimate_cost(...)
qd.compare_encodings(...)
qd.ComparisonResult
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
| `quprep.compare` | `compare_encodings`, `ComparisonResult` |
| `quprep.validation` | `DataSchema`, `FeatureSpec`, `SchemaViolationError`, `validate_dataset`, `warn_qubit_mismatch`, `QuPrepWarning` |
| `quprep.validation.cost` | `CostEstimate`, `estimate_cost` |
| `quprep.qubo` | `to_qubo`, `QUBOResult`, `qubo_to_ising`, `ising_to_qubo`, `IsingResult` |
| `quprep.qubo.problems` | `max_cut`, `knapsack`, `tsp`, `portfolio`, `graph_color`, `scheduling`, `number_partition` |
| `quprep.qubo.solver` | `solve_brute`, `solve_sa`, `SolveResult` _(classical reference utilities — not in `quprep.qubo.__all__`)_ |
| `quprep.qubo.qaoa` | `qaoa_circuit` |
| `quprep.qubo.constraints` | `equality_penalty`, `inequality_penalty` |
| `quprep.qubo.ising` | `qubo_to_ising`, `ising_to_qubo` |
| `quprep.qubo.utils` | `add_qubo` |
| `quprep.qubo.visualize` | `draw_qubo`, `draw_ising` |

---

::: quprep
    options:
      show_root_heading: true
      show_source: false
      members:
        - prepare
        - Pipeline
        - recommend
        - compare_encodings
        - draw_ascii
        - draw_matplotlib
