# Examples

Each example is a runnable Python script. All examples use `import quprep as qd`.

| # | Topic | Script |
|---|---|---|
| 01 | Quickstart — `qd.prepare()` one-liner | `01_quickstart.py` |
| 02 | Full pipeline — clean → encode → export → save/load | `02_pipeline.py` |
| 03 | All encoders — Angle, Amplitude, Basis, IQP, EntangledAngle, ReUpload, Hamiltonian | `03_encoders.py` |
| 04 | Framework export — QASM, Qiskit, PennyLane, Cirq, TKET, batch save | `04_export.py` |
| 05 | Encoding recommendation — `qd.recommend()` | `05_recommend.py` |
| 06 | Circuit visualization — ASCII + matplotlib | `06_visualization.py` |
| 07 | QUBO / Ising — Max-Cut, Knapsack, solvers, D-Wave export, QAOA | `07_qubo.py` |
| 08 | Validation, schema & cost — `qd.DataSchema`, `qd.estimate_cost`, `result.summary()` | `08_validation.py` |
| 09 | Data drift detection — `DriftDetector`, pipeline integration, serialization | `09_drift.py` |
| 10 | Qubit suggestion — `suggest_qubits`, task hints, NISQ ceiling, pipeline integration | `10_suggest.py` |
| 11 | New encoders (v0.6.0) — ZZFeatureMap, PauliFeatureMap, RandomFourier, TensorProduct | `11_new_encoders.py` |
| 12 | New exporters (v0.6.0) — Amazon Braket, Q# (Azure Quantum), IQM native format | `12_exporters.py` |

## Run a script

```bash
pip install quprep
python examples/01_quickstart.py
```

## Optional dependencies

```bash
pip install quprep[qiskit]     # example 04
pip install quprep[pennylane]  # example 04
pip install quprep[cirq]       # example 04
pip install quprep[tket]       # example 04
pip install quprep[viz]        # example 06 (matplotlib diagrams)
pip install quprep[braket]     # example 12
pip install quprep[qsharp]     # example 12 (Azure Quantum submission)
pip install quprep[iqm]        # example 12 (IQM hardware submission)
```

Examples skip any framework that isn't installed rather than crashing.
