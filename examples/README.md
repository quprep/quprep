# Examples

Each example is a runnable Python script. All examples use `import quprep as qd`.

| # | Topic | Script |
|---|---|---|
| 01 | Quickstart — `qd.prepare()` one-liner | `01_quickstart.py` |
| 02 | Full pipeline — clean → encode → export | `02_pipeline.py` |
| 03 | All encoders — Angle, Amplitude, Basis, IQP, EntangledAngle, ReUpload, Hamiltonian | `03_encoders.py` |
| 04 | Framework export — QASM, Qiskit, PennyLane, Cirq, TKET | `04_export.py` |
| 05 | Encoding recommendation — `qd.recommend()` | `05_recommend.py` |
| 06 | Circuit visualization — ASCII + matplotlib | `06_visualization.py` |
| 07 | QUBO / Ising — Max-Cut, Knapsack, solvers, D-Wave export, QAOA | `07_qubo.py` |
| 08 | Validation, schema & cost — `qd.DataSchema`, `qd.estimate_cost`, `result.summary()` | `08_validation.py` |

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
```

Examples skip any framework that isn't installed rather than crashing.
