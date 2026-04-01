# Examples

Each example is available as a Python script and a Jupyter notebook.

| # | Topic | Script | Launch |
|---|---|---|---|
| 01 | Quickstart — `qd.prepare()` one-liner | `01_quickstart.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/01_quickstart.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F01_quickstart.ipynb) |
| 02 | Full pipeline — clean → encode → export → save/load | `02_pipeline.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/02_pipeline.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F02_pipeline.ipynb) |
| 03 | All encoders — Angle, Amplitude, Basis, IQP, EntangledAngle, ReUpload, Hamiltonian | `03_encoders.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/03_encoders.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F03_encoders.ipynb) |
| 04 | Framework export — QASM, Qiskit, PennyLane, Cirq, TKET, batch save | `04_export.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/04_export.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F04_export.ipynb) |
| 05 | Encoding recommendation — `qd.recommend()` | `05_recommend.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/05_recommend.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F05_recommend.ipynb) |
| 06 | Circuit visualization — ASCII + matplotlib | `06_visualization.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/06_visualization.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F06_visualization.ipynb) |
| 07 | QUBO / Ising — Max-Cut, Knapsack, solvers, D-Wave export, QAOA | `07_qubo.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/07_qubo.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F07_qubo.ipynb) |
| 08 | Validation, schema & cost — `qd.DataSchema`, `qd.estimate_cost`, `result.summary()` | `08_validation.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/08_validation.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F08_validation.ipynb) |
| 09 | Data drift detection — `DriftDetector`, pipeline integration, serialization | `09_drift.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/09_drift.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F09_drift.ipynb) |
| 10 | Qubit suggestion — `suggest_qubits`, task hints, NISQ ceiling, pipeline integration | `10_suggest.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/10_suggest.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F10_suggest.ipynb) |
| 11 | New encoders (v0.6.0) — ZZFeatureMap, PauliFeatureMap, RandomFourier, TensorProduct | `11_new_encoders.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/11_new_encoders.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F11_new_encoders.ipynb) |
| 12 | New exporters (v0.6.0) — Amazon Braket, Q# (Azure Quantum), IQM native format | `12_exporters.py` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/12_exporters.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.6.0?labpath=examples%2F12_exporters.ipynb) |

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
