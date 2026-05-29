# QuPrep Examples

Two categories, each available as a verified `.py` script and a Jupyter `.ipynb` notebook with Colab / Binder / qBraid launch buttons.

---

## Tutorials — start here

Step-by-step guides that build up from a raw dataset to production-ready quantum circuits.

| File | What you will learn |
|------|---------------------|
| [tutorials/01_your_first_quantum_ready_dataset](tutorials/01_your_first_quantum_ready_dataset.ipynb) | Load data, build a Pipeline, encode, draw and export circuits |
| [tutorials/02_real_world_messy_data](tutorials/02_real_world_messy_data.ipynb) | Handle NaN, outliers, and class imbalance on a realistic dataset |
| [tutorials/03_end_to_end_with_a_framework](tutorials/03_end_to_end_with_a_framework.ipynb) | Audit, auto-suggest a pipeline, verify, and export to OpenQASM |

---

## How-to guides — task-focused

| Task | Guide |
|------|-------|
| Pick the right encoder | [how-to/choose_an_encoder](how-to/choose_an_encoder.ipynb) |
| Export to Qiskit / PennyLane / Cirq / Braket | [how-to/export_to_frameworks](how-to/export_to_frameworks.ipynb) |
| Read gate angles and draw circuit diagrams | [how-to/inspect_a_circuit](how-to/inspect_a_circuit.ipynb) |
| Load from CSV, OpenML, HuggingFace, streaming | [how-to/load_external_data](how-to/load_external_data.ipynb) |
| Handle time series, sparse matrices, multi-label | [how-to/non_tabular_data](how-to/non_tabular_data.ipynb) |
| Fix class imbalance (SMOTE / oversample / undersample) | [how-to/fix_class_imbalance](how-to/fix_class_imbalance.ipynb) |
| Validate data before encoding (report, schema) | [how-to/validate_before_encoding](how-to/validate_before_encoding.ipynb) |
| Detect distribution drift in production | [how-to/detect_data_drift](how-to/detect_data_drift.ipynb) |
| Measure expressibility and barren plateau risk | [how-to/assess_encoding_quality](how-to/assess_encoding_quality.ipynb) |
| Map features to reliable qubits on NISQ hardware | [how-to/noise_aware_preprocessing](how-to/noise_aware_preprocessing.ipynb) |
| Formulate and solve QUBO / Ising problems | [how-to/solve_qubo](how-to/solve_qubo.ipynb) |
| Register a custom encoder in the plugin registry | [how-to/write_a_custom_encoder](how-to/write_a_custom_encoder.ipynb) |

---

## Running locally

```bash
uv run python examples/tutorials/01_your_first_quantum_ready_dataset.py
uv run python examples/how-to/choose_an_encoder.py
```

All `.py` files exit 0 with no unhandled exceptions.

## Optional dependencies

```bash
pip install quprep[qiskit]      # export_to_frameworks — Qiskit
pip install quprep[pennylane]   # export_to_frameworks — PennyLane
pip install quprep[cirq]        # export_to_frameworks — Cirq
pip install quprep[braket]      # export_to_frameworks — Amazon Braket
pip install quprep[openml]      # load_external_data — OpenML
pip install quprep[huggingface] # load_external_data — HuggingFace
pip install quprep[kaggle]      # load_external_data — Kaggle
```

Examples skip any framework that isn't installed rather than crashing.
