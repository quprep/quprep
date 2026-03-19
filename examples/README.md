# Examples

Each example is available as a runnable Python script and a Jupyter notebook.

| # | Topic | Script | Notebook |
|---|---|---|---|
| 01 | Quickstart — `prepare()` one-liner | `01_quickstart.py` | `01_quickstart.ipynb` |
| 02 | Full pipeline — clean → encode → export | `02_pipeline.py` | `02_pipeline.ipynb` |
| 03 | Encoders compared — Angle, Amplitude, Basis | `03_encoders.py` | `03_encoders.ipynb` |
| 04 | Qiskit export — `QuantumCircuit` output | `04_qiskit_export.py` | `04_qiskit_export.ipynb` |

## Run a script

```bash
pip install quprep
python examples/01_quickstart.py
```

## Run a notebook

```bash
pip install quprep jupyter
jupyter notebook examples/01_quickstart.ipynb
```

## Qiskit examples

Example 04 requires the Qiskit extra:

```bash
pip install quprep[qiskit]
```
