# Installation

## Requirements

- Python ≥ 3.10
- Core dependencies (installed automatically): `numpy`, `scipy`, `pandas`, `scikit-learn`

## Install

```bash
pip install quprep
```

## Optional dependencies

Install only what you need.

### Quantum framework exporters

```bash
pip install quprep[qiskit]     # Qiskit QuantumCircuit export
pip install quprep[pennylane]  # PennyLane QNode export
pip install quprep[cirq]       # Cirq Circuit export
pip install quprep[tket]       # TKET/pytket Circuit export
pip install quprep[braket]     # Amazon Braket Circuit export
pip install quprep[qsharp]     # Q# / Azure Quantum export
pip install quprep[iqm]        # IQM native format export
pip install quprep[frameworks] # All framework exporters at once
```

`draw_ascii()` is always available with no extra dependencies.

### Data modalities

```bash
pip install quprep[image]      # Image ingestion (Pillow)
pip install quprep[text]       # Text embeddings (sentence-transformers)
pip install quprep[modalities] # All modality extras at once
```

### Visualization

```bash
pip install quprep[viz]        # matplotlib circuit diagrams
```

### Mix and match

```bash
pip install quprep[iqm,text]          # IQM export + text ingestion
pip install quprep[frameworks,modalities,viz]  # everything
pip install quprep[all]               # all extras
```

## Verify

```bash
python -c "import quprep; print(quprep.__version__)"
```

## Development install

Install [uv](https://docs.astral.sh/uv/) first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then:

```bash
git clone https://github.com/quprep/quprep.git
cd quprep
uv sync --extra dev
uv run pytest
```

## CLI

After installing, the `quprep` command is available:

```bash
quprep --version
quprep convert data.csv --encoding angle
```
