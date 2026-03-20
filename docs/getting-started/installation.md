# Installation

## Requirements

- Python ≥ 3.10
- Core dependencies (installed automatically): `numpy`, `scipy`, `pandas`, `scikit-learn`

## Install

```bash
pip install quprep
```

## Optional framework dependencies

Framework-specific exporters are optional. Install only what you need:

```bash
pip install quprep[qiskit]     # Qiskit QuantumCircuit export
pip install quprep[pennylane]  # PennyLane QNode export
pip install quprep[cirq]       # Cirq Circuit export
pip install quprep[tket]       # TKET/pytket Circuit export
pip install quprep[viz]        # matplotlib circuit diagrams
pip install quprep[all]        # All framework exports + visualization
```

`draw_ascii()` is always available with no extra dependencies.

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
