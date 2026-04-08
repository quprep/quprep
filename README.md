# QuPrep — Quantum Data Preparation

**The missing preprocessing layer between classical datasets and quantum computing frameworks.**

[![PyPI version](https://img.shields.io/pypi/v/quprep.svg)](https://pypi.org/project/quprep/)
[![Python 3.10+](https://img.shields.io/pypi/pyversions/quprep.svg)](https://pypi.org/project/quprep/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1185611576.svg)](https://doi.org/10.5281/zenodo.19286258)
[![Documentation](https://readthedocs.org/projects/quprep/badge/?version=latest)](https://docs.quprep.org)
[![CI](https://github.com/quprep/quprep/actions/workflows/ci.yml/badge.svg)](https://github.com/quprep/quprep/actions/workflows/ci.yml)
[![codecov](https://codecov.io/github/quprep/quprep/graph/badge.svg?token=I26OBPRZ86)](https://codecov.io/github/quprep/quprep)
[![CodeQL](https://github.com/quprep/quprep/actions/workflows/codeql.yml/badge.svg)](https://github.com/quprep/quprep/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/quprep/quprep/badge)](https://scorecard.dev/viewer/?uri=github.com/quprep/quprep)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/12341/badge)](https://www.bestpractices.dev/projects/12341)
[![Hugging Face Demo](https://img.shields.io/badge/🤗%20Spaces-Demo-blue)](https://huggingface.co/spaces/quprep/demo)

---

QuPrep converts classical datasets into quantum-circuit-ready format. It is **not** a quantum computing framework, simulator, or training tool — it is the preprocessing step that feeds into [Qiskit](https://qiskit.org), [PennyLane](https://pennylane.ai), [Cirq](https://quantumai.google/cirq), [TKET](https://tket.quantinuum.com), and any other quantum workflow.

```
CSV / DataFrame / NumPy / images / text / graphs  →  QuPrep  →  circuit-ready output
```

## What QuPrep does

- Ingest tabular data, time series, images, text, and graphs — all in the same pipeline API
- Clean, normalize, and reduce dimensionality to fit your hardware qubit budget
- Encode data into circuits using 13 encoding methods (Angle, Amplitude, IQP, ZZFeatureMap, GraphState, and more)
- Recommend, compare, and auto-select the best encoding for your dataset and task
- Export circuits to 8 frameworks: OpenQASM 3.0, Qiskit, PennyLane, Cirq, TKET, Braket, Q#, IQM
- Formulate combinatorial optimization problems as QUBO / Ising models; export as QAOA circuit templates for your quantum framework

QuPrep does **not** train models, simulate circuits, run on quantum hardware, or optimize variational parameters.

---

## Installation

```bash
pip install quprep
```

With optional extras:

```bash
# Framework exporters
pip install quprep[qiskit]     # Qiskit QuantumCircuit
pip install quprep[pennylane]  # PennyLane QNode
pip install quprep[cirq]       # Cirq Circuit
pip install quprep[tket]       # TKET/pytket Circuit
pip install quprep[braket]     # Amazon Braket Circuit
pip install quprep[qsharp]     # Q# / Azure Quantum
pip install quprep[iqm]        # IQM native format
pip install quprep[frameworks] # all framework exporters at once

# Data modalities
pip install quprep[image]      # image ingestion (Pillow)
pip install quprep[text]       # text embeddings (sentence-transformers, ~2 GB)
pip install quprep[modalities] # image + text at once

# Other
pip install quprep[umap]       # UMAP dimensionality reduction
pip install quprep[viz]        # matplotlib circuit diagrams
pip install quprep[all]        # everything
```

**Requirements:** Python ≥ 3.10. Core dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`.

---

## Quickstart

### One-liner

```python
import quprep as qd

result = qd.prepare("data.csv", encoding="angle", framework="qasm")
print(result.circuit)
```

### Full pipeline

```python
import quprep as qd

pipeline = qd.Pipeline(
    cleaner=qd.Imputer(),
    reducer=qd.PCAReducer(n_components=8),
    encoder=qd.IQPEncoder(reps=2),
    exporter=qd.PennyLaneExporter(),   # pip install quprep[pennylane]
)
result = pipeline.fit_transform("data.csv")
qnode = result.circuit   # callable qml.QNode
```

### Data modalities — time series, images, text, graphs

```python
import quprep as qd

# Time series — sliding window then encode
from quprep.ingest.time_series_ingester import TimeSeriesIngester
from quprep.clean.window_transformer import WindowTransformer

result = qd.Pipeline(
    preprocessor=WindowTransformer(window_size=5, step=1),
    encoder=qd.AngleEncoder(),
).fit_transform(TimeSeriesIngester(time_column="date").load("sensor.csv"))

# Images — pip install quprep[image]
from quprep.ingest.image_ingester import ImageIngester
result = qd.prepare("images/", encoding="angle", ingester=ImageIngester(size=(8, 8), grayscale=True))

# Text — TF-IDF (no deps) or sentence-transformers (pip install quprep[text])
from quprep.ingest.text_ingester import TextIngester
texts = ["quantum computing is powerful", "machine learning meets QML", ...]
result = qd.prepare(texts, encoding="angle", ingester=TextIngester(method="tfidf", max_features=16))

# Graphs — lossless graph state encoding
from quprep.ingest.graph_ingester import GraphIngester
from quprep.encode.graph_state import GraphStateEncoder
import numpy as np
graph_list = [np.array([[0,1,1],[1,0,0],[1,0,0]], dtype=float), ...]  # adjacency matrices
result = qd.Pipeline(encoder=GraphStateEncoder()).fit_transform(
    GraphIngester(features="adjacency").load(graph_list)
)
```

---

## More features

| Feature | Docs |
|---|---|
| Encoding recommendation — ranked by dataset profile and task | [guide](https://docs.quprep.org/en/latest/guides/encodings/) |
| Qubit budget suggestion — NISQ-safe ceiling with reasoning | [API](https://docs.quprep.org/en/latest/api/) |
| Side-by-side encoder comparison — depth, gates, NISQ safety | [API](https://docs.quprep.org/en/latest/api/) |
| Data drift detection — warn when new data leaves training distribution | [API](https://docs.quprep.org/en/latest/api/) |
| Pipeline save / load — serialize fitted pipelines, no re-fitting | [API](https://docs.quprep.org/en/latest/api/) |
| Schema validation & cost estimation — gate count before encoding | [guide](https://docs.quprep.org/en/latest/guides/validation/) |
| QUBO / Ising formulation — Max-Cut, TSP, Knapsack, QAOA circuits, D-Wave export | [guide](https://docs.quprep.org/en/latest/guides/qubo/) |
| Plugin system — register custom encoders and exporters | [guide](https://docs.quprep.org/en/latest/guides/plugins/) |
| Circuit visualization — ASCII (no deps) or matplotlib | [API](https://docs.quprep.org/en/latest/api/) |
| Batch QASM export — save all samples to disk as individual files | [API](https://docs.quprep.org/en/latest/api/) |

---

## Supported encodings

| Encoding | Qubits | Depth | NISQ-safe | Best for |
|---|---|---|---|---|
| Angle (Ry/Rx/Rz) | n = d | O(1) | ✅ Excellent | Most QML tasks |
| Amplitude | ⌈log₂ d⌉ | O(2ⁿ) | ❌ Poor | Qubit-limited scenarios |
| Basis | n = d | O(1) | ✅ Excellent | Binary features / QAOA |
| Entangled Angle | n = d | O(d · layers) | ✅ Good | Feature correlations |
| IQP | n = d | O(d² · reps) | ⚠️ Medium | Kernel methods |
| Re-uploading | n = d | O(d · layers) | ✅ Good | High-expressivity QNNs |
| Hamiltonian | n = d | O(d · steps) | ⚠️ Medium | Physics simulation / VQE |
| ZZ Feature Map | n = d | O(d² · reps) | ⚠️ Medium | Quantum kernel methods |
| Pauli Feature Map | n = d | O(d² · reps) | ⚠️ Medium | Configurable kernel methods |
| Random Fourier | n_components | O(1) | ✅ Excellent | RBF kernel approximation |
| Tensor Product | ⌈d/2⌉ | O(1) | ✅ Excellent | Qubit-efficient encoding |
| QAOA Problem | n = d | O(p) | ✅ Good | QAOA warm-start, problem-inspired maps |
| Graph State | n = nodes | O(edges) | ✅ Good | Graph-structured data (lossless) |

---

## Supported export frameworks

| Framework | Install | Output |
|---|---|---|
| OpenQASM 3.0 | _(included)_ | `str` |
| Qiskit | `pip install quprep[qiskit]` | `QuantumCircuit` |
| PennyLane | `pip install quprep[pennylane]` | `qml.QNode` |
| Cirq | `pip install quprep[cirq]` | `cirq.Circuit` |
| TKET | `pip install quprep[tket]` | `pytket.Circuit` |
| Amazon Braket | `pip install quprep[braket]` | `braket.Circuit` |
| Q# | `pip install quprep[qsharp]` | Q# operation string |
| IQM | `pip install quprep[iqm]` | IQM circuit JSON |

---

## Documentation

Full documentation at **[docs.quprep.org](https://docs.quprep.org/en/latest/)**

- [Installation](https://docs.quprep.org/en/latest/getting-started/installation/)
- [Quickstart guide](https://docs.quprep.org/en/latest/getting-started/quickstart/)
- [Encoding guide](https://docs.quprep.org/en/latest/guides/encodings/)
- [API reference](https://docs.quprep.org/en/latest/api/)

---

## Examples

| # | Topic | Launch |
|---|---|---|
| 01 | Quickstart — `prepare()` one-liner | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/01_quickstart.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F01_quickstart.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/01_quickstart.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 02 | Full pipeline — clean → encode → export → save/load | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/02_pipeline.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F02_pipeline.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/02_pipeline.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 03 | All encoders compared | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/03_encoders.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F03_encoders.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/03_encoders.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 04 | Framework export — QASM, Qiskit, PennyLane, Cirq, TKET, Braket, Q#, IQM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/04_export.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F04_export.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/04_export.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 05 | Encoding recommendation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/05_recommend.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F05_recommend.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/05_recommend.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 06 | Circuit visualization — ASCII + matplotlib | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/06_visualization.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F06_visualization.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/06_visualization.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 07 | QUBO / Ising — Max-Cut, Knapsack, solvers, D-Wave export, QAOA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/07_qubo.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F07_qubo.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/07_qubo.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 08 | Validation, schema & cost | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/08_validation.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F08_validation.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/08_validation.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 09 | Data drift detection | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/09_drift.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F09_drift.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/09_drift.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 10 | Qubit suggestion — `suggest_qubits`, task hints, NISQ ceiling | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/10_suggest.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F10_suggest.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/10_suggest.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 11 | Plugin system — register custom encoders and exporters | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/11_plugins.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F11_plugins.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/11_plugins.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |
| 12 | Data modalities — time series, image, text, graph | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/quprep/quprep/blob/main/examples/12_modalities.ipynb) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/quprep/quprep/v0.7.0?labpath=examples%2F12_modalities.ipynb) <a href="https://account.qbraid.com?gitHubUrl=https://github.com/quprep/quprep/blob/main/examples/12_modalities.ipynb"><img src="https://qbraid-static.s3.amazonaws.com/logos/Launch_on_qBraid_white.png" width="93"/></a> |

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

- [Open an issue](https://github.com/quprep/quprep/issues) for bugs or feature requests
- [Start a discussion](https://github.com/quprep/quprep/discussions) for questions or ideas

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use QuPrep in your research, please cite:

```bibtex
@software{quprep2026,
  author    = {Perera, Hasarindu},
  title     = {QuPrep: Quantum Data Preparation},
  year      = {2026},
  publisher = {Zenodo},
  version   = {0.7.0},
  doi       = {10.5281/zenodo.19286258},
  url       = {https://doi.org/10.5281/zenodo.19286258},
  license   = {Apache-2.0},
}
```
