---
title: 'QuPrep: A Framework-Agnostic Quantum Data Preparation Library'
tags:
  - quantum computing
  - quantum machine learning
  - data encoding
  - preprocessing
  - Python
authors:
  - name: Hasarindu Perera
    orcid: 0000-0002-7897-9664
    affiliation: 1
affiliations:
  - name: Independent Researcher, Colombo, Sri Lanka
    index: 1
date: 15 April 2026
bibliography: paper.bib
---

# Summary

QuPrep is an open-source Python library that converts classical datasets into quantum-circuit-ready formats. It provides a modular, six-stage preprocessing pipeline — ingestion, cleaning, dimensionality reduction, normalisation, encoding, and circuit export — that is agnostic to the target quantum computing framework. QuPrep supports thirteen quantum encoding methods, a dataset-aware encoding recommendation engine, QUBO/Ising problem formulation, and export to eight quantum frameworks (Qiskit, PennyLane, Cirq, TKET, Amazon Braket, Q#/Azure Quantum, IQM, and OpenQASM 3.0). Data ingestion spans six modalities — tabular, time series, image, text, graph, and graph-structured quantum data — with connectors to HuggingFace Hub, OpenML, and Kaggle. The library requires only NumPy, SciPy, and scikit-learn for its core functionality; all quantum framework packages are optional dependencies.

# Statement of Need

Quantum machine learning (QML) and combinatorial quantum optimisation require classical data to be encoded into quantum states before any circuit can be executed. This encoding step is non-trivial: different encoding schemes impose different qubit counts, circuit depths, and normalisation requirements, and the choice of encoding materially affects model performance [@schuld2021; @havlicek2019; @huang2021]. Despite this, the encoding and preprocessing step is handled inconsistently across the field.

@mancilla2022 identify data preprocessing as an underexplored bottleneck for QML classification on NISQ hardware, noting that encoding choice and feature engineering directly determine whether quantum advantage is achievable in practice. A practitioner who wants to evaluate multiple encodings against the same dataset currently must either write substantial boilerplate or commit to a single QML framework before any preprocessing decisions are made.

A further gap exists beyond tabular data. Real-world QML applications increasingly involve graphs (for molecular property prediction), time series (for financial and sensor data), and images (for computer vision tasks). No existing standalone tool provides a unified preprocessing path from these modalities to quantum circuits. QuPrep fills this gap by providing a dedicated preprocessing layer that researchers can use regardless of their chosen quantum framework or data modality. The entire pipeline can be invoked with a single call:

```python
import quprep as qd
result = qd.prepare("dataset.csv", encoding="angle", framework="qiskit")
```

# State of the Field

Existing tools embed encoding within larger QML frameworks. Qiskit Machine Learning and PennyLane [@bergholm2018] provide encoding circuits but require practitioners to commit to a specific framework before preprocessing decisions are made. sQUlearn [@kreplin2023] is a QML library built on top of Qiskit and PennyLane; it includes encoding circuits but offers no standalone preprocessing pipeline or encoding recommendation. AutoQML [@autoqml2023] automates the full training pipeline including encoding search, but is not a controlled preprocessing layer — it optimises for model accuracy rather than giving practitioners explicit authority over encoding decisions.

QuPrep was built as a standalone library rather than a contribution to existing tools for two reasons. First, the problem it solves — systematic quantum data preparation — is upstream of and orthogonal to training or simulation, and coupling it to any single framework would undermine its central value proposition. Second, no existing tool handles the full preprocessing chain (imputation, scaling, dimensionality reduction, encoding, export) across multiple modalities and frameworks from a single, unified API.

| Feature | QuPrep | Qiskit ML | PennyLane | sQUlearn [@kreplin2023] | AutoQML [@autoqml2023] |
|---|---|---|---|---|---|
| Standalone preprocessing | Yes | No | No | No | Yes (automated) |
| Framework-agnostic | Yes (8 frameworks) | No | No | No | No |
| Encoding recommendation | Yes | No | No | No | AutoML search |
| Multi-modal data | Yes | No | No | No | No |
| External data connectors | Yes (HF/OpenML/Kaggle) | No | No | No | No |
| QUBO / Ising formulation | Yes | Partial | No | No | No |
| Reproducibility fingerprint | Yes | No | No | No | No |

# Software Design

**Pipeline architecture.** QuPrep implements a linear six-stage pipeline modelled on scikit-learn's `fit`/`transform` paradigm [@pedregosa2011]. Each stage is optional and independently composable. Fitted pipelines capture all learned parameters (imputer statistics, scaler bounds, PCA components) and can be applied to held-out test data without re-fitting, enabling correct train/test separation in QML experiments.

The normalisation stage is automatic: QuPrep maps data to the mathematically required range for the target encoding without user intervention — amplitude encoding requires L2-normalised rows; angle encoding requires features in $[0, \pi]$; basis encoding requires binary values. This eliminates a common source of silent error in QML experiments.

**Encoding methods.** QuPrep implements twelve encoding methods for classical feature vectors plus a graph-state encoder. Table 1 summarises the circuit complexity of each.

| Encoding | Qubits | Depth | NISQ-safe |
|---|---|---|---|
| Angle (Ry/Rx/Rz) | $d$ | $O(1)$ | Yes |
| Amplitude | $\lceil\log_2 d\rceil$ | $O(2^n)$ | No |
| Basis | $d$ | $O(1)$ | Yes |
| IQP [@havlicek2019] | $d$ | $O(d^2 \cdot \text{reps})$ | Conditional |
| Entangled Angle | $d$ | $O(d \cdot \text{layers})$ | Yes |
| Data Re-uploading [@perez2020] | $d$ | $O(d \cdot \text{layers})$ | Yes |
| Hamiltonian | $d$ | $O(d \cdot \text{steps})$ | No |
| ZZ Feature Map [@havlicek2019] | $d$ | $O(d^2 \cdot \text{reps})$ | Conditional |
| Pauli Feature Map | $d$ | $O(d^2 \cdot \text{reps})$ | Conditional |
| Random Fourier [@rahimi2007] | $n_\text{components}$ | $O(1)$ | Yes |
| Tensor Product | $\lceil d/2 \rceil$ | $O(1)$ | Yes |
| QAOA Problem | $d$ | $O(d \cdot p)$ | Yes |
| Graph State | $n_\text{nodes}$ | $O(\|E\|)$ | Yes |

**Encoding recommendation.** QuPrep's dataset-aware recommendation engine scores all twelve encodings against four dataset signals — missing value rate, sparsity, presence of negative values, and mean pairwise feature correlation — and returns a ranked list with explanations. The companion `compare_encodings()` function produces a side-by-side circuit cost table analytically, without generating any circuits.

**Multi-modal ingestion.** Each modality ingester produces a uniform `Dataset` object consumed by downstream stages. `TimeSeriesIngester` applies sliding-window segmentation; `ImageIngester` reads directory trees of labelled images; `TextIngester` uses TF-IDF or sentence-transformer embeddings; `GraphIngester` extracts Laplacian eigenvalue features or passes the raw adjacency matrix to the `GraphStateEncoder`, which prepares a graph state whose entanglement pattern mirrors the input graph topology.

**Design trade-offs.** All quantum framework packages are optional dependencies; the core library requires only NumPy, SciPy, and scikit-learn. This allows installation without any quantum SDK and avoids forcing framework choices on the user. The trade-off is that framework-specific exporters must be installed separately (`pip install quprep[qiskit]`, etc.). A plugin registry allows users to register custom encoders and exporters that integrate with the standard `prepare()` interface without modifying library source code.

**Reproducibility.** `fingerprint_pipeline()` computes a deterministic SHA-256 hash of every pipeline stage's class, constructor parameters, and installed dependency versions. The hash is stable across machines and Python sessions for the same configuration, making it suitable for methods sections in papers and for experiment tracking.

# Research Impact Statement

QuPrep has been publicly developed on GitHub since March 2026, with eight tagged releases spanning v0.1.0 through v0.8.0 (Zenodo DOI: 10.5281/zenodo.19286258). The development history shows iterative refinement: early releases covered tabular encoding only; later releases added dimensionality reduction, QUBO support, multi-modal ingestion, and external data connectors based on identified gaps in the QML tooling landscape.

The library is available as `pip install quprep` on PyPI under the Apache 2.0 licence. The `papers/benchmark.py` script included in this repository reproduces the circuit depth and gate count measurements in Table 1 on the UCI Iris and Heart Disease datasets using only the installed library — no quantum backend required. This provides a reproducible reference for comparing encoding overhead across all twelve encoders.

The library ships with 1,295 unit and integration tests with an 85% line coverage gate enforced in CI, covering all pipeline stages, encoding methods, exporters, and modality ingesters. Comprehensive API documentation is available at docs.quprep.org.

# AI Usage Disclosure

Generative AI was used in the development of this submission:

- **Tool:** Claude (Anthropic), accessed via the Claude Code CLI.
- **Where used:** Code generation (encoder implementations, exporter wrappers, test scaffolding), documentation authoring (API docstrings, guides), and drafting of this paper.
- **Nature of assistance:** The AI generated initial implementations and documentation text that were reviewed, corrected, and validated by the author. Core design decisions — the pipeline architecture, the encoding strategy, the plugin registry design, and the choice of supported frameworks — were made by the author.
- **Verification:** All AI-generated code was tested via the library's test suite. All claims in this paper were verified against the source code and test output by the author.

# Acknowledgements

The author thanks the open-source quantum computing community whose libraries QuPrep builds upon, and the scikit-learn project whose API conventions informed QuPrep's design. No financial support was received for this work.

# References
