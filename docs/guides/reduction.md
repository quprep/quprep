# Dimensionality Reduction

!!! note "Coming in v0.2.0"
    Dimensionality reduction is a Phase 2 feature. This page documents the planned API.

Current quantum hardware is limited to tens or hundreds of qubits. Most real-world datasets have far more features. Reduction brings your data into a range that hardware can handle.

---

## Why it matters

| Encoding | Qubits needed for d features |
|---|---|
| Angle | d qubits |
| Amplitude | log₂(d) qubits |
| Basis | d qubits |

If your dataset has 512 features and your backend has 127 qubits, angle encoding is infeasible without reduction.

---

## Planned reducers

### PCA

```python
# v0.2.0
from quprep.reduce.pca import PCAReducer

pipeline = Pipeline(
    reducer=PCAReducer(n_components=8),
    encoder=AngleEncoder(),
)
```

Good general-purpose default. `n_components` can be an integer (exact) or float (variance fraction, e.g. `0.95`).

### LDA

```python
# v0.2.0
from quprep.reduce.lda import LDAReducer

pipeline = Pipeline(
    reducer=LDAReducer(n_components=4),
    encoder=AngleEncoder(),
)
```

Maximises class separability. Research shows LDA outperforms PCA for quantum classification tasks (Mancilla & Pere, 2022). Requires class labels. Maximum components: n_classes − 1.

### Hardware-aware reduction

```python
# v0.2.0
from quprep.reduce.hardware_aware import HardwareAwareReducer

pipeline = Pipeline(
    reducer=HardwareAwareReducer(backend="ibm_brisbane", encoding="angle"),
    encoder=AngleEncoder(),
)
```

Automatically calculates the qubit budget for the target backend and applies the optimal reducer. No manual calculation needed.

### Spectral / DFT

Best for time-series and signal data. Most noise-robust encoding strategy according to 2025 research.

### t-SNE / UMAP

For visualization and local structure preservation. Best suited to 2–3 qubit circuits.

---

## References

- Mancilla & Pere (2022) — LDA vs PCA for quantum classification
- 2025 spectral encoding research — DFT as most noise-robust strategy
