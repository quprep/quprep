# Dimensionality Reduction

Current quantum hardware is limited to tens or hundreds of qubits. Most real-world datasets have far more features. Reduction brings your data into a range that hardware can handle.

---

## Why it matters

| Encoding | Qubits needed for d features |
|---|---|
| Angle | d qubits |
| Amplitude | log₂(d) qubits |
| Basis | d qubits |
| IQP | d qubits |

If your dataset has 512 features and your backend has 127 qubits, angle encoding is infeasible without reduction.

---

## PCA

Good general-purpose default. `n_components` can be an integer (exact) or float (variance fraction, e.g. `0.95`).

```python
from quprep import Pipeline
from quprep.reduce.pca import PCAReducer
from quprep.encode.angle import AngleEncoder

pipeline = Pipeline(
    reducer=PCAReducer(n_components=8),
    encoder=AngleEncoder(),
)
result = pipeline.fit_transform("data.csv")

# Access explained variance after fit
print(pipeline.reducer.explained_variance_ratio_)
```

---

## LDA

Maximises class separability. Research shows LDA outperforms PCA for quantum classification tasks (Mancilla & Pere, 2022). Requires class labels. Maximum components: n_classes − 1.

```python
from quprep.reduce.lda import LDAReducer

pipeline = Pipeline(
    reducer=LDAReducer(n_components=4, labels=y),
    encoder=AngleEncoder(),
)
```

---

## Spectral (DFT)

Row-wise FFT — keeps the first `n_components` frequency magnitudes. Outputs are always ≥ 0. Best for time-series and signal data.

```python
from quprep.reduce.spectral import SpectralReducer

pipeline = Pipeline(
    reducer=SpectralReducer(n_components=8),
    encoder=AngleEncoder(),
)
```

---

## t-SNE

Preserves local structure. Best suited to 2–3 qubit circuits and visualization tasks.

```python
from quprep.reduce.spectral import TSNEReducer

reducer = TSNEReducer(n_components=2, perplexity=30)
```

!!! warning
    t-SNE requires `perplexity < n_samples`. For small datasets use a lower perplexity.

---

## UMAP

```bash
pip install umap-learn
```

Faster than t-SNE and scales to large datasets. Preserves both local and global structure.

```python
from quprep.reduce.spectral import UMAPReducer

reducer = UMAPReducer(n_components=4)
```

Raises `ImportError` with install hint if `umap-learn` is not installed.

---

## Hardware-aware reduction

Automatically calculates the qubit budget for a target backend and reduces to fit. No manual calculation needed.

```python
from quprep.reduce.hardware_aware import HardwareAwareReducer

# By backend name
pipeline = Pipeline(
    reducer=HardwareAwareReducer(backend="ibm_brisbane"),
    encoder=AngleEncoder(),
)

# By qubit count
pipeline = Pipeline(
    reducer=HardwareAwareReducer(backend=16),
    encoder=AngleEncoder(),
)
```

Supported backends: `ibm_brisbane` (127), `ibm_kyiv` (127), `ibm_torino` (133), `ionq_harmony` (11), `quantinuum_h1` (20), and more. Pass an integer to specify any qubit count directly.

---

## References

- Mancilla & Pere (2022) — LDA vs PCA for quantum classification
