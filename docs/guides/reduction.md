# Dimensionality Reduction

Current quantum hardware is limited to tens or hundreds of qubits. Most real-world datasets have far more features. Reduction brings your data into a range that hardware can handle.

---

## Why it matters

| Encoding | Qubits needed for d features |
|---|---|
| Angle | d qubits |
| Amplitude | $\log_2(d)$ qubits |
| Basis | d qubits |
| IQP | d qubits |

If your dataset has 512 features and your backend has 127 qubits, angle encoding is infeasible without reduction.

---

## PCA

Good general-purpose default. `n_components` can be an integer (exact) or float (variance fraction, e.g. `0.95`).

```python
import quprep as qd

pipeline = qd.Pipeline(
    reducer=qd.PCAReducer(n_components=8),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform("data.csv")

# Access explained variance after fit
print(pipeline.reducer.explained_variance_ratio_)
```

---

## LDA

Maximises class separability. Research shows LDA outperforms PCA for quantum classification tasks. Requires class labels. Maximum components: n_classes − 1.

```python
import quprep as qd

pipeline = qd.Pipeline(
    reducer=qd.LDAReducer(n_components=4, labels=y),
    encoder=qd.AngleEncoder(),
)
```

---

## Spectral (DFT)

Row-wise FFT — keeps the first `n_components` frequency magnitudes. Outputs are always ≥ 0. Best for time-series and signal data.

```python
import quprep as qd

pipeline = qd.Pipeline(
    reducer=qd.SpectralReducer(n_components=8),
    encoder=qd.AngleEncoder(),
)
```

---

## t-SNE

Preserves local structure. Best suited to 2–3 qubit circuits and visualization tasks.

```python
import quprep as qd

reducer = qd.TSNEReducer(n_components=2, perplexity=30)
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
import quprep as qd

reducer = qd.UMAPReducer(n_components=4)
```

Raises `ImportError` with install hint if `umap-learn` is not installed.

---

## Hardware-aware reduction

Automatically calculates the qubit budget for a target backend and reduces to fit. No manual calculation needed.

```python
import quprep as qd

# By backend name
pipeline = qd.Pipeline(
    reducer=qd.HardwareAwareReducer(backend="ibm_brisbane"),
    encoder=qd.AngleEncoder(),
)

# By qubit count
pipeline = qd.Pipeline(
    reducer=qd.HardwareAwareReducer(backend=16),
    encoder=qd.AngleEncoder(),
)
```

Supported backends: `ibm_brisbane` (127), `ibm_kyiv` (127), `ibm_torino` (133), `ionq_harmony` (11), `quantinuum_h1` (20), and more. Pass an integer to specify any qubit count directly.

---

!!! note "References"
    Mancilla, J., & Pere, C. (2022). A preprocessing perspective for quantum machine learning classification advantage in finance using NISQ algorithms. *Entropy*, 24(11), 1656. [doi:10.3390/e24111656](https://doi.org/10.3390/e24111656){target="_blank"}
