# Encoding Quality Metrics

QuPrep can measure the *actual* quality of a quantum encoding on your data — not just static heuristics, but circuit-level metrics computed by simulating the encoding on samples from your dataset.

Three metrics are available:

| Metric | Interpretation | Lower is … | Higher is … |
|---|---|---|---|
| `expressibility` | KL divergence from Haar random | ← more expressive | less expressive |
| `entanglement_capability` | Avg. Meyer-Wallach measure [0, 1] | — | more entangled |
| `kernel_alignment` | Alignment of K(X,X) with label kernel | — | better class sep. |

Simulation uses a lightweight numpy statevector backend — no quantum framework required. Circuits with more than **12 qubits** return `None` (too large to simulate efficiently).

---

## Standalone usage

```python
import quprep as qd
from quprep.core.dataset import Dataset
import numpy as np

X = np.random.default_rng(0).uniform(0, np.pi, (100, 4))
y = (X[:, 0] > np.pi / 2).astype(float) * 2 - 1  # simple label
ds = Dataset(data=X, feature_names=["a", "b", "c", "d"], labels=y)

enc = qd.IQPEncoder()
m = qd.score_encoding(enc, ds)
print(m)
```

Output:

```
EncoderMetrics(iqp)
  expressibility         : 0.3142 (lower = better)
  entanglement_capability: 0.4817 (higher = better)
  kernel_alignment       : 0.2103 (higher = better)
  n_qubits               : 4
```

---

## Individual functions

```python
# Expressibility — KL divergence from Haar (lower = more expressive)
exp = qd.expressibility(qd.AngleEncoder(), ds, n_samples=500)

# Entanglement capability — avg. Meyer-Wallach measure [0, 1]
ent = qd.entanglement_capability(qd.IQPEncoder(), ds, n_samples=200)

# Kernel alignment — requires labelled data
ka  = qd.kernel_alignment(qd.ZZFeatureMapEncoder(), ds)
```

---

## Data-driven recommendation

Pass `use_metrics=True` to `recommend()` to augment the heuristic scores with simulation-based signal:

```python
rec = qd.recommend(ds, task="classification", use_metrics=True)
print(rec)
```

When enabled (and `n_features ≤ 12`), the following bonuses are added to the heuristic score of each encoding:

- **Expressibility**: up to **+8** for highly expressive circuits
- **Entanglement capability**: up to **+6** for classification / kernel tasks
- **Kernel alignment**: up to **±12** based on label separation

The recommendation is then re-ranked by the combined score.

---

## Background

**Expressibility** (Sim et al. 2019) quantifies how uniformly a parameterised circuit covers the Hilbert space. It computes the KL divergence between the fidelity distribution of the circuit under sampled input data and the Haar (uniformly random) distribution:

$$\mathcal{E} = D_{KL}\!\left(\hat{F}_\text{circuit} \;\middle\|\; F_\text{Haar}\right), \qquad F_\text{Haar}(F) = (2^n - 1)(1-F)^{2^n-2}$$

**Entanglement capability** uses the Meyer-Wallach measure:

$$\mathcal{Q} = \frac{2}{n}\sum_{k=0}^{n-1} \left(1 - \mathrm{Tr}(\rho_k^2)\right)$$

where $\rho_k$ is the reduced density matrix of qubit $k$ averaged over sampled states.

**Kernel alignment** measures how well the quantum kernel $K[i,j] = |\langle\psi(x_i)|\psi(x_j)\rangle|^2$ aligns with the target label kernel $K_y[i,j] = y_i y_j$:

$$A(K, K_y) = \frac{\langle K, K_y \rangle_F}{\|K\|_F \|K_y\|_F}$$

---

---

## Encoding sensitivity

`encoding_sensitivity` identifies which input features cause the largest changes in the encoded quantum state. It perturbs each feature independently by a small epsilon and measures the infidelity:

$$s_j = 1 - |\langle\psi(x)|\psi(x + \epsilon e_j)\rangle|^2$$

Higher score → that feature has stronger influence on the encoded state → more information is carried through the encoding for that feature.

```python
import quprep as qd

enc = qd.AngleEncoder(rotation="ry")
result = qd.encoding_sensitivity(enc, dataset, epsilon=0.01, n_samples=20, seed=42)

print(result.scores)                  # array shape (n_features,)
print(result.feature_names)           # ["f0", "f1", ...]
print(result.epsilon)                 # 0.01

# Top 3 most influential features
for name, score in result.most_sensitive(n=3):
    print(f"  {name}: {score:.4f}")
```

Sensitivity analysis is limited to circuits with ≤ 12 qubits (same numpy statevector simulator as other metrics). For encoders with more qubits, all scores are returned as zero.

---

## References

Sim S. et al. "Expressibility and Entangling Capability of Parameterized Quantum Circuits for Hybrid Quantum-Classical Algorithms." *Advanced Quantum Technologies* 2(12), 1900070, 2019. [doi:10.1002/qute.201900070](https://doi.org/10.1002/qute.201900070)

Meyer D.A., Wallach N.R. "Global entanglement in multiparticle systems." *Journal of Mathematical Physics* 43(9), 4273–4278, 2002. [doi:10.1063/1.1497700](https://doi.org/10.1063/1.1497700) — the entanglement-capability measure.
