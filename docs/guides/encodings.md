# Encoding Guide

Quantum encoding maps classical feature vectors into quantum states. Choosing the right encoding affects qubit count, circuit depth, NISQ suitability, and expressivity.

---

## Available encodings

### Angle encoding

Maps each feature to a single-qubit rotation gate.

$$|\psi(x)\rangle = \bigotimes_{i=1}^{d} R_G(x_i)|0\rangle$$

where $R_G$ is Ry (default), Rx, or Rz.

| Property | Value |
|---|---|
| Qubits | n = d (one per feature) |
| Depth | O(1) |
| NISQ-safe | ✅ Excellent |
| Best for | Most QML tasks — default recommendation |

```python
import quprep as qd

enc = qd.AngleEncoder(rotation="ry")  # or "rx", "rz"
result = enc.encode(x)                # x must be in [0, π] for ry
print(result.parameters)              # rotation angles
print(result.metadata)                # {"n_qubits": 4, "depth": 1, ...}
```

**Normalization:** Use `Scaler("minmax_pi")` for Ry, `Scaler("minmax_pm_pi")` for Rx/Rz. The pipeline applies this automatically.

---

### Amplitude encoding

Embeds the entire vector as quantum state amplitudes.

$$|\psi(x)\rangle = \sum_{i=0}^{d-1} x_i |i\rangle$$

Requires $\|x\|_2 = 1$. If $d$ is not a power of two, pads with zeros and re-normalizes.

| Property | Value |
|---|---|
| Qubits | $n = \lceil \log_2(d) \rceil$ |
| Depth | $O(2^n)$ — exponential |
| NISQ-safe | ❌ Poor — deep circuit |
| Best for | Qubit-limited scenarios, high expressivity |

```python
import quprep as qd
import numpy as np

# Normalize first (pipeline does this automatically)
x = np.array([1.0, 2.0, 3.0, 4.0])
x = x / np.linalg.norm(x)

enc = qd.AmplitudeEncoder(pad=True)  # pad=False raises if d is not power of two
result = enc.encode(x)
print(result.metadata["n_qubits"])   # 2 (log2(4))
print(result.metadata["padded"])     # False
```

**Normalization:** Requires L2 normalization. Use `Scaler("l2")` or let the pipeline handle it automatically.

!!! warning "Not NISQ-safe"
    Amplitude encoding requires exponential-depth state preparation circuits. Avoid on current hardware unless qubit count is the primary constraint.

---

### Basis encoding

Maps binary features to computational basis states via X gates.

$$|\psi(x)\rangle = |x_1 x_2 \ldots x_d\rangle$$

| Property | Value |
|---|---|
| Qubits | n = d |
| Depth | O(1) — X gates only |
| NISQ-safe | ✅ Excellent |
| Best for | Binary data, QAOA, combinatorial optimization |

```python
import quprep as qd

enc = qd.BasisEncoder(threshold=0.5)  # values >= 0.5 → |1⟩, else → |0⟩
result = enc.encode(x)
print(result.parameters)  # binary {0.0, 1.0}
```

**Normalization:** Binarized at 0.5 by default. Use `Scaler("binary")` or set a custom threshold on the encoder.

---

---

### IQP encoding

Havlíček et al. 2019 feature map. Applies Hadamards, then single-qubit Rz and pairwise ZZ interactions, repeated `reps` times.

$$|\psi(x)\rangle = U_\Phi(x) H^{\otimes n} |0\rangle^n$$

| Property | Value |
|---|---|
| Qubits | n = d |
| Depth | $O(d^2 \cdot \text{reps})$ |
| NISQ-safe | ⚠️ Medium — $d^2$ two-qubit gates |
| Best for | Kernel methods, quantum advantage arguments |

```python
import quprep as qd

enc = qd.IQPEncoder(reps=2)
result = enc.encode(x)   # x in [−π, π]
```

**Normalization:** `minmax_pm_pi` ($[-\pi, \pi]$). Applied automatically.

!!! note "References"
    Havlíček et al. (2019). Supervised learning with quantum-enhanced feature spaces. *Nature*, 567, 209–212. [doi:10.1038/s41586-019-0980-2](https://doi.org/10.1038/s41586-019-0980-2){target="_blank"}

---

### Entangled angle encoding

Alternates rotation layers (Ry/Rx/Rz per qubit) with CNOT entangling layers, repeated `layers` times. Three entanglement topologies: `linear`, `circular`, `full`.

| Property | Value |
|---|---|
| Qubits | n = d |
| Depth | (d + CNOTs) × layers |
| NISQ-safe | ✅ Good — controlled depth |
| Best for | Feature correlations, expressivity beyond angle encoding |

```python
import quprep as qd

enc = qd.EntangledAngleEncoder(rotation="ry", layers=2, entanglement="circular")
result = enc.encode(x)
print(result.metadata["cnot_pairs"])  # [(0,1), (1,2), (2,0)] for circular
```

---

### Data re-uploading

Pérez-Salinas et al. 2020. Applies the same rotation layer `layers` times, interleaved with trainable parameters. Proven universal approximator with enough layers.

| Property | Value |
|---|---|
| Qubits | n = d |
| Depth | d × layers |
| NISQ-safe | ✅ Good |
| Best for | High-expressivity QNNs |

```python
import quprep as qd

enc = qd.ReUploadEncoder(layers=3, rotation="ry")
result = enc.encode(x)   # x in [−π, π]
```

!!! note "References"
    Pérez-Salinas et al. (2020). Data re-uploading for a universal quantum classifier. *Quantum*, 4, 226. [doi:10.22331/q-2020-02-06-226](https://doi.org/10.22331/q-2020-02-06-226){target="_blank"}

---

### Hamiltonian encoding

Trotterized time evolution under a single-qubit Z Hamiltonian $H(x) = \sum_i x_i Z_i$. Applies Rz gates over `trotter_steps` steps.

| Property | Value |
|---|---|
| Qubits | n = d |
| Depth | d × trotter_steps |
| NISQ-safe | ⚠️ Medium — grows with steps |
| Best for | Physics simulation, VQE |

```python
import quprep as qd

enc = qd.HamiltonianEncoder(evolution_time=1.0, trotter_steps=4)
result = enc.encode(x)
```

**Normalization:** `zscore`. Applied automatically.

---

## Choosing an encoding

Not sure? Use `quprep.recommend()`:

```python
import quprep as qd
rec = qd.recommend("data.csv", task="classification", qubits=8)
print(rec)
```

Or follow this decision tree:

```
Is your data binary?
  └─ Yes → Basis encoding (QAOA, combinatorial)
  └─ No  → Is qubit count the main constraint?
              └─ Yes → Amplitude encoding (log₂ d qubits, deep)
              └─ No  → What is your task?
                         classification/regression → Angle or IQP
                         kernel methods            → IQP
                         high-expressivity QNN      → Data re-uploading
                         physics simulation/VQE     → Hamiltonian
                         feature correlations       → Entangled Angle
```

---

## Auto-normalization

The pipeline selects the correct normalization automatically:

```python
import quprep as qd

# No manual normalization needed — pipeline handles it
pipeline = qd.Pipeline(encoder=qd.AngleEncoder(rotation="ry"))
result = pipeline.fit_transform(data)
# data was automatically scaled to [0, π] before encoding
```

To override:

```python
import quprep as qd

pipeline = qd.Pipeline(
    encoder=qd.AngleEncoder(),
    normalizer=qd.Scaler("zscore"),  # explicit override
)
```
