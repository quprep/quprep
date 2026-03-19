# Encoding Guide

Quantum encoding maps classical feature vectors into quantum states. Choosing the right encoding affects qubit count, circuit depth, NISQ suitability, and expressivity.

---

## Available encodings (v0.1.0)

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
from quprep.encode.angle import AngleEncoder

enc = AngleEncoder(rotation="ry")   # or "rx", "rz"
result = enc.encode(x)              # x must be in [0, π] for ry
print(result.parameters)            # rotation angles
print(result.metadata)              # {"n_qubits": 4, "depth": 1, ...}
```

**Normalization:** Use `Scaler("minmax_pi")` for Ry, `Scaler("minmax_pm_pi")` for Rx/Rz. The pipeline applies this automatically.

---

### Amplitude encoding

Embeds the entire vector as quantum state amplitudes.

$$|\psi(x)\rangle = \sum_{i=0}^{d-1} x_i |i\rangle$$

Requires $\|x\|_2 = 1$. If $d$ is not a power of two, pads with zeros and re-normalizes.

| Property | Value |
|---|---|
| Qubits | n = ⌈log₂(d)⌉ |
| Depth | O(2ⁿ) — exponential |
| NISQ-safe | ❌ Poor — deep circuit |
| Best for | Qubit-limited scenarios, high expressivity |

```python
from quprep.encode.amplitude import AmplitudeEncoder
from quprep.normalize.scalers import Scaler
import numpy as np

# Normalize first (pipeline does this automatically)
x = np.array([1.0, 2.0, 3.0, 4.0])
x = x / np.linalg.norm(x)

enc = AmplitudeEncoder(pad=True)  # pad=False raises if d is not power of two
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
from quprep.encode.basis import BasisEncoder

enc = BasisEncoder(threshold=0.5)  # values >= 0.5 → |1⟩, else → |0⟩
result = enc.encode(x)
print(result.parameters)  # binary {0.0, 1.0}
```

**Normalization:** Binarized at 0.5 by default. Use `Scaler("binary")` or set a custom threshold on the encoder.

---

## Coming in v0.2.0

| Encoding | Description | Best for |
|---|---|---|
| **IQP** | Havlíček et al. 2019. Hadamards + ZZ interactions. | Kernel methods |
| **Entangled Angle** | Angle encoding with entangling layers. | Feature correlations |
| **Data re-uploading** | Pérez-Salinas et al. 2020. Features repeated L times. | High-expressivity QNNs |
| **Hamiltonian** | Time-evolution encoding $e^{-iH(x)T}$. | Physics simulation, VQE |

---

## Choosing an encoding

```
Is your data binary?
  └─ Yes → Basis encoding
  └─ No  → Is qubit count the main constraint?
              └─ Yes → Amplitude encoding (log₂ d qubits, deep circuit)
              └─ No  → Angle encoding (default — shallow, NISQ-safe)
```

For kernel methods (v0.2.0): IQP encoding.
For high-expressivity QNNs (v0.2.0): Data re-uploading.

---

## Auto-normalization

The pipeline selects the correct normalization automatically:

```python
from quprep import Pipeline
from quprep.encode.angle import AngleEncoder

# No manual normalization needed — pipeline handles it
pipeline = Pipeline(encoder=AngleEncoder(rotation="ry"))
result = pipeline.fit_transform(data)
# data was automatically scaled to [0, π] before encoding
```

To override:

```python
from quprep.normalize.scalers import Scaler

pipeline = Pipeline(
    encoder=AngleEncoder(),
    normalizer=Scaler("zscore"),  # explicit override
)
```
