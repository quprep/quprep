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

!!! note "References"
    Möttönen, Vartiainen, Bergholm, Salomaa (2005). Transformation of quantum states using uniformly controlled rotations. *Quantum Information & Computation*, 5(6), 467–473. [doi:10.26421/QIC5.6-5](https://doi.org/10.26421/QIC5.6-5){target="_blank"}

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

### ZZ feature map

Havlíček et al. 2019 ZZ feature map (Qiskit convention). Applies a Hadamard layer, single-qubit Rz gates, and pairwise ZZ interactions, repeated `reps` times.

$$\phi_i = 2(\pi - x_i), \quad \phi_{ij} = 2(\pi - x_i)(\pi - x_j)$$

| Property | Value |
|---|---|
| Qubits | n = d |
| Depth | $O(d^2 \cdot \text{reps})$ |
| NISQ-safe | ⚠️ Medium — $d^2$ two-qubit gates |
| Best for | Kernel methods, QSVMs, quantum advantage |

```python
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder

enc = ZZFeatureMapEncoder(reps=2)
result = enc.encode(x)   # x in [0, π]
print(result.metadata["single_angles"])  # [2(π−x₀), 2(π−x₁), ...]
print(result.metadata["pair_angles"])    # pairwise ZZ angles
print(result.metadata["pairs"])          # [(0,1), (0,2), (1,2), ...]
```

**Normalization:** `minmax_pi` ($[0, \pi]$). Applied automatically.

!!! note "Qiskit compatibility"
    Produces the same circuit structure as Qiskit's `ZZFeatureMap`. Output circuits can be directly exported to Qiskit, QASM, Braket, Q#, or IQM.

---

### Pauli feature map

Generalized feature map using configurable Pauli string interactions. Extends ZZ feature map to arbitrary single-qubit and pairwise Pauli operators.

| Property | Value |
|---|---|
| Qubits | n = d |
| Depth | $O(d^2 \cdot \text{reps})$ with pair terms, $O(d \cdot \text{reps})$ without |
| NISQ-safe | ⚠️ Medium — depends on Pauli strings chosen |
| Best for | Expressive kernel circuits, custom feature maps |

```python
from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder

# Z single-qubit + ZZ pairwise (equivalent to ZZFeatureMap)
enc = PauliFeatureMapEncoder(paulis=["Z", "ZZ"], reps=2)

# Higher expressivity with mixed Paulis
enc = PauliFeatureMapEncoder(paulis=["Z", "X", "ZZ", "XZ"], reps=1)

result = enc.encode(x)
print(result.metadata["single_terms"])  # {"Z": [...], "X": [...]}
print(result.metadata["pair_terms"])    # {"ZZ": [(i,j,angle), ...]}
```

**Valid single Paulis:** `X`, `Y`, `Z`  
**Valid pair Paulis:** `XX`, `YY`, `ZZ`, `XZ`, `ZX`, `XY`, `YX`, `YZ`, `ZY`

---

### Random Fourier features

Approximates the RBF (Gaussian) kernel using Bochner's theorem. Samples random Fourier frequencies from $\mathcal{N}(0, 2\gamma)$ and encodes the cosine projection as rotation angles.

$$\phi(x) = \sqrt{\frac{2}{D}} \cos(Wx + b), \quad W \sim \mathcal{N}(0, 2\gamma), \quad b \sim \mathcal{U}(0, 2\pi)$$

| Property | Value |
|---|---|
| Qubits | n = `n_components` (fixed, regardless of input dim) |
| Depth | O(1) — single Ry layer |
| NISQ-safe | ✅ Excellent |
| Best for | Kernel approximation, dimensionality expansion |

```python
import numpy as np
from quprep.encode.random_fourier import RandomFourierEncoder

enc = RandomFourierEncoder(n_components=8, gamma=1.0, random_state=42)

# Must call fit() first — samples the random projection matrix
enc.fit(X_train)

result = enc.encode(x)   # always n_components qubits
```

!!! warning "Requires fit()"
    `RandomFourierEncoder` must be fitted on training data before encoding. Calling `encode()` without `fit()` raises `RuntimeError`.

!!! note "References"
    Rahimi & Recht (2007). Random Features for Large-Scale Kernel Machines. *Advances in Neural Information Processing Systems (NeurIPS)*, 20. [proceedings](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html){target="_blank"}

---

### Tensor product encoding

Encodes two features per qubit using a full Bloch sphere rotation (Ry + Rz). Requires $\lceil d/2 \rceil$ qubits. No entanglement — purely single-qubit.

$$|\psi_k\rangle = R_z(\theta_{2k+1}) R_y(\theta_{2k}) |0\rangle$$

| Property | Value |
|---|---|
| Qubits | $n = \lceil d/2 \rceil$ |
| Depth | O(1) |
| NISQ-safe | ✅ Excellent |
| Best for | Qubit-efficient encoding, full Bloch sphere expressivity |

```python
from quprep.encode.tensor_product import TensorProductEncoder

enc = TensorProductEncoder()
result = enc.encode(x)   # x in [0, π]; odd-length inputs zero-padded
print(result.metadata["ry_angles"])  # [θ₀, θ₁, ...]
print(result.metadata["rz_angles"])  # [φ₀, φ₁, ...]
print(result.metadata["n_qubits"])   # ceil(d/2)
```

**Normalization:** `minmax_pi` ($[0, \pi]$). Applied automatically.

---

### Dense angle encoding

Two rotation gates per qubit (Ry + Rz by default), halving the qubit count
compared to `AngleEncoder`. No entanglement — depth 2.

$$|\psi_k\rangle = R_2(x_{2k+1})\, R_1(x_{2k})\, |0\rangle$$

| Property | Value |
|---|---|
| Qubits | $n = \lceil d/2 \rceil$ |
| Depth | 2 |
| NISQ-safe | ✅ Excellent |
| Best for | Qubit-limited scenarios; paired features on a single qubit |

```python
import quprep as qd

enc = qd.DenseAngleEncoder()                        # default: Ry + Rz
enc = qd.DenseAngleEncoder(first_rotation="rx",
                            second_rotation="rz")   # configurable pair
result = enc.encode(x)   # x in [0, π]
print(result.metadata["n_qubits"])   # ceil(d/2)
print(result.metadata["depth"])      # 2
```

**Normalization:** `minmax_pi` ($[0, \pi]$). Applied automatically.

---

### Discretized encoding

Quantizes each continuous feature to `bits` binary digits (fixed-point), then
encodes the result as a computational basis state. Total qubits = d × bits.
The output binary vector is QUBO-compatible and can be passed directly to
`quprep.qubo.to_qubo()`.

$$v_i = \operatorname{round}\!\left(\frac{x_i - x_{\min}}{x_{\max} - x_{\min}} \cdot (2^b - 1)\right)$$

| Property | Value |
|---|---|
| Qubits | $n = d \times \text{bits}$ |
| Depth | 1 (X gates only) |
| NISQ-safe | ✅ Excellent |
| Best for | QUBO/Ising pipelines; continuous-to-binary relaxations; QAOA warm-start |

```python
import quprep as qd
import numpy as np

enc = qd.DiscretizedEncoder(bits=4, min_val=0.0, max_val=1.0)
result = enc.encode(x)
print(result.metadata["n_qubits"])        # d * 4
print(result.metadata["precision"])       # (max - min) / 15
print(result.metadata["qubo_variables"])  # {0: [0,1,2,3], 1: [4,5,6,7], ...}

# Roundtrip reconstruction
x_hat = enc.decode(result.parameters)
```

!!! tip "QUBO integration"
    `metadata["qubo_variables"]` maps each feature index to its qubit indices.
    Use this to build QUBO coefficient matrices aligned with the binary variables.

---

### QAOA problem encoding *(v0.6.0)*

Encodes a feature vector as a QAOA-inspired circuit where features become the cost Hamiltonian parameters. Each feature $x_i$ sets a local field $h_i = \gamma x_i$, and adjacent-pair products $x_i x_{i+1}$ set coupling angles. One layer applies $H^{\otimes d}$, cost unitaries (RZ + CNOT-RZ-CNOT), and a mixer (RX).

| Property | Value |
|---|---|
| Qubits | $n = d$ |
| Depth | O(p) linear; O(d·p) full |
| NISQ-safe | ✅ Yes (linear, p=1) |
| Best for | QAOA warm-starting, problem-inspired feature maps, NISQ kernel methods |

```python
from quprep.encode.qaoa_problem import QAOAProblemEncoder

enc = QAOAProblemEncoder(p=1, connectivity="linear")
result = enc.encode(x)   # x in [-π, π]
print(result.metadata["local_angles"])    # γ·xᵢ per qubit
print(result.metadata["coupling_angles"]) # γ·xᵢxⱼ per pair
print(result.metadata["depth"])           # 1 + p·(d + 3·(d−1)) for linear connectivity
```

**Normalization:** `minmax_pm_pi` ($[-\pi, \pi]$). Applied automatically.

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
        Need QUBO variables from continuous data?
  └─ Yes → Discretized encoding (continuous → binary, QUBO-ready)
  └─ No  → Is qubit count the main constraint?
              └─ Yes (log₂ d qubits) → Amplitude encoding (deep)
              └─ Yes (⌈d/2⌉ qubits) → Dense angle or Tensor product
              └─ No  → What is your task?
                         classification/regression → Angle or IQP
                         kernel methods            → IQP or ZZ feature map
                         high-expressivity QNN      → Data re-uploading
                         physics simulation/VQE     → Hamiltonian
                         feature correlations       → Entangled angle
                         QAOA warm-start / problem  → QAOA problem
```

---

## Inspecting encoded circuits

`inspect_encoding` gives a structured Python view of rotation angles and gate parameters after encoding — without parsing QASM strings:

```python
import numpy as np
import quprep as qd

x = np.array([0.5, 1.0, 1.5, 2.0])
enc = qd.AngleEncoder(rotation="ry")
result = enc.encode(x)

params = qd.inspect_encoding(result)
print(params.n_qubits)        # 4
print(params.encoding)        # "angle"
for g in params.gates:
    print(g.gate, g.qubit, g.angle)
# Ry  0  0.5
# Ry  1  1.0
# ...
```

`EncodingParams.gates` is a list of `GateParam` objects with fields `gate` (str), `qubit` (int), `angle` (float or None), `control` (int or None for entangled pairs), and `amplitudes` (ndarray or None for amplitude encoding). Works for all encoders; `angle` is `None` for non-rotation gates.

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
