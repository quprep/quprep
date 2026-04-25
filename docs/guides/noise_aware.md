# Noise-Aware Preprocessing

When running on real quantum hardware, not all qubits are equal. Gate error rates vary across the chip, coherence times differ by qubit, and two-qubit gates between non-adjacent qubits require inserted SWAP operations that add noise. `NoiseAwarePreprocessor` addresses all three issues before encoding begins.

---

## Quick example

```python
import numpy as np
import quprep as qd
from quprep.preprocess.noise_aware import NoiseProfile

# Describe your backend
profile = NoiseProfile(
    qubit_error_rates=[0.001, 0.003, 0.001, 0.002, 0.004],
    coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)],
    t1=[180.0, 120.0, 175.0, 160.0, 110.0],   # µs
    t2=[ 90.0,  65.0,  88.0,  80.0,  55.0],   # µs
)

prep = qd.NoiseAwarePreprocessor(profile, encoding="entangled_angle")

pipeline = qd.Pipeline(
    preprocessor=prep,
    encoder=qd.EntangledAngleEncoder(),
)
result = pipeline.fit_transform("data.csv")
```

After `fit_transform`, the dataset columns are reordered so the most informative features land on the quietest qubits, and adjacent logical qubits are physically connected on the chip.

---

## NoiseProfile

`NoiseProfile` is a dataclass that describes the noise characteristics of a backend.

```python
from quprep.preprocess.noise_aware import NoiseProfile

profile = NoiseProfile(
    qubit_error_rates=[0.001, 0.005, 0.002],   # required — one per qubit
    coupling_map=[(0, 1), (1, 2)],             # required — native 2Q connections
    t1=[150.0, 90.0, 160.0],                   # optional — relaxation times (µs)
    t2=[ 80.0, 45.0,  85.0],                   # optional — dephasing times (µs)
    cx_error_rates={(0, 1): 0.01, (1, 2): 0.012},  # optional — per-pair CX error
)
```

**`qubit_error_rates`** — Per-qubit single-qubit gate or readout error. The primary quality signal; lower is better.

**`coupling_map`** — Pairs of physically connected qubits. Any two-qubit gate between qubits *not* in this list requires SWAP insertion by the hardware compiler.

**`t1` / `t2`** — Coherence times in microseconds. Qubits with shorter coherence are penalised in the quality ranking. Omit if unavailable.

**`cx_error_rates`** — Per-pair CX error rates, stored for informational purposes.

Where to get these values: IBM Quantum / IQM / IonQ provider dashboards, or from `qiskit_ibm_runtime.IBMBackend.properties()`.

---

## Three optimisations

### 1 — Qubit assignment

Features are ranked by variance; qubits are ranked by a combined quality score (error rate + 1/T1 + 1/T2). The highest-variance feature is assigned to the lowest-score (least noisy) qubit, and so on.

This matters because high-variance features carry the most information. Placing them on noisy qubits wastes discriminative capacity.

```python
prep.fit(dataset)

prep.qubit_assignment_
# e.g. [2, 0, 1] — feature 0 → qubit 2, feature 1 → qubit 0, feature 2 → qubit 1
```

### 2 — Topology-aware reordering

Applies to entangled encodings: `entangled_angle`, `iqp`, `zz_feature_map`, `pauli_feature_map`, `reupload`.

After selecting the best *n* qubits, the preprocessor greedily threads them into a path through the coupling map — so that adjacent logical qubits (which the encoder connects with CNOT/CZ gates) are physically adjacent on the chip. This eliminates or reduces compiler-inserted SWAPs.

```python
prep.estimated_swaps_before_   # SWAPs with noise-ranked but topology-naive assignment
prep.estimated_swaps_after_    # SWAPs after topology path optimisation
```

For single-qubit encodings (`angle`, `basis`, `amplitude`) both estimates are 0 — there are no two-qubit gates.

### 3 — Angle dead-zone remapping

The rotation gates Ry(0) = |0⟩ and Ry(π) = |1⟩ produce computational-basis states with no superposition. Encoded angles near these poles are less discriminative and more sensitive to certain noise channels.

Set `angle_deadzone` to push all angles away from 0 and π:

```python
prep = qd.NoiseAwarePreprocessor(
    profile,
    encoding="angle",
    angle_deadzone=0.05,   # maps [0, π] → [0.05π, 0.95π]
)
```

!!! warning "Normalise first"
    `angle_deadzone` remaps values assuming they are in `[0, π]`. Apply
    `Scaler(method="minmax_pi")` **before** `NoiseAwarePreprocessor` in the pipeline,
    or use it standalone after normalisation.

---

## Standalone usage

`NoiseAwarePreprocessor` follows the sklearn `fit / transform / fit_transform` pattern and can be used outside a `Pipeline`:

```python
from quprep.core.dataset import Dataset
from quprep.preprocess.noise_aware import NoiseAwarePreprocessor, NoiseProfile
import numpy as np

profile = NoiseProfile(
    qubit_error_rates=[0.001, 0.002, 0.003, 0.001],
    coupling_map=[(0, 1), (1, 2), (2, 3)],
)

rng = np.random.default_rng(0)
data = rng.standard_normal((200, 3))
data[:, 2] *= 4   # feature 2 has highest variance
ds = Dataset(data=data, feature_names=["a", "b", "c"])

prep = NoiseAwarePreprocessor(profile, encoding="entangled_angle")
result = prep.fit_transform(ds)

print(prep.qubit_assignment_)        # [2, 0, 1] — feature 2 (high-var) → qubit 0 (best)
print(prep.estimated_swaps_before_)  # SWAPs without topology opt
print(prep.estimated_swaps_after_)   # SWAPs after topology opt
print(result.metadata["noise_aware"])  # True
```

---

## In a Pipeline

The preprocessor slot accepts a single transformer or a list:

```python
pipeline = qd.Pipeline(
    preprocessor=[
        qd.WindowTransformer(window_size=8),   # modality-specific step first
        qd.NoiseAwarePreprocessor(profile, encoding="angle"),
    ],
    encoder=qd.AngleEncoder(),
)
```

---

## When more features than qubits

If the dataset has more features than the noise profile has qubits, `fit()` raises a `ValueError`. Reduce first:

```python
pipeline = qd.Pipeline(
    reducer=qd.HardwareAwareReducer(backend=5, encoding="angle"),  # cap at 5 features
    preprocessor=qd.NoiseAwarePreprocessor(profile, encoding="angle"),
    encoder=qd.AngleEncoder(),
)
```
