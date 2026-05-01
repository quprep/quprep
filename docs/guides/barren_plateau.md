# Barren Plateau Detection

Barren plateaus are a critical failure mode in variational quantum algorithms: gradients vanish exponentially with circuit width, making training practically impossible.

QuPrep's `detect_barren_plateau()` estimates this risk **analytically at preprocessing time** — no circuit simulation, no training required.

---

## Background

McClean et al. (2018) showed that for a global cost function with an expressive parameterised circuit, the gradient variance scales as:

$$\text{Var}\!\left[\frac{\partial C}{\partial \theta}\right] \leq 2^{1-n}$$

where $n$ is the number of qubits. This means gradients halve with every additional qubit — for $n = 12$, the variance is less than $0.001$.

Cerezo et al. (2021) showed that **local cost functions** (single-qubit Pauli observables) exhibit only polynomial decay:

$$\text{Var} \approx \frac{1}{n^2}$$

This is the primary mitigation strategy.

---

## Quick start

```python
import quprep as qd
from quprep.core.dataset import Dataset
import numpy as np

ds = Dataset(data=np.random.default_rng(0).uniform(0, 1, (50, 8)))
report = qd.detect_barren_plateau(qd.IQPEncoder(), ds)
print(report)
```

```
BarrenPlateauReport(iqp)
  n_qubits         : 8
  circuit_depth    : 64
  gradient_variance: 7.81e-03  (upper bound)
  risk_level       : mild
  mitigations:
    - Use a local cost function (single-qubit Pauli observables) — polynomial gradient decay instead of exponential
```

---

## Risk levels

| Risk | Gradient variance | Typical qubit count |
|---|---|---|
| `none` | > 0.05 | ≤ 5 |
| `mild` | 0.005 – 0.05 | 6 – 8 |
| `high` | 0.0005 – 0.005 | 9 – 11 |
| `severe` | ≤ 0.0005 | ≥ 12 |

---

## Global vs local cost

```python
r_global = qd.detect_barren_plateau(qd.IQPEncoder(), ds, cost_type="global")
r_local  = qd.detect_barren_plateau(qd.IQPEncoder(), ds, cost_type="local")

print(r_global.risk_level)   # high
print(r_local.risk_level)    # mild
```

Pass `cost_type="local"` when your training objective is a sum of single-qubit expectations (e.g. Pauli-Z on each qubit). This substantially improves the bound and is the recommended default for all circuits with $n \geq 8$.

---

## Mitigation strategies

The report's `mitigations` list grows with risk level:

| Risk | Mitigations suggested |
|---|---|
| `mild` | Use local cost function |
| `high` | + Layer-wise training, identity-block initialisation |
| `severe` | + Reduce qubit count, prefer shallower encodings |

```python
for tip in report.mitigations:
    print("-", tip)
```

---

## Sweep across encoders

```python
encoders = {
    "angle"    : qd.AngleEncoder(),
    "iqp"      : qd.IQPEncoder(),
    "reupload" : qd.ReUploadEncoder(),
}

for name, enc in encoders.items():
    r = qd.detect_barren_plateau(enc, ds)
    print(f"{name:12s}  n_qubits={r.n_qubits}  risk={r.risk_level}")
```

---

## References

McClean J.R. et al. "Barren plateaus in quantum neural network training landscapes." *Nature Communications* 9, 4812 (2018). [doi:10.1038/s41467-018-07090-4](https://doi.org/10.1038/s41467-018-07090-4)

Cerezo M. et al. "Cost function dependent barren plateaus in shallow parametrized quantum circuits." *Nature Communications* 12, 1791 (2021). [doi:10.1038/s41467-021-21728-w](https://doi.org/10.1038/s41467-021-21728-w)
