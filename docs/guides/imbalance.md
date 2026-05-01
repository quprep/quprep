# Class Imbalance Handling

QuPrep's `ImbalanceHandler` balances skewed class distributions before encoding — a critical step because quantum kernels and variational classifiers are sensitive to majority-class bias.

Four strategies are available:

| Strategy | Deps | How it works |
|---|---|---|
| `oversample` | *(none)* | Random duplication of minority samples |
| `undersample` | *(none)* | Random removal of majority samples |
| `smote` | scikit-learn *(core dep)* | Synthetic interpolated samples via k-NN |
| `adasyn` | `imbalanced-learn` | Adaptive density-based synthetic sampling |

---

## Basic usage

```python
import quprep as qd
from quprep.core.dataset import Dataset
import numpy as np

rng = np.random.default_rng(0)
X = rng.uniform(0, 1, (110, 4))
y = np.array([0] * 100 + [1] * 10)   # 90/10 imbalance
ds = Dataset(data=X, labels=y)

handler = qd.ImbalanceHandler(strategy="smote")
ds_bal = handler.fit_transform(ds)

from collections import Counter
print(Counter(ds_bal.labels))   # Counter({0: 100, 1: 100})
```

---

## Strategies

### Random oversample / undersample

No extra dependencies. Oversample duplicates minority-class rows; undersample discards majority-class rows.

```python
# Oversample to majority count (default)
qd.ImbalanceHandler(strategy="oversample").fit_transform(ds)

# Undersample to minority count (default)
qd.ImbalanceHandler(strategy="undersample").fit_transform(ds)
```

### SMOTE

Synthetic Minority Over-sampling Technique. For each minority sample, interpolates towards a random k-nearest neighbour to create a new synthetic point. SMOTE is generally preferred over random oversample because the synthetic samples are not duplicates.

```python
handler = qd.ImbalanceHandler(strategy="smote", k_neighbors=5)
ds_bal = handler.fit_transform(ds)
```

!!! note
    SMOTE falls back to random oversampling for any class with fewer than 2 samples and emits a `QuPrepWarning`.

### ADASYN

Adaptive Density-based Synthetic Sampling. Like SMOTE but focuses synthetic samples on regions where the classifier has more difficulty. Requires `imbalanced-learn`:

```bash
pip install quprep[imbalance]
```

```python
handler = qd.ImbalanceHandler(strategy="adasyn")
ds_bal = handler.fit_transform(ds)
```

---

## Controlling the target ratio

```python
# Balance all classes to 80% of the majority count
handler = qd.ImbalanceHandler(strategy="oversample", sampling_strategy=0.8)
```

`sampling_strategy="auto"` (default) balances to the majority class count for oversampling and to the minority class count for undersampling.

---

## Multi-class

All four strategies support more than two classes:

```python
y3 = np.array([0] * 50 + [1] * 20 + [2] * 5)
ds3 = Dataset(data=X3, labels=y3)
ds3_bal = qd.ImbalanceHandler(strategy="oversample").fit_transform(ds3)
# All classes balanced to 50 samples
```

---

## In a pipeline

`ImbalanceHandler` is a standalone clean-stage transformer — not wired into `Pipeline` directly (labels are required and pipelines are typically label-unaware). Apply it before constructing your pipeline:

```python
ds_balanced = qd.ImbalanceHandler(strategy="smote").fit_transform(ds_raw)

pipeline = qd.Pipeline(
    encoder=qd.IQPEncoder(),
    exporter=qd.QASMExporter(),
)
result = pipeline.fit_transform(ds_balanced)
```
