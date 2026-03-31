# Qubit Suggestion

QuPrep can recommend a qubit count and encoding for your dataset before you build a pipeline. This is useful when you don't know how many qubits your hardware supports or which encoding fits your task.

---

## Basic usage

```python
import quprep as qd

suggestion = qd.suggest_qubits("data.csv", task="classification")
print(suggestion)
# Suggested qubits : 8
# Dataset features : 8
# NISQ-safe        : yes
# Encoding hint    : angle
# Reasoning        : ...
```

`suggest_qubits` reads the dataset, counts features, and returns a `QubitSuggestion` with a recommended qubit count, an encoding hint, and a NISQ safety flag.

---

## Checking the result

```python
suggestion = qd.suggest_qubits("data.csv", task="classification")

suggestion.n_qubits        # recommended qubit count
suggestion.n_features      # original feature count before any reduction
suggestion.encoding_hint   # e.g. "angle", "basis", "iqp"
suggestion.nisq_safe       # True if n_qubits ≤ 20
suggestion.reasoning       # human-readable explanation
suggestion.warning         # set if reduction is recommended, else None

print(suggestion.reasoning)
# "dataset has 8 feature(s) — one qubit per feature fits within the NISQ
#  budget of 20; angle encoding is NISQ-safe and widely applicable"
```

---

## Task-aware hints

The `task` parameter adjusts which encoding is recommended:

| `task` | Preferred encoding |
|---|---|
| `"classification"` (default) | `angle` |
| `"regression"` | `angle` |
| `"kernel"` (≤ 8 features) | `iqp` |
| `"kernel"` (> 8 features) | `angle` |
| `"qaoa"` | `basis` |
| `"simulation"` | `hamiltonian` |

Valid task values: `"classification"`, `"regression"`, `"qaoa"`, `"kernel"`, `"simulation"`.

```python
qd.suggest_qubits("data.csv", task="kernel")
# QubitSuggestion(n_qubits=8, encoding_hint='iqp', nisq_safe=True)

qd.suggest_qubits("data.csv", task="qaoa")
# QubitSuggestion(n_qubits=8, encoding_hint='basis', nisq_safe=True)
```

---

## NISQ ceiling

By default the suggestion is capped at 20 qubits — a practical limit for current NISQ hardware. If your dataset has more features than this cap, `suggestion.warning` explains that reduction is recommended.

```python
# Dataset with 50 features
suggestion = qd.suggest_qubits("wide_data.csv")
print(suggestion.n_qubits)   # 20
print(suggestion.warning)
# "Dataset has 50 features but qubit budget is 20. Apply a reducer
#  (e.g. PCAReducer(n_components=20)) before encoding to avoid information loss."
```

Override the ceiling for fault-tolerant or simulated backends:

```python
suggestion = qd.suggest_qubits("data.csv", max_qubits=50)
```

---

## Accepting the suggestion in a pipeline

```python
import quprep as qd

s = qd.suggest_qubits("data.csv", task="classification")

pipeline = qd.Pipeline(
    reducer=qd.PCAReducer(n_components=s.n_qubits),
    encoder=qd.AngleEncoder(),
)
result = pipeline.fit_transform("data.csv")
```

---

## CLI

```bash
quprep suggest data.csv
quprep suggest data.csv --task kernel
quprep suggest data.csv --task qaoa --max-qubits 50
```

Output:

```
Suggested qubits : 8
Dataset features : 8
NISQ-safe        : yes
Encoding hint    : angle
Reasoning        : dataset has 8 feature(s) — one qubit per feature fits
                   within the NISQ budget of 20; angle encoding is NISQ-safe
                   and widely applicable
```
