# Pipeline & Qubit Suggestion

QuPrep can recommend a full preprocessing pipeline — or just a qubit count — for your dataset before you write any pipeline code.

---

## Auto-suggest a full pipeline

`suggest_pipeline` analyses your dataset and returns a `PipelineSuggestion` with recommended imputer, outlier handler, reducer, normalizer, and encoder. Call `.build()` to get a ready-to-use `Pipeline`:

```python
import quprep as qd

suggestion = qd.suggest_pipeline(dataset, task="classification", qubits=8)
print(suggestion)
# PipelineSuggestion(encoder='iqp', normalizer='minmax_2pi',
#                    reducer='pca', reducer_n_components=8,
#                    imputer='median', outlier_handler=None)

pipeline = suggestion.build()
result = pipeline.fit_transform(dataset)
```

### What `suggest_pipeline` checks

| Dataset property | Suggestion |
|---|---|
| Any NaN values | `imputer='mean'` or `'median'` (based on skewness) |
| Feature IQR ratio > 10 | `outlier_handler='iqr'` |
| `n_features > qubits` | `reducer='pca'`, `reducer_n_components=qubits` |
| Chosen encoder | Matching `normalizer` from built-in map |

### `PipelineSuggestion` attributes

```python
suggestion.encoder            # str, e.g. "iqp"
suggestion.normalizer         # str, e.g. "minmax_2pi"
suggestion.reducer            # str or None, e.g. "pca"
suggestion.reducer_n_components  # int or None
suggestion.imputer            # str or None, e.g. "median"
suggestion.outlier_handler    # str or None, e.g. "iqr"
```

---

## Preprocessing report

`preprocessing_report` gives you a structured, human-readable audit of your dataset before encoding — identifying issues and recommending specific QuPrep components:

```python
import quprep as qd

report = qd.preprocessing_report(dataset, encoder=qd.AngleEncoder(), qubit_budget=8)
print(f"{report.n_issues} issues found")
for rec in report.recommendations:
    print(" •", rec)
```

Example output:
```
3 issues found
 • Missing values detected — consider Imputer(strategy='median')
 • Outliers detected in 2 features — consider OutlierHandler(method='iqr')
 • 15 features exceed qubit budget of 8 — consider PCAReducer(n_components=8)
```

What the report checks:

| Check | Recommendation |
|---|---|
| NaN values | `Imputer` |
| Large IQR outliers | `OutlierHandler` |
| `n_features > qubit_budget` | `PCAReducer` |
| `n_features > 20` (no budget) | `PCAReducer` or `HardwareAwareReducer` |
| Class imbalance ratio > 3:1 | `ImbalanceHandler` |
| Encoder-specific range issues | From `check_compatibility` |

---

## Qubit suggestion (simple)

---

## Basic usage (suggest_qubits)

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
suggestion.encoding_hint   # e.g. "angle", "iqp", "qaoa_problem"
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
| `"qaoa"` | `qaoa_problem` |
| `"simulation"` | `hamiltonian` |

Valid task values: `"classification"`, `"regression"`, `"qaoa"`, `"kernel"`, `"simulation"`.

```python
qd.suggest_qubits("data.csv", task="kernel")
# QubitSuggestion(n_qubits=8, encoding_hint='iqp', nisq_safe=True)

qd.suggest_qubits("data.csv", task="qaoa")
# QubitSuggestion(n_qubits=8, encoding_hint='qaoa_problem', nisq_safe=True)
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
