# Encoding Comparison & Smart Recommendations

QuPrep can compare all encoding methods side-by-side and recommend the best one for your dataset — analytically, before a single circuit is built.

---

## Encoding comparison

`compare_encodings()` profiles your dataset and runs cost estimation for every encoder, returning a table of qubit count, gate count, circuit depth, 2-qubit gates, and NISQ safety.

```python
import quprep as qd

result = qd.compare_encodings("data.csv")
print(result)
```

```
Encoding            Qubits    Gate Count  Depth    2Q Gates   NISQ Safe
------------------  --------  -----------  -------  ----------  ----------
angle               8         8            1        0           Yes
amplitude           3         16           8        8           Yes
basis               8         8            1        0           Yes
iqp                 8         84           64       28          Yes
reupload            8         24           24       0           Yes
entangled_angle     8         15           8        7           Yes
hamiltonian         8         8            8        0           Yes
```

No circuits are generated — costs are estimated analytically, so comparison is fast even for large datasets.

---

## Filtering encoders

```python
# Only compare a subset
result = qd.compare_encodings(X, include=["angle", "iqp", "amplitude"])

# Exclude encoders you know won't fit
result = qd.compare_encodings(X, exclude=["amplitude", "hamiltonian"])
```

---

## Task-aware recommendation

Pass `task=` to highlight the best encoding for your use case:

```python
result = qd.compare_encodings("data.csv", task="classification")
print(result)
# The recommended encoding is starred in the table
```

Valid tasks: `classification`, `regression`, `qaoa`, `kernel`, `simulation`.

---

## Qubit budget

```python
result = qd.compare_encodings("data.csv", qubits=8)
# Encoders requiring more than 8 qubits have nisq_safe=False and a budget warning
```

---

## Picking the best

```python
best = result.best(prefer="nisq")    # NISQ-safe, then lowest depth (default)
best = result.best(prefer="depth")   # Globally shallowest circuit
best = result.best(prefer="gates")   # Fewest total gates
best = result.best(prefer="qubits")  # Fewest qubits
print(best.encoding, best.n_qubits, best.circuit_depth)
```

---

## Programmatic access

```python
for row in result.to_dict():
    print(row["encoding"], row["nisq_safe"], row["circuit_depth"])
```

---

## CLI

```bash
quprep compare data.csv
quprep compare data.csv --task classification --qubits 8
quprep compare data.csv --include angle,iqp,amplitude
quprep compare data.csv --exclude amplitude,hamiltonian
```

---

## Smart recommendation engine

`recommend()` goes beyond a fixed lookup table — it adapts its scores based on what it finds in your data:

| Signal | How it affects the recommendation |
|---|---|
| `n_samples > 500` | Penalises amplitude (expensive state prep per sample); rewards reupload |
| `n_samples < 20` | Penalises reupload (high expressivity → overfitting risk) |
| `missing_rate > 10%` | Penalises amplitude (requires exact unit norm) |
| Negative values in data | Rewards amplitude (handles negatives naturally via superposition); penalises basis (all negatives → 0 after binarization) |
| Sparse data (many zeros) | Boosts basis encoding |
| Correlated features | Boosts IQP and entangled angle (entanglement captures inter-feature structure) |
| `n_features > 15` | Penalises IQP (depth grows as O(d²)) |

```python
rec = qd.recommend("data.csv", task="classification", qubits=8)
print(rec)
# Recommended encoding : iqp
# Qubits needed        : 8
# Circuit depth        : O(d²·reps)
# NISQ safe            : yes
# Score                : 54.0
# Reason               : best fit for classification tasks; continuous features
#                        map naturally to rotation angles; NISQ-safe (shallow circuit).
# Alternatives         :
#   angle            score=45.0  O(d)
#   reupload         score=45.0  O(d·layers)
#   ...
```

The `reason` field always explains which dataset signals drove the recommendation.

---

## Combining both

```python
# Compare first, then apply the best one
result = qd.compare_encodings("data.csv", task="classification", qubits=8)
best = result.best(prefer="nisq")

pipeline = qd.Pipeline(
    encoder=getattr(qd, f"{best.encoding.title().replace('_', '')}Encoder")(),
    exporter=qd.QASMExporter(),
)
pipeline_result = pipeline.fit_transform("data.csv")
```

Or use `recommend()` directly with `.apply()`:

```python
rec = qd.recommend("data.csv", task="classification")
pipeline_result = rec.apply("data.csv")
```
