# Data Drift Detection

QuPrep can warn you when data passed to `transform()` is statistically different from the data the pipeline was fitted on. This is useful in production scenarios where incoming data may shift away from the training distribution over time.

---

## Basic usage

```python
import quprep as qd

det = qd.DriftDetector()

pipeline = qd.Pipeline(
    encoder=qd.AngleEncoder(),
    drift_detector=det,
)
pipeline.fit(X_train)

result = pipeline.transform(X_test)
print(result.drift_report)
# DriftReport: no drift detected
```

If drift is detected, a `QuPrepWarning` is issued automatically and `result.drift_report.overall_drift` is `True`.

---

## Checking the report

```python
report = result.drift_report

report.overall_drift          # True / False
report.n_features_drifted     # int
report.drifted_features       # list of feature names
report.feature_stats          # per-feature dict

# Per-feature detail
stats = report.feature_stats["age"]
stats["train_mean"]           # mean at fit time
stats["new_mean"]             # mean in new data
stats["mean_shift_sigmas"]    # shift in units of training std
stats["std_ratio"]            # new_std / train_std
```

---

## Thresholds

```python
det = qd.DriftDetector(
    mean_threshold=3.0,   # flag when mean shifts > 3σ (default)
    std_threshold=2.0,    # flag when std doubles or halves (default)
    warn=True,            # issue QuPrepWarning when drift found (default)
)
```

Lower `mean_threshold` → more sensitive. Set `warn=False` to use the report programmatically without warnings.

---

## Standalone usage (without Pipeline)

```python
import quprep as qd

det = qd.DriftDetector(warn=False)
det.fit(train_dataset)

report = det.check(new_dataset)
if report.overall_drift:
    print(report)
```

---

## Drift after pipeline serialization

Drift detector state is preserved through `save()`/`load()`:

```python
pipeline.save("pipeline.pkl")
loaded = qd.Pipeline.load("pipeline.pkl")

result = loaded.transform(X_new)
print(result.drift_report)   # uses training stats from original fit
```

---

## Suppressing warnings selectively

```python
import warnings
from quprep import QuPrepWarning

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = pipeline.transform(X_new)

# Still check programmatically
if result.drift_report.overall_drift:
    print(result.drift_report.drifted_features)
```
