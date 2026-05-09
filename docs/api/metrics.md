# Encoding Quality Metrics

::: quprep.metrics
    options:
      show_root_heading: true
      show_source: false

::: quprep.metrics.expressibility
    options:
      show_root_heading: false
      members:
        - expressibility
        - entanglement_capability

::: quprep.metrics.kernel
    options:
      show_root_heading: false
      members:
        - kernel_alignment
        - EncoderMetrics
        - score_encoding

::: quprep.metrics.sensitivity
    options:
      show_root_heading: false
      members:
        - encoding_sensitivity
        - SensitivityResult

---

## Examples

### Identify sensitive features

```python
import quprep as qd

enc = qd.AngleEncoder(rotation="ry")
result = qd.encoding_sensitivity(enc, dataset, n_samples=20, seed=42)

print(result.scores)          # array of per-feature infidelity scores
print(result.feature_names)   # ["f0", "f1", ...]
for name, score in result.most_sensitive(n=3):
    print(f"{name}: {score:.4f}")
```
