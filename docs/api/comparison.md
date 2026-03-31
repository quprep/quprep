# Comparison API

API reference for `quprep.compare` — encoding comparison and side-by-side cost analysis.

All symbols are available on the top-level namespace:

```python
import quprep as qd

result = qd.compare_encodings("data.csv", task="classification")
best   = result.best(prefer="nisq")
```

---

## compare_encodings

::: quprep.compare.compare_encodings
    options:
      show_source: false

---

## ComparisonResult

::: quprep.compare.ComparisonResult
    options:
      show_source: false
      members:
        - best
        - to_dict

---

## EncodingRecommendation

::: quprep.core.recommender.EncodingRecommendation
    options:
      show_source: false
      members:
        - apply

---

## recommend

::: quprep.core.recommender.recommend
    options:
      show_source: false
