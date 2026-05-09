# Pipeline & Qubit Suggestion

---

## suggest_pipeline

::: quprep.core.recommender.suggest_pipeline
    options:
      show_source: false

---

## PipelineSuggestion

::: quprep.core.recommender.PipelineSuggestion
    options:
      show_source: false

---

## preprocessing_report

::: quprep.ingest.profiler.preprocessing_report
    options:
      show_source: false

---

## PreprocessingReport

::: quprep.ingest.profiler.PreprocessingReport
    options:
      show_source: false

---

## suggest_qubits

::: quprep.core.qubit_suggestion.suggest_qubits
    options:
      show_source: true

---

## QubitSuggestion

::: quprep.core.qubit_suggestion.QubitSuggestion
    options:
      show_source: false

---

## Examples

### Auto-suggest and build a pipeline

```python
import quprep as qd

suggestion = qd.suggest_pipeline(dataset, task="classification", qubits=8)
print(suggestion)            # PipelineSuggestion(encoder='iqp', normalizer='minmax_2pi', ...)

pipeline = suggestion.build()
result = pipeline.fit_transform(dataset)
```

### Preprocessing report before encoding

```python
import quprep as qd

report = qd.preprocessing_report(dataset, encoder=qd.AngleEncoder(), qubit_budget=8)
print(f"{report.n_issues} issues found")
for rec in report.recommendations:
    print(" •", rec)
```
