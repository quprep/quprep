# Dataset Inspection & Encoder Benchmarking

Two CLI commands for quickly profiling a dataset and measuring real encoder performance before committing to a full pipeline.

---

## `quprep inspect` — dataset profile

`inspect` loads a dataset and prints shape, feature types, missing-value counts, sparsity, per-feature statistics, and an encoding recommendation — all without encoding a single circuit.

```bash
quprep inspect data.csv
quprep inspect data.csv --task kernel --qubits 8
quprep inspect data.csv --no-recommend
```

### Example output

```
Source   : data.csv
Shape    : 150 samples × 4 features
Columns  : sepal_length, sepal_width, petal_length, petal_width
Types    : continuous: 4
Missing  : none
Sparsity : 0.0% zeros (0/600)

Feature stats (first 4):
  sepal_length  [4.3, 7.9]  mean=5.84  std=0.83
  sepal_width   [2.0, 4.4]  mean=3.05  std=0.43
  petal_length  [1.0, 6.9]  mean=3.76  std=1.77
  petal_width   [0.1, 2.5]  mean=1.20  std=0.76

Recommended encoding : angle
Qubits needed        : 4
...
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--task` | `classification` | Task for encoding recommendation |
| `--qubits` | none | Maximum qubit budget for recommendation |
| `--no-recommend` | off | Skip the encoding recommendation section |

**When to use**: before building a pipeline, to quickly understand what you're working with.

---

## `quprep benchmark` — encoder timing table

`benchmark` actually encodes a sample of your data with every encoder and reports gate count, circuit depth, 2-qubit gate count, and wall-clock encoding time per sample.

```bash
quprep benchmark data.csv
quprep benchmark data.csv --task classification --samples 10
quprep benchmark data.csv --include angle,iqp,amplitude
quprep benchmark data.csv --exclude hamiltonian --output results.json
```

### Example output

```
Source   : data.csv
Shape    : 150 samples × 4 features  (benchmarking on 5 samples)

Encoding              Qubits    Gates    Depth    2Q-Gates    Time/sample    NISQ
--------------------  -------  -------  -------  ---------  -------------  ---------
angle  *                   4        4        1          0         0.12 ms        yes
amplitude                  4        8        4          0         0.18 ms        yes
basis                      4        4        1          0         0.09 ms        yes
iqp                        4       16        6          4         0.31 ms        yes
reupload                   4       12       12          0         0.14 ms        yes
entangled_angle            4        7        4          3         0.21 ms        yes
hamiltonian                4        4        4          0         0.13 ms        yes
qaoa_problem               4       21        6          6         0.22 ms        yes

* recommended for task=classification
```

### Difference from `quprep compare`

| | `compare` | `benchmark` |
|---|---|---|
| Gate count / depth | Heuristic (analytical formula) | Heuristic (same formula) |
| Encoding time | — | Yes — actual wall-clock |
| Encodes real data | No | Yes |
| Speed | Instant | Proportional to `--samples` |

Use `compare` when you just want cost estimates. Use `benchmark` when you want to measure actual encoding throughput on your hardware before choosing an encoder.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--samples N` | `5` | Number of samples to encode per encoder |
| `--task` | none | Highlight recommended encoder for this task |
| `--include` | all | Comma-separated encoders to include |
| `--exclude` | none | Comma-separated encoders to exclude |
| `--output FILE` | none | Save results as JSON to FILE |

### JSON output

With `--output results.json` the benchmark saves a machine-readable report:

```json
{
  "source": "data.csv",
  "n_samples": 150,
  "n_features": 4,
  "n_bench_samples": 5,
  "task": "classification",
  "recommended": "angle",
  "results": [
    {
      "encoding": "angle",
      "n_qubits": 4,
      "gate_count": 4,
      "circuit_depth": 1,
      "two_qubit_gates": 0,
      "time_per_sample_ms": 0.12,
      "nisq_safe": true,
      "warning": null
    }
  ]
}
```
