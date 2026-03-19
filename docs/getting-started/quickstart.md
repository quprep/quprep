# Quickstart

## One-liner

Convert a CSV file to quantum circuits with a single call:

```python
import quprep

result = quprep.prepare("data.csv", encoding="angle")
print(result.circuit)   # first sample as OpenQASM 3.0 string
print(len(result.circuits))  # one circuit per row
```

Default framework is `qasm` (OpenQASM 3.0) — no optional dependencies required.

### With Qiskit

```bash
pip install quprep[qiskit]
```

```python
result = quprep.prepare("data.csv", encoding="angle", framework="qiskit")
qc = result.circuit      # qiskit.QuantumCircuit
qc.draw("mpl")
```

### Encoding options

```python
# Angle encoding — Ry rotation per qubit (default)
result = quprep.prepare("data.csv", encoding="angle", rotation="ry")

# Amplitude encoding — entire vector as quantum state amplitudes
result = quprep.prepare("data.csv", encoding="amplitude")

# Basis encoding — binary features to computational basis states
result = quprep.prepare("data.csv", encoding="basis")
```

---

## Pipeline API

For more control, build a pipeline manually:

```python
from quprep import Pipeline
from quprep.encode.angle import AngleEncoder
from quprep.export.qasm_export import QASMExporter

pipeline = Pipeline(
    encoder=AngleEncoder(rotation="ry"),
    exporter=QASMExporter(),
)

result = pipeline.fit_transform("data.csv")
print(result.circuits[0])   # QASM string for first sample
print(result.dataset)       # processed Dataset object
print(result.encoded[0].metadata)  # encoding metadata
```

### With cleaning

```python
from quprep import Pipeline
from quprep.clean.imputer import Imputer
from quprep.clean.outlier import OutlierHandler
from quprep.encode.angle import AngleEncoder
from quprep.export.qasm_export import QASMExporter

pipeline = Pipeline(
    cleaner=Imputer(strategy="knn"),
    encoder=AngleEncoder(),
    exporter=QASMExporter(),
)
result = pipeline.fit_transform("data.csv")
```

### From a NumPy array

```python
import numpy as np
from quprep import Pipeline
from quprep.encode.angle import AngleEncoder

data = np.random.rand(100, 8)

pipeline = Pipeline(encoder=AngleEncoder())
result = pipeline.fit_transform(data)
print(len(result.encoded))  # 100 EncodedResult objects
```

### From a Pandas DataFrame

```python
import pandas as pd
from quprep import Pipeline
from quprep.encode.basis import BasisEncoder

df = pd.read_csv("data.csv")

pipeline = Pipeline(encoder=BasisEncoder())
result = pipeline.fit_transform(df)
```

---

## CLI

```bash
# Convert to OpenQASM 3.0 (stdout)
quprep convert data.csv

# Save to file
quprep convert data.csv --output circuit.qasm

# Choose encoding and rotation
quprep convert data.csv --encoding angle --rotation rx

# Limit to first 5 samples
quprep convert data.csv --samples 5

# Basis encoding
quprep convert data.csv --encoding basis
```

---

## What happens automatically

When you call `prepare()` or `Pipeline.fit_transform()`, QuPrep automatically applies the correct normalization for your encoding:

| Encoding | Auto-normalization |
|---|---|
| `angle` (Ry) | Scale features to [0, π] |
| `angle` (Rx/Rz) | Scale features to [−π, π] |
| `amplitude` | L2-normalize each sample (‖x‖₂ = 1) |
| `basis` | Binarize features to {0, 1} |

You never need to think about this unless you want to override it.
