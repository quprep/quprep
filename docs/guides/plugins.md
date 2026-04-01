# Plugin System

QuPrep's plugin registry lets you register custom encoders and exporters and use them end-to-end through `prepare()` — no need to fork the library or patch internals.

---

## Registering an encoder

Decorate your encoder class with `@register_encoder("name")`:

```python
from quprep.plugins import register_encoder
from quprep.encode.base import BaseEncoder, EncodedResult
import numpy as np

@register_encoder("my_encoder")
class MyEncoder(BaseEncoder):
    def encode(self, x: np.ndarray) -> EncodedResult:
        params = np.sin(x)
        return EncodedResult(
            parameters=params,
            metadata={
                "encoding": "my_encoder",
                "n_qubits": len(x),
                "depth": 1,
            },
        )
```

Once registered, use it anywhere a built-in encoder name is accepted:

```python
import quprep as qd

result = qd.prepare(data, encoding="my_encoder")
print(result.encoded[0].metadata["encoding"])  # "my_encoder"
```

---

## Registering an exporter

```python
from quprep.plugins import register_exporter

@register_exporter("my_backend")
class MyExporter:
    def export(self, encoded):
        # convert EncodedResult to your backend's format
        return {"angles": encoded.parameters.tolist()}

    def export_batch(self, encoded_list):
        return [self.export(e) for e in encoded_list]
```

Use it via `prepare()`:

```python
result = qd.prepare(data, encoding="angle", framework="my_backend")
# result.circuits[0] is whatever your exporter returns
```

---

## Listing registered plugins

```python
from quprep.plugins import list_encoders, list_exporters

print(list_encoders())   # ["my_encoder", ...]
print(list_exporters())  # ["my_backend", ...]
```

These return only custom-registered names — built-in encoders/exporters are not listed here.

---

## Unregistering

```python
from quprep.plugins import unregister_encoder, unregister_exporter

unregister_encoder("my_encoder")
unregister_exporter("my_backend")
```

Useful in tests to clean up after each test case.

---

## Direct lookup

```python
from quprep.plugins import get_encoder_class, get_exporter_class

cls = get_encoder_class("my_encoder")
enc = cls()
result = enc.encode(x)
```

---

## Example: plugin encoder with QASM export

Custom encoders that produce standard angle-like output can be exported to any framework that supports `angle` encoding:

```python
import numpy as np
import quprep as qd
from quprep.plugins import register_encoder
from quprep.encode.base import BaseEncoder, EncodedResult

@register_encoder("sigmoid_angle")
class SigmoidAngleEncoder(BaseEncoder):
    """Encodes features as sigmoid-scaled Ry angles."""

    def encode(self, x: np.ndarray) -> EncodedResult:
        angles = np.pi / (1 + np.exp(-x))   # sigmoid scaled to (0, π)
        return EncodedResult(
            parameters=angles,
            metadata={
                "encoding": "angle",   # reuse angle encoding's QASM path
                "rotation": "ry",
                "n_qubits": len(x),
                "depth": 1,
            },
        )

result = qd.prepare(data, encoding="sigmoid_angle", framework="qasm")
print(result.circuit)
```

!!! tip
    Setting `metadata["encoding"] = "angle"` tells the QASM (and other) exporters to use the angle encoding circuit template. This is the easiest way to make custom encoders compatible with all existing exporters.
