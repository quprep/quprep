"""
11 — Plugin System
==================
Register custom encoders and exporters via the plugin registry.
Use them end-to-end with prepare(), the CLI, and existing exporters —
no fork required.

    uv run python examples/11_plugins.py
"""

import numpy as np

import quprep as qd
from quprep.encode.base import BaseEncoder, EncodedResult
from quprep.plugins import (
    list_encoders,
    list_exporters,
    register_encoder,
    register_exporter,
    unregister_encoder,
    unregister_exporter,
)

rng = np.random.default_rng(42)
data = rng.uniform(0, 1, size=(8, 3))

# ── 1. Custom encoder ─────────────────────────────────────────────────────────
#
#   Wrap any function as an encoder with @register_encoder.
#   The decorated class must implement encode(x) → EncodedResult.

print("=" * 55)
print("Custom encoder — SigmoidAngle")
print("=" * 55)


@register_encoder("sigmoid_angle")
class SigmoidAngleEncoder(BaseEncoder):
    """Features encoded as sigmoid-scaled Ry angles in (0, π)."""

    @property
    def n_qubits(self):
        return None  # data-dependent

    @property
    def depth(self):
        return 1

    def encode(self, x: np.ndarray) -> EncodedResult:
        angles = np.pi / (1.0 + np.exp(-x))   # sigmoid → (0, π)
        return EncodedResult(
            parameters=angles,
            metadata={
                "encoding": "angle",   # reuse angle QASM path
                "rotation": "ry",
                "n_qubits": len(x),
                "depth": 1,
            },
        )


result = qd.prepare(data, encoding="sigmoid_angle", framework="qasm")
print(f"Encoding     : {result.encoded[0].metadata['encoding']}")
print(f"n_qubits     : {result.encoded[0].metadata['n_qubits']}")
print(f"Circuits     : {len(result.circuits)}")
print()
print("First QASM circuit:")
print(result.circuit)

# ── 2. Custom encoder — no QASM export ───────────────────────────────────────
#
#   Set metadata["encoding"] to a built-in name → that encoder's export path.
#   Or set a unique name → only usable with custom exporters (see section 4).

print("=" * 55)
print("Custom encoder — FourierAngle")
print("=" * 55)


@register_encoder("fourier_angle")
class FourierAngleEncoder(BaseEncoder):
    """Encode features via their discrete cosine transform."""

    @property
    def n_qubits(self):
        return None

    @property
    def depth(self):
        return 1

    def encode(self, x: np.ndarray) -> EncodedResult:
        # DCT-II coefficients, clipped to [-π, π]
        dct = np.fft.rfft(x, norm="ortho").real
        angles = np.clip(dct, -np.pi, np.pi)
        n = len(angles)
        return EncodedResult(
            parameters=angles,
            metadata={
                "encoding": "angle",
                "rotation": "rz",
                "n_qubits": n,
                "depth": 1,
            },
        )


x = rng.uniform(-1, 1, 3)
enc = FourierAngleEncoder()
r = enc.encode(x)
print(f"Input   : {x.round(4)}")
print(f"Angles  : {r.parameters.round(4)}")
print(f"n_qubits: {r.metadata['n_qubits']}")
print()

# ── 3. List registered plugins ────────────────────────────────────────────────
#
#   list_encoders() / list_exporters() return custom-registered names only.

print("=" * 55)
print("Registered plugins")
print("=" * 55)

print(f"Encoders  : {list_encoders()}")
print(f"Exporters : {list_exporters()}")
print()

# ── 4. Custom exporter ────────────────────────────────────────────────────────
#
#   @register_exporter("name") wires the class into prepare(framework="name").

print("=" * 55)
print("Custom exporter — JSON angles")
print("=" * 55)


@register_exporter("json_angles")
class JsonAnglesExporter:
    """Export encoded circuit as a JSON-serialisable dict of angles."""

    def export(self, encoded: EncodedResult) -> dict:
        return {
            "encoding": encoded.metadata.get("encoding"),
            "n_qubits": encoded.metadata.get("n_qubits"),
            "angles": encoded.parameters.tolist(),
        }

    def export_batch(self, encoded_list):
        return [self.export(e) for e in encoded_list]


result_json = qd.prepare(data, encoding="angle", framework="json_angles")
print(f"Type of circuit   : {type(result_json.circuits[0])}")
print(f"First circuit     : {result_json.circuits[0]}")
print()

# ── 5. Plugin encoder + plugin exporter together ──────────────────────────────

print("=" * 55)
print("Plugin encoder + plugin exporter")
print("=" * 55)

result_combo = qd.prepare(data, encoding="sigmoid_angle", framework="json_angles")
first = result_combo.circuits[0]
print(f"Encoding : {first['encoding']}")
print(f"n_qubits : {first['n_qubits']}")
print(f"Angles   : {[round(a, 4) for a in first['angles']]}")
print()

# ── 6. Unregister ─────────────────────────────────────────────────────────────
#
#   Clean up when done — useful in tests and notebooks.

print("=" * 55)
print("Unregister")
print("=" * 55)

unregister_encoder("sigmoid_angle")
unregister_encoder("fourier_angle")
unregister_exporter("json_angles")

print(f"Encoders after cleanup  : {list_encoders()}")
print(f"Exporters after cleanup : {list_exporters()}")
