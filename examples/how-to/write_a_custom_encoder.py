"""
How to Write a Custom Encoder
===============================
QuPrep's plugin registry lets you register custom encoders and exporters
without forking the package. Once registered, they work everywhere QuPrep's
built-in encoders do: prepare(), Pipeline, CLI, and all exporters.

    uv run python examples/how-to/write_a_custom_encoder.py
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning
from quprep.encode.base import BaseEncoder, EncodedResult
from quprep.plugins import get_encoder_class, list_encoders, register_encoder, unregister_encoder

print(f"quprep {qd.__version__}\n")


# ── 1. Define a custom encoder ────────────────────────────────────────────────
#
# Subclass BaseEncoder and implement encode(). The method receives a 1D numpy
# array (one row of the preprocessed dataset) and must return an EncodedResult.
# EncodedResult takes: a circuit_fn (callable → QASM string), metadata dict,
# and parameters dict (the raw angles/values).

@register_encoder("hadamard_angle")
class HadamardAngleEncoder(BaseEncoder):
    """
    Applies H then Ry(x_i) per qubit.
    Creates superposition before rotation — a simple custom variant.
    """

    name = "hadamard_angle"

    @property
    def n_qubits(self):
        return None  # data-dependent: one qubit per feature

    @property
    def depth(self):
        return 2  # H + Ry per qubit

    def encode(self, x: np.ndarray) -> EncodedResult:
        n = len(x)
        angles = x.tolist()

        def circuit_fn():
            lines = ["OPENQASM 3.0;", 'include "stdgates.inc";', f"qubit[{n}] q;"]
            for i, a in enumerate(angles):
                lines.append(f"h q[{i}];")
                lines.append(f"ry({a}) q[{i}];")
            return "\n".join(lines)

        return EncodedResult(
            circuit_fn=circuit_fn,
            metadata={"encoding": self.name, "n_qubits": n, "depth": 2},
            parameters={"angles": angles},
        )


# ── 2. Registration via @register_encoder decorator happens at class definition.
#    Confirm it landed in the registry.

print("── 1. Register ──────────────────────────────────────────────────────────")
print(f"   Registered encoders : {list_encoders()}")
print()


# ── 3. Use it in a Pipeline ───────────────────────────────────────────────────

print("── 2. Use in Pipeline ───────────────────────────────────────────────────")
rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, (5, 3))
ds = qd.NumpyIngester().load(X)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = qd.Pipeline(
        encoder=HadamardAngleEncoder(),
    ).fit_transform(ds)

print(f"   Circuits : {len(result.encoded)}")
print(f"   Qubits   : {result.encoded[0].metadata['n_qubits']}")
print(f"   Depth    : {result.encoded[0].metadata['depth']}")
print()

print("── 3. QASM output ───────────────────────────────────────────────────────")
qasm = qd.QASMExporter().export(result.encoded[0])
print(qasm)


# ── 4. Confirm the encoder is reachable from the registry ────────────────────
#
# The registry is useful for framework-agnostic code that selects encoders by
# name (e.g., config files, CLI flags). Retrieve the class and instantiate it.

print("── 4. Retrieve encoder by name ──────────────────────────────────────────")
EncoderCls = get_encoder_class("hadamard_angle")
enc_from_registry = EncoderCls()
sample = X[0]
encoded_sample = enc_from_registry.encode(sample)
print(f"   Retrieved class : {EncoderCls.__name__}")
print(f"   Sample qubits   : {encoded_sample.metadata['n_qubits']}")
print()


# ── 5. Unregister ─────────────────────────────────────────────────────────────

unregister_encoder("hadamard_angle")
print("── 5. Unregistered ──────────────────────────────────────────────────────")
print(f"   Registered encoders : {list_encoders()}")
