"""
How to Inspect a Circuit
=========================
After encoding, QuPrep gives you three ways to inspect the result:
ASCII diagrams for quick terminal output, matplotlib diagrams for
publication, and structured parameter inspection via inspect_encoding().

    uv run python examples/how-to/inspect_a_circuit.py

For matplotlib output: pip install quprep[viz]
"""

import warnings

import numpy as np

import quprep as qd
from quprep import QuPrepWarning

rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, (3, 4))

with warnings.catch_warnings():
    warnings.simplefilter("ignore", QuPrepWarning)
    result = qd.Pipeline(
        normalizer=qd.Scaler(strategy="minmax_pi"),
        encoder=qd.AngleEncoder(),
    ).fit_transform(qd.NumpyIngester().load(X))

encoded = result.encoded
print(f"quprep {qd.__version__} | {len(encoded)} circuits encoded\n")


# ── 1. ASCII diagram ──────────────────────────────────────────────────────────

print("── 1. draw_ascii() ──────────────────────────────────────────────────────")
print(qd.draw_ascii(encoded[0]))


# ── 2. All encoders — ASCII overview ─────────────────────────────────────────

print("── 2. ASCII for different encoder types ─────────────────────────────────")

X2 = rng.uniform(0, 1, (2, 3))
ds2 = qd.NumpyIngester().load(X2)

encoders = [
    ("angle",        qd.AngleEncoder(),        qd.Scaler("minmax_pi")),
    ("basis",        qd.BasisEncoder(),         qd.Scaler("binary")),
    ("dense_angle",  qd.DenseAngleEncoder(),    qd.Scaler("minmax_pi")),
]

for name, enc, scaler in encoders:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", QuPrepWarning)
        r = qd.Pipeline(normalizer=scaler, encoder=enc).fit_transform(ds2)
    print(f"  {name}:")
    print(qd.draw_ascii(r.encoded[0]))


# ── 3. Structured parameter inspection ───────────────────────────────────────
#
# inspect_encoding() returns an EncodingParams dataclass with the rotation
# angles and gate parameters as Python objects — no QASM parsing required.
# Useful for debugging, logging, or feeding angles back into a classical loop.

print("── 3. inspect_encoding() ────────────────────────────────────────────────")
params = qd.inspect_encoding(encoded[0])
print(f"   Type     : {type(params).__name__}")
print(f"   encoding : {params.encoding}")
print(f"   n_qubits : {params.n_qubits}")
print(f"   depth    : {params.depth}")
print(f"   angles   : {params.angles}")
print()
for gp in params.gates:
    print(f"   {gp}")
print()


# ── 4. Matplotlib diagram (optional) ─────────────────────────────────────────

print("── 4. draw_matplotlib()  (pip install quprep[viz]) ──────────────────────")
try:
    path = "/tmp/quprep_circuit.png"
    qd.draw_matplotlib(encoded[0], filename=path)
    print(f"   Saved to {path}")
except Exception as e:
    print(f"   skipped: {e}")
