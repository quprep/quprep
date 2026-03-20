"""
06 — Circuit Visualization
===========================
draw_ascii() and draw_matplotlib() for all encoding types.

    uv run python examples/06_visualization.py

For matplotlib output:
    pip install quprep[viz]
    uv run python examples/06_visualization.py
"""

import numpy as np

from quprep.encode.angle import AngleEncoder
from quprep.encode.basis import BasisEncoder
from quprep.encode.entangled_angle import EntangledAngleEncoder
from quprep.encode.hamiltonian import HamiltonianEncoder
from quprep.encode.iqp import IQPEncoder
from quprep.encode.reupload import ReUploadEncoder
from quprep.export.visualize import draw_ascii, draw_matplotlib

x = np.array([0.5, 1.2, 0.75])   # 3-feature vector

# ── 1. ASCII diagrams ─────────────────────────────────────────────────────────

print("=" * 55)
print("Angle encoding  (Ry)")
print("=" * 55)
print(draw_ascii(AngleEncoder(rotation="ry").encode(x * np.pi)))

print("=" * 55)
print("Basis encoding")
print("=" * 55)
print(draw_ascii(BasisEncoder().encode(np.array([1.0, 0.0, 1.0]))))

print("=" * 55)
print("Entangled angle  (full, 2 layers)")
print("=" * 55)
print(draw_ascii(EntangledAngleEncoder(layers=2, entanglement="full").encode(x * np.pi)))

print("=" * 55)
print("IQP  (reps=1)")
print("=" * 55)
print(draw_ascii(IQPEncoder(reps=1).encode(x * np.pi)))

print("=" * 55)
print("Data re-uploading  (3 layers)")
print("=" * 55)
print(draw_ascii(ReUploadEncoder(layers=3).encode(x * np.pi)))

print("=" * 55)
print("Hamiltonian  (2 Trotter steps)")
print("=" * 55)
print(draw_ascii(HamiltonianEncoder(trotter_steps=2).encode(x)))

# ── 2. matplotlib ─────────────────────────────────────────────────────────────

try:
    import matplotlib.pyplot as plt  # noqa: F401
except ImportError:
    print("matplotlib not installed — skipping. Run: pip install quprep[viz]")
else:
    print("=" * 55)
    print("Saving matplotlib diagrams...")
    print("=" * 55)

    encodings = [
        ("angle",           AngleEncoder(rotation="ry").encode(x * np.pi)),
        ("entangled_full",  EntangledAngleEncoder(entanglement="full").encode(x * np.pi)),
        ("iqp",             IQPEncoder(reps=1).encode(x * np.pi)),
        ("reupload",        ReUploadEncoder(layers=2).encode(x * np.pi)),
        ("hamiltonian",     HamiltonianEncoder(trotter_steps=2).encode(x)),
    ]

    for name, enc in encodings:
        path = f"circuit_{name}.png"
        draw_matplotlib(enc, filename=path)
        print(f"  Saved: {path}")

    print()
    print("Done.")
