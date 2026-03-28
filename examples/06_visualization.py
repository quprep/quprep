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

import quprep as qd

x = np.array([0.5, 1.2, 0.75])   # 3-feature vector

# ── 1. ASCII diagrams ─────────────────────────────────────────────────────────

print("=" * 55)
print("Angle encoding  (Ry)")
print("=" * 55)
print(qd.draw_ascii(qd.AngleEncoder(rotation="ry").encode(x * np.pi)))

print("=" * 55)
print("Basis encoding")
print("=" * 55)
print(qd.draw_ascii(qd.BasisEncoder().encode(np.array([1.0, 0.0, 1.0]))))

print("=" * 55)
print("Entangled angle  (full, 2 layers)")
print("=" * 55)
print(qd.draw_ascii(qd.EntangledAngleEncoder(layers=2, entanglement="full").encode(x * np.pi)))

print("=" * 55)
print("IQP  (reps=1)")
print("=" * 55)
print(qd.draw_ascii(qd.IQPEncoder(reps=1).encode(x * np.pi)))

print("=" * 55)
print("Data re-uploading  (3 layers)")
print("=" * 55)
print(qd.draw_ascii(qd.ReUploadEncoder(layers=3).encode(x * np.pi)))

print("=" * 55)
print("Hamiltonian  (2 Trotter steps)")
print("=" * 55)
print(qd.draw_ascii(qd.HamiltonianEncoder(trotter_steps=2).encode(x)))

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
        ("angle",           qd.AngleEncoder(rotation="ry").encode(x * np.pi)),
        ("entangled_full",  qd.EntangledAngleEncoder(entanglement="full").encode(x * np.pi)),
        ("iqp",             qd.IQPEncoder(reps=1).encode(x * np.pi)),
        ("reupload",        qd.ReUploadEncoder(layers=2).encode(x * np.pi)),
        ("hamiltonian",     qd.HamiltonianEncoder(trotter_steps=2).encode(x)),
    ]

    for name, enc in encodings:
        path = f"/tmp/circuit_{name}.png"
        qd.draw_matplotlib(enc, filename=path)
        print(f"  Saved: {path}")

    print()
    print("Done.")
