"""
17 — Barren Plateau Detection
================================
Analytically estimate barren plateau risk before training a variational
quantum model.  No circuit simulation required.

    uv run python examples/17_barren_plateau.py
"""

import numpy as np

import quprep as qd
from quprep.core.dataset import Dataset

# ── Shared data ───────────────────────────────────────────────────────────────

rng = np.random.default_rng(0)

# ── 1. Shallow circuit — no risk ─────────────────────────────────────────────

print("=" * 60)
print("AngleEncoder on 4 features  (shallow circuit)")
print("=" * 60)

ds4 = Dataset(data=rng.uniform(0, 1, (50, 4)))
report = qd.detect_barren_plateau(qd.AngleEncoder(), ds4)
print(report)
print()

# ── 2. IQP on 8 features — mild risk ─────────────────────────────────────────

print("=" * 60)
print("IQPEncoder on 8 features")
print("=" * 60)

ds8 = Dataset(data=rng.uniform(0, 1, (50, 8)))
report_iqp = qd.detect_barren_plateau(qd.IQPEncoder(), ds8)
print(report_iqp)
print()

# ── 3. Large circuit — severe risk with mitigations ──────────────────────────

print("=" * 60)
print("IQPEncoder on 14 features  (large circuit)")
print("=" * 60)

ds14 = Dataset(data=rng.uniform(0, 1, (50, 14)))
report_large = qd.detect_barren_plateau(qd.IQPEncoder(), ds14)
print(report_large)
print()

# ── 4. Local vs global cost ───────────────────────────────────────────────────

print("=" * 60)
print("Effect of cost function type  (12 features)")
print("=" * 60)

ds12 = Dataset(data=rng.uniform(0, 1, (50, 12)))
r_global = qd.detect_barren_plateau(qd.IQPEncoder(), ds12, cost_type="global")
r_local  = qd.detect_barren_plateau(qd.IQPEncoder(), ds12, cost_type="local")

print(f"Global cost → risk: {r_global.risk_level:8s}  "
      f"(Var ≈ {r_global.gradient_variance:.2e})")
print(f"Local  cost → risk: {r_local.risk_level:8s}  "
      f"(Var ≈ {r_local.gradient_variance:.2e})")
print()

# ── 5. Sweep across encoder widths ───────────────────────────────────────────

print("=" * 60)
print("Risk level vs qubit count  (IQP encoder)")
print("=" * 60)
print(f"{'n_features':>12}  {'n_qubits':>9}  {'gradient_var':>14}  {'risk':>8}")
print("-" * 50)

for n in [2, 4, 6, 8, 10, 12, 14, 16]:
    ds_n = Dataset(data=rng.uniform(0, 1, (20, n)))
    r = qd.detect_barren_plateau(qd.IQPEncoder(), ds_n)
    print(f"{n:>12}  {r.n_qubits:>9}  {r.gradient_variance:>14.2e}  {r.risk_level:>8}")
