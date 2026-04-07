"""
Example 07 — QUBO / Ising

Demonstrates:
  - Problem formulation (Max-Cut, Knapsack)
  - Brute-force and simulated-annealing solvers
  - QUBO ↔ Ising round-trip
  - D-Wave export
  - QAOA circuit generation
"""

import numpy as np

from quprep.qubo import (
    ising_to_qubo,
    knapsack,
    max_cut,
    qaoa_circuit,
    qubo_to_ising,
)
from quprep.qubo.solver import solve_brute, solve_sa  # classical reference utilities

# ── Max-Cut ───────────────────────────────────────────────────────────────────
print("── Max-Cut ──────────────────────────────────────────────────────────────")

adj = np.array([[0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0]], dtype=float)

q = max_cut(adj)
print(f"Variables : {q.Q.shape[0]}")
print(f"Offset    : {q.offset}")

sol = solve_brute(q)
bits = "".join(str(int(b)) for b in sol.x)
print(f"Exact solution : x={bits}  energy={sol.energy:.4f}")

# Evaluate a specific assignment manually
x_test = np.array([0.0, 1.0, 0.0, 1.0])
print(f"evaluate([0,1,0,1]) = {q.evaluate(x_test):.4f}")

# ── Ising round-trip ──────────────────────────────────────────────────────────
print("\n── Ising round-trip ─────────────────────────────────────────────────────")

ising = q.to_ising()
print(f"Ising h : {ising.h}")
print(f"Ising J matrix:\n{ising.J}")

q2 = ising.to_qubo()
print(f"Round-trip Q matches: {np.allclose(q.Q, q2.Q)}")

# ── D-Wave export ─────────────────────────────────────────────────────────────
print("\n── D-Wave export ────────────────────────────────────────────────────────")

dwave = q.to_dwave()
print(f"D-Wave BQM dict ({len(dwave)} entries):")
for (i, j), v in list(dwave.items())[:5]:
    print(f"  ({i},{j}): {v:.4f}")

# Standalone ising_to_qubo
q3 = ising_to_qubo(qubo_to_ising(q))
print(f"ising_to_qubo() round-trip matches: {np.allclose(q.Q, q3.Q)}")

# ── Knapsack with simulated annealing ─────────────────────────────────────────
print("\n── Knapsack (SA solver) ─────────────────────────────────────────────────")

weights = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
values  = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
kq = knapsack(weights, values, capacity=10.0)
print(f"Variables : {kq.Q.shape[0]}")

sol_sa = solve_sa(kq, n_steps=20_000, restarts=3, seed=42)
bits_sa = "".join(str(int(b)) for b in sol_sa.x[:len(weights)])
print(f"SA solution : x={bits_sa}  energy={sol_sa.energy:.4f}")

sol_exact = solve_brute(kq)
bits_exact = "".join(str(int(b)) for b in sol_exact.x[:len(weights)])
print(f"Exact       : x={bits_exact}  energy={sol_exact.energy:.4f}")

# ── QAOA circuit ─────────────────────────────────────────────────────────────
print("\n── QAOA circuit ─────────────────────────────────────────────────────────")

small_adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
qsmall = max_cut(small_adj)
qasm = qaoa_circuit(qsmall, p=2)
lines = qasm.strip().splitlines()
print(f"QAOA circuit ({len(lines)} lines):")
for line in lines[:6]:
    print(f"  {line}")
print("  ...")

print("\nDone.")
