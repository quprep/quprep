"""
How to Solve a QUBO / Ising Problem
=====================================
QuPrep includes a lightweight QUBO module for formulating combinatorial
optimisation problems and solving them classically or exporting to QAOA circuits.

    uv run python examples/how-to/solve_qubo.py
"""

import numpy as np

import quprep as qd
from quprep.qubo import (
    ising_to_qubo,
    knapsack,
    max_cut,
    qaoa_circuit,
    qubo_to_ising,
)
from quprep.qubo.solver import solve_brute, solve_sa

print(f"quprep {qd.__version__}\n")


# ── 1. Max-Cut formulation ────────────────────────────────────────────────────
#
# max_cut() takes a weighted adjacency matrix. Here we build a 4-cycle graph:
# vertices 0–3, edges (0,1), (1,2), (2,3), (0,3) — each with weight 1.

print("── 1. Max-Cut ───────────────────────────────────────────────────────────")
adj = np.array(
    [[0, 1, 0, 1],
     [1, 0, 1, 0],
     [0, 1, 0, 1],
     [1, 0, 1, 0]],
    dtype=float,
)
q_maxcut = max_cut(adj)
print(f"   QUBO matrix shape : {q_maxcut.Q.shape}")
print(f"   Offset            : {q_maxcut.offset}")
print()


# ── 2. Brute-force solver ─────────────────────────────────────────────────────
#
# solve_brute() enumerates all 2^n binary strings and finds the minimum.
# Only practical for n ≤ ~20. SolveResult.x is the optimal assignment vector.

print("── 2. Brute-force ───────────────────────────────────────────────────────")
sol_brute = solve_brute(q_maxcut)
print(f"   Best assignment : {sol_brute.x.astype(int).tolist()}")
print(f"   Objective value : {sol_brute.energy:.4f}")
print()


# ── 3. Simulated annealing ────────────────────────────────────────────────────

print("── 3. Simulated annealing ───────────────────────────────────────────────")
sol_sa = solve_sa(q_maxcut, n_steps=500, seed=42)
print(f"   Best assignment : {sol_sa.x.astype(int).tolist()}")
print(f"   Objective value : {sol_sa.energy:.4f}")
print()


# ── 4. Knapsack problem ───────────────────────────────────────────────────────
#
# Knapsack: select items to maximise value without exceeding capacity.
# knapsack(weights, values, capacity) — weights first, values second.

print("── 4. Knapsack ──────────────────────────────────────────────────────────")
weights  = np.array([2, 3, 4, 5], dtype=float)
values   = np.array([3, 4, 5, 6], dtype=float)
capacity = 8
q_knap = knapsack(weights, values, capacity=capacity)
sol_knap = solve_brute(q_knap)
print(f"   Values/weights : {list(zip(values.astype(int).tolist(), weights.astype(int).tolist()))}")
print(f"   Capacity       : {capacity}")
print(f"   Best items     : {sol_knap.x.astype(int).tolist()}")
print(f"   Objective      : {sol_knap.energy:.4f}")
print()


# ── 5. QUBO ↔ Ising round-trip ───────────────────────────────────────────────
#
# qubo_to_ising() / ising_to_qubo() convert between QUBO and Ising forms.
# IsingResult has fields: h (linear), J (quadratic), offset.

print("── 5. QUBO ↔ Ising conversion ───────────────────────────────────────────")
ising = qubo_to_ising(q_maxcut)
print(f"   h (linear)     : {ising.h}")
print(f"   J (quadratic)  :\n{ising.J}")
print(f"   offset         : {ising.offset:.4f}")
q_back = ising_to_qubo(ising)
print(f"   Round-trip OK  : {np.allclose(q_maxcut.Q, q_back.Q)}")
print()


# ── 6. QAOA circuit export ────────────────────────────────────────────────────
#
# qaoa_circuit() returns an OpenQASM 3.0 string ready for any QASM-compatible
# backend (Qiskit, Braket, etc.).

print("── 6. QAOA circuit ──────────────────────────────────────────────────────")
qasm_str = qaoa_circuit(q_maxcut, p=1)
lines = qasm_str.strip().splitlines()
print(f"   QASM lines : {len(lines)}")
print(f"   Header     : {lines[0]}")
print("   Preview:\n")
print("\n".join(f"     {ln}" for ln in lines[:8]))
