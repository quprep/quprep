# QUBO / Ising

QUBO (Quadratic Unconstrained Binary Optimization) is the standard input format for quantum annealers (D-Wave) and QAOA circuits. QuPrep handles the full pipeline: problem formulation → QUBO matrix → Ising model → QAOA circuit or annealer export.

---

## Quick example

```python
import numpy as np
from quprep.qubo import max_cut, solve_brute, qaoa_circuit

# Max-Cut on a triangle graph
adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)

print(q)             # QUBOResult(n_variables=3, offset=0.0)
print(q.Q)           # 3×3 upper-triangular Q matrix

# Brute-force solve (exact, n ≤ 20)
sol = solve_brute(q)
print(sol.energy)    # -2.0

# Generate a QAOA circuit
qasm = qaoa_circuit(q, p=2)
print(qasm[:60])     # OPENQASM 3.0; ...
```

---

## QUBOResult

All problem functions return a `QUBOResult`:

```python
from quprep.qubo import max_cut
import numpy as np

q = max_cut(np.array([[0,1],[1,0]], dtype=float))

q.Q              # np.ndarray (n, n) — upper-triangular Q matrix
q.offset         # float — constant energy offset
q.variable_map   # dict — name → index mapping
q.n_original     # int — original variable count (before slack)

# Evaluate a solution
q.evaluate(np.array([0.0, 1.0]))   # x^T Q x + offset

# Convert to Ising
ising = q.to_ising()
ising.h   # bias vector
ising.J   # coupling matrix (upper-triangular)

# Serialize
d = q.to_dict()          # JSON-compatible dict
q2 = QUBOResult.from_dict(d)

# D-Wave Ocean SDK export
bqm_dict = q.to_dwave()  # {(i, j): coeff}  i <= j
```

---

## Problem library

Seven NP-hard combinatorial problems, ready to formulate as QUBO:

```python
import numpy as np
from quprep.qubo import (
    max_cut, knapsack, tsp, portfolio,
    graph_color, scheduling, number_partition,
)

# Max-Cut
adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)

# 0/1 Knapsack
q = knapsack(
    weights=np.array([2.0, 3.0, 4.0]),
    values=np.array([3.0, 4.0, 5.0]),
    capacity=5.0,
)

# Travelling Salesman Problem (n² variables)
D = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=float)
q = tsp(D)

# Markowitz Portfolio Optimization
mu = np.array([0.1, 0.2, 0.15, 0.05])
Sigma = np.eye(4) * 0.01
q = portfolio(mu, Sigma, budget=2)

# Graph Colouring
q = graph_color(adj, n_colors=3)

# Job Scheduling (load balancing)
q = scheduling(processing_times=np.array([3.0,1.0,4.0,2.0]), n_machines=2)

# Number Partitioning
q = number_partition(values=np.array([3.0,1.0,1.0,2.0,2.0,1.0]))
```

---

## Solvers

```python
from quprep.qubo import solve_brute, solve_sa, max_cut
import numpy as np

adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)

# Exact — exhaustive 2^n search (n ≤ 20)
sol = solve_brute(q)
print(sol.x)        # optimal binary vector
print(sol.energy)   # optimal energy

# Simulated annealing — scales to n ~ 500+
sol = solve_sa(q, n_steps=10_000, restarts=3, seed=42)
print(sol.energy)
```

---

## Constraints

```python
from quprep.qubo import to_qubo
from quprep.qubo.constraints import equality_penalty, inequality_penalty
import numpy as np

# Equality constraint: x0 + x1 + x2 = 1
Q_pen, offset = equality_penalty(
    A=np.array([[1.0, 1.0, 1.0]]),
    b=np.array([1.0]),
    penalty=10.0,
)

# Inequality constraint: x0 + x1 ≤ 1 (adds slack variables)
Q_pen, offset, n_slack = inequality_penalty(
    A=np.array([[1.0, 1.0]]),
    b=np.array([1.0]),
    penalty=10.0,
)

# Or pass constraints directly to to_qubo()
cost = np.diag([-1.0, -2.0, -3.0])
q = to_qubo(cost, constraints=[
    {"type": "eq",   "A": np.ones((1, 3)), "b": np.array([1.0]), "penalty": 10.0},
    {"type": "ineq", "A": np.array([[1.0, 1.0, 0.0]]), "b": np.array([1.0])},
])
```

---

## Ising conversion

```python
from quprep.qubo import qubo_to_ising, ising_to_qubo, max_cut
import numpy as np

q = max_cut(np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float))

# QUBO → Ising
ising = qubo_to_ising(q)   # or: q.to_ising()
print(ising.h)             # bias vector h_i
print(ising.J)             # coupling matrix J_ij

# Ising → QUBO (round-trip)
q2 = ising_to_qubo(ising)  # or: ising.to_qubo()
```

---

## QAOA circuit generation

```python
from quprep.qubo import qaoa_circuit, max_cut
import numpy as np

adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)

# Generate p-layer QAOA circuit (OpenQASM 3.0)
qasm = qaoa_circuit(q, p=2)

# Custom parameters
qasm = qaoa_circuit(q, p=2, gamma=[0.5, 0.3], beta=[0.2, 0.1])

# Write to file
from pathlib import Path
Path("qaoa_maxcut.qasm").write_text(qasm)
```

---

## Combining QUBOs

```python
from quprep.qubo import add_qubo, max_cut
from quprep.qubo.constraints import equality_penalty
from quprep.qubo.converter import QUBOResult
import numpy as np

q_obj = max_cut(np.array([[0,1],[1,0]], dtype=float))
Q_pen, off = equality_penalty(np.array([[1.0, 1.0]]), np.array([1.0]), 5.0)
q_pen = QUBOResult(Q=Q_pen, offset=off)

q_combined = add_qubo(q_obj, q_pen, weight=2.0)
```

---

## Visualization

Requires `pip install quprep[viz]`.

```python
from quprep.qubo import draw_qubo, draw_ising, max_cut
import numpy as np

adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)

# Heatmap of Q matrix
ax = draw_qubo(q, title="Max-Cut QUBO")

# Circular Ising graph
ax = draw_ising(q.to_ising(), title="Max-Cut Ising")
```

---

## CLI

```bash
# Problem formulation
quprep qubo maxcut --adjacency "0,1,1;1,0,1;1,1,0" --solve
quprep qubo knapsack --weights "2,3,4" --values "3,4,5" --capacity 5 --solve
quprep qubo tsp --distances "0,1,2;1,0,1;2,1,0"
quprep qubo schedule --times "3,1,4,2" --machines 2
quprep qubo partition --values "3,1,1,2,2,1" --solve
quprep qubo portfolio --returns "0.5,0.3,0.2" --covariance "0.1,0.02,0.01;0.02,0.05,0.01;0.01,0.01,0.08" --budget 2
quprep qubo graphcolor --adjacency "0,1,1;1,0,1;1,1,0" --colors 3

# QAOA circuit
quprep qubo qaoa maxcut --adjacency "0,1,1;1,0,1;1,1,0" --p 2 --output circuit.qasm

# Export Q matrix
quprep qubo export maxcut --adjacency "0,1,1;1,0,1;1,1,0" --format json --output q.json
quprep qubo export knapsack --weights "2,3,4" --values "3,4,5" --capacity 5 --format npy
```

`--solve` automatically uses exact brute-force for n ≤ 20 and simulated annealing for larger problems.
