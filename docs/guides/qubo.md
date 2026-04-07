# QUBO / Ising

QUBO (Quadratic Unconstrained Binary Optimization) is the standard input format for quantum annealers and QAOA circuits.

The objective is to minimize:

$$\min_{x \in \{0,1\}^n} x^T Q x + c$$

where $Q$ is an upper-triangular $n \times n$ real matrix and $c$ is a constant offset.

QuPrep handles the full pipeline: problem formulation → QUBO matrix → Ising model → QAOA circuit or annealer export.

---

## Quick example

```python
import numpy as np
from quprep.qubo import max_cut, qaoa_circuit
from quprep.qubo.solver import solve_brute  # classical reference utility

adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)

print(q)           # QUBOResult(n_variables=3, offset=0.0)
print(q.Q)         # 3×3 upper-triangular Q matrix

sol = solve_brute(q)
print(sol.energy)  # -2.0

qasm = qaoa_circuit(q, p=2)
```

---

## QUBOResult

All problem functions return a `QUBOResult`. Key attributes:

| Attribute | Type | Description |
|---|---|---|
| `Q` | `np.ndarray (n, n)` | Upper-triangular QUBO matrix |
| `offset` | `float` | Constant energy offset $c$ |
| `variable_map` | `dict` | Variable name → matrix index |
| `n_original` | `int` | Original variable count (before slack) |

```python
from quprep.qubo import max_cut
import numpy as np

q = max_cut(np.array([[0,1],[1,0]], dtype=float))

# Evaluate objective: x^T Q x + offset
q.evaluate(np.array([0.0, 1.0]))  # → -1.0

# Convert to Ising
ising = q.to_ising()
print(ising.h)  # bias vector
print(ising.J)  # coupling matrix (upper-triangular)

# Serialize / deserialize
d = q.to_dict()
q2 = QUBOResult.from_dict(d)

# D-Wave export
bqm_dict = q.to_dwave()  # {(i, j): coeff}
```

---

## Problem library

Seven NP-hard combinatorial problems, each returning a `QUBOResult`:

```python
import numpy as np
from quprep.qubo import (
    max_cut, knapsack, tsp, portfolio,
    graph_color, scheduling, number_partition,
)

# Max-Cut: partition vertices to maximise cut weight
adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
q = max_cut(adj)

# 0/1 Knapsack
q = knapsack(
    weights=np.array([2.0, 3.0, 4.0]),
    values=np.array([3.0, 4.0, 5.0]),
    capacity=5.0,
)

# Travelling Salesman (n² variables)
D = np.array([[0,1,2],[1,0,1],[2,1,0]], dtype=float)
q = tsp(D)

# Markowitz Portfolio Optimisation
mu    = np.array([0.1, 0.2, 0.15, 0.05])
Sigma = np.eye(4) * 0.01
q = portfolio(mu, Sigma, budget=2)

# Graph Colouring
q = graph_color(adj, n_colors=3)

# Job Scheduling (load balancing)
q = scheduling(processing_times=np.array([3.0,1.0,4.0,2.0]), n_machines=2)

# Number Partitioning
q = number_partition(values=np.array([3.0,1.0,1.0,2.0,2.0,1.0]))
```

### Problem formulations

| Problem | Variables | Objective |
|---|---|---|
| Max-Cut | $n$ | $-\sum_{(i,j)\in E} w_{ij}(x_i + x_j - 2x_ix_j)$ |
| Knapsack | $n$ | $-\sum_i v_i x_i + \lambda(\sum_i w_i x_i - W)^2$ |
| TSP | $n^2$ | $\sum_{i,j,t} D_{ij} x_{i,t} x_{j,(t+1)\bmod n}$ + penalties |
| Portfolio | $n$ | $-\sum_i \mu_i x_i + \lambda_r x^T\Sigma x + \lambda_b(\sum_i x_i - K)^2$ |
| Scheduling | $n \times m$ | $\sum_k (\sum_i p_i x_{i,k})^2$ + assignment penalties |
| Number Partition | $n$ | $(\sum_i v_i(2x_i-1))^2$ |

---

## Classical reference solvers

These utilities compute the optimal classical solution — useful for benchmarking
QAOA results against the known optimum. They are not part of the QuPrep
preprocessing workflow; import them directly from the submodule.

```python
from quprep.qubo.solver import solve_brute, solve_sa
import numpy as np
from quprep.qubo import max_cut

q = max_cut(np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float))

# Exact — exhaustive 2ⁿ search (n ≤ 20)
sol = solve_brute(q)
print(sol.x)       # optimal binary vector
print(sol.energy)  # optimal energy

# Simulated annealing — scales to n ~ 500+
sol = solve_sa(q, n_steps=10_000, restarts=3, seed=42)
print(sol.energy)
```

| Solver | Method | Limit |
|---|---|---|
| `solve_brute` | Exhaustive enumeration | $n \leq 20$ |
| `solve_sa` | Simulated annealing | $n \sim 500+$ |

---

## Ising conversion

The QUBO and Ising models are related via $x_i = (s_i + 1)/2$, $s_i \in \{-1, +1\}$:

$$J_{ij} = \frac{Q_{ij}}{4} \quad (i < j) \qquad h_i = \frac{Q_{ii}}{2} + \sum_{j \neq i} \frac{Q^{\text{sym}}_{ij}}{4}$$

```python
from quprep.qubo import qubo_to_ising, ising_to_qubo, max_cut
import numpy as np

q = max_cut(np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float))

ising = qubo_to_ising(q)   # or: q.to_ising()
print(ising.h)             # bias vector $h$
print(ising.J)             # coupling matrix $J$

q2 = ising_to_qubo(ising)  # round-trip
```

---

## Constraints

Constraints are added as quadratic penalty terms $\lambda (Ax - b)^2$:

```python
from quprep.qubo import to_qubo
import numpy as np

# Equality:   x₀ + x₁ + x₂ = 1
# Inequality: x₀ + x₁ ≤ 1  (adds slack variables)
cost = np.diag([-1.0, -2.0, -3.0])
q = to_qubo(cost, constraints=[
    {"type": "eq",   "A": np.ones((1, 3)), "b": np.array([1.0]), "penalty": 10.0},
    {"type": "ineq", "A": np.array([[1.0, 1.0, 0.0]]), "b": np.array([1.0])},
])
```

---

## QAOA circuit generation

```python
from quprep.qubo import qaoa_circuit, max_cut
import numpy as np

q = max_cut(np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float))

# p-layer QAOA circuit (OpenQASM 3.0)
qasm = qaoa_circuit(q, p=2)

# Custom initial parameters
qasm = qaoa_circuit(q, p=2, gamma=[0.5, 0.3], beta=[0.2, 0.1])

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

q = max_cut(np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float))

ax = draw_qubo(q, title="Max-Cut QUBO")       # heatmap of Q
ax = draw_ising(q.to_ising(), title="Ising")  # circular graph
```

---

## CLI

```bash
# Problem formulation
quprep qubo maxcut   --adjacency "0,1,1;1,0,1;1,1,0" --solve
quprep qubo knapsack --weights "2,3,4" --values "3,4,5" --capacity 5 --solve
quprep qubo tsp      --distances "0,1,2;1,0,1;2,1,0"
quprep qubo schedule --times "3,1,4,2" --machines 2
quprep qubo partition --values "3,1,1,2,2,1" --solve

# QAOA circuit
quprep qubo qaoa maxcut --adjacency "0,1,1;1,0,1;1,1,0" --p 2 --output circuit.qasm

# Export Q matrix
quprep qubo export maxcut --adjacency "0,1,1;1,0,1;1,1,0" --format json --output q.json
```

!!! tip
    `--solve` automatically uses exact brute-force for $n \leq 20$ and simulated annealing for larger problems.
