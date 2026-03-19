# QUBO / Ising

!!! note "Coming in v0.3.0"
    QUBO and Ising conversion is a Phase 3 feature. This page documents the planned API.

QUBO (Quadratic Unconstrained Binary Optimization) is the input format for quantum annealers (D-Wave) and QAOA circuits. QuPrep will handle the conversion from classical cost matrices to QUBO/Ising form.

---

## Planned API

```python
# v0.3.0
import quprep

# Convert a cost matrix to QUBO
qubo = quprep.to_qubo(cost_matrix, constraints, penalty=10.0)
print(qubo.Q)          # upper-triangular Q matrix

# Convert to Ising form
ising = qubo.to_ising()
print(ising.h)         # bias vector
print(ising.J)         # coupling matrix

# Built-in problem formulations
tsp = quprep.problems.TSP(distance_matrix)
qubo = tsp.to_qubo()
```

---

## Planned problem library

| Problem | Description |
|---|---|
| Max-Cut | Graph partitioning |
| TSP | Travelling salesman |
| Knapsack | Resource allocation |
| Portfolio | Financial optimization |
| Graph colouring | Constraint satisfaction |
| Scheduling | Job scheduling |

---

## Why separate from encoding?

QUBO is a parallel path to the QML pipeline. Instead of encoding continuous data for a variational circuit, QUBO formulates a combinatorial problem for an annealer or QAOA circuit. Both paths end at a quantum circuit — they start from different problem types.
