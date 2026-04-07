# QUBO / Ising API

The `quprep.qubo` module provides QUBO and Ising problem formulation, QAOA circuit generation, and D-Wave export.

Classical reference solvers (`solve_brute`, `solve_sa`) are available in
`quprep.qubo.solver` for benchmarking quantum results against classical baselines.
They are not part of the public `quprep.qubo` namespace.

---

## Core types

::: quprep.qubo.converter.QUBOResult
    options:
      show_source: false

::: quprep.qubo.ising.IsingResult
    options:
      show_source: false

---

## Conversion

::: quprep.qubo.converter.to_qubo
    options:
      show_source: true

::: quprep.qubo.ising.qubo_to_ising
    options:
      show_source: true

::: quprep.qubo.ising.ising_to_qubo
    options:
      show_source: true

---

## Problem library

::: quprep.qubo.problems.max_cut
    options:
      show_source: true

::: quprep.qubo.problems.knapsack
    options:
      show_source: true

::: quprep.qubo.problems.tsp
    options:
      show_source: true

::: quprep.qubo.problems.portfolio
    options:
      show_source: true

::: quprep.qubo.problems.graph_color
    options:
      show_source: true

::: quprep.qubo.problems.scheduling
    options:
      show_source: true

::: quprep.qubo.problems.number_partition
    options:
      show_source: true

---

## Classical reference solvers

Import from `quprep.qubo.solver` — not part of the `quprep.qubo` public namespace.

```python
from quprep.qubo.solver import solve_brute, solve_sa, SolveResult
```

::: quprep.qubo.solver.SolveResult
    options:
      show_source: false

::: quprep.qubo.solver.solve_brute
    options:
      show_source: true

::: quprep.qubo.solver.solve_sa
    options:
      show_source: true

---

## QAOA

::: quprep.qubo.qaoa.qaoa_circuit
    options:
      show_source: true

---

## Constraints

::: quprep.qubo.constraints.equality_penalty
    options:
      show_source: true

::: quprep.qubo.constraints.inequality_penalty
    options:
      show_source: true

---

## Utilities

::: quprep.qubo.utils.add_qubo
    options:
      show_source: true

::: quprep.qubo.visualize.draw_qubo
    options:
      show_source: false

::: quprep.qubo.visualize.draw_ising
    options:
      show_source: false
