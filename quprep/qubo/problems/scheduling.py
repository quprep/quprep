r"""Job Scheduling QUBO formulation.

Scheduling: assign n jobs to m machines to minimize total squared load
(a proxy for balanced makespan minimization).

QUBO uses $n \cdot m$ binary variables $x_{i,k}$ where $x_{i,k}=1$ means
job $i$ is assigned to machine $k$.

Objective (minimize load imbalance):

$$\min \sum_k \left(\sum_i p_i\, x_{i,k}\right)^2$$

where $p_i$ is the processing time of job $i$.

Constraint: each job assigned to exactly one machine:
$\sum_k x_{i,k} = 1$ for all $i$.

Variable index: $v(i, k) = i \cdot m + k$.

Notes
-----
This is a load-balancing formulation. It does not model job dependencies,
deadlines, or preemption. For those, see the Travelling Salesman or QUBO
references for richer scheduling models.

References
----------
Lucas, A. (2014). Ising formulations of many NP problems.
    *Frontiers in Physics*, 2, 5. [doi:10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005){target="_blank"}
"""

from __future__ import annotations

import numpy as np

from quprep.qubo.converter import QUBOResult


def scheduling(
    processing_times: np.ndarray,
    n_machines: int,
    penalty: float | None = None,
) -> QUBOResult:
    """
    Build the QUBO for the job scheduling (load balancing) problem.

    Parameters
    ----------
    processing_times : np.ndarray, shape (n,)
        Processing time of each job (non-negative).
    n_machines : int
        Number of machines m available.
    penalty : float, optional
        Lagrange multiplier for the assignment constraint (each job assigned
        to exactly one machine). Defaults to sum(processing_times)^2 + 1,
        which is always large enough to enforce feasibility.

    Returns
    -------
    QUBOResult
        n * n_machines binary variables. variable_map["x_{i}_{k}"] gives the
        index of x_{i,k}. A feasible solution has exactly n variables set to 1,
        one per job.
    """
    p = np.asarray(processing_times, dtype=float)
    n = len(p)
    m = n_machines

    if penalty is None:
        penalty = float(np.sum(p) ** 2) + 1.0

    N = n * m
    Q = np.zeros((N, N))
    offset = 0.0

    def idx(job: int, machine: int) -> int:
        return job * m + machine

    # Objective: minimize sum_k (sum_i p_i x[i,k])^2
    # Expanding: sum_k sum_{i,j} p_i p_j x[i,k] x[j,k]
    # For i == j (same job, same machine): p_i^2 x[i,k] (diagonal)
    # For i != j (diff jobs, same machine): 2*p_i*p_j x[i,k]*x[j,k] (off-diagonal)
    for k in range(m):
        for i in range(n):
            Q[idx(i, k), idx(i, k)] += p[i] ** 2
        for i in range(n):
            for j in range(i + 1, n):
                a = idx(i, k)
                b = idx(j, k)
                Q[a, b] += 2.0 * p[i] * p[j]

    # Constraint: each job assigned to exactly one machine
    # penalty * (sum_k x[i,k] - 1)^2 for each job i
    for i in range(n):
        for k1 in range(m):
            a = idx(i, k1)
            Q[a, a] += penalty * (1.0 - 2.0)   # -penalty per variable
            for k2 in range(k1 + 1, m):
                b = idx(i, k2)
                Q[a, b] += penalty * 2.0
    offset += penalty * n   # n constraints, each contributes penalty * 1^2

    var_map = {f"x_{i}_{k}": idx(i, k) for i in range(n) for k in range(m)}
    return QUBOResult(Q=Q, offset=offset, variable_map=var_map)
