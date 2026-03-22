"""QUBO solvers.

solve_brute   — exact exhaustive solver for small instances (n <= 20).
solve_sa      — simulated annealing heuristic for larger instances.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SolveResult:
    """
    Result of a QUBO solve.

    Attributes
    ----------
    x : np.ndarray, shape (n,)
        Optimal binary assignment vector (values in {0, 1}).
    energy : float
        Optimal objective value including the constant offset.
    n_evaluated : int
        Number of binary strings evaluated.
    """

    x: np.ndarray
    energy: float
    n_evaluated: int

    def __repr__(self) -> str:
        bits = "".join(str(int(b)) for b in self.x)
        return f"SolveResult(x={bits}, energy={self.energy:.6f})"


def solve_brute(qubo, max_n: int = 20) -> SolveResult:
    """
    Find the exact minimum of a QUBO by exhaustive enumeration.

    Evaluates all $2^n$ binary strings and returns the one with the lowest
    objective value $x^T Q x + \text{offset}$.

    Parameters
    ----------
    qubo : QUBOResult
        The QUBO problem to solve.
    max_n : int
        Safety limit on problem size. Raises ValueError if n > max_n.
        Default is 20 (2^20 = ~1M evaluations, runs in < 1s).

    Returns
    -------
    SolveResult
        Best solution found, its energy, and the number of states evaluated.

    Raises
    ------
    ValueError
        If n > max_n.

    Examples
    --------
    >>> from quprep.qubo.problems.maxcut import max_cut
    >>> from quprep.qubo.solver import solve_brute
    >>> import numpy as np
    >>> adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
    >>> result = solve_brute(max_cut(adj))
    >>> result.energy   # max cut of triangle = -1 (one node vs two)
    -1.0
    """
    Q = qubo.Q
    n = Q.shape[0]
    if n > max_n:
        raise ValueError(
            f"n={n} exceeds max_n={max_n}. "
            "Pass max_n=<larger value> or use a heuristic solver."
        )

    best_x = np.zeros(n)
    best_energy = float("inf")

    for bits in range(1 << n):
        x = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        energy = float(x @ Q @ x) + qubo.offset
        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return SolveResult(x=best_x, energy=best_energy, n_evaluated=1 << n)


def solve_sa(
    qubo,
    n_steps: int = 10_000,
    T_start: float | None = None,
    T_end: float = 0.01,
    seed: int | None = None,
    restarts: int = 1,
) -> SolveResult:
    """
    Find a near-optimal QUBO solution via simulated annealing.

    Uses an incremental O(n)-per-step energy update and a geometric
    cooling schedule. Suitable for problems where ``solve_brute`` is
    impractical (n > 20).

    Parameters
    ----------
    qubo : QUBOResult
        The QUBO problem to solve.
    n_steps : int
        Number of single-bit-flip proposals per restart. Default 10 000.
    T_start : float or None
        Initial temperature. If None (default), auto-set to
        ``max(|Q|) * n``, which is a reasonable scale for most problems.
    T_end : float
        Final temperature. Default 0.01.
    seed : int or None
        Random seed for reproducibility.
    restarts : int
        Number of independent restarts. The best result is returned.
        Default 1.

    Returns
    -------
    SolveResult
        ``n_evaluated`` is ``restarts * n_steps`` (proposals, not full
        evaluations).

    Examples
    --------
    >>> from quprep.qubo.problems.maxcut import max_cut
    >>> from quprep.qubo.solver import solve_sa
    >>> import numpy as np
    >>> adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
    >>> sol = solve_sa(max_cut(adj), seed=0)
    >>> sol.energy
    -1.0
    """
    Q = qubo.Q
    n = Q.shape[0]

    if T_start is None:
        max_q = np.max(np.abs(Q))
        T_start = float(max_q * n) if max_q > 0 else 1.0

    # Symmetric Q for O(n) delta computation: delta = flip * (Q_sym[k,:] @ x) + Q[k,k]
    Q_sym = Q + Q.T  # Q_sym[i,j] = Q[i,j] + Q[j,i]; diagonal = 2*Q[i,i]

    rng = np.random.default_rng(seed)
    alpha = (T_end / T_start) ** (1.0 / max(n_steps - 1, 1))

    best_x = np.zeros(n)
    best_energy = float("inf")

    for _ in range(restarts):
        x = rng.integers(0, 2, size=n).astype(float)
        fields = Q_sym @ x  # fields[k] = Q_sym[k,:] @ x
        energy = float(x @ Q @ x) + qubo.offset

        T = T_start
        for _ in range(n_steps):
            k = int(rng.integers(0, n))
            flip = 1.0 - 2.0 * x[k]
            delta = flip * fields[k] + Q[k, k]

            if delta < 0.0 or rng.random() < np.exp(-delta / T):
                fields += flip * Q_sym[:, k]
                x[k] = 1.0 - x[k]
                energy += delta

            T *= alpha

        if energy < best_energy:
            best_energy = energy
            best_x = x.copy()

    return SolveResult(x=best_x, energy=best_energy, n_evaluated=restarts * n_steps)
