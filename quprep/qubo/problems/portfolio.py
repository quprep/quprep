"""Markowitz Portfolio Optimization QUBO formulation.

Portfolio optimization: given n assets with expected returns mu_i and
covariance matrix Sigma, select exactly K assets to maximize risk-adjusted
return.

QUBO formulation (minimization):
    minimize  -sum_i mu_i * x_i
              + risk_penalty * x^T Sigma x
              + budget_penalty * (sum_i x_i - K)^2

References
----------
Mugel et al. (2022). Dynamic portfolio optimization with real datasets using
    quantum processors and quantum-inspired tensor networks.
    Physical Review Research, 4(1), 013006.
"""

from __future__ import annotations

import numpy as np

from quprep.qubo.converter import QUBOResult


def portfolio(
    returns: np.ndarray,
    covariance: np.ndarray,
    budget: int,
    risk_penalty: float = 1.0,
    budget_penalty: float | None = None,
) -> QUBOResult:
    """
    Build the QUBO for Markowitz portfolio optimization.

    Parameters
    ----------
    returns : np.ndarray, shape (n,)
        Expected return for each asset.
    covariance : np.ndarray, shape (n, n)
        Return covariance matrix (positive semi-definite).
    budget : int
        Number of assets to select (K).
    risk_penalty : float
        Lagrange multiplier for the risk (variance) term. Higher values
        favour lower-risk portfolios. Default is 1.0.
    budget_penalty : float, optional
        Lagrange multiplier enforcing the budget constraint sum(x) = K.
        Defaults to max(|returns|) * n, which is generally strong enough.

    Returns
    -------
    QUBOResult
        Variable x_i = 1 means asset i is selected.
    """
    mu = np.asarray(returns, dtype=float)
    Sigma = np.asarray(covariance, dtype=float)
    n = len(mu)
    if Sigma.shape != (n, n):
        raise ValueError(f"covariance must be ({n}, {n}), got {Sigma.shape}.")

    if budget_penalty is None:
        budget_penalty = float(np.max(np.abs(mu)) * n) + 1.0

    Q = np.zeros((n, n))
    offset = 0.0

    # Objective term: -mu_i * x_i
    np.fill_diagonal(Q, Q.diagonal() - mu)

    # Risk term: risk_penalty * x^T Sigma x
    # Diagonal: risk_penalty * Sigma[i,i]
    np.fill_diagonal(Q, Q.diagonal() + risk_penalty * np.diag(Sigma))
    # Off-diagonal: risk_penalty * 2 * Sigma[i,j] (symmetrize)
    Sigma_sym = (Sigma + Sigma.T) / 2.0
    Q += risk_penalty * 2.0 * np.triu(Sigma_sym, k=1)

    # Budget constraint: budget_penalty * (sum x_i - K)^2
    # Diagonal: budget_penalty * (1 - 2K)
    np.fill_diagonal(Q, Q.diagonal() + budget_penalty * (1.0 - 2.0 * budget))
    # Off-diagonal: budget_penalty * 2
    ones = np.ones((n, n))
    Q += budget_penalty * 2.0 * np.triu(ones, k=1)
    offset += budget_penalty * float(budget) ** 2

    var_map = {f"x{i}": i for i in range(n)}
    return QUBOResult(Q=Q, offset=offset, variable_map=var_map)
