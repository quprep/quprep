"""Tests for quprep.qubo — QUBO/Ising conversion and problem library."""

from __future__ import annotations

import numpy as np
import pytest

from quprep.qubo.constraints import equality_penalty
from quprep.qubo.converter import QUBOResult, to_qubo
from quprep.qubo.ising import qubo_to_ising
from quprep.qubo.problems.knapsack import knapsack
from quprep.qubo.problems.maxcut import max_cut
from quprep.qubo.problems.portfolio import portfolio
from quprep.qubo.problems.tsp import tsp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _eval_qubo(Q: np.ndarray, x: np.ndarray) -> float:
    """Evaluate x^T Q x for binary vector x."""
    return float(x @ Q @ x)


def _eval_ising(h: np.ndarray, J: np.ndarray, s: np.ndarray) -> float:
    """Evaluate sum_{i<j} J_ij s_i s_j + sum_i h_i s_i."""
    return float(np.sum(np.triu(J, k=1) * np.outer(s, s)) + h @ s)


# ---------------------------------------------------------------------------
# to_qubo / QUBOResult
# ---------------------------------------------------------------------------

class TestToQubo:
    def test_symmetric_diagonal_only(self):
        M = np.diag([1.0, 2.0, 3.0])
        r = to_qubo(M)
        assert isinstance(r, QUBOResult)
        assert r.Q.shape == (3, 3)
        # Diagonal should match
        np.testing.assert_array_almost_equal(np.diag(r.Q), [1.0, 2.0, 3.0])
        # Off-diagonal should be zero
        assert np.sum(np.abs(np.triu(r.Q, k=1))) == 0.0

    def test_symmetric_matrix(self):
        M = np.array([[1.0, 2.0], [2.0, 3.0]])
        r = to_qubo(M)
        # Q[0,1] = M[0,1] + M[1,0] = 4
        assert r.Q[0, 1] == pytest.approx(4.0)
        assert r.Q[1, 0] == 0.0  # strictly upper-triangular

    def test_asymmetric_matrix_symmetrized(self):
        M = np.array([[0.0, 3.0], [1.0, 0.0]])
        r = to_qubo(M)
        # Q[0,1] = M[0,1] + M[1,0] = 4
        assert r.Q[0, 1] == pytest.approx(4.0)

    def test_upper_triangular_output(self):
        rng = np.random.default_rng(42)
        M = rng.standard_normal((5, 5))
        r = to_qubo(M)
        # Lower triangle must be zero
        lower = np.tril(r.Q, k=-1)
        assert np.allclose(lower, 0.0)

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            to_qubo(np.ones((2, 3)))

    def test_with_equality_constraint(self):
        # sum(x) = 1 on a 3-variable problem
        M = np.zeros((3, 3))
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([1.0])
        r = to_qubo(M, constraints=[{"A": A, "b": b, "penalty": 5.0}])
        # x = [1,0,0] should satisfy constraint; penalty ~ 0
        x_good = np.array([1.0, 0.0, 0.0])
        x_bad = np.array([1.0, 1.0, 0.0])
        assert _eval_qubo(r.Q, x_good) < _eval_qubo(r.Q, x_bad)

    def test_repr(self):
        r = to_qubo(np.eye(3))
        assert "n_variables=3" in repr(r)


# ---------------------------------------------------------------------------
# equality_penalty
# ---------------------------------------------------------------------------

class TestEqualityPenalty:
    def test_single_constraint(self):
        A = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0])
        Q, offset = equality_penalty(A, b, penalty=1.0)
        assert Q.shape == (3, 3)
        # offset = penalty * b^2 = 1
        assert offset == pytest.approx(1.0)

    def test_penalty_enforced(self):
        # x0 + x1 = 1 with penalty 10
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        Q, offset = equality_penalty(A, b, penalty=10.0)
        # x=[1,0]: (1+0-1)^2=0 → total penalty 0
        # x=[1,1]: (1+1-1)^2=1 → total penalty 10
        x_good = np.array([1.0, 0.0])
        x_bad = np.array([1.0, 1.0])
        assert _eval_qubo(Q, x_good) + offset == pytest.approx(0.0)
        assert _eval_qubo(Q, x_bad) + offset == pytest.approx(10.0)

    def test_multiple_constraints(self):
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 0.0])
        Q, offset = equality_penalty(A, b, penalty=2.0)
        assert Q.shape == (2, 2)
        # Two constraints contribute 2 * penalty * b^2 = 2*(1+0) = 2
        assert offset == pytest.approx(2.0)

    def test_mismatch_raises(self):
        A = np.ones((3, 2))
        b = np.ones(2)
        with pytest.raises(ValueError):
            equality_penalty(A, b, penalty=1.0)


# ---------------------------------------------------------------------------
# QUBO <-> Ising round-trip
# ---------------------------------------------------------------------------

class TestIsingRoundTrip:
    def _make_qubo(self, n=3, seed=0):
        rng = np.random.default_rng(seed)
        M = rng.standard_normal((n, n))
        return to_qubo(M)

    def test_qubo_to_ising_shapes(self):
        q = self._make_qubo(4)
        ising = qubo_to_ising(q)
        assert ising.h.shape == (4,)
        assert ising.J.shape == (4, 4)

    def test_ising_J_upper_triangular(self):
        q = self._make_qubo(4)
        ising = qubo_to_ising(q)
        lower = np.tril(ising.J, k=-1)
        assert np.allclose(lower, 0.0)

    def test_qubo_ising_qubo_round_trip(self):
        """QUBO -> Ising -> QUBO should recover original Q and offset."""
        q = self._make_qubo(3)
        ising = qubo_to_ising(q)
        q2 = ising.to_qubo()
        np.testing.assert_array_almost_equal(q.Q, q2.Q)
        assert q.offset == pytest.approx(q2.offset)

    def test_energy_consistency(self):
        """For every binary x, QUBO energy == Ising energy (with offset)."""
        q = self._make_qubo(3)
        ising = qubo_to_ising(q)
        for bits in range(8):
            x = np.array([(bits >> i) & 1 for i in range(3)], dtype=float)
            s = 2.0 * x - 1.0
            e_qubo = _eval_qubo(q.Q, x) + q.offset
            e_ising = _eval_ising(ising.h, ising.J, s) + ising.offset
            assert e_qubo == pytest.approx(e_ising, abs=1e-10)

    def test_repr(self):
        q = self._make_qubo(2)
        ising = qubo_to_ising(q)
        assert "IsingResult" in repr(ising)


# ---------------------------------------------------------------------------
# MaxCut
# ---------------------------------------------------------------------------

class TestMaxCut:
    def test_triangle_graph(self):
        # Triangle: all edges weight 1
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        r = max_cut(adj)
        assert r.Q.shape == (3, 3)
        assert isinstance(r, QUBOResult)

    def test_variable_map(self):
        r = max_cut(np.zeros((3, 3)))
        assert "x0" in r.variable_map

    def test_optimal_cut(self):
        # Simple 2-node graph: edge weight 1
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        r = max_cut(adj)
        # Optimal: one node in each partition -> x=[1,0] or x=[0,1]
        x_cut = np.array([1.0, 0.0])
        x_no_cut = np.array([0.0, 0.0])
        # Cut should have lower QUBO energy (we minimize negative cut)
        assert _eval_qubo(r.Q, x_cut) < _eval_qubo(r.Q, x_no_cut)

    def test_weighted_graph(self):
        adj = np.array([[0, 5, 0], [5, 0, 3], [0, 3, 0]], dtype=float)
        r = max_cut(adj)
        assert r.Q.shape == (3, 3)
        # Lower triangle is zero
        assert np.allclose(np.tril(r.Q, k=-1), 0.0)


# ---------------------------------------------------------------------------
# Knapsack
# ---------------------------------------------------------------------------

class TestKnapsack:
    def test_basic(self):
        w = np.array([2.0, 3.0, 4.0])
        v = np.array([3.0, 4.0, 5.0])
        r = knapsack(w, v, capacity=5.0)
        assert r.Q.shape == (3, 3)
        assert isinstance(r, QUBOResult)

    def test_variable_map(self):
        r = knapsack(np.ones(4), np.ones(4), capacity=2.0)
        assert "x0" in r.variable_map
        assert "x3" in r.variable_map

    def test_feasible_lower_than_infeasible(self):
        # 3 items: weights [3,3,3], values [5,5,5], capacity 3
        w = np.array([3.0, 3.0, 3.0])
        v = np.array([5.0, 5.0, 5.0])
        r = knapsack(w, v, capacity=3.0, penalty=20.0)
        x_feasible = np.array([1.0, 0.0, 0.0])
        x_infeasible = np.array([1.0, 1.0, 0.0])
        assert _eval_qubo(r.Q, x_feasible) < _eval_qubo(r.Q, x_infeasible)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            knapsack(np.ones(3), np.ones(4), capacity=5.0)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class TestPortfolio:
    def test_basic(self):
        mu = np.array([0.1, 0.2, 0.15])
        Sigma = np.eye(3) * 0.01
        r = portfolio(mu, Sigma, budget=2)
        assert r.Q.shape == (3, 3)
        assert isinstance(r, QUBOResult)

    def test_variable_map(self):
        mu = np.ones(4) * 0.1
        Sigma = np.eye(4) * 0.01
        r = portfolio(mu, Sigma, budget=2)
        assert "x0" in r.variable_map
        assert "x3" in r.variable_map

    def test_cov_shape_mismatch_raises(self):
        mu = np.ones(3)
        Sigma = np.eye(4)
        with pytest.raises(ValueError):
            portfolio(mu, Sigma, budget=1)

    def test_budget_constraint_enforced(self):
        mu = np.array([0.1, 0.1, 0.1])
        Sigma = np.eye(3) * 0.0
        # With no risk, only objective and budget matter
        r = portfolio(mu, Sigma, budget=1, risk_penalty=0.0, budget_penalty=100.0)
        x_good = np.array([1.0, 0.0, 0.0])  # exactly 1 asset
        x_bad = np.array([1.0, 1.0, 0.0])   # 2 assets — violates budget=1
        assert _eval_qubo(r.Q, x_good) < _eval_qubo(r.Q, x_bad)


# ---------------------------------------------------------------------------
# TSP
# ---------------------------------------------------------------------------

class TestTSP:
    def test_basic_3city(self):
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        r = tsp(D)
        assert r.Q.shape == (9, 9)  # 3^2 variables
        assert isinstance(r, QUBOResult)

    def test_variable_map(self):
        D = np.ones((3, 3)) - np.eye(3)
        r = tsp(D)
        assert "x_0_0" in r.variable_map
        assert "x_2_2" in r.variable_map
        assert len(r.variable_map) == 9

    def test_non_square_raises(self):
        with pytest.raises(ValueError):
            tsp(np.ones((2, 3)))

    def test_upper_triangular(self):
        D = np.ones((3, 3)) - np.eye(3)
        r = tsp(D)
        assert np.allclose(np.tril(r.Q, k=-1), 0.0)

    def test_feasible_lower_than_infeasible(self):
        # 3-city tour: symmetric distances
        D = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
        r = tsp(D, penalty=50.0)

        # Build a valid tour: city 0->1->2->0
        # x[i,t]=1 means city i at time t
        # x_0_0=1, x_1_1=1, x_2_2=1 -> valid tour
        x_feasible = np.zeros(9)
        x_feasible[0 * 3 + 0] = 1  # city 0 at t=0
        x_feasible[1 * 3 + 1] = 1  # city 1 at t=1
        x_feasible[2 * 3 + 2] = 1  # city 2 at t=2

        # Invalid: city 0 at both t=0 and t=1
        x_infeasible = np.zeros(9)
        x_infeasible[0 * 3 + 0] = 1
        x_infeasible[0 * 3 + 1] = 1
        x_infeasible[1 * 3 + 2] = 1

        assert _eval_qubo(r.Q, x_feasible) < _eval_qubo(r.Q, x_infeasible)


# ---------------------------------------------------------------------------
# Module-level import test
# ---------------------------------------------------------------------------

def test_qubo_module_imports():
    import quprep.qubo as qubo
    assert hasattr(qubo, "to_qubo")
    assert hasattr(qubo, "QUBOResult")
    assert hasattr(qubo, "qubo_to_ising")
    assert hasattr(qubo, "IsingResult")
    assert hasattr(qubo, "equality_penalty")
    assert hasattr(qubo, "inequality_penalty")
    assert hasattr(qubo, "max_cut")
    assert hasattr(qubo, "tsp")
    assert hasattr(qubo, "knapsack")
    assert hasattr(qubo, "portfolio")
    assert hasattr(qubo, "graph_color")
    assert hasattr(qubo, "scheduling")
    assert hasattr(qubo, "qaoa_circuit")
    # solve_brute/solve_sa are demoted to quprep.qubo.solver — not in __all__
    assert "solve_brute" not in qubo.__all__
    assert "solve_sa" not in qubo.__all__


# ---------------------------------------------------------------------------
# inequality_penalty
# ---------------------------------------------------------------------------

class TestInequalityPenalty:
    def test_basic_shape(self):
        from quprep.qubo.constraints import inequality_penalty
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        Q, offset, n_slack = inequality_penalty(A, b, penalty=5.0)
        assert n_slack >= 1
        assert Q.shape == (2 + n_slack, 2 + n_slack)

    def test_upper_triangular(self):
        from quprep.qubo.constraints import inequality_penalty
        A = np.array([[1.0, 2.0, 3.0]])
        b = np.array([4.0])
        Q, _, _ = inequality_penalty(A, b, penalty=5.0)
        assert np.allclose(np.tril(Q, k=-1), 0.0)

    def test_feasible_solution_lower_energy(self):
        from quprep.qubo.constraints import inequality_penalty
        # x0 + x1 <= 1: feasible x=[1,0], infeasible x=[1,1]
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        Q, offset, n_slack = inequality_penalty(A, b, penalty=20.0)
        # Extend binary vectors to include slack variables
        n = 2
        N = n + n_slack
        x_good = np.zeros(N)
        x_good[0] = 1.0
        # Set slack so equality holds: 1 + slack = 1 -> slack = 0
        x_bad = np.zeros(N)
        x_bad[0] = 1.0
        x_bad[1] = 1.0
        # x_good satisfies constraint, x_bad violates it
        assert _eval_qubo(Q, x_good) + offset <= _eval_qubo(Q, x_bad) + offset

    def test_infeasible_raises(self):
        from quprep.qubo.constraints import inequality_penalty
        # All-positive a with b < 0 is always infeasible
        A = np.array([[1.0, 1.0]])
        b = np.array([-5.0])
        with pytest.raises(ValueError):
            inequality_penalty(A, b, penalty=5.0)

    def test_to_qubo_ineq_constraint(self):
        # to_qubo with ineq constraint expands Q
        M = np.zeros((3, 3))
        A = np.array([[1.0, 1.0, 1.0]])
        b = np.array([2.0])
        r = to_qubo(M, constraints=[{"type": "ineq", "A": A, "b": b, "penalty": 5.0}])
        assert r.Q.shape[0] > 3
        assert r.n_original == 3


# ---------------------------------------------------------------------------
# solve_brute
# ---------------------------------------------------------------------------

class TestSolveBrute:
    def test_simple_minimum(self):
        from quprep.qubo.solver import solve_brute
        # Q = -I -> minimum at x = [1,1,1], energy = -3
        Q = -np.eye(3)
        from quprep.qubo.converter import QUBOResult
        qubo = QUBOResult(Q=Q, offset=0.0)
        sol = solve_brute(qubo)
        assert np.allclose(sol.x, [1, 1, 1])
        assert sol.energy == pytest.approx(-3.0)

    def test_maxcut_triangle(self):
        from quprep.qubo.problems.maxcut import max_cut
        from quprep.qubo.solver import solve_brute
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        r = max_cut(adj)
        sol = solve_brute(r)
        # Max cut of triangle is 2 edges -> QUBO energy = -2
        assert sol.energy == pytest.approx(-2.0)
        assert sol.n_evaluated == 8

    def test_too_large_raises(self):
        from quprep.qubo.converter import QUBOResult
        from quprep.qubo.solver import solve_brute
        qubo = QUBOResult(Q=np.eye(25), offset=0.0)
        with pytest.raises(ValueError):
            solve_brute(qubo, max_n=20)

    def test_repr(self):
        from quprep.qubo.solver import SolveResult
        sol = SolveResult(x=np.array([1.0, 0.0]), energy=-1.0, n_evaluated=4)
        assert "10" in repr(sol)


# ---------------------------------------------------------------------------
# QAOA
# ---------------------------------------------------------------------------

class TestQAOACircuit:
    def test_basic_output(self):
        from quprep.qubo.problems.maxcut import max_cut
        from quprep.qubo.qaoa import qaoa_circuit
        adj = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        qasm = qaoa_circuit(max_cut(adj), p=1, gamma=[0.5], beta=[0.3])
        assert "OPENQASM 3.0" in qasm
        assert "qubit[3]" in qasm
        assert "h q[0]" in qasm
        assert "measure" in qasm

    def test_p_layers(self):
        from quprep.qubo.problems.maxcut import max_cut
        from quprep.qubo.qaoa import qaoa_circuit
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        qasm1 = qaoa_circuit(max_cut(adj), p=1)
        qasm2 = qaoa_circuit(max_cut(adj), p=2)
        # More layers -> more lines
        assert len(qasm2.splitlines()) > len(qasm1.splitlines())

    def test_default_angles(self):
        from quprep.qubo.converter import QUBOResult
        from quprep.qubo.qaoa import qaoa_circuit
        qubo = QUBOResult(Q=np.eye(2), offset=0.0)
        qasm = qaoa_circuit(qubo, p=1)
        assert isinstance(qasm, str)

    def test_wrong_gamma_length_raises(self):
        from quprep.qubo.converter import QUBOResult
        from quprep.qubo.qaoa import qaoa_circuit
        qubo = QUBOResult(Q=np.eye(2), offset=0.0)
        with pytest.raises(ValueError):
            qaoa_circuit(qubo, p=2, gamma=[0.5], beta=[0.3, 0.3])


# ---------------------------------------------------------------------------
# GraphColor
# ---------------------------------------------------------------------------

class TestGraphColor:
    def test_basic(self):
        from quprep.qubo.problems.graph_color import graph_color
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        r = graph_color(adj, n_colors=2)
        assert r.Q.shape == (6, 6)  # 3 nodes * 2 colors

    def test_variable_map(self):
        from quprep.qubo.problems.graph_color import graph_color
        adj = np.zeros((3, 3))
        r = graph_color(adj, n_colors=3)
        assert "x_0_0" in r.variable_map
        assert "x_2_2" in r.variable_map
        assert len(r.variable_map) == 9

    def test_valid_coloring_lower_than_invalid(self):
        from quprep.qubo.problems.graph_color import graph_color
        # Path graph: 0-1-2, 2 colors needed
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        r = graph_color(adj, n_colors=2, penalty=10.0)
        # Valid 2-coloring: node0=color0, node1=color1, node2=color0
        x_valid = np.zeros(6)
        x_valid[0 * 2 + 0] = 1  # node0 -> color0
        x_valid[1 * 2 + 1] = 1  # node1 -> color1
        x_valid[2 * 2 + 0] = 1  # node2 -> color0
        # Invalid: node0 and node1 both color0
        x_invalid = np.zeros(6)
        x_invalid[0 * 2 + 0] = 1
        x_invalid[1 * 2 + 0] = 1
        x_invalid[2 * 2 + 1] = 1
        assert _eval_qubo(r.Q, x_valid) < _eval_qubo(r.Q, x_invalid)


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------

class TestScheduling:
    def test_basic(self):
        from quprep.qubo.problems.scheduling import scheduling
        p = np.array([3.0, 1.0, 4.0, 2.0])
        r = scheduling(p, n_machines=2)
        assert r.Q.shape == (8, 8)  # 4 jobs * 2 machines

    def test_variable_map(self):
        from quprep.qubo.problems.scheduling import scheduling
        p = np.array([1.0, 2.0, 3.0])
        r = scheduling(p, n_machines=2)
        assert "x_0_0" in r.variable_map
        assert "x_2_1" in r.variable_map
        assert len(r.variable_map) == 6

    def test_balanced_assignment_preferred(self):
        from quprep.qubo.problems.scheduling import scheduling
        # 2 equal jobs on 2 machines: balanced is optimal
        p = np.array([1.0, 1.0])
        r = scheduling(p, n_machines=2, penalty=100.0)
        # Balanced: job0->m0, job1->m1
        x_balanced = np.array([1.0, 0.0, 0.0, 1.0])
        # Imbalanced: both jobs on m0
        x_imbalanced = np.array([1.0, 0.0, 1.0, 0.0])
        assert _eval_qubo(r.Q, x_balanced) <= _eval_qubo(r.Q, x_imbalanced)


# ---------------------------------------------------------------------------
# NumberPartition
# ---------------------------------------------------------------------------

class TestNumberPartition:
    def test_basic(self):
        from quprep.qubo.problems.number_partition import number_partition
        v = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 1.0])
        r = number_partition(v)
        assert r.Q.shape == (6, 6)
        assert isinstance(r, QUBOResult)

    def test_perfect_partition(self):
        from quprep.qubo.problems.number_partition import number_partition
        from quprep.qubo.solver import solve_brute
        # [3,1,1,2,2,1] sums to 10 -> perfect split at 5
        v = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 1.0])
        sol = solve_brute(number_partition(v))
        assert abs(sol.energy) < 1e-9

    def test_upper_triangular(self):
        from quprep.qubo.problems.number_partition import number_partition
        r = number_partition(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(np.tril(r.Q, k=-1), 0.0)

    def test_variable_map(self):
        from quprep.qubo.problems.number_partition import number_partition
        r = number_partition(np.array([1.0, 2.0, 3.0]))
        assert "x0" in r.variable_map
        assert "x2" in r.variable_map


# ---------------------------------------------------------------------------
# add_qubo + serialization
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_add_qubo(self):
        from quprep.qubo.utils import add_qubo
        q1 = QUBOResult(Q=np.eye(3), offset=1.0)
        q2 = QUBOResult(Q=np.eye(3) * 2, offset=0.5)
        combined = add_qubo(q1, q2)
        np.testing.assert_array_almost_equal(combined.Q, np.eye(3) * 3)
        assert combined.offset == pytest.approx(1.5)

    def test_add_qubo_shape_mismatch_raises(self):
        from quprep.qubo.utils import add_qubo
        q1 = QUBOResult(Q=np.eye(2), offset=0.0)
        q2 = QUBOResult(Q=np.eye(3), offset=0.0)
        with pytest.raises(ValueError):
            add_qubo(q1, q2)

    def test_add_qubo_weight(self):
        from quprep.qubo.utils import add_qubo
        q1 = QUBOResult(Q=np.eye(2), offset=0.0)
        q2 = QUBOResult(Q=np.eye(2), offset=2.0)
        combined = add_qubo(q1, q2, weight=0.5)
        assert combined.offset == pytest.approx(1.0)

    def test_to_dict_from_dict(self):
        from quprep.qubo.problems.maxcut import max_cut
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        q = max_cut(adj)
        d = q.to_dict()
        assert "Q" in d and "offset" in d
        q2 = QUBOResult.from_dict(d)
        np.testing.assert_array_almost_equal(q.Q, q2.Q)
        assert q.offset == pytest.approx(q2.offset)

    def test_to_dict_json_serializable(self):
        import json

        from quprep.qubo.problems.knapsack import knapsack
        r = knapsack(np.array([2.0, 3.0]), np.array([3.0, 4.0]), capacity=3.0)
        # Should not raise
        json.dumps(r.to_dict())

    def test_draw_qubo_requires_matplotlib(self):
        from quprep.qubo.problems.maxcut import max_cut
        from quprep.qubo.visualize import draw_qubo
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        try:
            ax = draw_qubo(max_cut(adj))
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not installed")

    def test_draw_ising_requires_matplotlib(self):
        from quprep.qubo.problems.maxcut import max_cut
        from quprep.qubo.visualize import draw_ising
        adj = np.array([[0, 1], [1, 0]], dtype=float)
        try:
            ax = draw_ising(max_cut(adj).to_ising())
            assert ax is not None
        except ImportError:
            pytest.skip("matplotlib not installed")


# ---------------------------------------------------------------------------
# QUBOResult.evaluate + to_dwave
# ---------------------------------------------------------------------------

class TestEvaluateAndDwave:
    def test_evaluate_known(self):
        # max_cut on triangle: x=[0,1,1] should give energy -2.0
        adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        q = max_cut(adj)
        assert q.evaluate(np.array([0.0, 1.0, 1.0])) == pytest.approx(-2.0)

    def test_evaluate_matches_manual(self):
        Q = np.array([[1.0, 2.0, 0.0],
                      [0.0, 3.0, 4.0],
                      [0.0, 0.0, 5.0]])
        r = QUBOResult(Q=Q, offset=1.5)
        x = np.array([1.0, 0.0, 1.0])
        expected = float(x @ Q @ x) + 1.5
        assert r.evaluate(x) == pytest.approx(expected)

    def test_evaluate_zero_vector(self):
        r = QUBOResult(Q=np.eye(3), offset=2.0)
        assert r.evaluate(np.zeros(3)) == pytest.approx(2.0)

    def test_to_dwave_triangle(self):
        adj = np.array([[0,1],[1,0]], dtype=float)
        d = max_cut(adj).to_dwave()
        assert (0, 0) in d
        assert (1, 1) in d
        assert (0, 1) in d
        assert d[(0, 1)] == pytest.approx(2.0)

    def test_to_dwave_omits_zeros(self):
        Q = np.array([[1.0, 0.0], [0.0, 0.0]])
        r = QUBOResult(Q=Q)
        d = r.to_dwave()
        assert (0, 0) in d
        assert (1, 1) not in d
        assert (0, 1) not in d

    def test_to_dwave_no_lower_triangle(self):
        adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        d = max_cut(adj).to_dwave()
        for i, j in d:
            assert i <= j


# ---------------------------------------------------------------------------
# ising_to_qubo
# ---------------------------------------------------------------------------

class TestIsingToQubo:
    def test_roundtrip_maxcut(self):
        from quprep.qubo.ising import ising_to_qubo, qubo_to_ising
        adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        q = max_cut(adj)
        q2 = ising_to_qubo(qubo_to_ising(q))
        np.testing.assert_allclose(q.Q, q2.Q, atol=1e-10)
        assert q.offset == pytest.approx(q2.offset)

    def test_roundtrip_knapsack(self):
        from quprep.qubo.ising import ising_to_qubo, qubo_to_ising
        from quprep.qubo.problems.knapsack import knapsack
        w = np.array([2.0, 3.0, 4.0])
        v = np.array([3.0, 4.0, 5.0])
        q = knapsack(w, v, capacity=5.0)
        q2 = ising_to_qubo(qubo_to_ising(q))
        np.testing.assert_allclose(q.Q, q2.Q, atol=1e-10)

    def test_exported_from_top_level(self):
        from quprep.qubo import ising_to_qubo
        assert callable(ising_to_qubo)

    def test_energy_preserved(self):
        from quprep.qubo.ising import ising_to_qubo, qubo_to_ising
        adj = np.array([[0,1],[1,0]], dtype=float)
        q = max_cut(adj)
        q2 = ising_to_qubo(qubo_to_ising(q))
        for bits in range(4):
            x = np.array([(bits >> i) & 1 for i in range(2)], dtype=float)
            assert q.evaluate(x) == pytest.approx(q2.evaluate(x))


# ---------------------------------------------------------------------------
# solve_sa
# ---------------------------------------------------------------------------

class TestSolveSA:
    def test_triangle_maxcut(self):
        from quprep.qubo.solver import solve_sa
        adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        sol = solve_sa(max_cut(adj), seed=42, n_steps=5000)
        assert sol.energy == pytest.approx(-2.0, abs=1e-9)

    def test_perfect_partition(self):
        from quprep.qubo.problems.number_partition import number_partition
        from quprep.qubo.solver import solve_sa
        v = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 1.0])
        sol = solve_sa(number_partition(v), seed=0, n_steps=20_000, restarts=3)
        assert sol.energy == pytest.approx(0.0, abs=1e-9)

    def test_returns_solve_result(self):
        from quprep.qubo.solver import SolveResult, solve_sa
        adj = np.array([[0,1],[1,0]], dtype=float)
        sol = solve_sa(max_cut(adj), seed=1)
        assert isinstance(sol, SolveResult)
        assert sol.x.shape == (2,)
        assert all(v in (0.0, 1.0) for v in sol.x)

    def test_n_evaluated(self):
        from quprep.qubo.solver import solve_sa
        adj = np.array([[0,1],[1,0]], dtype=float)
        sol = solve_sa(max_cut(adj), n_steps=500, restarts=3, seed=0)
        assert sol.n_evaluated == 1500

    def test_restarts_improves_result(self):
        from quprep.qubo.problems.knapsack import knapsack
        from quprep.qubo.solver import solve_sa
        w = np.array([2.0, 3.0, 4.0, 5.0, 1.0])
        v = np.array([3.0, 4.0, 5.0, 6.0, 2.0])
        q = knapsack(w, v, capacity=8.0)
        sol1 = solve_sa(q, seed=0, n_steps=2000, restarts=1)
        sol5 = solve_sa(q, seed=0, n_steps=2000, restarts=5)
        assert sol5.energy <= sol1.energy + 1e-9

    def test_accessible_via_submodule(self):
        from quprep.qubo.solver import solve_sa
        assert callable(solve_sa)

    def test_auto_T_start(self):
        from quprep.qubo.solver import solve_sa
        # Should not raise when T_start is None (default)
        adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        sol = solve_sa(max_cut(adj), T_start=None, seed=7)
        assert sol.energy <= 0.0

    def test_larger_problem(self):
        # n=25 — too large for brute force, SA should find a reasonable answer
        from quprep.qubo.solver import solve_sa
        rng = np.random.default_rng(99)
        Q = np.triu(rng.uniform(-1, 1, (25, 25)))
        r = QUBOResult(Q=Q, offset=0.0)
        sol = solve_sa(r, n_steps=20_000, restarts=2, seed=99)
        assert isinstance(sol.energy, float)


# ---------------------------------------------------------------------------
# CLI — portfolio + graphcolor
# ---------------------------------------------------------------------------

class TestCLIPortfolioGraphcolor:
    def test_portfolio_basic(self):
        from quprep.cli import main
        ret = main([
            "qubo", "portfolio",
            "--returns", "0.5,0.3,0.2,0.1",
            "--covariance",
            "0.1,0.02,0.01,0.0;0.02,0.05,0.01,0.0;0.01,0.01,0.08,0.0;0.0,0.0,0.0,0.04",
            "--budget", "2",
        ])
        assert ret == 0

    def test_portfolio_with_solve(self):
        from quprep.cli import main
        ret = main([
            "qubo", "portfolio",
            "--returns", "0.5,0.3",
            "--covariance", "0.1,0.02;0.02,0.05",
            "--budget", "1",
            "--solve",
        ])
        assert ret == 0

    def test_graphcolor_basic(self):
        from quprep.cli import main
        ret = main([
            "qubo", "graphcolor",
            "--adjacency", "0,1,1;1,0,1;1,1,0",
            "--colors", "3",
        ])
        assert ret == 0

    def test_graphcolor_with_solve(self):
        from quprep.cli import main
        ret = main([
            "qubo", "graphcolor",
            "--adjacency", "0,1,0;1,0,1;0,1,0",
            "--colors", "2",
            "--solve",
        ])
        assert ret == 0
