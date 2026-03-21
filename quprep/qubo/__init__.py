"""QUBO / Ising conversion for combinatorial optimization problems."""

from quprep.qubo.constraints import equality_penalty, inequality_penalty
from quprep.qubo.converter import QUBOResult, to_qubo
from quprep.qubo.ising import IsingResult, ising_to_qubo, qubo_to_ising
from quprep.qubo.problems import (
    graph_color, knapsack, max_cut, number_partition, portfolio, scheduling, tsp,
)
from quprep.qubo.qaoa import qaoa_circuit
from quprep.qubo.solver import SolveResult, solve_brute, solve_sa
from quprep.qubo.utils import add_qubo
from quprep.qubo.visualize import draw_ising, draw_qubo

__all__ = [
    "to_qubo",
    "QUBOResult",
    "qubo_to_ising",
    "ising_to_qubo",
    "IsingResult",
    "equality_penalty",
    "inequality_penalty",
    "max_cut",
    "tsp",
    "knapsack",
    "portfolio",
    "graph_color",
    "scheduling",
    "number_partition",
    "qaoa_circuit",
    "solve_brute",
    "solve_sa",
    "SolveResult",
    "add_qubo",
    "draw_qubo",
    "draw_ising",
]
