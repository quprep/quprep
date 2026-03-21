"""Built-in QUBO problem formulations.

Available problems:
    MaxCut       — Max-Cut graph partitioning
    TSP          — Travelling Salesman Problem
    Knapsack     — 0/1 Knapsack
    Portfolio    — Markowitz portfolio optimization
    GraphColor   — Graph colouring
    Scheduling   — Job scheduling (load balancing)
"""

from quprep.qubo.problems.graph_color import graph_color
from quprep.qubo.problems.knapsack import knapsack
from quprep.qubo.problems.maxcut import max_cut
from quprep.qubo.problems.number_partition import number_partition
from quprep.qubo.problems.portfolio import portfolio
from quprep.qubo.problems.scheduling import scheduling
from quprep.qubo.problems.tsp import tsp

__all__ = [
    "max_cut", "tsp", "knapsack", "portfolio",
    "graph_color", "scheduling", "number_partition",
]
