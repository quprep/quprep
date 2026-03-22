r"""QUBO -> QAOA circuit generator.

Converts a QUBOResult (via its Ising form) into a parameterized QAOA circuit
ready for quantum hardware or simulation.

QAOA circuit structure ($p$ layers):

1. Initialize: $|{+}\rangle^{\otimes n} = H^{\otimes n}|0\rangle^n$
2. For each layer $l = 0 \ldots p-1$:

   a. Cost unitary $U_C(\gamma_l)$:
      - $RZ(2\gamma_l h_i)$ on each qubit $i$ (single-qubit Z bias)
      - $\text{CNOT}_{ij} \cdot RZ(2\gamma_l J_{ij}) \cdot \text{CNOT}_{ij}$ (ZZ coupling)

   b. Mixer unitary $U_B(\beta_l)$: $RX(2\beta_l)$ on each qubit $i$

3. Measure all qubits

The output is an OpenQASM 3.0 string by default, or can be integrated with
any framework via the existing QuPrep exporters.

References
----------
Farhi, E., Goldstone, J., Gutmann, S. (2014). A Quantum Approximate
    Optimization Algorithm. arXiv:1411.4028.
"""

from __future__ import annotations

import numpy as np


def qaoa_circuit(
    qubo,
    p: int = 1,
    gamma: list[float] | np.ndarray | None = None,
    beta: list[float] | np.ndarray | None = None,
) -> str:
    """
    Generate a QAOA circuit for a QUBO problem as OpenQASM 3.0 string.

    The QUBO is first converted to Ising form (h, J) via ``qubo.to_ising()``,
    then a p-layer QAOA ansatz is constructed.

    Parameters
    ----------
    qubo : QUBOResult
        The QUBO problem. Use any of the problem library functions
        (max_cut, knapsack, tsp, portfolio) or to_qubo().
    p : int
        Number of QAOA layers (circuit depth). Default is 1.
    gamma : list of float, optional
        Cost unitary angles, length p. Defaults to [pi/4] * p.
        Optimal values depend on the problem; use a classical optimizer
        (e.g. scipy.optimize.minimize) to tune them.
    beta : list of float, optional
        Mixer unitary angles, length p. Defaults to [pi/8] * p.

    Returns
    -------
    str
        OpenQASM 3.0 circuit string. Can be run directly on Qiskit, Cirq,
        or any QASM-compatible backend.

    Examples
    --------
    >>> from quprep.qubo.problems.maxcut import max_cut
    >>> from quprep.qubo.qaoa import qaoa_circuit
    >>> import numpy as np
    >>> adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
    >>> qasm = qaoa_circuit(max_cut(adj), p=2)
    >>> print(qasm[:60])
    OPENQASM 3.0;
    include "stdgates.inc";
    """
    if gamma is None:
        gamma = [np.pi / 4] * p
    if beta is None:
        beta = [np.pi / 8] * p

    gamma = list(gamma)
    beta = list(beta)
    if len(gamma) != p or len(beta) != p:
        raise ValueError(f"gamma and beta must each have length p={p}.")

    # Convert QUBO -> Ising
    ising = qubo.to_ising()
    h = ising.h
    J = ising.J
    n = len(h)

    lines: list[str] = []
    lines.append("OPENQASM 3.0;")
    lines.append('include "stdgates.inc";')
    lines.append(f"qubit[{n}] q;")
    lines.append(f"bit[{n}] c;")
    lines.append("")
    lines.append("// Initialize |+>^n")
    for i in range(n):
        lines.append(f"h q[{i}];")

    for layer in range(p):
        g = gamma[layer]
        b = beta[layer]
        lines.append("")
        lines.append(f"// Layer {layer}: cost unitary U_C(gamma={g:.6f})")

        # Single-qubit Z terms: exp(-i*gamma*h_i*Z_i) = RZ(2*gamma*h_i)
        for i in range(n):
            if abs(h[i]) > 1e-12:
                angle = 2.0 * g * h[i]
                lines.append(f"rz({angle:.6f}) q[{i}];")

        # Two-qubit ZZ terms: exp(-i*gamma*J_ij*Z_i*Z_j)
        # = CNOT_{i,j} RZ(2*gamma*J_ij)_j CNOT_{i,j}
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    angle = 2.0 * g * J[i, j]
                    lines.append(f"cx q[{i}], q[{j}];")
                    lines.append(f"rz({angle:.6f}) q[{j}];")
                    lines.append(f"cx q[{i}], q[{j}];")

        lines.append("")
        lines.append(f"// Layer {layer}: mixer U_B(beta={b:.6f})")
        for i in range(n):
            lines.append(f"rx({2.0 * b:.6f}) q[{i}];")

    lines.append("")
    lines.append("// Measurement")
    lines.append("c = measure q;")

    return "\n".join(lines)
