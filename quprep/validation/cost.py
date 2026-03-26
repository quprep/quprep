"""Circuit cost estimation — gate count and depth before encoding runs."""

from __future__ import annotations

import math
from dataclasses import dataclass

# Thresholds for NISQ-safe classification
_NISQ_DEPTH_LIMIT = 200
_NISQ_CNOT_LIMIT = 50


@dataclass
class CostEstimate:
    """
    Gate count and circuit depth estimate for an encoder configuration.

    Attributes
    ----------
    encoding : str
        Name of the encoding method.
    n_features : int
        Number of input features.
    n_qubits : int
        Number of qubits required by this encoding.
    gate_count : int
        Total gate count (1-qubit + 2-qubit) per circuit.
    circuit_depth : int
        Critical-path depth estimate.
    two_qubit_gates : int
        Number of 2-qubit gates (CNOTs). Most relevant for NISQ hardware.
    nisq_safe : bool
        ``True`` if ``circuit_depth < 200`` and ``two_qubit_gates < 50``.
    warning : str or None
        Human-readable warning if the depth is prohibitively high.
    """

    encoding: str
    n_features: int
    n_qubits: int
    gate_count: int
    circuit_depth: int
    two_qubit_gates: int
    nisq_safe: bool
    warning: str | None


def estimate_cost(encoder, n_features: int) -> CostEstimate:
    """
    Estimate gate count and circuit depth for an encoder configuration.

    Parameters
    ----------
    encoder : BaseEncoder
        A configured encoder instance.
    n_features : int
        Number of features in the dataset (after any reduction).

    Returns
    -------
    CostEstimate
    """
    from quprep.encode.amplitude import AmplitudeEncoder
    from quprep.encode.angle import AngleEncoder
    from quprep.encode.basis import BasisEncoder
    from quprep.encode.entangled_angle import EntangledAngleEncoder
    from quprep.encode.hamiltonian import HamiltonianEncoder
    from quprep.encode.iqp import IQPEncoder
    from quprep.encode.reupload import ReUploadEncoder

    d = n_features

    if isinstance(encoder, AngleEncoder):
        return CostEstimate(
            encoding="angle",
            n_features=d,
            n_qubits=d,
            gate_count=d,
            circuit_depth=1,
            two_qubit_gates=0,
            nisq_safe=True,
            warning=None,
        )

    if isinstance(encoder, AmplitudeEncoder):
        n_qubits = max(1, math.ceil(math.log2(max(d, 2))))
        depth = 2**n_qubits
        gates = 2 * depth
        cnots = depth
        nisq_safe = depth < _NISQ_DEPTH_LIMIT and cnots < _NISQ_CNOT_LIMIT
        warning = (
            f"AmplitudeEncoder requires depth ~{depth} — likely infeasible on NISQ "
            "hardware. Consider AngleEncoder or add a reducer first."
        ) if not nisq_safe else None
        return CostEstimate(
            encoding="amplitude",
            n_features=d,
            n_qubits=n_qubits,
            gate_count=gates,
            circuit_depth=depth,
            two_qubit_gates=cnots,
            nisq_safe=nisq_safe,
            warning=warning,
        )

    if isinstance(encoder, BasisEncoder):
        return CostEstimate(
            encoding="basis",
            n_features=d,
            n_qubits=d,
            gate_count=d,
            circuit_depth=1,
            two_qubit_gates=0,
            nisq_safe=True,
            warning=None,
        )

    if isinstance(encoder, IQPEncoder):
        reps = encoder.reps
        pairs = d * (d - 1) // 2
        cnots = pairs * reps
        gates = d * reps + cnots
        depth = d**2 * reps
        nisq_safe = depth < _NISQ_DEPTH_LIMIT and cnots < _NISQ_CNOT_LIMIT
        warning = (
            f"IQPEncoder depth ~{depth} may be infeasible on NISQ hardware. "
            "Reduce features or reps."
        ) if not nisq_safe else None
        return CostEstimate(
            encoding="iqp",
            n_features=d,
            n_qubits=d,
            gate_count=gates,
            circuit_depth=depth,
            two_qubit_gates=cnots,
            nisq_safe=nisq_safe,
            warning=warning,
        )

    if isinstance(encoder, ReUploadEncoder):
        layers = encoder.layers
        gates = d * layers
        depth = d * layers
        nisq_safe = depth < _NISQ_DEPTH_LIMIT
        return CostEstimate(
            encoding="reupload",
            n_features=d,
            n_qubits=d,
            gate_count=gates,
            circuit_depth=depth,
            two_qubit_gates=0,
            nisq_safe=nisq_safe,
            warning=None,
        )

    if isinstance(encoder, EntangledAngleEncoder):
        layers = encoder.layers
        entanglement = encoder.entanglement
        if entanglement == "linear":
            pairs_per_layer = max(d - 1, 0)
        elif entanglement == "circular":
            pairs_per_layer = d
        else:  # full
            pairs_per_layer = d * (d - 1) // 2
        cnots = pairs_per_layer * layers
        gates = d * layers + cnots
        depth = (d + pairs_per_layer) * layers
        nisq_safe = depth < _NISQ_DEPTH_LIMIT and cnots < _NISQ_CNOT_LIMIT
        warning = (
            f"EntangledAngleEncoder depth ~{depth} may be infeasible on NISQ hardware."
        ) if not nisq_safe else None
        return CostEstimate(
            encoding="entangled_angle",
            n_features=d,
            n_qubits=d,
            gate_count=gates,
            circuit_depth=depth,
            two_qubit_gates=cnots,
            nisq_safe=nisq_safe,
            warning=warning,
        )

    if isinstance(encoder, HamiltonianEncoder):
        steps = encoder.trotter_steps
        pairs = d * (d - 1) // 2
        cnots = pairs * steps * 2  # each ZZ interaction ~ 2 CNOTs
        gates = d * steps + cnots
        depth = d * steps
        nisq_safe = depth < _NISQ_DEPTH_LIMIT and cnots < _NISQ_CNOT_LIMIT
        warning = (
            f"HamiltonianEncoder depth ~{depth} may be infeasible on NISQ hardware. "
            "Reduce trotter_steps or features."
        ) if not nisq_safe else None
        return CostEstimate(
            encoding="hamiltonian",
            n_features=d,
            n_qubits=d,
            gate_count=gates,
            circuit_depth=depth,
            two_qubit_gates=cnots,
            nisq_safe=nisq_safe,
            warning=warning,
        )

    # Fallback for custom/unknown encoders
    return CostEstimate(
        encoding=type(encoder).__name__,
        n_features=d,
        n_qubits=d,
        gate_count=d,
        circuit_depth=d,
        two_qubit_gates=0,
        nisq_safe=True,
        warning=None,
    )
