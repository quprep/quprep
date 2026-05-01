"""Analytical barren plateau risk estimation (McClean et al. 2018, Cerezo et al. 2021)."""

from dataclasses import dataclass, field

from quprep.core.dataset import Dataset


@dataclass  # pragma: no cover
class BarrenPlateauReport:
    """
    Barren plateau risk report for a quantum encoding.

    Attributes
    ----------
    encoding : str
        Encoder name (lower-case, without "Encoder" suffix).
    n_qubits : int
        Number of qubits determined by cost estimation.
    circuit_depth : int
        Estimated circuit depth.
    gradient_variance : float
        Analytical upper bound on the gradient variance for the given cost type.
        Derived from the formula for the specified *cost_type* — no simulation
        is performed.
    risk_level : str
        One of ``"none"``, ``"mild"``, ``"high"``, ``"severe"``.
    mitigations : list[str]
        Suggested mitigation strategies (empty when risk is "none").
    """

    encoding: str
    n_qubits: int
    circuit_depth: int
    gradient_variance: float
    risk_level: str
    mitigations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"BarrenPlateauReport({self.encoding})",
            f"  n_qubits         : {self.n_qubits}",
            f"  circuit_depth    : {self.circuit_depth}",
            f"  gradient_variance: {self.gradient_variance:.2e}  (upper bound)",
            f"  risk_level       : {self.risk_level}",
        ]
        if self.mitigations:
            lines.append("  mitigations:")
            for tip in self.mitigations:
                lines.append(f"    - {tip}")
        return "\n".join(lines)


def _gradient_variance(n_qubits: int, cost_type: str) -> float:
    if cost_type == "local":
        # Cerezo et al. 2021: polynomial decay for local (single-qubit) observables
        return 1.0 / (n_qubits**2)
    # McClean et al. 2018: exponential decay for global cost
    return 2.0 ** (1 - n_qubits)


def _risk_level(var: float) -> str:
    if var > 0.05:
        return "none"
    if var > 0.005:
        return "mild"
    if var > 0.0005:
        return "high"
    return "severe"


_MITIGATIONS = {  # pragma: no cover
    "local_cost": (
        "Use a local cost function (single-qubit Pauli observables) — "
        "polynomial gradient decay instead of exponential"
    ),
    "layerwise": (
        "Layer-wise training: freeze earlier layers and train one block at a time"
    ),
    "identity_init": (
        "Identity-block initialisation: start near-identity gate parameters "
        "to avoid flat initial landscapes"
    ),
    "reduce_qubits": (
        "Reduce qubit count via dimensionality reduction (PCAReducer, LDAReducer) "
        "before encoding"
    ),
    "shallow_enc": (
        "Prefer shallower encodings (AngleEncoder, BasisEncoder) — lower "
        "expressibility means weaker barren plateaus"
    ),
}


def _mitigations(risk: str) -> list[str]:
    if risk == "none":
        return []
    tips = [_MITIGATIONS["local_cost"]]
    if risk in ("high", "severe"):
        tips += [_MITIGATIONS["layerwise"], _MITIGATIONS["identity_init"]]
    if risk == "severe":
        tips += [_MITIGATIONS["reduce_qubits"], _MITIGATIONS["shallow_enc"]]
    return tips


def detect_barren_plateau(
    encoder,
    dataset: Dataset,
    *,
    cost_type: str = "global",
) -> BarrenPlateauReport:
    """
    Analytically estimate barren plateau risk for a quantum encoding.

    No circuit simulation is performed.  Risk is derived from qubit count
    using the theoretical gradient variance bounds:

    - **Global cost** (McClean et al. 2018): ``Var[∂C/∂θ] ≤ 2^(1−n)``
      — exponential decay with qubit count.
    - **Local cost** (Cerezo et al. 2021): ``Var[∂C/∂θ] ≈ 1/n²``
      — polynomial decay; strongly preferred for large circuits.

    Parameters
    ----------
    encoder : BaseEncoder
        A QuPrep encoder.  Does not need to be fitted.
    dataset : Dataset
        Used only to determine qubit count and circuit depth via cost
        estimation.
    cost_type : {"global", "local"}
        Cost function type used during training.

    Returns
    -------
    BarrenPlateauReport

    Examples
    --------
    >>> import numpy as np
    >>> import quprep as qd
    >>> from quprep.core.dataset import Dataset
    >>> ds = Dataset(data=np.random.default_rng(0).uniform(0, 1, (50, 8)))
    >>> report = qd.detect_barren_plateau(qd.IQPEncoder(), ds)
    >>> print(report.risk_level)
    mild

    References
    ----------
    McClean J.R. et al. "Barren plateaus in quantum neural network training
    landscapes." *Nature Communications* 9, 4812 (2018).

    Cerezo M. et al. "Cost function dependent barren plateaus in shallow
    parametrized quantum circuits." *Nature Communications* 12, 1791 (2021).
    """
    if cost_type not in ("global", "local"):
        raise ValueError(f"cost_type must be 'global' or 'local', got {cost_type!r}")

    from quprep.validation.cost import estimate_cost

    cost = estimate_cost(encoder, dataset.n_features)
    n = cost.n_qubits
    var = _gradient_variance(n, cost_type)
    risk = _risk_level(var)
    encoding_name = type(encoder).__name__.replace("Encoder", "").lower()

    return BarrenPlateauReport(
        encoding=encoding_name,
        n_qubits=n,
        circuit_depth=cost.circuit_depth,
        gradient_variance=var,
        risk_level=risk,
        mitigations=_mitigations(risk),
    )
