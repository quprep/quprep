"""Encoding recommendation engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EncodingRecommendation:
    """
    Result of the encoding recommendation engine.

    Attributes
    ----------
    method : str
        Recommended encoding method (e.g. 'angle_ry').
    qubits : int
        Number of qubits required.
    depth : int or str
        Circuit depth (integer or asymptotic expression).
    nisq_safe : bool
        Whether the encoding is suitable for NISQ devices.
    reason : str
        Human-readable explanation of the recommendation.
    alternatives : list of EncodingRecommendation
        Runner-up options ranked by suitability.
    """

    method: str
    qubits: int
    depth: int | str
    nisq_safe: bool
    reason: str
    alternatives: list[EncodingRecommendation] | None = None

    def apply(self, source, *, framework: str = "qiskit"):
        """Apply this recommendation to source data and export a circuit."""
        raise NotImplementedError("EncodingRecommendation.apply() — coming in v0.1.0")


def recommend(
    source, *, task: str = "classification", qubits: int | None = None, **kwargs
) -> EncodingRecommendation:
    """
    Recommend the best encoding for a dataset and task.

    Considers data type (continuous/binary/mixed), target algorithm,
    available qubit count, and noise characteristics to rank encodings.

    Parameters
    ----------
    source : str, Path, np.ndarray, or pd.DataFrame
        Input data.
    task : str
        Target task: 'classification', 'regression', 'qaoa', 'kernel', 'simulation'.
    qubits : int, optional
        Maximum number of qubits available. Used to filter infeasible encodings.
    **kwargs
        Additional hints (e.g. backend name for hardware-aware mode).

    Returns
    -------
    EncodingRecommendation
    """
    raise NotImplementedError("recommend() — coming in v0.2.0")
