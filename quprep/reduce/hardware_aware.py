"""Hardware-aware dimensionality reduction — auto-reduces to qubit budget."""

from __future__ import annotations

BACKEND_PROFILES: dict[str, dict] = {
    # IBM
    "ibm_brisbane": {"qubits": 127, "architecture": "superconducting"},
    "ibm_torino": {"qubits": 133, "architecture": "superconducting"},
    # IonQ
    "ionq_forte": {"qubits": 36, "architecture": "trapped_ion"},
    # Quantinuum
    "quantinuum_h2": {"qubits": 56, "architecture": "trapped_ion"},
    # IQM
    "iqm_crystal": {"qubits": 54, "architecture": "superconducting"},
}


def _max_features_for_encoding(encoding: str, qubit_budget: int) -> int:
    """Return the max feature count implied by an encoding on a qubit budget."""
    if encoding in ("amplitude",):
        # amplitude needs log2(d) qubits → d = 2^qubits, but cap at 512
        return min(2**qubit_budget, 512)
    # angle, basis, iqp, reupload, hamiltonian: 1 feature per qubit
    return qubit_budget


class HardwareAwareReducer:
    """
    Automatically reduce features to fit a target backend's qubit count.

    Calculates the qubit budget for the given backend and encoding, then
    applies PCA to reduce n_features down to that budget.

    Parameters
    ----------
    backend : str or int
        Backend name (e.g. ``'ibm_brisbane'``) or a raw qubit count.
    encoding : str
        Target encoding — determines the feature-to-qubit mapping.
        ``'angle'`` and ``'basis'`` use 1 qubit per feature.
        ``'amplitude'`` uses log₂(n_features) qubits.
    method : str
        Reduction method. Currently ``'pca'`` and ``'auto'`` (defaults to PCA).
    """

    def __init__(
        self,
        backend: str | int,
        encoding: str = "angle",
        method: str = "auto",
    ):
        self.backend = backend
        self.encoding = encoding
        self.method = method

    def fit_transform(self, dataset):
        """Reduce to the backend's qubit budget and return Dataset."""
        n_features = dataset.data.shape[1]
        budget = self._qubit_budget()
        max_feat = _max_features_for_encoding(self.encoding, budget)

        if n_features <= max_feat:
            # Already within budget — passthrough
            return dataset

        from quprep.reduce.pca import PCAReducer

        reducer = PCAReducer(n_components=max_feat)
        result = reducer.fit_transform(dataset)
        # Overwrite reducer tag so caller knows what happened
        result.metadata["reducer"] = "hardware_aware_pca"
        result.metadata["backend"] = self.backend
        result.metadata["qubit_budget"] = budget
        return result

    def _qubit_budget(self) -> int:
        if isinstance(self.backend, int):
            return self.backend
        profile = BACKEND_PROFILES.get(self.backend)
        if profile is None:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                f"Known backends: {list(BACKEND_PROFILES)}. "
                "Pass an integer qubit count instead."
            )
        return profile["qubits"]
