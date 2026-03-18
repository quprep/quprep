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


class HardwareAwareReducer:
    """
    Automatically reduce features to fit a target backend's qubit count.

    Selects the most appropriate reduction method based on data type and
    the chosen encoding, then reduces n_features to at most the qubit
    capacity implied by the encoding on the specified backend.

    Parameters
    ----------
    backend : str
        Backend name (e.g. 'ibm_brisbane') or qubit count (int).
    encoding : str
        Target encoding — determines how features map to qubits.
    method : str or 'auto'
        Reduction method to use. 'auto' picks the best method for the task.
    """

    def __init__(self, backend: str | int, encoding: str = "angle", method: str = "auto"):
        self.backend = backend
        self.encoding = encoding
        self.method = method

    def fit_transform(self, dataset):
        """Reduce to the backend's qubit budget and return Dataset."""
        raise NotImplementedError("HardwareAwareReducer.fit_transform() — coming in v0.2.0")

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
