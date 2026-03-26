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
    r"""
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
        ``'amplitude'`` uses $\log_2(\text{n\_features})$ qubits.
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
        self._fitted = False
        self._inner_reducer = None  # PCAReducer if reduction was needed

    def fit(self, dataset) -> HardwareAwareReducer:
        """
        Fit the hardware-aware reducer on dataset.

        If the dataset already fits within the qubit budget, this is a no-op.
        Otherwise, fits a PCAReducer to the required feature count.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        HardwareAwareReducer
            Returns ``self`` for chaining.

        Raises
        ------
        ValueError
            If ``backend`` is an unrecognised string name.
        """
        n_features = dataset.data.shape[1]
        budget = self._qubit_budget()
        max_feat = _max_features_for_encoding(self.encoding, budget)

        if n_features <= max_feat:
            self._inner_reducer = None
        else:
            from quprep.reduce.pca import PCAReducer
            self._inner_reducer = PCAReducer(n_components=max_feat)
            self._inner_reducer.fit(dataset)

        self._budget = budget
        self._fitted = True
        return self

    def transform(self, dataset) -> object:
        """
        Reduce features to fit the backend's qubit budget.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            Dataset with at most ``max_features`` columns. Passthrough if
            already within budget.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit()`` has not been called yet.
        """
        from sklearn.exceptions import NotFittedError

        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit()' before 'transform()'."
            )

        if self._inner_reducer is None:
            return dataset

        result = self._inner_reducer.transform(dataset)
        result.metadata["reducer"] = "hardware_aware_pca"
        result.metadata["backend"] = self.backend
        result.metadata["qubit_budget"] = self._budget
        return result

    def fit_transform(self, dataset):
        """
        Reduce features to fit the backend's qubit budget.

        If the dataset already fits within the budget, it is returned unchanged.
        Otherwise PCA is applied to reduce to the maximum feature count allowed
        by the encoding on the target backend.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.

        Returns
        -------
        Dataset
            Dataset with at most ``max_features`` columns for the configured
            backend and encoding. Passthrough if already within budget.

        Raises
        ------
        ValueError
            If ``backend`` is an unrecognised string name.
        """
        return self.fit(dataset).transform(dataset)

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
