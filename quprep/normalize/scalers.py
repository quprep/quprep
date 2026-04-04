r"""Normalization strategies with auto-selection per encoding type.

Encoding–normalization mapping
-------------------------------
Amplitude  → L2-normalize ($\|x\|_2 = 1$). Amplitudes must form a valid quantum state.
Angle Ry   → scale to $[0, \pi]$. Maps to rotation angles on the Bloch sphere.
Angle Rx   → scale to $[-\pi, \pi]$.
Basis      → binarize to $\{0, 1\}$. Qubits are $|0\rangle$ or $|1\rangle$ only.
IQP        → scale to $[-\pi, \pi]$ (feature products handled by the IQP encoder).
QUBO/Ising → binary $\{0,1\}$ or signed $\{-1,+1\}$.

Users can override by passing a Scaler explicitly to the Pipeline.
"""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset

_VALID_STRATEGIES = (
    "l2",
    "minmax",
    "minmax_pi",
    "minmax_pm_pi",
    "zscore",
    "binary",
    "pm_one",
)

ENCODING_NORMALIZER_MAP: dict[str, str] = {
    "amplitude": "l2",
    "angle_ry": "minmax_pi",
    "angle_rx": "minmax_pm_pi",
    "angle_rz": "minmax_pm_pi",
    "basis": "binary",
    "iqp": "minmax_pm_pi",
    "qubo": "binary",
    "ising": "pm_one",
    "hamiltonian": "zscore",
}


def auto_normalizer(encoding: str) -> Scaler:
    """
    Return the correct Scaler for a given encoding name.

    Parameters
    ----------
    encoding : str
        One of the keys in ENCODING_NORMALIZER_MAP.

    Returns
    -------
    Scaler

    Raises
    ------
    ValueError
        If the encoding name is not recognised.
    """
    key = ENCODING_NORMALIZER_MAP.get(encoding)
    if key is None:
        raise ValueError(
            f"Unknown encoding '{encoding}'. Known: {sorted(ENCODING_NORMALIZER_MAP)}"
        )
    return Scaler(strategy=key)


class Scaler:
    r"""
    Apply a normalization strategy to a Dataset.

    All strategies operate column-wise (per feature) except 'l2' which
    operates row-wise (per sample), as required by amplitude encoding.

    Parameters
    ----------
    strategy : str
        'l2'           — unit L2 norm per sample: $x / \|x\|_2$.
                         Zero-norm rows are left as-is (all-zero vector).
        'minmax'       — scale each feature to $[0, 1]$.
        'minmax_pi'    — scale each feature to $[0, \pi]$.  (angle Ry)
        'minmax_pm_pi' — scale each feature to $[-\pi, \pi]$. (angle Rx/Rz, IQP)
        'zscore'       — zero mean, unit std per feature.
                         Constant features (std = 0) are left as zero.
        'binary'       — threshold each feature at 0.5 → $\{0.0, 1.0\}$.
        'pm_one'       — threshold each feature at 0.5 → $\{-1.0, +1.0\}$. (Ising)
    threshold : float
        Binarization cutoff for 'binary' and 'pm_one'. Default 0.5.

    Raises
    ------
    ValueError
        If ``strategy`` is not one of the supported strategies.
    """

    def __init__(self, strategy: str = "minmax", threshold: float = 0.5):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'"
            )
        self.strategy = strategy
        self.threshold = threshold
        self._fitted = False
        self._col_min: np.ndarray | None = None
        self._col_max: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def fit(self, dataset: Dataset) -> Scaler:
        """
        Learn normalization parameters from dataset.

        For stateless strategies (``'l2'``, ``'binary'``, ``'pm_one'``) this
        is a no-op that marks the scaler as fitted.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Scaler
            Returns ``self`` for chaining.
        """
        data = dataset.data
        if self.strategy in ("minmax", "minmax_pi", "minmax_pm_pi", "pm_one"):
            self._col_min = np.nanmin(data, axis=0)
            self._col_max = np.nanmax(data, axis=0)
        elif self.strategy == "zscore":
            self._mean = np.nanmean(data, axis=0)
            self._std = np.nanstd(data, axis=0)
        # l2 and binary are stateless — nothing to learn
        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Apply learned normalization and return a new Dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            Same metadata, feature_names, feature_types, and categorical_data.
            Only ``data`` is modified.

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
        data = dataset.data.copy()
        data = self._apply(data)
        return Dataset(
            data=data,
            feature_names=list(dataset.feature_names),
            feature_types=list(dataset.feature_types),
            categorical_data=dict(dataset.categorical_data),
            metadata=dict(dataset.metadata),
            labels=dataset.labels,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Normalize and return a new Dataset with scaled data.

        Parameters
        ----------
        dataset : Dataset
            Input dataset to normalize.

        Returns
        -------
        Dataset
            Same metadata, feature_names, feature_types, and categorical_data.
            Only `data` is modified.
        """
        return self.fit(dataset).transform(dataset)

    def _apply(self, data: np.ndarray) -> np.ndarray:
        """Apply fitted parameters (or stateless transform) to data."""
        if self.strategy == "l2":
            return _l2_normalize(data)

        if self.strategy == "minmax":
            return _apply_minmax(data, self._col_min, self._col_max, low=0.0, high=1.0)

        if self.strategy == "minmax_pi":
            return _apply_minmax(data, self._col_min, self._col_max, low=0.0, high=np.pi)

        if self.strategy == "minmax_pm_pi":
            return _apply_minmax(data, self._col_min, self._col_max, low=-np.pi, high=np.pi)

        if self.strategy == "zscore":
            std = np.where(self._std == 0, 1.0, self._std)
            return (data - self._mean) / std

        if self.strategy == "binary":
            return (data >= self.threshold).astype(float)

        if self.strategy == "pm_one":
            return np.where(data >= self.threshold, 1.0, -1.0)

        raise ValueError(f"Unknown strategy '{self.strategy}'")  # unreachable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _l2_normalize(data: np.ndarray) -> np.ndarray:
    """Normalise each row to unit L2 norm. Zero rows are unchanged."""
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return data / norms


def _apply_minmax(
    data: np.ndarray,
    col_min: np.ndarray,
    col_max: np.ndarray,
    low: float,
    high: float,
) -> np.ndarray:
    """Scale each column to [low, high] using pre-fitted min/max."""
    col_range = col_max - col_min
    col_range = np.where(col_range == 0, 1.0, col_range)
    scaled = (data - col_min) / col_range   # → [0, 1]
    return scaled * (high - low) + low      # → [low, high]
