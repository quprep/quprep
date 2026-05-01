"""Class imbalance handling — oversample, undersample, SMOTE, ADASYN."""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np

from quprep.core.dataset import Dataset
from quprep.validation.input_validator import QuPrepWarning


class ImbalanceHandler:
    """
    Balance class distributions before quantum encoding.

    Supports four strategies:

    - ``"oversample"`` — random duplication of minority samples (no extra deps).
    - ``"undersample"`` — random removal of majority samples (no extra deps).
    - ``"smote"`` — Synthetic Minority Over-sampling Technique; interpolates
      in feature space using k-nearest neighbours (requires scikit-learn,
      already a core dependency).
    - ``"adasyn"`` — Adaptive Density-based Synthetic sampling; focuses
      synthetic samples on harder-to-learn regions (requires
      ``imbalanced-learn``: ``pip install quprep[imbalance]``).

    Parameters
    ----------
    strategy : {"oversample", "undersample", "smote", "adasyn"}
        Resampling strategy.
    sampling_strategy : float or "auto"
        - ``"auto"`` balances all classes to the majority class count
          (oversampling) or the minority class count (undersampling).
        - A float ``r`` targets ``majority_count × r`` samples per class for
          oversampling, or ``minority_count / r`` for undersampling.
    k_neighbors : int
        Number of nearest neighbours for SMOTE and ADASYN.
    random_state : int
        Seed for reproducibility.

    Examples
    --------
    >>> import numpy as np
    >>> import quprep as qd
    >>> from quprep.core.dataset import Dataset
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 1, (110, 4))
    >>> y = np.array([0] * 100 + [1] * 10)
    >>> ds = Dataset(data=X, labels=y)
    >>> handler = qd.ImbalanceHandler(strategy="smote")
    >>> ds_bal = handler.fit_transform(ds)
    >>> from collections import Counter
    >>> print(Counter(ds_bal.labels))
    Counter({0: 100, 1: 100})
    """

    _VALID_STRATEGIES = {"oversample", "undersample", "smote", "adasyn"}

    def __init__(
        self,
        strategy: str = "oversample",
        sampling_strategy: float | str = "auto",
        k_neighbors: int = 5,
        random_state: int = 42,
    ) -> None:
        if strategy not in self._VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {self._VALID_STRATEGIES}, got {strategy!r}"
            )
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self._fitted = False
        self._class_counts: Counter = Counter()
        self._target_count: int = 0

    def fit(self, dataset: Dataset) -> ImbalanceHandler:
        """
        Compute class distribution and target count from *dataset*.

        Parameters
        ----------
        dataset : Dataset
            Must have ``labels`` set (1-D array, single-target only).
        """
        if dataset.labels is None:
            raise ValueError("ImbalanceHandler requires Dataset.labels to be set.")
        labels = np.asarray(dataset.labels)
        if labels.ndim > 1:
            raise ValueError(
                "ImbalanceHandler supports single-target labels only (1-D array)."
            )
        self._class_counts = Counter(labels)
        majority = max(self._class_counts.values())
        minority = min(self._class_counts.values())

        if self.sampling_strategy == "auto":
            self._target_count = (
                majority if self.strategy in ("oversample", "smote", "adasyn") else minority
            )
        else:
            ratio = float(self.sampling_strategy)
            if self.strategy in ("oversample", "smote", "adasyn"):
                self._target_count = int(majority * ratio)
            else:
                self._target_count = int(minority / ratio)

        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Apply the fitted resampling strategy to *dataset*.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            New Dataset with resampled data and labels (shuffled).
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        if dataset.labels is None:
            raise ValueError("ImbalanceHandler requires Dataset.labels.")
        labels = np.asarray(dataset.labels)
        X = dataset.data
        rng = np.random.default_rng(self.random_state)

        dispatch = {
            "oversample": self._random_oversample,
            "undersample": self._random_undersample,
            "smote": self._smote,
            "adasyn": self._adasyn,
        }
        X_res, y_res = dispatch[self.strategy](X, labels, rng)

        idx = rng.permutation(len(X_res))
        ds = dataset.copy()
        ds.data = X_res[idx]
        ds.labels = y_res[idx]
        return ds

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fit and transform in one step."""
        return self.fit(dataset).transform(dataset)

    # ── private helpers ──────────────────────────────────────────────────────

    def _random_oversample(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        parts_X, parts_y = [X], [y]
        for cls, count in self._class_counts.items():
            if count >= self._target_count:
                continue
            idx = np.where(y == cls)[0]
            extra = rng.choice(idx, size=self._target_count - count, replace=True)
            parts_X.append(X[extra])
            parts_y.append(y[extra])
        return np.vstack(parts_X), np.concatenate(parts_y)

    def _random_undersample(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        parts_X, parts_y = [], []
        for cls, count in self._class_counts.items():
            idx = np.where(y == cls)[0]
            kept = rng.choice(idx, size=min(count, self._target_count), replace=False)
            parts_X.append(X[kept])
            parts_y.append(y[kept])
        return np.vstack(parts_X), np.concatenate(parts_y)

    def _smote(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as exc:  # pragma: no cover
            raise ImportError("SMOTE requires scikit-learn.") from exc

        parts_X, parts_y = [X], [y]
        for cls, count in self._class_counts.items():
            if count >= self._target_count:
                continue
            idx = np.where(y == cls)[0]
            X_min = X[idx]
            k = min(self.k_neighbors, len(idx) - 1)

            if k < 1:
                warnings.warn(
                    f"Class {cls!r} has only {len(idx)} sample(s) — cannot apply "
                    "SMOTE (need ≥ 2). Falling back to random oversampling for this class.",
                    QuPrepWarning,
                    stacklevel=3,
                )
                extra = rng.choice(idx, size=self._target_count - count, replace=True)
                parts_X.append(X[extra])
                parts_y.append(y[extra])
                continue

            nn = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
            _, neighbors = nn.kneighbors(X_min)
            neighbors = neighbors[:, 1:]  # drop self

            n_new = self._target_count - count
            synth = np.empty((n_new, X.shape[1]))
            for i in range(n_new):
                src = rng.integers(len(X_min))
                nn_pick = rng.integers(k)
                lam = rng.random()
                synth[i] = X_min[src] + lam * (X_min[neighbors[src, nn_pick]] - X_min[src])
            parts_X.append(synth)
            parts_y.append(np.full(n_new, cls, dtype=y.dtype))

        return np.vstack(parts_X), np.concatenate(parts_y)

    def _adasyn(  # pragma: no cover
        self, X: np.ndarray, y: np.ndarray, _rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            from imblearn.over_sampling import ADASYN
        except ImportError as exc:
            raise ImportError(
                "ADASYN requires imbalanced-learn: pip install quprep[imbalance]"
            ) from exc

        if self.sampling_strategy == "auto":
            strategy: str | dict = "auto"
        else:
            minority_cls = min(self._class_counts, key=self._class_counts.get)
            strategy = {minority_cls: self._target_count}

        ada = ADASYN(
            sampling_strategy=strategy,
            n_neighbors=self.k_neighbors,
            random_state=self.random_state,
        )
        X_res, y_res = ada.fit_resample(X, y)
        return np.asarray(X_res), np.asarray(y_res)
