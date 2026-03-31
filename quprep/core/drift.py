"""Data drift detection — warn when new data is outside the training distribution."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriftReport:
    """
    Summary of drift detected between training and new data.

    Attributes
    ----------
    drifted_features : list of str
        Feature names (or indices) where drift was detected.
    feature_stats : dict
        Per-feature drift details: ``{name: {"train_mean", "new_mean",
        "train_std", "new_std", "mean_shift_sigmas", "std_ratio"}}``.
    n_features_drifted : int
        Number of features that exceeded the drift threshold.
    overall_drift : bool
        ``True`` if any feature exceeded the threshold.
    """

    drifted_features: list[str] = field(default_factory=list)
    feature_stats: dict = field(default_factory=dict)
    n_features_drifted: int = 0
    overall_drift: bool = False

    def __str__(self) -> str:
        if not self.overall_drift:
            return "DriftReport: no drift detected"
        lines = [
            f"DriftReport: {self.n_features_drifted} feature(s) drifted",
        ]
        for name, stats in self.feature_stats.items():
            if name in self.drifted_features:
                lines.append(
                    f"  {name}: mean {stats['train_mean']:.4g} → {stats['new_mean']:.4g} "
                    f"({stats['mean_shift_sigmas']:.1f}σ shift), "
                    f"std ratio {stats['std_ratio']:.2f}"
                )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"DriftReport(n_features_drifted={self.n_features_drifted}, "
            f"overall_drift={self.overall_drift})"
        )


class DriftDetector:
    """
    Detects statistical drift between training data and new data.

    Fitted during ``Pipeline.fit()`` on the post-cleaning, post-reduction
    feature matrix. On ``transform()``, compares new data against stored
    training statistics and issues a ``QuPrepWarning`` if drift is detected.

    Two signals are checked per feature:

    - **Mean shift** — the difference in feature means, expressed in units
      of the training standard deviation. Flagged when
      ``|new_mean - train_mean| / train_std > mean_threshold`` (default 3σ).
    - **Std ratio** — the ratio of new std to training std. Flagged when
      the ratio is outside ``[1/std_threshold, std_threshold]``
      (default 2×, i.e. std has doubled or halved).

    Parameters
    ----------
    mean_threshold : float
        Number of training standard deviations a mean shift must exceed
        to be flagged (default: 3.0).
    std_threshold : float
        Maximum ratio of new std to training std before flagging (default: 2.0).
        A ratio of 2.0 means the new data is twice as spread out (or half).
    warn : bool
        Whether to issue a ``QuPrepWarning`` when drift is detected (default:
        ``True``). Set to ``False`` to use :meth:`check` programmatically
        without side effects.
    """

    def __init__(
        self,
        mean_threshold: float = 3.0,
        std_threshold: float = 2.0,
        warn: bool = True,
    ):
        self.mean_threshold = mean_threshold
        self.std_threshold = std_threshold
        self.warn = warn
        self._train_mean: np.ndarray | None = None
        self._train_std: np.ndarray | None = None
        self._feature_names: list[str] | None = None
        self._fitted = False

    def fit(self, dataset) -> DriftDetector:
        """
        Record training distribution statistics.

        Parameters
        ----------
        dataset : Dataset
            Training data (post-cleaning, post-reduction). NaN values are
            excluded from statistics using nan-safe functions.

        Returns
        -------
        DriftDetector
            Returns ``self``.
        """
        data = dataset.data
        self._train_mean = np.nanmean(data, axis=0)
        self._train_std = np.nanstd(data, axis=0)
        self._feature_names = list(dataset.feature_names) if dataset.feature_names else [
            f"feature[{i}]" for i in range(data.shape[1])
        ]
        self._fitted = True
        return self

    def check(self, dataset) -> DriftReport:
        """
        Check new data for drift against the training distribution.

        Parameters
        ----------
        dataset : Dataset
            New data to check. Must have the same number of features as the
            training data.

        Returns
        -------
        DriftReport

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        ValueError
            If ``dataset`` has a different number of features than training data.
        """
        if not self._fitted:
            raise RuntimeError(
                "DriftDetector has not been fitted. Call fit() first."
            )

        data = dataset.data
        n_train_features = len(self._train_mean)
        if data.shape[1] != n_train_features:
            raise ValueError(
                f"Feature count mismatch: training had {n_train_features} features, "
                f"new data has {data.shape[1]}."
            )

        new_mean = np.nanmean(data, axis=0)
        new_std = np.nanstd(data, axis=0)

        drifted = []
        feature_stats = {}

        for i, name in enumerate(self._feature_names):
            t_mean = float(self._train_mean[i])
            t_std = float(self._train_std[i])
            n_mean = float(new_mean[i])
            n_std = float(new_std[i])

            # Mean shift in units of training std
            if t_std > 0:
                mean_shift_sigmas = abs(n_mean - t_mean) / t_std
            else:
                mean_shift_sigmas = 0.0 if abs(n_mean - t_mean) < 1e-9 else float("inf")

            # Std ratio (avoid div-by-zero)
            if t_std > 0:
                std_ratio = n_std / t_std if t_std > 0 else 1.0
            else:
                std_ratio = 1.0

            is_drifted = (
                mean_shift_sigmas > self.mean_threshold
                or std_ratio > self.std_threshold
                or (std_ratio > 0 and (1.0 / std_ratio) > self.std_threshold)
            )

            feature_stats[name] = {
                "train_mean": t_mean,
                "new_mean": n_mean,
                "train_std": t_std,
                "new_std": n_std,
                "mean_shift_sigmas": mean_shift_sigmas,
                "std_ratio": std_ratio,
            }

            if is_drifted:
                drifted.append(name)

        overall_drift = len(drifted) > 0
        report = DriftReport(
            drifted_features=drifted,
            feature_stats=feature_stats,
            n_features_drifted=len(drifted),
            overall_drift=overall_drift,
        )

        if overall_drift and self.warn:
            from quprep.validation.input_validator import QuPrepWarning
            warnings.warn(
                f"Data drift detected in {len(drifted)} feature(s): "
                f"{', '.join(drifted[:5])}"
                + (" ..." if len(drifted) > 5 else "")
                + ". New data may be outside the training distribution.",
                QuPrepWarning,
                stacklevel=3,
            )

        return report
