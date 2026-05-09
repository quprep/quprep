"""Dataset profiling — statistics, distributions, feature types, class balance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quprep.core.dataset import Dataset


@dataclass
class DatasetProfile:
    """
    Summary statistics computed during ingestion.

    Attributes
    ----------
    n_samples : int
    n_features : int
    feature_names : list of str
    feature_types : list of str
        Per-feature type: 'continuous', 'discrete', 'binary', 'categorical'.
    missing_counts : np.ndarray
        Number of missing values per feature.
    means : np.ndarray
    stds : np.ndarray
    mins : np.ndarray
    maxs : np.ndarray
    class_balance : dict or None
        Class frequencies for classification datasets.
    """

    n_samples: int = 0
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)
    feature_types: list[str] = field(default_factory=list)
    missing_counts: np.ndarray | None = None
    means: np.ndarray | None = None
    stds: np.ndarray | None = None
    mins: np.ndarray | None = None
    maxs: np.ndarray | None = None
    class_balance: dict | None = None

    def __str__(self) -> str:
        lines = [
            "DatasetProfile",
            f"  samples   : {self.n_samples}",
            f"  features  : {self.n_features}",
        ]
        if self.feature_names:
            lines.append(f"  names     : {self.feature_names}")
        if self.feature_types:
            counts = {}
            for t in self.feature_types:
                counts[t] = counts.get(t, 0) + 1
            lines.append(f"  types     : {counts}")
        if self.missing_counts is not None:
            total_missing = int(self.missing_counts.sum())
            lines.append(f"  missing   : {total_missing} total")
        return "\n".join(lines)


@dataclass
class PreprocessingReport:
    """
    Actionable preprocessing recommendations for a dataset.

    Produced by :func:`preprocessing_report`. Each entry in
    ``recommendations`` is a concrete action the user should take.

    Attributes
    ----------
    recommendations : list[str]
        Ordered list of actionable recommendations.
    n_issues : int
        Number of recommendations (0 = dataset is ready to encode).
    """

    recommendations: list[str] = field(default_factory=list)
    n_issues: int = 0

    def __str__(self) -> str:
        if not self.recommendations:
            return "PreprocessingReport: no issues found — dataset is ready to encode"
        lines = [f"PreprocessingReport: {self.n_issues} recommendation(s)"]
        for i, rec in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"PreprocessingReport(n_issues={self.n_issues})"


def preprocessing_report(
    dataset: Dataset,
    *,
    encoder=None,
    qubit_budget: int | None = None,
) -> PreprocessingReport:
    """
    Produce actionable preprocessing recommendations for a dataset.

    Extends :func:`profile` from statistics to concrete action items: which
    columns need imputation, whether outlier removal is advisable, whether
    dimensionality reduction is needed for the qubit budget, encoder value-
    range compatibility issues, and class imbalance warnings.

    Parameters
    ----------
    dataset : Dataset
    encoder : BaseEncoder, optional
        If provided, check value-range compatibility and suggest the correct
        normalizer.
    qubit_budget : int, optional
        Maximum qubit count. Flag when ``n_features`` exceeds this value.

    Returns
    -------
    PreprocessingReport
    """
    recs: list[str] = []
    data = dataset.data
    feature_names = (
        list(dataset.feature_names)
        if dataset.feature_names
        else [f"feature[{i}]" for i in range(dataset.n_features)]
    )

    # ── Missing values ────────────────────────────────────────────────────────
    nan_per_col = np.isnan(data).sum(axis=0)
    nan_col_idx = np.where(nan_per_col > 0)[0]
    if nan_col_idx.size > 0:
        names = [feature_names[i] for i in nan_col_idx[:5]]
        suffix = " ..." if nan_col_idx.size > 5 else ""
        recs.append(
            f"{nan_col_idx.size} column(s) need imputation "
            f"({int(nan_per_col.sum())} NaN total): "
            f"{', '.join(names)}{suffix} — add Imputer(strategy='mean')"
        )

    # ── Outliers ──────────────────────────────────────────────────────────────
    with np.errstate(invalid="ignore"):
        q1 = np.nanpercentile(data, 25, axis=0)
        q3 = np.nanpercentile(data, 75, axis=0)
        iqr = q3 - q1
        d_range = np.nanmax(data, axis=0) - np.nanmin(data, axis=0)
        outlier_idx = np.where(
            (iqr > 0) & (d_range / np.where(iqr > 0, iqr, 1.0) > 10)
        )[0]
    if outlier_idx.size > 0:
        names = [feature_names[i] for i in outlier_idx[:5]]
        suffix = " ..." if outlier_idx.size > 5 else ""
        recs.append(
            f"{outlier_idx.size} outlier-prone column(s): "
            f"{', '.join(names)}{suffix} — add OutlierHandler(method='iqr')"
        )

    # ── Qubit budget ──────────────────────────────────────────────────────────
    n_features = dataset.n_features
    if qubit_budget is not None and n_features > qubit_budget:
        recs.append(
            f"{n_features} features exceed qubit budget {qubit_budget} — "
            f"add PCAReducer(n_components={qubit_budget}) or HardwareAwareReducer"
        )
    elif qubit_budget is None and n_features > 20:
        recs.append(
            f"{n_features} features detected — consider PCAReducer or "
            "HardwareAwareReducer to fit a NISQ qubit budget"
        )

    # ── Encoder compatibility ─────────────────────────────────────────────────
    if encoder is not None:
        from quprep.validation.compatibility import check_compatibility
        compat = check_compatibility(encoder, dataset)
        for e in compat.errors:
            recs.append(f"encoder error: {e}")
        for w in compat.warnings:
            recs.append(f"encoder warning: {w}")

    # ── Class imbalance ───────────────────────────────────────────────────────
    if dataset.labels is not None:
        unique, counts = np.unique(dataset.labels, return_counts=True)
        if len(unique) >= 2:
            ratio = float(counts.max()) / float(counts.min())
            if ratio > 3.0:
                recs.append(
                    f"class imbalance detected (ratio {ratio:.1f}:1) — "
                    "add ImbalanceHandler(strategy='smote') or strategy='oversample'"
                )

    return PreprocessingReport(recommendations=recs, n_issues=len(recs))


def profile(dataset: Dataset) -> DatasetProfile:
    """
    Compute a DatasetProfile for a Dataset.

    Parameters
    ----------
    dataset : Dataset
        Input dataset to profile.

    Returns
    -------
    DatasetProfile
        Summary statistics including shape, feature types, and missing value counts.
    """
    data = dataset.data
    n_samples, n_features = data.shape

    missing_counts = np.isnan(data).sum(axis=0)

    with np.errstate(invalid="ignore"):
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        mins = np.nanmin(data, axis=0)
        maxs = np.nanmax(data, axis=0)

    return DatasetProfile(
        n_samples=n_samples,
        n_features=n_features,
        feature_names=list(dataset.feature_names),
        feature_types=list(dataset.feature_types),
        missing_counts=missing_counts,
        means=means,
        stds=stds,
        mins=mins,
        maxs=maxs,
    )
