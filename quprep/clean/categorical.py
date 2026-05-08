"""Categorical feature encoding — converts string columns to numeric."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset

_VALID_STRATEGIES = ("onehot", "label", "ordinal")


class CategoricalEncoder:
    """
    Encode categorical columns stored in Dataset.categorical_data into
    numeric columns and merge them into Dataset.data.

    Parameters
    ----------
    strategy : str
        'onehot'  — one binary column per category (default).
                    Increases dimensionality; use a reducer afterwards.
        'label'   — integer code per unique value, arbitrary order.
                    Compact but implies ordinal relationship.
        'ordinal' — integer code respecting a provided category order.
    handle_unknown : str
        'ignore' — unknown categories at transform time become NaN (onehot)
                   or -1 (label/ordinal).
        'error'  — raise ValueError on unknown categories.
    cardinality_threshold : int or None
        If set, issues a ``QuPrepWarning`` when a column has more unique
        categories than this value. Useful for catching accidental qubit
        explosion before encoding. Default None (no check).
    min_frequency : int or None
        If set, categories appearing fewer than this many times in training
        data are grouped into a single ``"_other"`` category. Applied during
        ``fit()``. Default None (no grouping).
    """

    def __init__(
        self,
        strategy: str = "onehot",
        handle_unknown: str = "ignore",
        cardinality_threshold: int | None = None,
        min_frequency: int | None = None,
    ):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'")
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(f"handle_unknown must be 'ignore' or 'error', got '{handle_unknown}'")
        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self.cardinality_threshold = cardinality_threshold
        self.min_frequency = min_frequency
        self._fitted = False
        self._categories: dict[str, list] = {}  # col_name → ordered category list
        # col_name → set of rare values grouped as "_other"
        self._rare_categories: dict[str, set] = {}

    def fit(self, dataset: Dataset) -> CategoricalEncoder:
        """
        Learn category mappings from dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        CategoricalEncoder
            Returns ``self`` for chaining.
        """
        import warnings

        import pandas as pd

        from quprep.validation.input_validator import QuPrepWarning

        self._categories = {}
        self._rare_categories = {}
        for col_name, values in dataset.categorical_data.items():
            s = pd.Series(values, dtype="category")
            categories = s.cat.categories.tolist()

            over_threshold = (
                self.cardinality_threshold is not None
                and len(categories) > self.cardinality_threshold
            )
            if over_threshold:
                warnings.warn(
                    f"Column '{col_name}' has {len(categories)} unique categories "
                    f"(cardinality_threshold={self.cardinality_threshold}). "
                    "Consider using a reducer after encoding to control qubit count.",
                    QuPrepWarning,
                    stacklevel=2,
                )

            if self.min_frequency is not None:
                counts = s.value_counts()
                rare = set(counts[counts < self.min_frequency].index.tolist())
                if rare:
                    self._rare_categories[col_name] = rare
                    categories = [c for c in categories if c not in rare] + ["_other"]

            self._categories[col_name] = categories
        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Encode learned categorical columns and return a numeric-only Dataset.

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
            ``categorical_data`` will be empty.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If ``fit()`` has not been called yet.
        ValueError
            If an unknown category is found and ``handle_unknown='error'``.
        """
        from sklearn.exceptions import NotFittedError

        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit()' before 'transform()'."
            )

        if not dataset.categorical_data:
            return dataset

        data = dataset.data.copy()
        feature_names = list(dataset.feature_names)
        feature_types = list(dataset.feature_types)

        for col_name, values in dataset.categorical_data.items():
            categories = self._categories.get(col_name, [])
            rare = self._rare_categories.get(col_name)
            if rare:
                values = ["_other" if v in rare else v for v in values]
            encoded, new_names, new_types = self._encode_with_categories(
                col_name, values, categories
            )
            data = np.hstack([data, encoded])
            feature_names.extend(new_names)
            feature_types.extend(new_types)

        return Dataset(
            data=data,
            feature_names=feature_names,
            feature_types=feature_types,
            categorical_data={},
            metadata=dict(dataset.metadata),
            labels=dataset.labels,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Encode all categorical columns and return a Dataset with only
        numeric data (categorical_data will be empty afterwards).

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        Dataset
        """
        return self.fit(dataset).transform(dataset)

    def _encode_with_categories(self, col_name: str, values: list, categories: list):
        """Encode a column using a pre-fitted category list."""
        import pandas as pd

        if self.strategy == "onehot":
            s = pd.Series(values, dtype="category")
            dummies = pd.get_dummies(s, prefix=col_name, dtype=float)
            # Align columns to fitted categories
            expected_cols = [f"{col_name}_{cat}" for cat in categories]
            for col in expected_cols:
                if col not in dummies.columns:
                    dummies[col] = 0.0
            dummies = dummies.reindex(columns=expected_cols, fill_value=0.0)
            encoded = dummies.to_numpy(dtype=float)
            new_names = list(dummies.columns)
            new_types = ["binary"] * encoded.shape[1]

        else:  # label / ordinal
            mapping = {cat: float(i) for i, cat in enumerate(categories)}
            encoded_vals = np.array([mapping.get(v, np.nan) for v in values], dtype=float)
            if self.handle_unknown == "error" and np.isnan(encoded_vals).any():
                raise ValueError(f"Unknown categories in column '{col_name}'")
            encoded_vals = np.where(np.isnan(encoded_vals), -1.0, encoded_vals)
            encoded = encoded_vals.reshape(-1, 1)
            new_names = [col_name]
            new_types = ["discrete"]

        return encoded, new_names, new_types
