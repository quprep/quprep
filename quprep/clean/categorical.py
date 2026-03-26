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
    """

    def __init__(self, strategy: str = "onehot", handle_unknown: str = "ignore"):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(f"strategy must be one of {_VALID_STRATEGIES}, got '{strategy}'")
        if handle_unknown not in ("ignore", "error"):
            raise ValueError(f"handle_unknown must be 'ignore' or 'error', got '{handle_unknown}'")
        self.strategy = strategy
        self.handle_unknown = handle_unknown
        self._fitted = False
        self._categories: dict[str, list] = {}  # col_name → ordered category list

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
        import pandas as pd

        self._categories = {}
        for col_name, values in dataset.categorical_data.items():
            s = pd.Series(values, dtype="category")
            self._categories[col_name] = s.cat.categories.tolist()
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
