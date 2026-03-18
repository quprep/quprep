"""Categorical feature encoding."""

from __future__ import annotations


class CategoricalEncoder:
    """
    Encode categorical features as numeric values.

    Parameters
    ----------
    strategy : str
        'onehot', 'label', 'ordinal', 'target', or 'binary'.
        'binary' is recommended for high-cardinality features.
    handle_unknown : str
        'ignore' or 'error'. Default 'ignore'.
    """

    def __init__(self, strategy: str = "onehot", handle_unknown: str = "ignore"):
        self.strategy = strategy
        self.handle_unknown = handle_unknown

    def fit_transform(self, dataset):
        """Encode categoricals and return Dataset."""
        raise NotImplementedError("CategoricalEncoder.fit_transform() — coming in v0.1.0")
