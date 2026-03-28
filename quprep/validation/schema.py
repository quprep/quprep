"""Schema enforcement — declare expected features, types, and value ranges."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quprep.core.dataset import Dataset


@dataclass
class FeatureSpec:
    """
    Specification for a single feature column.

    Parameters
    ----------
    name : str
        Expected column name.
    dtype : str
        Expected feature type: ``'continuous'``, ``'discrete'``, or ``'binary'``.
    min_value : float, optional
        Minimum allowed value (inclusive). ``None`` means no lower bound.
    max_value : float, optional
        Maximum allowed value (inclusive). ``None`` means no upper bound.
    nullable : bool
        Whether NaN is permitted. Default ``False``.
    """

    name: str
    dtype: str
    min_value: float | None = None
    max_value: float | None = None
    nullable: bool = False


class SchemaViolationError(ValueError):
    """Raised when a Dataset violates a DataSchema contract."""


class DataSchema:
    """
    Declare expected features, types, and value ranges for pipeline input.

    Attach to a Pipeline via ``schema=`` to enforce the contract at entry.
    Also usable standalone via :meth:`validate`.

    Parameters
    ----------
    features : list of FeatureSpec
        One spec per expected feature, in column order.

    Examples
    --------
    >>> schema = DataSchema([
    ...     FeatureSpec("age", dtype="continuous", min_value=0, max_value=120),
    ...     FeatureSpec("income", dtype="continuous", min_value=0),
    ... ])
    >>> schema.validate(dataset)  # raises SchemaViolationError on mismatch
    """

    def __init__(self, features: list[FeatureSpec]):
        self.features = features

    def validate(self, dataset: Dataset) -> None:
        """
        Validate a Dataset against this schema.

        All violations are collected and reported together so the caller gets
        the full picture in a single error.

        Parameters
        ----------
        dataset : Dataset

        Raises
        ------
        SchemaViolationError
            If any violations are found.
        """
        violations: list[str] = []

        n_expected = len(self.features)
        n_actual = dataset.n_features
        if n_actual != n_expected:
            violations.append(
                f"Feature count mismatch: expected {n_expected}, got {n_actual}."
            )

        check_count = min(n_expected, n_actual)
        for i, spec in enumerate(self.features[:check_count]):
            col = dataset.data[:, i]
            col_name = (
                dataset.feature_names[i]
                if i < len(dataset.feature_names)
                else f"feature[{i}]"
            )

            if dataset.feature_names and col_name != spec.name:
                violations.append(
                    f"Feature {i}: expected name '{spec.name}', got '{col_name}'."
                )

            if not spec.nullable and np.isnan(col).any():
                n_nan = int(np.isnan(col).sum())
                violations.append(
                    f"Feature '{spec.name}': {n_nan} NaN value(s) not allowed "
                    "(set nullable=True to permit)."
                )

            valid = col[~np.isnan(col)]
            if spec.min_value is not None and valid.size > 0 and valid.min() < spec.min_value:
                violations.append(
                    f"Feature '{spec.name}': min value {valid.min():.4g} "
                    f"< allowed minimum {spec.min_value}."
                )
            if spec.max_value is not None and valid.size > 0 and valid.max() > spec.max_value:
                violations.append(
                    f"Feature '{spec.name}': max value {valid.max():.4g} "
                    f"> allowed maximum {spec.max_value}."
                )

            if spec.dtype == "binary":
                unique = set(np.unique(valid).tolist())
                if not unique.issubset({0.0, 1.0}):
                    shown = sorted(unique)[:5]
                    violations.append(
                        f"Feature '{spec.name}': expected binary {{0, 1}}, "
                        f"got values {shown}."
                    )

        if violations:
            bullet = "\n  - ".join(violations)
            raise SchemaViolationError(
                f"DataSchema validation failed with {len(violations)} "
                f"violation(s):\n  - {bullet}"
            )

    @classmethod
    def infer(cls, dataset: Dataset) -> DataSchema:
        """
        Infer a DataSchema from an existing Dataset.

        Parameters
        ----------
        dataset : Dataset
            Reference dataset to infer schema from.

        Returns
        -------
        DataSchema
            Schema with inferred names, types, and value ranges.
        """
        features: list[FeatureSpec] = []
        for i in range(dataset.n_features):
            col = dataset.data[:, i]
            valid = col[~np.isnan(col)]
            name = (
                dataset.feature_names[i]
                if i < len(dataset.feature_names)
                else f"feature_{i}"
            )
            dtype = (
                dataset.feature_types[i]
                if i < len(dataset.feature_types)
                else "continuous"
            )
            features.append(
                FeatureSpec(
                    name=name,
                    dtype=dtype,
                    min_value=float(valid.min()) if valid.size > 0 else None,
                    max_value=float(valid.max()) if valid.size > 0 else None,
                    nullable=bool(np.isnan(col).any()),
                )
            )
        return cls(features)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> list[dict]:
        """
        Serialise this schema to a plain list of dicts.

        Each dict has keys ``name``, ``dtype``, and optionally ``min_value``,
        ``max_value``, and ``nullable`` (only included when non-default so the
        output stays terse).

        Returns
        -------
        list[dict]
        """
        out = []
        for spec in self.features:
            entry: dict = {"name": spec.name, "dtype": spec.dtype}
            if spec.min_value is not None:
                entry["min_value"] = spec.min_value
            if spec.max_value is not None:
                entry["max_value"] = spec.max_value
            if spec.nullable:
                entry["nullable"] = spec.nullable
            out.append(entry)
        return out

    def to_json(self, indent: int = 2) -> str:
        """
        Serialise this schema to a JSON string.

        Parameters
        ----------
        indent : int
            JSON indentation level (default 2).

        Returns
        -------
        str
        """
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: list[dict]) -> DataSchema:
        """
        Build a DataSchema from a list of dicts (e.g. loaded from JSON).

        Parameters
        ----------
        data : list[dict]
            Each dict must have ``name`` and ``dtype``; ``min_value``,
            ``max_value``, and ``nullable`` are optional.

        Returns
        -------
        DataSchema
        """
        features = [
            FeatureSpec(
                name=entry["name"],
                dtype=entry["dtype"],
                min_value=entry.get("min_value"),
                max_value=entry.get("max_value"),
                nullable=entry.get("nullable", False),
            )
            for entry in data
        ]
        return cls(features)

    @classmethod
    def from_json(cls, s: str) -> DataSchema:
        """
        Build a DataSchema from a JSON string.

        Parameters
        ----------
        s : str
            JSON string produced by :meth:`to_json`.

        Returns
        -------
        DataSchema
        """
        import json
        return cls.from_dict(json.loads(s))
