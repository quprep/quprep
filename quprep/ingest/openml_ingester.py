"""OpenML dataset ingestion."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset
from quprep.ingest.csv_ingester import _detect_feature_types


class OpenMLIngester:
    """
    Load an OpenML dataset into a Dataset.

    Requires ``pip install quprep[openml]``.

    OpenML datasets are identified by an integer task/dataset ID or by name.
    The ingester calls :func:`openml.datasets.get_dataset` and uses
    :meth:`~openml.datasets.OpenMLDataset.get_data` to retrieve a pandas
    DataFrame, then processes it the same way as
    :class:`~quprep.ingest.csv_ingester.CSVIngester`.

    Parameters
    ----------
    target_column : str or None
        Name of the target / label column.  When ``None`` (default) the
        dataset's default target is used if it exists; otherwise no labels
        are extracted.
    numeric_only : bool
        If ``True`` (default), drop non-numeric columns after label
        extraction.  If ``False``, non-numeric columns are stored in
        ``Dataset.categorical_data``.
    version : int or None
        Specific dataset version to load.  ``None`` uses the latest active
        version.
    cache_format : str
        Cache format for the downloaded dataset files.  ``"pickle"``
        (default) or ``"feather"``.

    Examples
    --------
    By dataset ID::

        from quprep.ingest.openml_ingester import OpenMLIngester

        ds = OpenMLIngester(target_column="class").load(61)   # iris

    By dataset name::

        ds = OpenMLIngester(target_column="class").load("iris")

    No target (unsupervised)::

        ds = OpenMLIngester().load(554)   # MNIST_784

    Full pipeline::

        import quprep as qd

        result = qd.Pipeline(encoder=qd.AngleEncoder()).fit_transform(
            qd.OpenMLIngester(target_column="class").load("iris")
        )
    """

    def __init__(
        self,
        target_column: str | None = None,
        numeric_only: bool = True,
        version: int | None = None,
        cache_format: str = "pickle",
    ):
        self.target_column = target_column
        self.numeric_only = numeric_only
        self.version = version
        self.cache_format = cache_format

    def load(self, dataset_id: int | str) -> Dataset:
        """
        Load an OpenML dataset by ID or name.

        Parameters
        ----------
        dataset_id : int or str
            OpenML dataset ID (e.g. ``61`` for Iris) or dataset name
            (e.g. ``"iris"``).  When a name is given, the most recently
            published version matching the name is used (or the version
            specified by the ``version`` parameter).

        Returns
        -------
        Dataset

        Raises
        ------
        ImportError
            If ``openml`` is not installed.
        ValueError
            If no numeric columns remain after filtering, or the dataset
            cannot be found.
        """
        try:
            import openml
        except ImportError as e:
            raise ImportError(
                "OpenMLIngester requires the 'openml' package. "
                "Install it with: pip install quprep[openml]"
            ) from e

        import pandas as pd

        # Resolve name → id if a string was passed
        resolved_id = self._resolve_id(dataset_id, openml)

        oml_dataset = openml.datasets.get_dataset(
            resolved_id,
            download_data=True,
            version=self.version,
            cache_format=self.cache_format,
        )

        # Determine which column to use as the label
        target = self.target_column or oml_dataset.default_target_attribute

        X, y, categorical_indicator, attribute_names = oml_dataset.get_data(
            target=target,
            dataset_format="dataframe",
        )

        df: pd.DataFrame = X  # type: ignore[assignment]
        labels: np.ndarray | None = None
        if y is not None:
            arr = np.asarray(y)
            labels = arr if arr.ndim > 1 else arr.ravel()

        all_feature_names = list(df.columns)
        all_feature_types = _detect_feature_types(df)

        numeric_mask = [
            not (
                isinstance(df[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(df[col])
                or df[col].dtype.name == "string"
            )
            for col in df.columns
        ]
        numeric_cols = [col for col, keep in zip(df.columns, numeric_mask) if keep]
        cat_cols = [col for col, keep in zip(df.columns, numeric_mask) if not keep]

        if not numeric_cols:
            raise ValueError(
                f"No numeric columns found in OpenML dataset '{dataset_id}'. "
                f"Available columns: {all_feature_names}. "
                "Check target_column or set numeric_only=False."
            )

        data = df[numeric_cols].to_numpy(dtype=float)
        numeric_types = [
            t for t, keep in zip(all_feature_types, numeric_mask) if keep
        ]
        categorical_data = (
            {} if self.numeric_only
            else {col: df[col].tolist() for col in cat_cols}
        )

        return Dataset(
            data=data,
            feature_names=numeric_cols,
            feature_types=numeric_types,
            categorical_data=categorical_data,
            metadata={
                "source": f"openml:{resolved_id}",
                "dataset_id": resolved_id,
                "dataset_name": oml_dataset.name,
                "version": oml_dataset.version,
                "original_columns": all_feature_names,
                "original_types": all_feature_types,
                "n_dropped_categorical": len(cat_cols),
                "target_column": target,
            },
            labels=labels,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_id(self, dataset_id: int | str, openml) -> int:
        """Return the integer dataset ID, resolving names via the OpenML API."""
        if isinstance(dataset_id, int):
            return dataset_id

        # String name — search for it
        datasets = openml.datasets.list_datasets(output_format="dataframe")
        match = datasets[datasets["name"] == dataset_id]
        if match.empty:
            raise ValueError(
                f"No OpenML dataset found with name '{dataset_id}'. "
                "Check the name at https://www.openml.org/search?type=data "
                "or pass the integer dataset ID directly."
            )
        # Pick the version requested or the highest active version
        if self.version is not None:
            row = match[match["version"] == self.version]
            if row.empty:
                raise ValueError(
                    f"OpenML dataset '{dataset_id}' has no version {self.version}. "
                    f"Available versions: {sorted(match['version'].tolist())}"
                )
            return int(row.iloc[0]["did"])
        # Default: highest version
        return int(match.sort_values("version", ascending=False).iloc[0]["did"])
