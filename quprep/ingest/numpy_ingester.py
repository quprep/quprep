"""NumPy array and Pandas DataFrame ingestion."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset
from quprep.ingest.csv_ingester import _detect_feature_types


class NumpyIngester:
    """Wrap a NumPy array or Pandas DataFrame as a Dataset."""

    def load(self, data) -> Dataset:
        """
        Convert array-like data to a Dataset.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            2-D numeric array or DataFrame. 1-D arrays are treated as a
            single-feature column.

        Returns
        -------
        Dataset

        Raises
        ------
        TypeError
            If data is not a NumPy array or Pandas DataFrame.
        ValueError
            If data has more than 2 dimensions.
        """
        try:
            import pandas as pd
            is_dataframe = isinstance(data, pd.DataFrame)
        except ImportError:
            is_dataframe = False

        if is_dataframe:
            import pandas as pd
            df = data
            feature_names = list(df.columns.astype(str))
            feature_types = _detect_feature_types(df)
            numeric = df.select_dtypes(include=[np.number])
            arr = numeric.to_numpy(dtype=float)
            return Dataset(
                data=arr,
                feature_names=feature_names,
                feature_types=feature_types,
            )

        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data, dtype=float)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Expected np.ndarray or pd.DataFrame, got {type(data).__name__}"
                ) from e

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if data.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {data.shape}")

        data = data.astype(float)
        n_features = data.shape[1]
        feature_names = [f"x{i}" for i in range(n_features)]
        feature_types = ["continuous"] * n_features

        return Dataset(
            data=data,
            feature_names=feature_names,
            feature_types=feature_types,
        )
