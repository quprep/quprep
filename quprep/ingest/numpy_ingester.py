"""NumPy array and Pandas DataFrame ingestion."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset
from quprep.ingest.csv_ingester import _detect_feature_types


class NumpyIngester:
    """Wrap a NumPy array, Pandas DataFrame, or SciPy sparse matrix as a Dataset."""

    def load(self, data, y=None) -> Dataset:
        """
        Convert array-like data to a Dataset.

        Parameters
        ----------
        data : np.ndarray, pd.DataFrame, or scipy.sparse matrix
            2-D numeric array or DataFrame. 1-D arrays are treated as a
            single-feature column. Sparse matrices are converted to dense.
        y : np.ndarray or array-like, optional
            Target labels. Shape (n_samples,) for single-target or
            (n_samples, n_labels) for multi-label. Stored in ``Dataset.labels``.

        Returns
        -------
        Dataset

        Raises
        ------
        TypeError
            If data is not a recognisable array-like type.
        ValueError
            If data has more than 2 dimensions.
        """
        # --- sparse matrix support ---
        try:
            import scipy.sparse as _sp
            if _sp.issparse(data):
                data = data.toarray()
        except ImportError:
            pass

        labels = np.asarray(y) if y is not None else None

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
                labels=labels,
            )

        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data, dtype=float)
            except (TypeError, ValueError) as e:
                raise TypeError(
                    f"Expected np.ndarray, pd.DataFrame, or scipy.sparse matrix, "
                    f"got {type(data).__name__}"
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
            labels=labels,
        )
