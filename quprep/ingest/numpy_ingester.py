"""NumPy array and Pandas DataFrame ingestion."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset


class NumpyIngester:
    """Wrap a NumPy array or Pandas DataFrame as a Dataset."""

    def load(self, data) -> Dataset:
        """
        Convert array-like data to a Dataset.

        Parameters
        ----------
        data : np.ndarray or pd.DataFrame
            Input data.

        Returns
        -------
        Dataset
        """
        raise NotImplementedError("NumpyIngester.load() — coming in v0.1.0")
