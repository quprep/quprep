"""Sliding-window transformer for time series data."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset


class WindowTransformer:
    """
    Convert a time series Dataset into a set of sliding-window samples.

    Each window becomes one row in the output Dataset, with features laid
    out as ``[feat0_lag(W-1), feat1_lag(W-1), ..., feat0_lag0, feat1_lag0]``
    (oldest timestep first). The output is a standard 2-D Dataset that can
    be fed directly into any QuPrep encoder.

    Parameters
    ----------
    window_size : int
        Number of consecutive timesteps per window. Must be ≤ the number
        of rows in the input Dataset.
    step : int
        Stride between consecutive windows. Default 1 (fully overlapping).
        Set to ``window_size`` for non-overlapping windows.
    flatten : bool
        If ``True`` (default), each window is flattened to a 1-D vector of
        length ``window_size × n_features``. If ``False``, the raw
        ``(window_size, n_features)`` array is stored per window — useful
        for inspection but not compatible with the standard encoders.

    Examples
    --------
    >>> from quprep.ingest.timeseries_ingester import TimeSeriesIngester
    >>> from quprep.preprocess.window import WindowTransformer
    >>> from quprep.encode.angle import AngleEncoder
    >>> from quprep.core.pipeline import Pipeline
    >>>
    >>> pipeline = Pipeline(
    ...     ingester=TimeSeriesIngester(time_column="date"),
    ...     preprocessor=WindowTransformer(window_size=8, step=1),
    ...     encoder=AngleEncoder(),
    ... )
    >>> result = pipeline.fit_transform("timeseries.csv")
    """

    def __init__(
        self,
        window_size: int = 16,
        step: int = 1,
        flatten: bool = True,
    ):
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}")
        self.window_size = window_size
        self.step = step
        self.flatten = flatten
        self._n_features: int | None = None

    def fit(self, dataset: Dataset) -> WindowTransformer:
        """
        Fit the transformer (records input feature count).

        Parameters
        ----------
        dataset : Dataset

        Returns
        -------
        WindowTransformer
            Returns ``self`` for chaining.
        """
        self._n_features = dataset.n_features
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Apply sliding-window extraction and return a new Dataset.

        Parameters
        ----------
        dataset : Dataset
            Time series dataset with shape (n_timesteps, n_features).

        Returns
        -------
        Dataset
            Shape (n_windows, window_size × n_features) where
            ``n_windows = (n_timesteps - window_size) // step + 1``.

        Raises
        ------
        ValueError
            If ``n_timesteps < window_size``.
        """
        X = dataset.data
        n_timesteps, n_features = X.shape

        if n_timesteps < self.window_size:
            raise ValueError(
                f"Time series has {n_timesteps} timesteps but "
                f"window_size={self.window_size}. Reduce window_size or "
                "provide more data."
            )

        time_index = dataset.metadata.get("time_index")

        windows: list[np.ndarray] = []
        window_time_index: list = []

        for i in range(0, n_timesteps - self.window_size + 1, self.step):
            window = X[i : i + self.window_size]
            windows.append(window.flatten() if self.flatten else window)
            if time_index is not None:
                # tag each window with the timestamp of its last (most recent) step
                window_time_index.append(time_index[i + self.window_size - 1])

        windows_array = np.array(windows)

        # build feature names: <original_feat>_lag<k> where k=0 is most recent
        feat_names = dataset.feature_names or [f"x{i}" for i in range(n_features)]
        if self.flatten:
            feature_names = [
                f"{feat}_lag{self.window_size - 1 - t}"
                for t in range(self.window_size)
                for feat in feat_names
            ]
        else:
            feature_names = feat_names

        meta = {k: v for k, v in dataset.metadata.items() if k != "time_index"}
        meta.update({
            "window_size": self.window_size,
            "step": self.step,
            "original_n_timesteps": n_timesteps,
            "window_time_index": window_time_index if window_time_index else None,
            "modality": "time_series_windowed",
        })

        # propagate labels if they exist (window-aligned label = label at last timestep)
        labels = None
        if dataset.labels is not None:
            y = dataset.labels
            label_windows = []
            for i in range(0, n_timesteps - self.window_size + 1, self.step):
                label_windows.append(y[i + self.window_size - 1])
            labels = np.array(label_windows)

        return Dataset(
            data=windows_array,
            feature_names=feature_names,
            feature_types=["continuous"] * len(feature_names),
            metadata=meta,
            labels=labels,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fit and transform in one call."""
        return self.fit(dataset).transform(dataset)
