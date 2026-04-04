"""Time series CSV ingestion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quprep.core.dataset import Dataset


class TimeSeriesIngester:
    """
    Ingest a time-series CSV file into a Dataset.

    Reads a CSV where rows are timesteps and columns are features. An
    optional datetime column is extracted as a time index and stored in
    ``Dataset.metadata["time_index"]`` rather than treated as a feature.

    The resulting Dataset preserves temporal ordering. Pass it to a
    :class:`~quprep.preprocess.window.WindowTransformer` to produce
    sliding-window samples ready for quantum encoding.

    Parameters
    ----------
    time_column : str or None
        Name of the column containing timestamps. Parsed with
        ``pandas.to_datetime``. If ``None``, no time column is extracted
        and an integer index is stored instead.
    delimiter : str or None
        Field delimiter. Auto-detected from file extension if None:
        '.tsv' → tab, everything else → comma.
    encoding : str
        File encoding. Defaults to 'utf-8'.
    target_columns : str or list of str, optional
        Column name(s) to treat as labels rather than features. Stored in
        ``Dataset.labels``.
    """

    def __init__(
        self,
        time_column: str | None = None,
        delimiter: str | None = None,
        encoding: str = "utf-8",
        target_columns: str | list[str] | None = None,
    ):
        self.time_column = time_column
        self.delimiter = delimiter
        self.encoding = encoding
        self.target_columns = target_columns

    def load(self, path: str | Path) -> Dataset:
        """
        Load a time-series CSV and return a Dataset.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Dataset
            ``metadata["time_index"]`` holds the parsed timestamps (list of
            ``pandas.Timestamp``) or a plain integer range if no
            ``time_column`` was specified.
            ``metadata["modality"]`` is set to ``"time_series"``.

        Raises
        ------
        FileNotFoundError
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        delimiter = self.delimiter
        if delimiter is None:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

        df = pd.read_csv(path, delimiter=delimiter, encoding=self.encoding)

        # --- extract time column ---
        if self.time_column and self.time_column in df.columns:
            time_index = pd.to_datetime(df[self.time_column], errors="coerce").tolist()
            df = df.drop(columns=[self.time_column])
        else:
            time_index = list(range(len(df)))

        # --- extract label columns ---
        labels = None
        if self.target_columns is not None:
            cols = (
                [self.target_columns]
                if isinstance(self.target_columns, str)
                else list(self.target_columns)
            )
            labels = df[cols].to_numpy()
            if labels.shape[1] == 1:
                labels = labels.ravel()
            df = df.drop(columns=cols)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        data = df[numeric_cols].to_numpy(dtype=float)

        return Dataset(
            data=data,
            feature_names=numeric_cols,
            feature_types=["continuous"] * len(numeric_cols),
            metadata={
                "source": str(path),
                "time_index": time_index,
                "modality": "time_series",
            },
            labels=labels,
        )
