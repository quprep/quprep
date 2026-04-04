"""CSV and delimiter-separated file ingestion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from quprep.core.dataset import Dataset


def _detect_feature_types(df: pd.DataFrame) -> list[str]:
    types = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            types.append("binary")
        elif isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(s):
            types.append("categorical")
        elif pd.api.types.is_integer_dtype(s):
            unique_ratio = s.nunique() / max(len(s), 1)
            if s.nunique() == 2:
                types.append("binary")
            elif unique_ratio < 0.05 or s.nunique() <= 10:
                types.append("discrete")
            else:
                types.append("continuous")
        else:
            types.append("continuous")
    return types


class CSVIngester:
    """
    Ingest CSV and TSV files into a Dataset.

    Supports automatic type detection (continuous, discrete, binary, categorical)
    and basic dataset profiling on load.

    Parameters
    ----------
    delimiter : str or None
        Field delimiter. Auto-detected from file extension if None:
        '.tsv' → tab, everything else → comma.
    encoding : str
        File encoding. Defaults to 'utf-8'.
    target_columns : str or list of str, optional
        Column name(s) to treat as labels rather than features. These columns
        are extracted and stored in ``Dataset.labels`` instead of ``Dataset.data``.
        Supports single-target (str) and multi-label (list of str) use cases.
    """

    def __init__(
        self,
        delimiter: str | None = None,
        encoding: str = "utf-8",
        target_columns: str | list[str] | None = None,
    ):
        self.delimiter = delimiter
        self.encoding = encoding
        self.target_columns = target_columns

    def load(self, path: str | Path) -> Dataset:
        """
        Load a CSV/TSV file and return a Dataset.

        Numeric columns go into `data`. Non-numeric (categorical) columns
        are stored in `categorical_data` for CategoricalEncoder to process.
        NaN values are preserved as-is for the Imputer to handle.
        Columns listed in ``target_columns`` are extracted as ``Dataset.labels``.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        Dataset

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

        # --- extract label columns before feature processing ---
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

        all_feature_names = list(df.columns)
        all_feature_types = _detect_feature_types(df)

        numeric_mask = [
            not (
                isinstance(df[col].dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(df[col])
            )
            for col in df.columns
        ]
        numeric_cols = [col for col, keep in zip(df.columns, numeric_mask) if keep]
        cat_cols = [col for col, keep in zip(df.columns, numeric_mask) if not keep]

        data = df[numeric_cols].to_numpy(dtype=float) if numeric_cols else np.empty((len(df), 0))

        numeric_types = [
            t for t, keep in zip(all_feature_types, numeric_mask) if keep
        ]

        categorical_data = {col: df[col].tolist() for col in cat_cols}

        return Dataset(
            data=data,
            feature_names=numeric_cols,
            feature_types=numeric_types,
            categorical_data=categorical_data,
            metadata={
                "source": str(path),
                "original_columns": all_feature_names,
                "original_types": all_feature_types,
            },
            labels=labels,
        )
