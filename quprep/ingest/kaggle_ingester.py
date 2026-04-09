"""Kaggle dataset ingestion."""

from __future__ import annotations

import tempfile
from pathlib import Path

from quprep.core.dataset import Dataset
from quprep.ingest.csv_ingester import _detect_feature_types


class KaggleIngester:
    """
    Load a Kaggle dataset or competition file into a Dataset.

    Requires ``pip install quprep[kaggle]`` and a Kaggle API token stored at
    ``~/.kaggle/kaggle.json`` (or set via the ``KAGGLE_USERNAME`` /
    ``KAGGLE_KEY`` environment variables).

    The ingester downloads the dataset to a temporary directory, finds the
    first (or specified) CSV file, and ingests it exactly like
    :class:`~quprep.ingest.csv_ingester.CSVIngester`.

    Parameters
    ----------
    target_columns : str or list of str, optional
        Column name(s) to treat as labels rather than features.
    numeric_only : bool
        If ``True`` (default), drop non-numeric columns after label
        extraction.  If ``False``, non-numeric columns are stored in
        ``Dataset.categorical_data``.
    file_name : str, optional
        Specific file to download from the dataset (e.g. ``"train.csv"``).
        When ``None`` (default), all files are downloaded and the first
        CSV found is used.
    force : bool
        Re-download even if the file already exists locally. Default ``False``.

    Examples
    --------
    Dataset (``owner/name`` format)::

        from quprep.ingest.kaggle_ingester import KaggleIngester

        ds = KaggleIngester(target_columns="label").load("heptapod/titanic")

    Competition data::

        ds = KaggleIngester(file_name="train.csv").load_competition("titanic")

    Specific file::

        ds = KaggleIngester(file_name="test.csv").load("owner/dataset-name")
    """

    def __init__(
        self,
        target_columns: str | list[str] | None = None,
        numeric_only: bool = True,
        file_name: str | None = None,
        force: bool = False,
    ):
        self.target_columns = target_columns
        self.numeric_only = numeric_only
        self.file_name = file_name
        self.force = force

    def load(self, dataset: str) -> Dataset:
        """
        Download and load a Kaggle dataset.

        Parameters
        ----------
        dataset : str
            Kaggle dataset identifier in ``"owner/dataset-name"`` format,
            e.g. ``"heptapod/titanic"`` or ``"zillow/zecon"``.

        Returns
        -------
        Dataset

        Raises
        ------
        ImportError
            If ``kaggle`` is not installed.
        FileNotFoundError
            If no CSV file is found in the downloaded dataset.
        ValueError
            If no numeric columns remain after filtering.
        """
        api = self._get_api()

        with tempfile.TemporaryDirectory() as tmpdir:
            if self.file_name is not None:
                api.dataset_download_file(
                    dataset,
                    file_name=self.file_name,
                    path=tmpdir,
                    force=self.force,
                    quiet=True,
                )
            else:
                api.dataset_download_files(
                    dataset,
                    path=tmpdir,
                    force=self.force,
                    quiet=True,
                    unzip=True,
                )

            csv_path = self._find_csv(tmpdir, self.file_name)
            return self._ingest_csv(
                csv_path,
                source=f"kaggle:dataset:{dataset}",
                extra_meta={"dataset": dataset},
            )

    def load_competition(self, competition: str) -> Dataset:
        """
        Download and load a Kaggle competition data file.

        Parameters
        ----------
        competition : str
            Competition identifier, e.g. ``"titanic"`` or
            ``"house-prices-advanced-regression-techniques"``.

        Returns
        -------
        Dataset

        Raises
        ------
        ImportError
            If ``kaggle`` is not installed.
        FileNotFoundError
            If no CSV file is found in the downloaded competition data.
        ValueError
            If no numeric columns remain after filtering.
        """
        api = self._get_api()

        with tempfile.TemporaryDirectory() as tmpdir:
            if self.file_name is not None:
                api.competition_download_file(
                    competition,
                    file_name=self.file_name,
                    path=tmpdir,
                    force=self.force,
                    quiet=True,
                )
            else:
                api.competition_download_files(
                    competition,
                    path=tmpdir,
                    force=self.force,
                    quiet=True,
                )
                # competition files download as individual files, not a zip
                self._unzip_all(tmpdir)

            csv_path = self._find_csv(tmpdir, self.file_name)
            return self._ingest_csv(
                csv_path,
                source=f"kaggle:competition:{competition}",
                extra_meta={"competition": competition},
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_api(self):
        """Return an authenticated KaggleApiExtended instance."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApiExtended
        except ImportError as e:
            raise ImportError(
                "KaggleIngester requires the 'kaggle' package. "
                "Install it with: pip install quprep[kaggle]"
            ) from e

        api = KaggleApiExtended()
        api.authenticate()
        return api

    def _find_csv(self, directory: str, preferred: str | None) -> Path:
        """Return the path to the CSV to load from *directory*."""
        dirpath = Path(directory)

        # If a specific file was requested, look for it (possibly unzipped)
        if preferred is not None:
            stem = Path(preferred).stem  # e.g. "train.csv" → "train"
            for candidate in dirpath.rglob("*.csv"):
                if candidate.stem == stem or candidate.name == preferred:
                    return candidate

        # Otherwise take the first CSV found (sorted for determinism)
        csvs = sorted(dirpath.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"No CSV file found in downloaded Kaggle data at '{directory}'. "
                "Use file_name= to specify a particular file, or check that the "
                "dataset contains CSV files."
            )
        return csvs[0]

    def _unzip_all(self, directory: str) -> None:
        """Unzip any .zip files found in *directory* in-place."""
        import zipfile

        for zf in Path(directory).rglob("*.zip"):
            with zipfile.ZipFile(zf) as z:
                z.extractall(zf.parent)
            try:
                zf.unlink()
            except OSError:
                pass

    def _ingest_csv(self, csv_path: Path, source: str, extra_meta: dict) -> Dataset:
        """Load *csv_path* using CSVIngester logic and return a Dataset."""
        import pandas as pd

        df = pd.read_csv(csv_path)

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
                or df[col].dtype.name == "string"
            )
            for col in df.columns
        ]
        numeric_cols = [col for col, keep in zip(df.columns, numeric_mask) if keep]
        cat_cols = [col for col, keep in zip(df.columns, numeric_mask) if not keep]

        if not numeric_cols:
            raise ValueError(
                f"No numeric columns found in '{csv_path.name}'. "
                f"Available columns: {all_feature_names}. "
                "Check target_columns or set numeric_only=False."
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
                "source": source,
                "file": csv_path.name,
                "original_columns": all_feature_names,
                "original_types": all_feature_types,
                "n_dropped_categorical": len(cat_cols),
                **extra_meta,
            },
            labels=labels,
        )
