"""Unit tests for KaggleIngester (Kaggle API fully mocked)."""

from __future__ import annotations

import sys
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quprep.ingest.kaggle_ingester import KaggleIngester

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(directory: str, filename: str, df: pd.DataFrame) -> Path:
    """Write df as CSV inside directory and return the path."""
    path = Path(directory) / filename
    df.to_csv(path, index=False)
    return path


def _write_zip_csv(directory: str, zip_name: str, csv_name: str, df: pd.DataFrame) -> Path:
    """Write df as CSV inside a zip file and return the zip path."""
    zip_path = Path(directory) / zip_name
    csv_path = Path(directory) / csv_name
    df.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(csv_path, arcname=csv_name)
    csv_path.unlink()
    return zip_path


@contextmanager
def _mock_kaggle_api(download_side_effect=None, competition_side_effect=None):
    """Patch KaggleApiExtended so no real network calls are made."""
    mock_api = MagicMock()
    mock_api.authenticate.return_value = None
    if download_side_effect:
        mock_api.dataset_download_files.side_effect = download_side_effect
        mock_api.dataset_download_file.side_effect = download_side_effect
    if competition_side_effect:
        mock_api.competition_download_files.side_effect = competition_side_effect
        mock_api.competition_download_file.side_effect = competition_side_effect

    with patch(
        "quprep.ingest.kaggle_ingester.KaggleIngester._get_api",
        return_value=mock_api,
    ):
        yield mock_api


# ---------------------------------------------------------------------------
# ImportError
# ---------------------------------------------------------------------------

def test_import_error_when_kaggle_missing():
    old = sys.modules.get("kaggle.api.kaggle_api_extended")
    sys.modules["kaggle.api.kaggle_api_extended"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(ImportError, match="pip install quprep\\[kaggle\\]"):
            # Force _get_api to run without mock
            ingester = KaggleIngester()
            ingester._get_api()
    finally:
        if old is None:
            sys.modules.pop("kaggle.api.kaggle_api_extended", None)
        else:
            sys.modules["kaggle.api.kaggle_api_extended"] = old


# ---------------------------------------------------------------------------
# load() — dataset
# ---------------------------------------------------------------------------

def test_load_basic_shape():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

    def fake_download(dataset, path, force, quiet, unzip):
        _write_csv(path, "data.csv", df)

    with _mock_kaggle_api(download_side_effect=fake_download):
        ds = KaggleIngester().load("owner/dataset")

    assert ds.data.shape == (3, 2)
    assert list(ds.feature_names) == ["a", "b"]
    assert ds.metadata["source"] == "kaggle:dataset:owner/dataset"
    assert ds.metadata["dataset"] == "owner/dataset"


def test_load_extracts_single_label():
    df = pd.DataFrame({"feat": [1.0, 2.0], "label": [0, 1]})

    def fake_download(dataset, path, force, quiet, unzip):
        _write_csv(path, "data.csv", df)

    with _mock_kaggle_api(download_side_effect=fake_download):
        ds = KaggleIngester(target_columns="label").load("owner/dataset")

    assert ds.labels is not None
    assert ds.labels.tolist() == [0, 1]
    assert "label" not in ds.feature_names


def test_load_extracts_multi_label():
    df = pd.DataFrame({"x": [1.0, 2.0], "y1": [0, 1], "y2": [1, 0]})

    def fake_download(dataset, path, force, quiet, unzip):
        _write_csv(path, "data.csv", df)

    with _mock_kaggle_api(download_side_effect=fake_download):
        ds = KaggleIngester(target_columns=["y1", "y2"]).load("owner/dataset")

    assert ds.labels.shape == (2, 2)
    assert ds.feature_names == ["x"]


def test_load_specific_file():
    df = pd.DataFrame({"x": [1.0]})

    def fake_download_file(dataset, file_name, path, force, quiet):
        _write_csv(path, file_name, df)

    with _mock_kaggle_api() as mock_api:
        mock_api.dataset_download_file.side_effect = fake_download_file
        ds = KaggleIngester(file_name="train.csv").load("owner/dataset")

    assert ds.metadata["file"] == "train.csv"
    mock_api.dataset_download_file.assert_called_once()


def test_load_no_csv_raises():
    def fake_download(dataset, path, force, quiet, unzip):
        pass  # write nothing

    with _mock_kaggle_api(download_side_effect=fake_download):
        with pytest.raises(FileNotFoundError, match="No CSV file found"):
            KaggleIngester().load("owner/empty-dataset")


def test_load_no_numeric_columns_raises():
    df = pd.DataFrame({"text": ["hello", "world"]})

    def fake_download(dataset, path, force, quiet, unzip):
        _write_csv(path, "data.csv", df)

    with _mock_kaggle_api(download_side_effect=fake_download):
        with pytest.raises(ValueError, match="No numeric columns"):
            KaggleIngester().load("owner/text-only")


def test_load_categorical_stored_when_numeric_only_false():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})

    def fake_download(dataset, path, force, quiet, unzip):
        _write_csv(path, "data.csv", df)

    with _mock_kaggle_api(download_side_effect=fake_download):
        ds = KaggleIngester(numeric_only=False).load("owner/dataset")

    assert "cat" in ds.categorical_data


def test_load_categorical_dropped_when_numeric_only_true():
    df = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})

    def fake_download(dataset, path, force, quiet, unzip):
        _write_csv(path, "data.csv", df)

    with _mock_kaggle_api(download_side_effect=fake_download):
        ds = KaggleIngester(numeric_only=True).load("owner/dataset")

    assert ds.categorical_data == {}


# ---------------------------------------------------------------------------
# load_competition()
# ---------------------------------------------------------------------------

def test_load_competition_basic():
    df = pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 1], "Age": [22.0, 38.0]})

    def fake_comp_download(competition, path, force, quiet):
        _write_csv(path, "train.csv", df)

    with _mock_kaggle_api(competition_side_effect=fake_comp_download):
        ds = KaggleIngester(target_columns="Survived").load_competition("titanic")

    assert ds.metadata["source"] == "kaggle:competition:titanic"
    assert ds.metadata["competition"] == "titanic"
    assert ds.labels.tolist() == [0, 1]
    assert "Survived" not in ds.feature_names


def test_load_competition_specific_file():
    df = pd.DataFrame({"x": [1.0, 2.0]})

    def fake_file_download(competition, file_name, path, force, quiet):
        _write_csv(path, file_name, df)

    with _mock_kaggle_api() as mock_api:
        mock_api.competition_download_file.side_effect = fake_file_download
        ds = KaggleIngester(file_name="test.csv").load_competition("titanic")

    mock_api.competition_download_file.assert_called_once()
    assert ds.data.shape == (2, 1)


# ---------------------------------------------------------------------------
# _find_csv helper
# ---------------------------------------------------------------------------

def test_find_csv_picks_first_when_no_preference():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "b.csv").write_text("x\n1\n")
        Path(tmpdir, "a.csv").write_text("x\n2\n")
        ingester = KaggleIngester()
        found = ingester._find_csv(tmpdir, preferred=None)
    assert found.name == "a.csv"  # sorted → alphabetically first


def test_find_csv_by_preferred_name():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "train.csv").write_text("x\n1\n")
        Path(tmpdir, "test.csv").write_text("x\n2\n")
        ingester = KaggleIngester()
        found = ingester._find_csv(tmpdir, preferred="test.csv")
    assert found.name == "test.csv"


def test_find_csv_raises_when_none():
    with tempfile.TemporaryDirectory() as tmpdir:
        ingester = KaggleIngester()
        with pytest.raises(FileNotFoundError, match="No CSV file found"):
            ingester._find_csv(tmpdir, preferred=None)


# ---------------------------------------------------------------------------
# _unzip_all helper
# ---------------------------------------------------------------------------

def test_unzip_all_extracts_csv():
    df = pd.DataFrame({"x": [1.0]})
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_zip_csv(tmpdir, "data.zip", "data.csv", df)
        ingester = KaggleIngester()
        ingester._unzip_all(tmpdir)
        assert (Path(tmpdir) / "data.csv").exists()
        assert not (Path(tmpdir) / "data.zip").exists()
