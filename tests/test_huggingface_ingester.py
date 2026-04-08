"""Unit tests for HuggingFaceIngester — all modalities mocked."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from quprep.ingest.huggingface_ingester import HuggingFaceIngester

# ---------------------------------------------------------------------------
# Helpers — fake HuggingFace feature types and dataset
# ---------------------------------------------------------------------------

# The ingester uses type(feat).__name__ for modality detection, so these fake
# feature classes must have exactly the right class name.
_FakeValue = type("Value", (), {"__init__": lambda self, dtype: setattr(self, "dtype", dtype)})
_FakeImage = type("Image", (), {})
_FakeAudio = type("Audio", (), {})


def _make_mock_hf(df: pd.DataFrame, features: dict | None = None):
    """Return a mock HF dataset whose .to_pandas() returns df."""
    mock = MagicMock()
    mock.to_pandas.return_value = df
    # Support column access: dataset["col"] → list
    mock.__getitem__.side_effect = lambda col: df[col].tolist()
    mock.features = features or {
        col: _FakeValue("float32") for col in df.columns
    }
    mock.__len__ = lambda _: len(df)
    return mock


@contextmanager
def _patch_datasets(mock_hf):
    """Swap sys.modules['datasets'] with a fake module, then restore.

    Deliberately avoids patch.dict() — that tool takes a full snapshot of
    sys.modules and removes any C extensions that get imported during the
    context, causing "cannot load module more than once per process" errors.
    We only touch the one key we care about.
    """
    fake_mod = MagicMock()
    fake_mod.load_dataset.return_value = mock_hf
    _sentinel = object()
    old = sys.modules.get("datasets", _sentinel)
    sys.modules["datasets"] = fake_mod
    try:
        yield fake_mod
    finally:
        if old is _sentinel:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = old


# ---------------------------------------------------------------------------
# ImportError
# ---------------------------------------------------------------------------

def test_import_error_when_datasets_missing():
    _sentinel = object()
    old = sys.modules.get("datasets", _sentinel)
    sys.modules["datasets"] = None  # None → Python raises ImportError on import
    try:
        with pytest.raises(ImportError, match="pip install quprep\\[huggingface\\]"):
            HuggingFaceIngester().load("some/dataset")
    finally:
        if old is _sentinel:
            sys.modules.pop("datasets", None)
        else:
            sys.modules["datasets"] = old


# ---------------------------------------------------------------------------
# Bad modality in constructor
# ---------------------------------------------------------------------------

def test_invalid_modality_raises():
    with pytest.raises(ValueError, match="modality must be one of"):
        HuggingFaceIngester(modality="audio")


# ---------------------------------------------------------------------------
# Tabular — basic
# ---------------------------------------------------------------------------

def test_tabular_basic_shape():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(split="train").load("fake/dataset")
    assert ds.data.shape == (3, 2)
    assert list(ds.feature_names) == ["a", "b"]


def test_tabular_extracts_single_label():
    df = pd.DataFrame({"feat": [1.0, 2.0], "label": [0, 1]})
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(target_columns="label").load("fake/dataset")
    assert ds.labels is not None
    assert ds.labels.shape == (2,)
    assert "label" not in ds.feature_names


def test_tabular_extracts_multi_label():
    df = pd.DataFrame({"x": [1.0, 2.0], "y1": [0, 1], "y2": [1, 0]})
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(target_columns=["y1", "y2"]).load("fake/dataset")
    assert ds.labels.shape == (2, 2)
    assert ds.feature_names == ["x"]


def test_tabular_no_numeric_raises():
    df = pd.DataFrame({"text": ["hello", "world"]})
    mock_hf = _make_mock_hf(df, features={"text": _FakeValue("string")})
    with _patch_datasets(mock_hf):
        # all-string dataset without numeric → ValueError from tabular handler
        # (auto-detection picks text because no numeric cols)
        # so pass modality="tabular" explicitly to force tabular path
        with pytest.raises(ValueError, match="No numeric columns"):
            HuggingFaceIngester(modality="tabular").load("fake/dataset")


def test_tabular_categorical_stored_when_numeric_only_false():
    df = pd.DataFrame({
        "num": [1.0, 2.0],
        "cat": pd.Categorical(["a", "b"]),
    })
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(numeric_only=False).load("fake/ds")
    assert "cat" in ds.categorical_data


def test_tabular_categorical_dropped_when_numeric_only_true():
    df = pd.DataFrame({
        "num": [1.0, 2.0],
        "cat": pd.Categorical(["a", "b"]),
    })
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(numeric_only=True).load("fake/ds")
    assert ds.categorical_data == {}


def test_tabular_metadata():
    df = pd.DataFrame({"a": [1.0], "b": [2.0]})
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(split="test").load("my/dataset", config_name="en")
    assert ds.metadata["source"] == "huggingface:my/dataset"
    assert ds.metadata["split"] == "test"
    assert ds.metadata["config"] == "en"
    assert ds.metadata["modality"] == "tabular"


def test_token_forwarded():
    df = pd.DataFrame({"x": [1.0]})
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf) as fake_mod:
        HuggingFaceIngester(token="hf_abc").load("gated/dataset")
    assert fake_mod.load_dataset.call_args[1].get("token") == "hf_abc"


def test_split_forwarded():
    df = pd.DataFrame({"x": [1.0]})
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf) as fake_mod:
        HuggingFaceIngester(split="validation").load("some/dataset")
    assert fake_mod.load_dataset.call_args[1].get("split") == "validation"


def test_config_name_forwarded():
    df = pd.DataFrame({"x": [1.0]})
    mock_hf = _make_mock_hf(df)
    with _patch_datasets(mock_hf) as fake_mod:
        HuggingFaceIngester().load("multi/dataset", config_name="fr")
    assert fake_mod.load_dataset.call_args[1].get("name") == "fr"


# ---------------------------------------------------------------------------
# Modality auto-detection
# ---------------------------------------------------------------------------

def test_autodetect_image_when_image_feature_present():
    """Auto-detection picks 'image' when an Image feature column exists."""
    from PIL import Image as PILImage

    raw_img = PILImage.fromarray(np.zeros((8, 8), dtype=np.uint8), mode="L")
    df = pd.DataFrame({"image": [None], "label": [3]})
    mock_hf = _make_mock_hf(df, features={
        "image": _FakeImage(),
        "label": _FakeValue("int64"),
    })
    mock_hf.__getitem__.side_effect = lambda col: (
        [raw_img] if col == "image" else [3]
    )

    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(target_columns="label", image_size=(4, 4)).load("mnist/fake")

    assert ds.metadata["modality"] == "image"
    assert ds.data.shape == (1, 16)


def test_autodetect_text_when_only_string_columns():
    """Auto-detection picks 'text' for all-string (no numeric) datasets."""
    texts = ["hello world", "quantum rocks"]
    df = pd.DataFrame({"text": texts, "label": [0, 1]})
    mock_hf = _make_mock_hf(df, features={
        "text": _FakeValue("string"),
        "label": _FakeValue("int64"),
    })
    mock_hf.__getitem__.side_effect = lambda col: (
        texts if col == "text" else [0, 1]
    )

    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(target_columns="label", max_features=4).load("text/fake")

    assert ds.metadata["modality"] == "text"
    assert ds.data.shape == (2, 4)


def test_autodetect_tabular_when_numeric_present():
    """Mixed text+numeric falls through to tabular (string cols dropped)."""
    df = pd.DataFrame({"num": [1.0, 2.0], "desc": ["a", "b"]})
    mock_hf = _make_mock_hf(df, features={
        "num": _FakeValue("float32"),
        "desc": _FakeValue("string"),
    })

    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester().load("mixed/fake")

    assert ds.metadata["modality"] == "tabular"
    assert list(ds.feature_names) == ["num"]


def test_unsupported_audio_raises():
    """Pure audio dataset with no numeric fallback raises NotImplementedError."""
    df = pd.DataFrame({"audio": [None]})
    mock_hf = _make_mock_hf(df, features={"audio": _FakeAudio()})
    with _patch_datasets(mock_hf):
        with pytest.raises(NotImplementedError, match="audio"):
            HuggingFaceIngester().load("audio/dataset")


# ---------------------------------------------------------------------------
# Image modality
# ---------------------------------------------------------------------------

def test_image_load_shape():
    """Image ingestion should flatten and stack pixel arrays (real PIL images)."""
    from PIL import Image as PILImage

    # Create two real 28×28 grayscale PIL images
    raw_img = PILImage.fromarray(np.zeros((28, 28), dtype=np.uint8), mode="L")

    df = pd.DataFrame({"image": [None, None], "label": [0, 1]})
    mock_hf = _make_mock_hf(df, features={
        "image": _FakeImage(),
        "label": _FakeValue("int64"),
    })
    mock_hf.__getitem__.side_effect = lambda col: (
        [raw_img, raw_img] if col == "image" else [0, 1]
    )

    with _patch_datasets(mock_hf):
        ingester = HuggingFaceIngester(
            modality="image",
            image_column="image",
            target_columns="label",
            image_size=(4, 4),   # resize to 4×4 → 16 pixels
        )
        ds = ingester.load("fake/mnist")

    assert ds.data.shape == (2, 16)   # 4×4 grayscale
    assert ds.metadata["modality"] == "image"
    assert ds.metadata["image_column"] == "image"
    assert ds.labels is not None


# ---------------------------------------------------------------------------
# Text modality
# ---------------------------------------------------------------------------

def test_text_tfidf_load():
    texts = ["quantum computing rocks", "machine learning is great", "hello world"]
    labels = [0, 1, 0]
    df = pd.DataFrame({"text": texts, "label": labels})
    mock_hf = _make_mock_hf(df, features={
        "text": _FakeValue("string"),
        "label": _FakeValue("int64"),
    })
    mock_hf.__getitem__.side_effect = lambda col: (
        texts if col == "text" else labels
    )
    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(
            modality="text",
            text_column="text",
            target_columns="label",
            max_features=8,
        ).load("fake/text")

    assert ds.data.shape == (3, 8)
    assert ds.metadata["modality"] == "text"
    assert ds.metadata["text_column"] == "text"
    assert ds.labels.tolist() == labels


def test_text_invalid_method_raises():
    texts = ["hello"]
    df = pd.DataFrame({"text": texts})
    mock_hf = _make_mock_hf(df, features={"text": _FakeValue("string")})
    mock_hf.__getitem__.side_effect = lambda col: texts

    with _patch_datasets(mock_hf):
        with pytest.raises(ValueError, match="text_method must be"):
            HuggingFaceIngester(
                modality="text", text_column="text", text_method="bert"
            ).load("fake/text")


# ---------------------------------------------------------------------------
# Graph modality
# ---------------------------------------------------------------------------

def test_graph_load_basic():
    # Two tiny graphs: triangle (3 nodes, 3 edges) and path (4 nodes, 3 edges)
    edge_index_0 = np.array([[0, 1, 2], [1, 2, 0]])
    edge_index_1 = np.array([[0, 1, 2], [1, 2, 3]])
    labels = [0, 1]

    df = pd.DataFrame({"edge_index": [edge_index_0, edge_index_1], "y": labels})
    mock_hf = _make_mock_hf(df, features={
        "edge_index": MagicMock(),
        "y": _FakeValue("int64"),
    })
    mock_hf.__getitem__.side_effect = lambda col: (
        [edge_index_0, edge_index_1] if col == "edge_index" else labels
    )
    mock_hf.__len__ = lambda _: 2

    with _patch_datasets(mock_hf):
        ds = HuggingFaceIngester(
            modality="graph",
            target_columns="y",
            n_graph_features=6,
        ).load("fake/graph")

    assert ds.data.shape == (2, 6)
    assert ds.metadata["modality"] == "graph"
    assert ds.labels.tolist() == labels


def test_graph_missing_edge_column_raises():
    df = pd.DataFrame({"node_feat": [[1.0, 2.0]]})
    mock_hf = _make_mock_hf(df, features={"node_feat": MagicMock()})
    with _patch_datasets(mock_hf):
        with pytest.raises(ValueError, match="edge_index_column"):
            HuggingFaceIngester(modality="graph").load("fake/graph")
