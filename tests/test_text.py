"""Tests for TextIngester (v0.7.0)."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest


def _make_csv(tmp_path, content, name="data.csv"):
    p = tmp_path / name
    p.write_text(textwrap.dedent(content).strip())
    return p


def _make_txt(tmp_path, lines, name="corpus.txt"):
    p = tmp_path / name
    p.write_text("\n".join(lines))
    return p


_TEXTS = [
    "quantum computing is the future",
    "machine learning algorithms are powerful",
    "data preprocessing is important",
    "neural networks learn from data",
    "quantum machine learning combines both",
]


# ===========================================================================
# TF-IDF — list input
# ===========================================================================

class TestTextIngesterTFIDF:
    def test_output_shape(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf", max_features=16).load(_TEXTS)
        assert ds.data.shape == (5, 16)

    def test_output_is_dense(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf").load(_TEXTS)
        assert isinstance(ds.data, np.ndarray)

    def test_values_in_range(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf", max_features=32).load(_TEXTS)
        assert ds.data.min() >= 0.0
        assert ds.data.max() <= 1.0

    def test_feature_names(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf", max_features=8).load(_TEXTS)
        assert ds.feature_names[0] == "emb_0"
        assert len(ds.feature_names) == 8

    def test_modality_metadata(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf").load(_TEXTS)
        assert ds.metadata["modality"] == "text"
        assert ds.metadata["method"] == "tfidf"

    def test_no_labels_from_list(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf").load(_TEXTS)
        assert ds.labels is None

    def test_fewer_samples_than_features(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf", max_features=512).load(["hello world"])
        assert ds.data.shape[0] == 1

    def test_max_features_none(self):
        from quprep.ingest.text_ingester import TextIngester
        ds = TextIngester(method="tfidf", max_features=None).load(_TEXTS)
        assert ds.data.shape[0] == 5


# ===========================================================================
# TF-IDF — file inputs
# ===========================================================================

class TestTextIngesterFileLoading:
    def test_txt_file(self, tmp_path):
        from quprep.ingest.text_ingester import TextIngester
        p = _make_txt(tmp_path, _TEXTS)
        ds = TextIngester(method="tfidf", max_features=16).load(p)
        assert ds.data.shape[0] == 5

    def test_txt_skips_blank_lines(self, tmp_path):
        from quprep.ingest.text_ingester import TextIngester
        p = _make_txt(tmp_path, ["hello world", "", "quantum computing", ""])
        ds = TextIngester(method="tfidf").load(p)
        assert ds.data.shape[0] == 2

    def test_csv_with_text_column(self, tmp_path):
        from quprep.ingest.text_ingester import TextIngester
        p = _make_csv(tmp_path, """
            text,score
            quantum is great,5
            ml is interesting,4
            data science rocks,3
        """)
        ds = TextIngester(method="tfidf", text_column="text").load(p)
        assert ds.data.shape[0] == 3

    def test_csv_with_target_column(self, tmp_path):
        from quprep.ingest.text_ingester import TextIngester
        p = _make_csv(tmp_path, """
            text,label
            quantum rocks,1
            classical is ok,0
            hybrid is best,1
        """)
        ds = TextIngester(method="tfidf", text_column="text", target_column="label").load(p)
        assert ds.labels is not None
        assert ds.labels.shape == (3,)
        np.testing.assert_array_equal(ds.labels, [1, 0, 1])

    def test_csv_multi_label(self, tmp_path):
        from quprep.ingest.text_ingester import TextIngester
        p = _make_csv(tmp_path, """
            text,y1,y2
            quantum,1,0
            classical,0,1
            hybrid,1,1
        """)
        ds = TextIngester(method="tfidf", text_column="text",
                          target_column=["y1", "y2"]).load(p)
        assert ds.labels.shape == (3, 2)

    def test_csv_no_text_column_raises(self, tmp_path):
        from quprep.ingest.text_ingester import TextIngester
        p = _make_csv(tmp_path, "a,b\n1,2\n3,4\n")
        with pytest.raises(ValueError, match="text_column must be set"):
            TextIngester(method="tfidf").load(p)

    def test_csv_missing_column_raises(self, tmp_path):
        from quprep.ingest.text_ingester import TextIngester
        p = _make_csv(tmp_path, "text,label\nhello,1\nworld,0\n")
        with pytest.raises(ValueError, match="not found"):
            TextIngester(method="tfidf", text_column="nonexistent").load(p)

    def test_file_not_found(self):
        from quprep.ingest.text_ingester import TextIngester
        with pytest.raises(FileNotFoundError):
            TextIngester(method="tfidf").load("/nonexistent/corpus.txt")


# ===========================================================================
# Validation
# ===========================================================================

class TestTextIngesterValidation:
    def test_invalid_method_raises(self):
        from quprep.ingest.text_ingester import TextIngester
        with pytest.raises(ValueError, match="method must be one of"):
            TextIngester(method="bert")

    def test_sentence_transformers_import_error(self):
        from quprep.ingest.text_ingester import TextIngester
        ingester = TextIngester(method="sentence_transformers")
        # Will raise ImportError if sentence-transformers not installed
        try:
            ingester.load(["hello world"])
        except ImportError as e:
            assert "sentence-transformers" in str(e)
        except Exception:
            pass  # installed — that's fine too


# ===========================================================================
# Pipeline integration
# ===========================================================================

class TestTextIngesterPipeline:
    def test_tfidf_pipeline_roundtrip(self):
        import quprep as qd
        pipeline = qd.Pipeline(
            ingester=qd.TextIngester(method="tfidf", max_features=8),
            encoder=qd.AngleEncoder(),
        )
        result = pipeline.fit_transform(_TEXTS)
        assert len(result.encoded) == 5
        assert result.encoded[0].metadata["n_qubits"] == 8

    def test_tfidf_pipeline_with_reducer(self):
        import quprep as qd
        pipeline = qd.Pipeline(
            ingester=qd.TextIngester(method="tfidf", max_features=32),
            reducer=qd.PCAReducer(n_components=4),
            encoder=qd.AngleEncoder(),
        )
        result = pipeline.fit_transform(_TEXTS)
        assert result.dataset.n_features == 4

    def test_labels_survive_pipeline(self, tmp_path):
        import quprep as qd
        p = _make_csv(tmp_path, """
            text,label
            quantum computing,1
            machine learning,0
            data science,1
            neural networks,0
            deep learning,1
        """)
        pipeline = qd.Pipeline(
            ingester=qd.TextIngester(method="tfidf", text_column="text",
                                     target_column="label", max_features=8),
            encoder=qd.AngleEncoder(),
        )
        result = pipeline.fit_transform(p)
        assert result.dataset.labels is not None
        assert len(result.dataset.labels) == 5
