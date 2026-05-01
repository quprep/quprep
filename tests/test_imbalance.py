"""Tests for quprep.clean.imbalance.ImbalanceHandler."""

from collections import Counter

import numpy as np
import pytest

import quprep as qd
from quprep.clean.imbalance import ImbalanceHandler
from quprep.core.dataset import Dataset


@pytest.fixture
def imbalanced_ds():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (110, 4))
    y = np.array([0] * 100 + [1] * 10)
    return Dataset(data=X, labels=y)


@pytest.fixture
def balanced_ds():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (60, 4))
    y = np.array([0] * 30 + [1] * 30)
    return Dataset(data=X, labels=y)


class TestRandomOversample:
    def test_balances_classes(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="oversample")
        ds = h.fit_transform(imbalanced_ds)
        counts = Counter(ds.labels)
        assert counts[0] == counts[1] == 100

    def test_original_samples_preserved(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="oversample")
        ds = h.fit_transform(imbalanced_ds)
        assert len(ds.data) == 200

    def test_no_change_when_balanced(self, balanced_ds):
        h = ImbalanceHandler(strategy="oversample")
        ds = h.fit_transform(balanced_ds)
        counts = Counter(ds.labels)
        assert counts[0] == counts[1]


class TestRandomUndersample:
    def test_balances_classes(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="undersample")
        ds = h.fit_transform(imbalanced_ds)
        counts = Counter(ds.labels)
        assert counts[0] == counts[1] == 10

    def test_output_size_reduced(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="undersample")
        ds = h.fit_transform(imbalanced_ds)
        assert len(ds.data) == 20


class TestSMOTE:
    def test_balances_classes(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="smote")
        ds = h.fit_transform(imbalanced_ds)
        counts = Counter(ds.labels)
        assert counts[0] == counts[1] == 100

    def test_synthetic_samples_in_feature_range(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="smote")
        ds = h.fit_transform(imbalanced_ds)
        assert ds.data.min() >= 0.0
        assert ds.data.max() <= 1.0 + 1e-9


class TestImbalanceHandlerGeneral:
    def test_requires_labels(self):
        ds = Dataset(data=np.ones((10, 2)))
        h = ImbalanceHandler()
        with pytest.raises(ValueError, match="labels"):
            h.fit(ds)

    def test_multilabel_raises(self):
        ds = Dataset(
            data=np.ones((10, 2)),
            labels=np.zeros((10, 2)),
        )
        h = ImbalanceHandler()
        with pytest.raises(ValueError, match="single-target"):
            h.fit(ds)

    def test_invalid_strategy(self):
        with pytest.raises(ValueError, match="strategy"):
            ImbalanceHandler(strategy="magic")

    def test_transform_before_fit_raises(self, imbalanced_ds):
        h = ImbalanceHandler()
        with pytest.raises(RuntimeError, match="fit"):
            h.transform(imbalanced_ds)

    def test_labels_preserved_dtype(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="oversample")
        ds = h.fit_transform(imbalanced_ds)
        assert ds.labels is not None

    def test_data_shape_consistent(self, imbalanced_ds):
        h = ImbalanceHandler(strategy="oversample")
        ds = h.fit_transform(imbalanced_ds)
        assert ds.data.shape[1] == imbalanced_ds.data.shape[1]

    def test_top_level_import(self, imbalanced_ds):
        h = qd.ImbalanceHandler(strategy="undersample")
        ds = h.fit_transform(imbalanced_ds)
        assert isinstance(ds, Dataset)

    def test_multiclass(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(0, 1, (60, 3))
        y = np.array([0] * 40 + [1] * 15 + [2] * 5)
        ds = Dataset(data=X, labels=y)
        h = ImbalanceHandler(strategy="oversample")
        ds_bal = h.fit_transform(ds)
        counts = Counter(ds_bal.labels)
        assert counts[0] == counts[1] == counts[2] == 40
