"""Tests for sklearn-compatible fit/transform API across all stages."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from quprep.clean.imputer import Imputer
from quprep.clean.outlier import OutlierHandler
from quprep.clean.selector import FeatureSelector
from quprep.core.dataset import Dataset
from quprep.core.pipeline import Pipeline, PipelineResult
from quprep.encode.angle import AngleEncoder
from quprep.normalize.scalers import Scaler
from quprep.reduce.pca import PCAReducer


def _ds(n=20, d=4, seed=0):
    rng = np.random.default_rng(seed)
    return Dataset(
        data=rng.random((n, d)).astype(np.float64),
        feature_names=[f"f{i}" for i in range(d)],
        feature_types=["continuous"] * d,
    )


# ---------------------------------------------------------------------------
# Dataset.copy()
# ---------------------------------------------------------------------------

def test_dataset_copy_independence():
    ds = _ds()
    copy = ds.copy()
    copy.data[0, 0] = 999.0
    assert ds.data[0, 0] != 999.0


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------

def test_scaler_fit_transform_unchanged():
    ds = _ds()
    s = Scaler(strategy="minmax")
    result_ft = s.fit_transform(ds)
    s2 = Scaler(strategy="minmax")
    result_chain = s2.fit(ds).transform(ds)
    np.testing.assert_allclose(result_ft.data, result_chain.data)


def test_scaler_not_fitted_raises():
    with pytest.raises(NotFittedError):
        Scaler().transform(_ds())


def test_scaler_train_test_split():
    rng = np.random.default_rng(1)
    train = Dataset(data=rng.random((50, 3)).astype(np.float64))
    test = Dataset(data=rng.random((10, 3)).astype(np.float64))
    s = Scaler(strategy="minmax")
    s.fit(train)
    out = s.transform(test)
    # test data scaled with train min/max — values may exceed [0,1] but dtype is float
    assert out.data.dtype == np.float64


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------

def test_imputer_fit_transform_unchanged():
    data = np.random.default_rng(0).random((20, 4))
    data[0, 1] = np.nan
    ds = Dataset(data=data.astype(np.float64))
    result_ft = Imputer().fit_transform(ds)
    result_chain = Imputer().fit(ds).transform(ds)
    np.testing.assert_allclose(result_ft.data, result_chain.data)


def test_imputer_not_fitted_raises():
    with pytest.raises(NotFittedError):
        Imputer().transform(_ds())


# ---------------------------------------------------------------------------
# OutlierHandler
# ---------------------------------------------------------------------------

def test_outlier_fit_transform_unchanged():
    ds = _ds()
    result_ft = OutlierHandler().fit_transform(ds)
    result_chain = OutlierHandler().fit(ds).transform(ds)
    np.testing.assert_allclose(result_ft.data, result_chain.data)


def test_outlier_not_fitted_raises():
    with pytest.raises(NotFittedError):
        OutlierHandler().transform(_ds())


# ---------------------------------------------------------------------------
# FeatureSelector
# ---------------------------------------------------------------------------

def test_selector_fit_transform_unchanged():
    ds = _ds(d=6)
    result_ft = FeatureSelector(method="variance", threshold=0.0).fit_transform(ds)
    result_chain = (
        FeatureSelector(method="variance", threshold=0.0).fit(ds).transform(ds)
    )
    np.testing.assert_allclose(result_ft.data, result_chain.data)


def test_selector_not_fitted_raises():
    with pytest.raises(NotFittedError):
        FeatureSelector().transform(_ds())


# ---------------------------------------------------------------------------
# PCAReducer
# ---------------------------------------------------------------------------

def test_pca_fit_transform_unchanged():
    ds = _ds()
    result_ft = PCAReducer(n_components=2).fit_transform(ds)
    result_chain = PCAReducer(n_components=2).fit(ds).transform(ds)
    np.testing.assert_allclose(result_ft.data, result_chain.data)


def test_pca_not_fitted_raises():
    with pytest.raises(NotFittedError):
        PCAReducer().transform(_ds())


def test_pca_train_test():
    rng = np.random.default_rng(42)
    train = Dataset(data=rng.random((50, 5)).astype(np.float64))
    test = Dataset(data=rng.random((10, 5)).astype(np.float64))
    reducer = PCAReducer(n_components=2).fit(train)
    out = reducer.transform(test)
    assert out.data.shape == (10, 2)


# ---------------------------------------------------------------------------
# Pipeline.fit / transform
# ---------------------------------------------------------------------------

def test_pipeline_fit_returns_self():
    pipeline = Pipeline(encoder=AngleEncoder())
    result = pipeline.fit(_ds())
    assert result is pipeline


def test_pipeline_transform_requires_fit():
    with pytest.raises(RuntimeError, match="not been fitted"):
        Pipeline(encoder=AngleEncoder()).transform(_ds())


def test_pipeline_fit_transform_equiv():
    ds = _ds()
    p1 = Pipeline(encoder=AngleEncoder())
    r1 = p1.fit_transform(ds)

    p2 = Pipeline(encoder=AngleEncoder())
    p2.fit(ds)
    r2 = p2.transform(ds)

    # Both should produce the same encoded parameters
    np.testing.assert_allclose(
        r1.encoded[0].parameters,
        r2.encoded[0].parameters,
    )


def test_pipeline_get_params():
    enc = AngleEncoder()
    p = Pipeline(encoder=enc)
    params = p.get_params()
    assert params["encoder"] is enc
    assert "cleaner" in params
    assert "schema" in params


def test_pipeline_set_params():
    p = Pipeline()
    enc = AngleEncoder()
    p.set_params(encoder=enc)
    assert p.encoder is enc


def test_pipeline_set_params_invalid():
    with pytest.raises(ValueError, match="Invalid parameter"):
        Pipeline().set_params(nonexistent=42)


# ---------------------------------------------------------------------------
# Pipeline with cleaner (train/test split)
# ---------------------------------------------------------------------------

def test_pipeline_cleaner_train_test():
    rng = np.random.default_rng(7)
    data_train = rng.random((40, 4)).astype(np.float64)
    data_test = rng.random((10, 4)).astype(np.float64)
    data_train[2, 0] = np.nan

    train_ds = Dataset(data=data_train)
    test_ds = Dataset(data=data_test)

    pipeline = Pipeline(cleaner=Imputer(strategy="mean"), encoder=AngleEncoder())
    pipeline.fit(train_ds)
    result = pipeline.transform(test_ds)
    assert isinstance(result, PipelineResult)
    assert result.encoded is not None
    assert len(result.encoded) == 10


# ---------------------------------------------------------------------------
# Pipeline returns PipelineResult
# ---------------------------------------------------------------------------

def test_pipeline_result_type():
    result = Pipeline(encoder=AngleEncoder()).fit_transform(_ds())
    assert isinstance(result, PipelineResult)
    assert result.encoded is not None


# ---------------------------------------------------------------------------
# Scaler.inverse_transform
# ---------------------------------------------------------------------------

def test_inverse_transform_minmax_roundtrip():
    ds = _ds()
    s = Scaler(strategy="minmax")
    scaled = s.fit_transform(ds)
    recovered = s.inverse_transform(scaled)
    np.testing.assert_allclose(recovered.data, ds.data, atol=1e-10)


def test_inverse_transform_zscore_roundtrip():
    ds = _ds()
    s = Scaler(strategy="zscore")
    scaled = s.fit_transform(ds)
    recovered = s.inverse_transform(scaled)
    np.testing.assert_allclose(recovered.data, ds.data, atol=1e-10)


def test_inverse_transform_minmax_pi_roundtrip():
    ds = _ds()
    s = Scaler(strategy="minmax_pi")
    scaled = s.fit_transform(ds)
    recovered = s.inverse_transform(scaled)
    np.testing.assert_allclose(recovered.data, ds.data, atol=1e-10)


def test_inverse_transform_non_invertible_raises():
    ds = _ds()
    s = Scaler(strategy="l2")
    s.fit(ds)
    with pytest.raises(ValueError, match="not invertible"):
        s.inverse_transform(s.transform(ds))


def test_inverse_transform_not_fitted_raises():
    with pytest.raises(NotFittedError):
        Scaler(strategy="minmax").inverse_transform(_ds())


# ---------------------------------------------------------------------------
# FeatureSelector.get_feature_names_out
# ---------------------------------------------------------------------------

def test_get_feature_names_out_returns_kept_names():
    ds = Dataset(
        data=np.random.default_rng(0).random((20, 4)).astype(np.float64),
        feature_names=["a", "b", "c", "d"],
        feature_types=["continuous"] * 4,
    )
    sel = FeatureSelector(method="variance", threshold=0.0).fit(ds)
    names = sel.get_feature_names_out()
    assert isinstance(names, list)
    assert all(n in ["a", "b", "c", "d"] for n in names)


def test_get_feature_names_out_not_fitted_raises():
    with pytest.raises(NotFittedError):
        FeatureSelector().get_feature_names_out()


# ---------------------------------------------------------------------------
# OutlierHandler.outlier_mask_
# ---------------------------------------------------------------------------

def test_outlier_mask_set_after_transform():
    rng = np.random.default_rng(0)
    data = rng.random((30, 3)).astype(np.float64)
    data[0, 0] = 1000.0  # clear outlier
    ds = Dataset(data=data, feature_types=["continuous"] * 3)
    handler = OutlierHandler(method="iqr", action="remove")
    handler.fit(ds)
    handler.transform(ds)
    assert handler.outlier_mask_ is not None
    assert handler.outlier_mask_.dtype == bool
    assert handler.outlier_mask_.any()  # at least one outlier detected


def test_outlier_mask_none_before_transform():
    ds = _ds()
    handler = OutlierHandler()
    handler.fit(ds)
    assert handler.outlier_mask_ is None
