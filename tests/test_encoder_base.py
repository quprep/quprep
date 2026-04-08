"""Tests for BaseEncoder and EncodedResult infrastructure."""

import numpy as np

from quprep.core.dataset import Dataset
from quprep.encode.angle import AngleEncoder
from quprep.encode.base import EncodedResult

# ---------------------------------------------------------------------------
# EncodedResult
# ---------------------------------------------------------------------------

def test_encoded_result_defaults():
    r = EncodedResult(parameters=np.array([1.0, 2.0]))
    assert r.circuit_fn is None
    assert r.metadata == {}


def test_encoded_result_with_circuit_fn():
    def fn(fw):
        return f"circuit({fw})"
    r = EncodedResult(parameters=np.array([0.5]), circuit_fn=fn, metadata={"n_qubits": 1})
    assert r.circuit_fn("qasm") == "circuit(qasm)"
    assert r.metadata["n_qubits"] == 1


def test_encoded_result_metadata_defaults_to_empty():
    r = EncodedResult(parameters=np.array([]))
    assert isinstance(r.metadata, dict)


# ---------------------------------------------------------------------------
# BaseEncoder.encode_batch via AngleEncoder
# ---------------------------------------------------------------------------

def test_encode_batch_returns_one_per_sample():
    enc = AngleEncoder()
    ds = Dataset(
        data=np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        feature_names=["a", "b", "c"],
    )
    results = enc.encode_batch(ds)
    assert len(results) == 2
    assert all(isinstance(r, EncodedResult) for r in results)


def test_encode_batch_correct_qubit_count():
    enc = AngleEncoder()
    ds = Dataset(data=np.ones((5, 4), dtype=np.float64))
    results = enc.encode_batch(ds)
    assert all(r.metadata["n_qubits"] == 4 for r in results)


# ---------------------------------------------------------------------------
# Dataset properties and copy
# ---------------------------------------------------------------------------

def test_dataset_n_categorical():
    ds = Dataset(
        data=np.ones((3, 2), dtype=np.float64),
        categorical_data={"color": ["red", "green", "blue"]},
    )
    assert ds.n_categorical == 1


def test_dataset_copy_labels_none():
    ds = Dataset(data=np.ones((2, 2), dtype=np.float64))
    copy = ds.copy()
    assert copy.labels is None


def test_dataset_copy_with_labels():
    labels = np.array([0.0, 1.0])
    ds = Dataset(data=np.ones((2, 2), dtype=np.float64), labels=labels)
    copy = ds.copy()
    copy.labels[0] = 99.0
    assert ds.labels[0] == 0.0  # original unaffected
