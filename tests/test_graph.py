"""Tests for GraphIngester and GraphStateEncoder (v0.7.0)."""

from __future__ import annotations

import numpy as np
import pytest


def _triangle():
    """3-node complete graph adjacency matrix."""
    return np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)


def _path4():
    """4-node path graph: 0-1-2-3."""
    a = np.zeros((4, 4))
    a[0, 1] = a[1, 0] = 1
    a[1, 2] = a[2, 1] = 1
    a[2, 3] = a[3, 2] = 1
    return a


# ===========================================================================
# GraphIngester — lossy path
# ===========================================================================

class TestGraphIngesterAdjacency:
    def test_basic_shape(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester().load(_triangle())
        assert ds.data.shape == (1, 6)  # all: 3 eigenvalues + 3 degrees

    def test_laplacian_eigenvalues_only(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester(features="laplacian_eigenvalues").load(_triangle())
        assert ds.data.shape == (1, 3)

    def test_degree_only(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester(features="degree").load(_triangle())
        assert ds.data.shape == (1, 3)
        # All degrees = 2 for complete graph
        np.testing.assert_array_almost_equal(ds.data[0], [2, 2, 2])

    def test_modality_metadata(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester().load(_triangle())
        assert ds.metadata["modality"] == "graph"
        assert ds.metadata["features"] == "all"

    def test_n_nodes_metadata(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester().load(_triangle())
        assert ds.metadata["n_nodes"] == [3]

    def test_feature_names(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester().load(_triangle())
        assert ds.feature_names[0] == "gfeat_0"

    def test_non_square_raises(self):
        from quprep.ingest.graph_ingester import GraphIngester
        with pytest.raises(ValueError, match="square"):
            GraphIngester().load(np.ones((3, 4)))

    def test_invalid_features_raises(self):
        from quprep.ingest.graph_ingester import GraphIngester
        with pytest.raises(ValueError, match="features must be one of"):
            GraphIngester(features="bad")


class TestGraphIngesterNFeatures:
    def test_pad_to_n_features(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester(n_features=8).load(_triangle())
        assert ds.data.shape == (1, 8)

    def test_truncate_to_n_features(self):
        from quprep.ingest.graph_ingester import GraphIngester
        ds = GraphIngester(n_features=2).load(_triangle())
        assert ds.data.shape == (1, 2)

    def test_batch_consistent_shape(self):
        from quprep.ingest.graph_ingester import GraphIngester
        graphs = [_triangle(), _path4()]
        ds = GraphIngester(n_features=8).load(graphs)
        assert ds.data.shape == (2, 8)

    def test_empty_list_raises(self):
        from quprep.ingest.graph_ingester import GraphIngester
        with pytest.raises(ValueError, match="Empty"):
            GraphIngester().load([])


class TestGraphIngesterNetworkx:
    def test_networkx_graph(self):
        nx = pytest.importorskip("networkx")
        from quprep.ingest.graph_ingester import GraphIngester
        G = nx.cycle_graph(5)
        ds = GraphIngester(n_features=8).load(G)
        assert ds.data.shape == (1, 8)

    def test_networkx_batch(self):
        nx = pytest.importorskip("networkx")
        from quprep.ingest.graph_ingester import GraphIngester
        graphs = [nx.path_graph(4), nx.complete_graph(3), nx.cycle_graph(5)]
        ds = GraphIngester(n_features=10).load(graphs)
        assert ds.data.shape == (3, 10)

    def test_unsupported_type_raises(self):
        from quprep.ingest.graph_ingester import GraphIngester
        with pytest.raises(TypeError, match="Unsupported graph type"):
            GraphIngester().load("not_a_graph")


class TestGraphIngesterPipeline:
    def test_pipeline_lossy(self):
        import quprep as qd
        ds = qd.GraphIngester(n_features=8).load(_triangle())
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder())
        result = pipeline.fit_transform(ds)
        assert len(result.encoded) == 1
        assert result.encoded[0].metadata["n_qubits"] == 8

    def test_pipeline_batch_lossy(self):
        import quprep as qd
        graphs = [_triangle(), _path4(), _path4()]
        ds = qd.GraphIngester(n_features=6).load(graphs)
        pipeline = qd.Pipeline(encoder=qd.AngleEncoder())
        result = pipeline.fit_transform(ds)
        assert len(result.encoded) == 3


# ===========================================================================
# GraphStateEncoder — lossless path
# ===========================================================================

class TestGraphStateEncoder:
    def test_encode_graph_shape(self):
        from quprep.encode.graph_state import GraphStateEncoder
        enc = GraphStateEncoder()
        result = enc.encode_graph(_triangle())
        assert result.metadata["n_qubits"] == 3
        assert result.metadata["n_edges"] == 3

    def test_edges_correct(self):
        from quprep.encode.graph_state import GraphStateEncoder
        enc = GraphStateEncoder()
        result = enc.encode_graph(_path4())
        assert result.metadata["edges"] == [(0, 1), (1, 2), (2, 3)]
        assert result.metadata["n_edges"] == 3

    def test_encoding_metadata(self):
        from quprep.encode.graph_state import GraphStateEncoder
        result = GraphStateEncoder().encode_graph(_triangle())
        assert result.metadata["encoding"] == "graph_state"

    def test_empty_graph_no_edges(self):
        from quprep.encode.graph_state import GraphStateEncoder
        adj = np.zeros((4, 4))
        result = GraphStateEncoder().encode_graph(adj)
        assert result.metadata["n_edges"] == 0
        assert result.metadata["n_qubits"] == 4

    def test_non_square_raises(self):
        from quprep.encode.graph_state import GraphStateEncoder
        with pytest.raises(ValueError, match="square"):
            GraphStateEncoder().encode_graph(np.ones((3, 4)))

    def test_encode_from_upper_triangle(self):
        from quprep.encode.graph_state import GraphStateEncoder
        # triangle: upper triangle = [1, 1, 1]
        enc = GraphStateEncoder()
        result = enc.encode(np.array([1.0, 1.0, 1.0]))
        assert result.metadata["n_qubits"] == 3
        assert result.metadata["n_edges"] == 3

    def test_invalid_upper_triangle_raises(self):
        from quprep.encode.graph_state import GraphStateEncoder
        with pytest.raises(ValueError, match="not a valid upper-triangle"):
            GraphStateEncoder().encode(np.array([1.0, 1.0]))

    def test_encode_batch_graphs(self):
        from quprep.encode.graph_state import GraphStateEncoder
        enc = GraphStateEncoder()
        results = enc.encode_batch_graphs([_triangle(), _path4()])
        assert len(results) == 2
        assert results[0].metadata["n_qubits"] == 3
        assert results[1].metadata["n_qubits"] == 4

    def test_n_qubits_property(self):
        from quprep.encode.graph_state import GraphStateEncoder
        assert GraphStateEncoder().n_qubits is None

    def test_depth_property(self):
        from quprep.encode.graph_state import GraphStateEncoder
        assert "E" in GraphStateEncoder().depth


# ===========================================================================
# QASM export — graph state
# ===========================================================================

class TestGraphStateQASM:
    def test_qasm_contains_h_gates(self):
        from quprep.encode.graph_state import GraphStateEncoder
        from quprep.export.qasm_export import QASMExporter
        result = GraphStateEncoder().encode_graph(_triangle())
        qasm = QASMExporter().export(result)
        assert qasm.count("h q[") == 3

    def test_qasm_contains_cz_gates(self):
        from quprep.encode.graph_state import GraphStateEncoder
        from quprep.export.qasm_export import QASMExporter
        result = GraphStateEncoder().encode_graph(_triangle())
        qasm = QASMExporter().export(result)
        assert qasm.count("cz q[") == 3

    def test_qasm_no_edges(self):
        from quprep.encode.graph_state import GraphStateEncoder
        from quprep.export.qasm_export import QASMExporter
        result = GraphStateEncoder().encode_graph(np.zeros((3, 3)))
        qasm = QASMExporter().export(result)
        assert "cz" not in qasm
        assert qasm.count("h q[") == 3

    def test_qasm_header(self):
        from quprep.encode.graph_state import GraphStateEncoder
        from quprep.export.qasm_export import QASMExporter
        result = GraphStateEncoder().encode_graph(_path4())
        qasm = QASMExporter().export(result)
        assert qasm.startswith("OPENQASM 3.0;")
        assert "qubit[4] q;" in qasm
