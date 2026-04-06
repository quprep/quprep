"""Graph ingestion — converts graph data to feature vectors (lossy path)."""

from __future__ import annotations

import numpy as np

from quprep.core.dataset import Dataset

_VALID_FEATURES = ("laplacian_eigenvalues", "degree", "all")


class GraphIngester:
    """
    Convert graph data into a Dataset of feature vectors (lossy path).

    Extracts a fixed-size feature vector from each graph so that existing
    encoders (AngleEncoder, AmplitudeEncoder, etc.) can be applied without
    modification. Features are drawn from the graph's Laplacian spectrum and
    degree sequence — both are proven to carry structural information relevant
    to graph classification tasks.

    For the structure-preserving (lossless) path, use
    :class:`~quprep.encode.graph_state.GraphStateEncoder` directly.

    Parameters
    ----------
    features : str
        Which features to extract:

        - ``'laplacian_eigenvalues'`` — sorted eigenvalues of the normalized
          Laplacian (captures global topology).
        - ``'degree'`` — sorted node degree sequence.
        - ``'all'`` (default) — concatenation of both.
    n_features : int or None
        Pad or truncate the feature vector to exactly this length.
        If ``None``, the vector length equals the number of nodes
        (or 2 × n_nodes for ``'all'``). Useful when loading a batch of
        graphs with different sizes.

    Examples
    --------
    From a NumPy adjacency matrix::

        import numpy as np
        import quprep as qd

        adj = np.array([[0,1,1],[1,0,1],[1,1,0]], dtype=float)
        dataset = qd.GraphIngester().load(adj)

    From a networkx graph::

        import networkx as nx
        G = nx.karate_club_graph()
        dataset = qd.GraphIngester(n_features=16).load(G)

    Batch of graphs (list)::

        graphs = [nx.path_graph(5), nx.cycle_graph(6), nx.complete_graph(4)]
        dataset = qd.GraphIngester(n_features=8).load(graphs)
        print(dataset.data.shape)   # (3, 8)
    """

    def __init__(
        self,
        features: str = "all",
        n_features: int | None = None,
    ):
        if features not in _VALID_FEATURES:
            raise ValueError(f"features must be one of {_VALID_FEATURES}, got '{features}'")
        self.features = features
        self.n_features = n_features

    def load(self, source) -> Dataset:
        """
        Load graph(s) and return a Dataset of feature vectors.

        Parameters
        ----------
        source : np.ndarray, networkx.Graph, or list
            - **np.ndarray** — square adjacency matrix, shape ``(n, n)``.
            - **networkx.Graph / DiGraph** — converted to adjacency matrix
              internally (requires ``networkx``; pure NumPy path available).
            - **list** — each element is a graph (ndarray or networkx Graph);
              all are embedded and stacked into a single Dataset.

        Returns
        -------
        Dataset
            ``data`` shape is ``(n_graphs, n_features)``.
            ``metadata["modality"]`` is ``"graph"``.
            ``metadata["features"]`` is the feature set used.
            ``metadata["n_nodes"]`` is a list of node counts per graph.

        Raises
        ------
        ValueError
            If the adjacency matrix is not square, or no graphs are provided.
        """
        if isinstance(source, list):
            if not source:
                raise ValueError("Empty graph list.")
            vectors = [self._graph_to_vec(g) for g in source]
            n_nodes = [self._n_nodes(g) for g in source]
        else:
            vectors = [self._graph_to_vec(source)]
            n_nodes = [self._n_nodes(source)]

        # pad / truncate to common length
        target_len = self.n_features or max(v.shape[0] for v in vectors)
        padded = np.stack([self._fit_length(v, target_len) for v in vectors], axis=0)

        n_feat = padded.shape[1]
        return Dataset(
            data=padded,
            feature_names=[f"gfeat_{i}" for i in range(n_feat)],
            feature_types=["continuous"] * n_feat,
            metadata={
                "modality": "graph",
                "features": self.features,
                "n_nodes": n_nodes,
                "n_features": n_feat,
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _graph_to_vec(self, graph) -> np.ndarray:
        """Extract a feature vector from a single graph."""
        adj = self._to_adj(graph)
        parts = []
        if self.features in ("laplacian_eigenvalues", "all"):
            parts.append(self._laplacian_eigenvalues(adj))
        if self.features in ("degree", "all"):
            parts.append(self._degree_sequence(adj))
        return np.concatenate(parts)

    def _to_adj(self, graph) -> np.ndarray:
        """Convert any supported graph type to a square float adjacency matrix."""
        if isinstance(graph, np.ndarray):
            if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
                raise ValueError(
                    f"Adjacency matrix must be square 2-D, got shape {graph.shape}"
                )
            return graph.astype(float)
        # Try networkx
        try:
            import networkx as nx
            if isinstance(graph, (nx.Graph, nx.DiGraph)):
                return nx.to_numpy_array(graph, dtype=float)
        except ImportError:
            pass
        raise TypeError(
            f"Unsupported graph type '{type(graph).__name__}'. "
            "Pass a square np.ndarray or a networkx Graph."
        )

    def _n_nodes(self, graph) -> int:
        if isinstance(graph, np.ndarray):
            return graph.shape[0]
        try:
            import networkx as nx
            if isinstance(graph, (nx.Graph, nx.DiGraph)):
                return graph.number_of_nodes()
        except ImportError:
            pass
        return 0

    def _laplacian_eigenvalues(self, adj: np.ndarray) -> np.ndarray:
        """Sorted eigenvalues of the normalized Laplacian (ascending)."""
        n = adj.shape[0]
        degree = adj.sum(axis=1)
        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(n) - D_inv_sqrt @ adj @ D_inv_sqrt
        eigvals = np.linalg.eigvalsh(L)
        return np.sort(eigvals)

    def _degree_sequence(self, adj: np.ndarray) -> np.ndarray:
        """Sorted node degree sequence (descending)."""
        degrees = adj.sum(axis=1)
        return np.sort(degrees)[::-1]

    @staticmethod
    def _fit_length(vec: np.ndarray, length: int) -> np.ndarray:
        """Pad with zeros or truncate to exactly `length`."""
        if len(vec) >= length:
            return vec[:length]
        return np.concatenate([vec, np.zeros(length - len(vec))])
