r"""Graph state encoding — structure-preserving lossless path.

Mathematical formulation
------------------------
A graph state :math:`|G\rangle` for graph :math:`G = (V, E)` with
:math:`|V| = n` nodes is defined as:

.. math::

    |G\rangle = \prod_{(i,j) \in E} CZ_{ij} \, |+\rangle^{\otimes n}

where :math:`|+\rangle = H|0\rangle`. The circuit construction is:

1. Apply Hadamard to every qubit: :math:`H^{\otimes n}|0\rangle^n`
2. Apply a CZ gate for every edge :math:`(i,j) \in E`

Properties
----------
Qubits : n = number of nodes
Depth  : O(|E|) — one CZ layer per edge (can be parallelized by colouring)
NISQ   : Moderate — scales with edges, not nodes squared

References
----------
Hein, M., et al. (2004). Multiparty entanglement in graph states.
    *Physical Review A*, 69(6), 062311.
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class GraphStateEncoder(BaseEncoder):
    """
    Encode a graph as a graph state circuit (lossless path).

    Produces :math:`|G\\rangle = \\prod_{(i,j) \\in E} CZ_{ij} H^{\\otimes n} |0\\rangle^n`.
    The circuit preserves the full graph structure — every edge becomes a
    CZ entangling gate.

    Two usage patterns:

    **Pipeline path** (recommended) — pair with
    :class:`~quprep.ingest.graph_ingester.GraphIngester` using
    ``features='adjacency'``::

        dataset = GraphIngester(features="adjacency").load(adj)
        result  = Pipeline(encoder=GraphStateEncoder()).fit_transform(dataset)

    **Direct path** — pass the adjacency matrix directly::

        result = GraphStateEncoder().encode_graph(adj)

    For feature-based (lossy) encoding, use
    :class:`~quprep.ingest.graph_ingester.GraphIngester` with any standard
    encoder instead.

    Examples
    --------
    Direct graph encoding::

        import numpy as np
        import quprep as qd

        adj = np.array([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], dtype=float)
        encoder = qd.GraphStateEncoder()
        result = encoder.encode_graph(adj)
        print(result.metadata["n_qubits"])   # 4
        print(result.metadata["edges"])      # [(0,1),(0,2),(1,2),(2,3)]
    """

    @property
    def n_qubits(self):
        return None  # data-dependent

    @property
    def depth(self):
        return "O(|E|)"

    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode a flattened upper-triangle adjacency vector as a graph state.

        Parameters
        ----------
        x : np.ndarray, shape (n*(n-1)//2,)
            Flattened upper triangle of the adjacency matrix (row-major,
            values thresholded at 0.5 to determine edge presence).
            Use :meth:`encode_graph` to pass an adjacency matrix directly.

        Returns
        -------
        EncodedResult
        """
        x = np.asarray(x, dtype=float)
        # reconstruct n from triangular number: k = n*(n-1)/2
        k = len(x)
        n = int((1 + np.sqrt(1 + 8 * k)) / 2)
        if n * (n - 1) // 2 != k:
            raise ValueError(
                f"Input length {k} is not a valid upper-triangle size. "
                "Expected n*(n-1)//2 for some integer n."
            )
        adj = np.zeros((n, n))
        idx = np.triu_indices(n, k=1)
        adj[idx] = x
        adj = adj + adj.T
        return self._from_adj(adj)

    def encode_graph(self, adj: np.ndarray) -> EncodedResult:
        """
        Encode directly from a square adjacency matrix.

        Parameters
        ----------
        adj : np.ndarray, shape (n, n)
            Square adjacency matrix. Values > 0.5 are treated as edges.

        Returns
        -------
        EncodedResult
        """
        adj = np.asarray(adj, dtype=float)
        if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
            raise ValueError(f"adj must be a square 2-D array, got shape {adj.shape}")
        return self._from_adj(adj)

    def encode_batch_graphs(self, graphs: list[np.ndarray]) -> list[EncodedResult]:
        """
        Encode a list of adjacency matrices.

        Parameters
        ----------
        graphs : list of np.ndarray
            Each element is a square adjacency matrix.

        Returns
        -------
        list of EncodedResult
        """
        return [self.encode_graph(g) for g in graphs]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _from_adj(self, adj: np.ndarray) -> EncodedResult:
        n = adj.shape[0]
        edges = [
            (int(i), int(j))
            for i in range(n)
            for j in range(i + 1, n)
            if adj[i, j] > 0.5
        ]
        # parameters: flattened upper triangle (for serialization)
        idx = np.triu_indices(n, k=1)
        params = adj[idx].copy()
        return EncodedResult(
            parameters=params,
            metadata={
                "encoding": "graph_state",
                "n_qubits": n,
                "edges": edges,
                "n_edges": len(edges),
                "depth": len(edges),
            },
        )
