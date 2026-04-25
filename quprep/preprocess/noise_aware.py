"""Noise-aware preprocessing — assign features to qubits to minimise circuit error.

Three optimisations are applied in sequence during ``fit`` / ``transform``:

1. **Qubit assignment** — feature variances are ranked; the highest-variance
   feature is assigned to the least-noisy physical qubit so that the most
   informative features are least corrupted by gate/readout errors.

2. **Topology-aware reordering** — for entangled encodings (angle+CNOT,
   IQP, ZZ, Pauli, re-upload) the selected qubits are greedily threaded
   into a path through the coupling map so that neighbouring logical qubits
   are physically adjacent.  This minimises the number of SWAP gates
   inserted by hardware compilers.

3. **Angle dead-zone remapping** (optional) — when ``angle_deadzone > 0``
   and the encoding uses rotation angles, data values are linearly remapped
   from ``[0, π]`` to ``[deadzone·π, (1−deadzone)·π]``.  The poles 0 and π
   correspond to computational basis states (no superposition) where
   single-qubit gates are least discriminative under noise.  Apply this
   **after** normalising the dataset to ``[0, π]``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from quprep.core.dataset import Dataset


@dataclass
class NoiseProfile:
    """
    Noise characteristics of a quantum backend.

    Parameters
    ----------
    qubit_error_rates : list of float
        Per-qubit single-qubit gate or readout error rate (lower = better).
        Length must equal the number of physical qubits on the device.
    coupling_map : list of (int, int)
        Available two-qubit connections.  Each pair ``(i, j)`` means a
        native two-qubit gate can run directly between physical qubits
        *i* and *j* without an inserted SWAP.  Pass an empty list for
        all-to-all devices (trapped-ion) or for single-qubit-only encodings.
    t1 : list of float, optional
        T1 relaxation times in microseconds, one per qubit.  Longer is better.
    t2 : list of float, optional
        T2 dephasing times in microseconds, one per qubit.  Longer is better.
    cx_error_rates : dict mapping (int, int) → float, optional
        Per-pair two-qubit (CX/CNOT) gate error rates.  Stored for
        informational purposes; the qubit-assignment algorithm uses
        ``qubit_error_rates`` and T1/T2 as the primary quality signal.

    Examples
    --------
    >>> profile = NoiseProfile(
    ...     qubit_error_rates=[0.001, 0.002, 0.003, 0.001, 0.002],
    ...     coupling_map=[(0, 1), (1, 2), (2, 3), (3, 4)],
    ...     t1=[150.0, 120.0, 180.0, 160.0, 140.0],
    ...     t2=[80.0, 70.0, 90.0, 85.0, 75.0],
    ... )
    >>> profile.n_qubits
    5
    """

    qubit_error_rates: list[float]
    coupling_map: list[tuple[int, int]]
    t1: list[float] | None = None
    t2: list[float] | None = None
    cx_error_rates: dict | None = None

    def __post_init__(self) -> None:
        n = len(self.qubit_error_rates)
        if self.t1 is not None and len(self.t1) != n:
            raise ValueError(
                f"t1 must have length {n} (one entry per qubit), got {len(self.t1)}"
            )
        if self.t2 is not None and len(self.t2) != n:
            raise ValueError(
                f"t2 must have length {n} (one entry per qubit), got {len(self.t2)}"
            )
        for a, b in self.coupling_map:
            if a >= n or b >= n or a < 0 or b < 0:
                raise ValueError(
                    f"Coupling map references qubit {max(a, b)} but "
                    f"qubit_error_rates has only {n} entries (indices 0–{n - 1})."
                )

    @property
    def n_qubits(self) -> int:
        """Number of physical qubits described by this profile."""
        return len(self.qubit_error_rates)

    def qubit_score(self, qubit: int) -> float:
        """
        Combined quality score for a single qubit.  **Lower is better.**

        Combines gate/readout error rate with inverse coherence times so
        that shorter T1/T2 (noisier qubits) produce a higher (worse) score.

        Parameters
        ----------
        qubit : int
            Physical qubit index.

        Returns
        -------
        float
            Quality score.  Qubits with lower scores should be preferred
            for high-variance features.
        """
        score = self.qubit_error_rates[qubit]
        if self.t1 is not None and self.t1[qubit] > 0:
            score += 1.0 / self.t1[qubit]
        if self.t2 is not None and self.t2[qubit] > 0:
            score += 1.0 / self.t2[qubit]
        return score


class NoiseAwarePreprocessor:
    """
    Reorder dataset features for noise-aware qubit assignment.

    Given a backend :class:`NoiseProfile`, this transformer:

    * assigns high-variance features to the least-noisy physical qubits,
    * reorders the selected qubits to form a path through the hardware
      coupling map (minimising SWAP overhead for entangled encodings), and
    * optionally remaps angle-encoded values away from 0 and π
      (requires data already normalised to ``[0, π]``).

    The output is a :class:`~quprep.core.dataset.Dataset` whose columns
    are reordered so that column *i* maps to the *i*-th qubit in
    :attr:`qubit_assignment_`.  Downstream encoders that follow the
    standard ``feature i → logical qubit i`` convention will therefore
    automatically use the noise-optimised assignment.

    Parameters
    ----------
    noise_profile : NoiseProfile
        Noise characteristics of the target backend.
    encoding : str
        Target encoding name.  Used to select the interaction graph for
        SWAP minimisation and to decide whether angle remapping applies.
        Recognised values: ``'angle'``, ``'entangled_angle'``, ``'basis'``,
        ``'amplitude'``, ``'iqp'``, ``'zz_feature_map'``,
        ``'pauli_feature_map'``, ``'reupload'``, ``'tensor_product'``.
        Unknown values are treated as single-qubit (no SWAP pressure).
    angle_deadzone : float
        Fraction of ``[0, π]`` to exclude at each pole.  For example,
        ``0.05`` remaps data from ``[0, π]`` to ``[0.05π, 0.95π]``, keeping
        all encoded angles at least 5 % away from the computational-basis
        poles.  Must be in ``[0, 0.5)``.  Default ``0`` (no remapping).
        Only applied when ``encoding`` uses rotation angles.  The input
        data must already be normalised to ``[0, π]`` for this to be
        meaningful; use :class:`~quprep.normalize.scalers.Scaler` with
        ``method='minmax_pi'`` before applying this transformer.

    Attributes
    ----------
    permutation_ : np.ndarray of int, shape (n_features,)
        ``permutation_[i]`` is the original feature index placed at output
        column *i*.  Available after :meth:`fit`.
    qubit_assignment_ : list of int, length n_features
        ``qubit_assignment_[j]`` is the physical qubit assigned to original
        feature *j*.  Available after :meth:`fit`.
    estimated_swaps_before_ : int
        Estimated number of SWAP gates needed without topology optimisation.
        Available after :meth:`fit`.
    estimated_swaps_after_ : int
        Estimated number of SWAP gates needed after topology optimisation.
        Always ≤ ``estimated_swaps_before_``.  Available after :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> from quprep.core.dataset import Dataset
    >>> from quprep.preprocess.noise_aware import NoiseAwarePreprocessor, NoiseProfile
    >>>
    >>> profile = NoiseProfile(
    ...     qubit_error_rates=[0.001, 0.005, 0.002, 0.003],
    ...     coupling_map=[(0, 1), (1, 2), (2, 3)],
    ... )
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((100, 3))
    >>> data[:, 2] *= 5          # feature 2 has highest variance
    >>> ds = Dataset(data=data, feature_names=["a", "b", "c"])
    >>> prep = NoiseAwarePreprocessor(profile, encoding="entangled_angle")
    >>> result = prep.fit_transform(ds)
    >>> prep.qubit_assignment_[2]   # high-variance feature → low-error qubit
    0
    """

    _ANGLE_ENCODINGS: frozenset[str] = frozenset({
        "angle", "entangled_angle", "reupload", "tensor_product",
        "iqp", "zz_feature_map", "pauli_feature_map",
    })
    _ENTANGLED_ENCODINGS: frozenset[str] = frozenset({
        "entangled_angle", "iqp", "zz_feature_map",
        "pauli_feature_map", "reupload",
    })

    def __init__(
        self,
        noise_profile: NoiseProfile,
        encoding: str = "angle",
        angle_deadzone: float = 0.0,
    ) -> None:
        if not 0.0 <= angle_deadzone < 0.5:
            raise ValueError(
                f"angle_deadzone must be in [0, 0.5), got {angle_deadzone}"
            )
        self.noise_profile = noise_profile
        self.encoding = encoding
        self.angle_deadzone = angle_deadzone
        self._fitted = False

        self.permutation_: np.ndarray | None = None
        self.qubit_assignment_: list[int] | None = None
        self.estimated_swaps_before_: int | None = None
        self.estimated_swaps_after_: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, dataset: Dataset) -> NoiseAwarePreprocessor:
        """
        Compute the noise-optimal feature permutation from dataset variances.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.  Per-feature variances are computed from
            ``dataset.data``.

        Returns
        -------
        NoiseAwarePreprocessor
            Returns ``self`` for chaining.

        Raises
        ------
        ValueError
            If the dataset has more features than qubits in the noise profile.
        """
        n_feat = dataset.n_features
        n_qubits = self.noise_profile.n_qubits

        if n_feat > n_qubits:
            raise ValueError(
                f"Dataset has {n_feat} features — more features than qubits "
                f"({n_qubits}) in the noise profile.  Reduce the feature count "
                "first (e.g. with PCAReducer or HardwareAwareReducer)."
            )

        variances = np.var(dataset.data, axis=0)

        qubit_scores = np.array([
            self.noise_profile.qubit_score(q) for q in range(n_qubits)
        ])
        # Best n_feat qubits, ranked by ascending score (lowest noise first).
        best_qubits_naive: list[int] = np.argsort(qubit_scores)[:n_feat].tolist()

        # SWAP pressure only exists for encodings that insert 2-qubit gates.
        if self.encoding in self._ENTANGLED_ENCODINGS:
            self.estimated_swaps_before_ = self._count_adjacent_swaps(best_qubits_naive)
            if self.noise_profile.coupling_map:
                qubit_path = self._connectivity_path(best_qubits_naive, qubit_scores)
                self.estimated_swaps_after_ = self._count_adjacent_swaps(qubit_path)
            else:
                qubit_path = best_qubits_naive
                self.estimated_swaps_after_ = 0  # all-to-all / no topology = no SWAPs
        else:
            qubit_path = best_qubits_naive
            self.estimated_swaps_before_ = 0
            self.estimated_swaps_after_ = 0

        # Assign: position 0 in path (least noisy) ← feature with highest variance.
        feat_order = np.argsort(variances)[::-1]  # indices, highest variance first
        self.permutation_ = feat_order.copy()

        self.qubit_assignment_ = [0] * n_feat
        for position, feat_idx in enumerate(feat_order):
            self.qubit_assignment_[int(feat_idx)] = qubit_path[position]

        self._fitted = True
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Reorder columns according to the fitted permutation.

        Parameters
        ----------
        dataset : Dataset
            Input dataset.  Must have the same feature count as the dataset
            used in :meth:`fit`.

        Returns
        -------
        Dataset
            Dataset with columns reordered so that column *i* corresponds
            to ``qubit_assignment_[original_feature_i]``.  Metadata is
            updated with noise-aware routing information.  If
            ``angle_deadzone > 0`` and the encoding uses rotation angles,
            all values are linearly remapped from ``[0, π]`` to the
            interior ``[deadzone·π, (1−deadzone)·π]``.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If :meth:`fit` has not been called.
        ValueError
            If the dataset's feature count differs from the fitted count.
        """
        from sklearn.exceptions import NotFittedError

        if not self._fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit()' before 'transform()'."
            )

        n_feat = dataset.n_features
        if n_feat != len(self.permutation_):
            raise ValueError(
                f"Dataset has {n_feat} features but this transformer was "
                f"fitted on {len(self.permutation_)} features."
            )

        X = dataset.data[:, self.permutation_]

        if self.angle_deadzone > 0.0 and self.encoding in self._ANGLE_ENCODINGS:
            lo = self.angle_deadzone * np.pi
            hi = (1.0 - self.angle_deadzone) * np.pi
            X = lo + (X / np.pi) * (hi - lo)

        feat_names = (
            [dataset.feature_names[int(i)] for i in self.permutation_]
            if dataset.feature_names
            else []
        )
        feat_types = (
            [dataset.feature_types[int(i)] for i in self.permutation_]
            if dataset.feature_types
            else []
        )

        meta = dict(dataset.metadata)
        meta.update({
            "noise_aware": True,
            "qubit_assignment": list(self.qubit_assignment_),
            "angle_deadzone": self.angle_deadzone,
            "encoding": self.encoding,
            "estimated_swaps_before": self.estimated_swaps_before_,
            "estimated_swaps_after": self.estimated_swaps_after_,
        })

        return Dataset(
            data=X,
            feature_names=feat_names,
            feature_types=feat_types,
            categorical_data=dict(dataset.categorical_data),
            metadata=meta,
            labels=dataset.labels,
        )

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """Fit and transform in one call."""
        return self.fit(dataset).transform(dataset)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adjacency(self) -> dict[int, set[int]]:
        adj: dict[int, set[int]] = defaultdict(set)
        for a, b in self.noise_profile.coupling_map:
            adj[a].add(b)
            adj[b].add(a)
        return dict(adj)

    def _connectivity_path(
        self,
        candidates: list[int],
        qubit_scores: np.ndarray,
    ) -> list[int]:
        """
        Greedy path through *candidates* that maximises coupling-map adjacency.

        Starts from the lowest-error candidate and at each step extends to
        the physically connected neighbour with the lowest error.  When no
        connected candidate remains, falls back to the globally best
        remaining candidate (a SWAP will still be needed, but we pick the
        least noisy option).
        """
        adj = self._adjacency()
        remaining = set(candidates[1:])
        path = [candidates[0]]

        while remaining:
            last = path[-1]
            connected = adj.get(last, set()) & remaining
            nxt = (
                min(connected, key=lambda q: qubit_scores[q])
                if connected
                else min(remaining, key=lambda q: qubit_scores[q])
            )
            path.append(nxt)
            remaining.discard(nxt)

        return path

    def _count_adjacent_swaps(self, qubit_path: list[int]) -> int:
        """
        Estimate total SWAP overhead for a linear sequence of qubit interactions.

        For each consecutive pair ``(path[i], path[i+1])`` that is *not*
        directly connected in the coupling map, one SWAP per additional hop
        is required.  This matches the interaction structure of linear
        entanglement patterns (EntangledAngle, IQP, ZZ with nearest-neighbour).
        """
        if len(qubit_path) < 2:
            return 0
        adj = self._adjacency()
        total = 0
        for i in range(len(qubit_path) - 1):
            a, b = qubit_path[i], qubit_path[i + 1]
            if b not in adj.get(a, set()):
                total += max(0, self._bfs_distance(a, b, adj) - 1)
        return total

    def _bfs_distance(self, src: int, dst: int, adj: dict[int, set[int]]) -> int:
        """Shortest path length (hop count) between *src* and *dst*."""
        if src == dst:
            return 0
        visited = {src}
        queue = [(src, 0)]
        while queue:
            node, dist = queue.pop(0)
            for nb in adj.get(node, set()):
                if nb == dst:
                    return dist + 1
                if nb not in visited:
                    visited.add(nb)
                    queue.append((nb, dist + 1))
        return self.noise_profile.n_qubits + 1  # unreachable: large penalty
