r"""Random Fourier features encoder (quantum-inspired kernel approximation).

Mathematical formulation
------------------------
Approximates the RBF (Gaussian) kernel $k(x, y) = \exp(-\gamma \|x-y\|^2)$
via Bochner's theorem (Rahimi & Recht, 2007).

Fit phase (on training data):
  Sample $W \in \mathbb{R}^{D \times d}$ i.i.d. from
  $\mathcal{N}(0, 2\gamma \cdot I)$ and bias $b \in [0, 2\pi)^D$.

Transform phase (per sample $x$):
  $z(x) = \sqrt{\tfrac{2}{D}} \cos(W x + b) \in \mathbb{R}^D$

The resulting $z(x)$ approximates the feature map of the RBF kernel.
Values are then scaled to $[0, \pi]$ for angle encoding on $D$ qubits.

Properties
----------
Qubits : D (n_components, user-specified)
Depth  : 1 (angle encoding after projection)
NISQ   : Excellent — shallow circuit, fixed qubit count.
Best for: Kernel-based QML when the exact qubit count must be controlled;
          bridging classical kernel methods with NISQ hardware.

Reference: Rahimi & Recht, *NeurIPS* (2007).
"""

from __future__ import annotations

import numpy as np

from quprep.encode.base import BaseEncoder, EncodedResult


class RandomFourierEncoder(BaseEncoder):
    """
    Random Fourier features encoder.

    Projects input features through a random Fourier basis to
    approximate the RBF kernel, then angle-encodes the result on a
    fixed number of qubits.  Must be fitted before use.

    Parameters
    ----------
    n_components : int
        Number of random Fourier features (= number of qubits). Default 8.
    gamma : float
        RBF kernel bandwidth parameter. Default 1.0.
    random_state : int or None
        Seed for reproducibility. Default None.
    """

    def __init__(
        self,
        n_components: int = 8,
        gamma: float = 1.0,
        random_state: int | None = None,
    ):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}.")
        if gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma}.")
        self.n_components = n_components
        self.gamma = gamma
        self.random_state = random_state
        self._W: np.ndarray | None = None
        self._b: np.ndarray | None = None

    @property
    def n_qubits(self):
        return self.n_components

    @property
    def depth(self):
        return 1

    def fit(self, X: np.ndarray) -> RandomFourierEncoder:
        """
        Sample the random projection matrix from the training data shape.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, d)
            Training data (used only for dimensionality).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        d = X.shape[1]
        rng = np.random.default_rng(self.random_state)
        self._W = rng.normal(0, np.sqrt(2.0 * self.gamma), size=(self.n_components, d))
        self._b = rng.uniform(0, 2 * np.pi, size=(self.n_components,))
        return self

    def encode(self, x: np.ndarray) -> EncodedResult:
        r"""
        Project ``x`` through random Fourier features and angle-encode.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Feature vector. Must match the dimensionality used in :meth:`fit`.

        Returns
        -------
        EncodedResult
            ``parameters`` = scaled angles in $[0, \pi]$ of length ``n_components``.
            ``metadata`` includes ``encoding``, ``n_qubits``, ``n_components``,
            ``gamma``, ``depth``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if self._W is None:
            raise RuntimeError(
                "RandomFourierEncoder must be fitted before encoding. Call fit(X) first."
            )
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) == 0:
            raise ValueError("RandomFourierEncoder.encode() expects a non-empty 1-D array.")

        z = np.sqrt(2.0 / self.n_components) * np.cos(self._W @ x + self._b)
        # z ∈ [-sqrt(2/D), sqrt(2/D)] approximately.  Scale to [0, π].
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            angles = (z - z_min) / (z_max - z_min) * np.pi
        else:
            angles = np.zeros_like(z)

        return EncodedResult(
            parameters=angles,
            metadata={
                "encoding": "random_fourier",
                "n_qubits": self.n_components,
                "n_components": self.n_components,
                "gamma": self.gamma,
                "depth": 1,
            },
        )
