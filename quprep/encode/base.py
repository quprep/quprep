"""Abstract base class for all quantum encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from quprep.core.dataset import Dataset


class BaseEncoder(ABC):
    r"""
    Abstract interface for quantum encoders.

    All encoders must implement `encode()`, which maps a normalized
    feature vector $x \in \mathbb{R}^d$ to a parameterized quantum circuit.

    Subclasses should document:
    - The mathematical mapping x → circuit parameters.
    - The number of qubits as a function of d.
    - The circuit depth as a function of d.
    - NISQ suitability.
    """

    @abstractmethod
    def encode(self, x: np.ndarray) -> EncodedResult:
        """
        Encode a single normalized feature vector.

        Parameters
        ----------
        x : np.ndarray, shape (d,)
            Normalized feature vector.

        Returns
        -------
        EncodedResult
            Circuit parameters and metadata for the encoded sample.
        """

    def encode_batch(self, dataset: Dataset) -> list[EncodedResult]:
        """
        Encode all samples in a Dataset.

        Parameters
        ----------
        dataset : Dataset
            Dataset whose rows are encoded one by one.

        Returns
        -------
        list of EncodedResult
            One EncodedResult per sample row.
        """
        return [self.encode(row) for row in dataset.data]

    @property
    @abstractmethod
    def n_qubits(self) -> int | None:
        """Number of qubits required. None if data-dependent."""

    @property
    @abstractmethod
    def depth(self) -> int | str | None:
        """Circuit depth (integer or asymptotic expression)."""


class EncodedResult:
    """
    Output of an encoder for a single sample.

    Attributes
    ----------
    parameters : np.ndarray
        Circuit parameter values.
    circuit_fn : callable or None
        A function(framework) → circuit object, filled in by the exporter.
    metadata : dict
        Encoding details (method, n_qubits, depth, etc.).
    """

    def __init__(self, parameters: np.ndarray, circuit_fn=None, metadata: dict | None = None):
        self.parameters = parameters
        self.circuit_fn = circuit_fn
        self.metadata = metadata or {}
