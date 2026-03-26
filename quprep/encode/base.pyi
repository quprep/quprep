from abc import ABC, abstractmethod

import numpy as np

from quprep.core.dataset import Dataset

class EncodedResult:
    parameters: np.ndarray
    circuit_fn: object | None
    metadata: dict

    def __init__(
        self,
        parameters: np.ndarray,
        circuit_fn: object | None = ...,
        metadata: dict | None = ...,
    ) -> None: ...

class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, x: np.ndarray) -> EncodedResult: ...
    def encode_batch(self, dataset: Dataset) -> list[EncodedResult]: ...
    @property
    @abstractmethod
    def n_qubits(self) -> int | None: ...
    @property
    @abstractmethod
    def depth(self) -> int | str | None: ...
