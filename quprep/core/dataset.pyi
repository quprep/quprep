import numpy as np

class Dataset:
    data: np.ndarray
    feature_names: list[str]
    feature_types: list[str]
    categorical_data: dict[str, list]
    metadata: dict

    def __init__(
        self,
        data: np.ndarray,
        feature_names: list[str] | None = ...,
        feature_types: list[str] | None = ...,
        categorical_data: dict[str, list] | None = ...,
        metadata: dict | None = ...,
    ) -> None: ...

    @property
    def n_samples(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    @property
    def n_categorical(self) -> int: ...
    def copy(self) -> Dataset: ...
