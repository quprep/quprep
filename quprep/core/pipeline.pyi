from quprep.core.dataset import Dataset
from quprep.encode.base import EncodedResult
from quprep.validation.schema import DataSchema

class PipelineResult:
    dataset: Dataset
    encoded: list[EncodedResult] | None
    circuits: list | None

    def __init__(
        self,
        dataset: Dataset,
        encoded: list[EncodedResult] | None,
        circuits: list | None,
    ) -> None: ...

    @property
    def circuit(self) -> object | None: ...

class Pipeline:
    ingester: object | None
    cleaner: object | None
    reducer: object | None
    normalizer: object | None
    encoder: object | None
    exporter: object | None
    schema: DataSchema | None

    def __init__(
        self,
        ingester: object | None = ...,
        cleaner: object | None = ...,
        reducer: object | None = ...,
        normalizer: object | None = ...,
        encoder: object | None = ...,
        exporter: object | None = ...,
        schema: DataSchema | None = ...,
    ) -> None: ...

    def fit(self, source: object, y: object = ...) -> Pipeline: ...
    def transform(self, source: object) -> PipelineResult: ...
    def fit_transform(self, source: object, y: object = ...) -> PipelineResult: ...
    def get_params(self, deep: bool = ...) -> dict: ...
    def set_params(self, **params: object) -> Pipeline: ...
