from quprep.core.dataset import Dataset

class QuPrepWarning(UserWarning): ...

class FeatureSpec:
    name: str
    dtype: str
    min_value: float | None
    max_value: float | None
    nullable: bool

    def __init__(
        self,
        name: str,
        dtype: str,
        min_value: float | None = ...,
        max_value: float | None = ...,
        nullable: bool = ...,
    ) -> None: ...

class SchemaViolationError(ValueError): ...

class DataSchema:
    features: list[FeatureSpec]

    def __init__(self, features: list[FeatureSpec]) -> None: ...
    def validate(self, dataset: Dataset) -> None: ...
    @classmethod
    def infer(cls, dataset: Dataset) -> DataSchema: ...

class CostEstimate:
    encoding: str
    n_features: int
    n_qubits: int
    gate_count: int
    circuit_depth: int
    two_qubit_gates: int
    nisq_safe: bool
    warning: str | None

def validate_dataset(dataset: Dataset, *, context: str = ...) -> None: ...
def warn_qubit_mismatch(n_features: int, n_qubits: int, encoding: str) -> None: ...
def estimate_cost(encoder: object, n_features: int) -> CostEstimate: ...
