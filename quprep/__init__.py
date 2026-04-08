"""
QuPrep — Quantum Data Preparation.

The missing preprocessing layer between classical datasets and quantum computing frameworks.

Both import styles are supported::

    import quprep
    import quprep as qd   # "quantum data" — preferred short alias

One-liner::

    circuit = qd.prepare("data.csv", encoding="angle", framework="qiskit")

Full pipeline::

    pipeline = qd.Pipeline(
        cleaner=qd.Imputer(strategy="knn"),
        reducer=qd.LDAReducer(n_components=4),
        encoder=qd.AngleEncoder(),
        exporter=qd.QiskitExporter(),
    )
    result = pipeline.fit_transform(df)
    result.summary()

Schema-validated pipeline::

    schema = qd.DataSchema([
        qd.FeatureSpec("age", dtype="continuous", min_value=0, max_value=120),
        qd.FeatureSpec("score", dtype="continuous", min_value=0.0, max_value=1.0),
    ])
    pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), schema=schema)

Recommendation::

    rec = qd.recommend(df, task="classification", qubits=8)
"""

__version__ = "0.7.0"
__author__ = "Hasarindu Perera"
__license__ = "Apache-2.0"

# Core
# Cleaners
from quprep.clean.categorical import CategoricalEncoder
from quprep.clean.imputer import Imputer
from quprep.clean.outlier import OutlierHandler
from quprep.clean.selector import FeatureSelector

# Comparison
from quprep.compare import ComparisonResult, compare_encodings
from quprep.core.drift import DriftDetector, DriftReport
from quprep.core.pipeline import Pipeline, PipelineResult
from quprep.core.qubit_suggestion import QubitSuggestion, suggest_qubits
from quprep.core.recommender import recommend

# Encoders
from quprep.encode.amplitude import AmplitudeEncoder
from quprep.encode.angle import AngleEncoder
from quprep.encode.basis import BasisEncoder
from quprep.encode.entangled_angle import EntangledAngleEncoder
from quprep.encode.graph_state import GraphStateEncoder
from quprep.encode.hamiltonian import HamiltonianEncoder
from quprep.encode.iqp import IQPEncoder
from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
from quprep.encode.qaoa_problem import QAOAProblemEncoder
from quprep.encode.random_fourier import RandomFourierEncoder
from quprep.encode.reupload import ReUploadEncoder
from quprep.encode.tensor_product import TensorProductEncoder
from quprep.encode.zz_feature_map import ZZFeatureMapEncoder

# Exporters
from quprep.export.qasm_export import QASMExporter
from quprep.export.visualize import draw_ascii, draw_matplotlib

# Ingesters
from quprep.ingest.graph_ingester import GraphIngester
from quprep.ingest.huggingface_ingester import HuggingFaceIngester
from quprep.ingest.image_ingester import ImageIngester
from quprep.ingest.text_ingester import TextIngester
from quprep.ingest.timeseries_ingester import TimeSeriesIngester

# Normalizer
from quprep.normalize.scalers import Scaler

# Plugins
from quprep.plugins import (
    get_encoder_class,
    get_exporter_class,
    list_encoders,
    list_exporters,
    register_encoder,
    register_exporter,
    unregister_encoder,
    unregister_exporter,
)

# Preprocessors
from quprep.preprocess.window import WindowTransformer

# QUBO / Ising / quantum optimization
from quprep.qubo import (
    IsingResult,
    QUBOResult,
    add_qubo,
    draw_ising,
    draw_qubo,
    equality_penalty,
    graph_color,
    inequality_penalty,
    ising_to_qubo,
    knapsack,
    max_cut,
    number_partition,
    portfolio,
    qaoa_circuit,
    qubo_to_ising,
    scheduling,
    to_qubo,
    tsp,
)

# Reducers
from quprep.reduce.hardware_aware import HardwareAwareReducer
from quprep.reduce.lda import LDAReducer
from quprep.reduce.pca import PCAReducer
from quprep.reduce.spectral import SpectralReducer, TSNEReducer, UMAPReducer

# Validation
from quprep.validation import (
    CostEstimate,
    DataSchema,
    FeatureSpec,
    QuPrepWarning,
    SchemaViolationError,
    estimate_cost,
)

__all__ = [
    "__version__",
    # Core
    "Pipeline",
    "PipelineResult",
    "recommend",
    "prepare",
    # Encoders
    "AngleEncoder",
    "EntangledAngleEncoder",
    "GraphStateEncoder",
    "AmplitudeEncoder",
    "BasisEncoder",
    "IQPEncoder",
    "ReUploadEncoder",
    "HamiltonianEncoder",
    "ZZFeatureMapEncoder",
    "PauliFeatureMapEncoder",
    "RandomFourierEncoder",
    "TensorProductEncoder",
    "QAOAProblemEncoder",
    # Cleaners
    "Imputer",
    "OutlierHandler",
    "CategoricalEncoder",
    "FeatureSelector",
    # Normalizer
    "Scaler",
    # Reducers
    "PCAReducer",
    "LDAReducer",
    "HardwareAwareReducer",
    "SpectralReducer",
    "TSNEReducer",
    "UMAPReducer",
    # Exporters
    "QASMExporter",
    "draw_ascii",
    "draw_matplotlib",
    # Comparison
    "compare_encodings",
    "ComparisonResult",
    # Qubit suggestion
    "suggest_qubits",
    "QubitSuggestion",
    # Drift detection
    "DriftDetector",
    "DriftReport",
    # Batch export
    "batch_export",
    # Ingesters
    "GraphIngester",
    "HuggingFaceIngester",
    "ImageIngester",
    "TextIngester",
    "TimeSeriesIngester",
    # Preprocessors
    "WindowTransformer",
    # QUBO / quantum optimization
    "to_qubo",
    "QUBOResult",
    "qubo_to_ising",
    "ising_to_qubo",
    "IsingResult",
    "equality_penalty",
    "inequality_penalty",
    "max_cut",
    "tsp",
    "knapsack",
    "portfolio",
    "graph_color",
    "scheduling",
    "number_partition",
    "qaoa_circuit",
    "add_qubo",
    "draw_qubo",
    "draw_ising",
    # Plugins
    "register_encoder",
    "register_exporter",
    "unregister_encoder",
    "unregister_exporter",
    "list_encoders",
    "list_exporters",
    "get_encoder_class",
    "get_exporter_class",
    # Validation
    "DataSchema",
    "FeatureSpec",
    "SchemaViolationError",
    "CostEstimate",
    "estimate_cost",
    "QuPrepWarning",
]


def prepare(
    source, *, encoding: str = "angle", framework: str = "qasm",
    ingester=None, preprocessor=None, **kwargs
):
    """
    Convert a dataset to quantum circuits in one call.

    Parameters
    ----------
    source : str, Path, np.ndarray, pd.DataFrame, or Dataset
        Input data — file path, in-memory array/frame, or a pre-loaded Dataset.
        For image directories, text files, or graph data pass a modality
        ingester via the ``ingester`` parameter.
    encoding : str
        Encoding method. One of: 'angle' (default), 'entangled_angle', 'amplitude',
        'basis', 'iqp', 'reupload', 'hamiltonian', 'zz_feature_map',
        'pauli_feature_map', 'random_fourier', 'tensor_product', 'qaoa_problem'.
        Plugin encoders registered via :func:`register_encoder` are also accepted.
    framework : str
        Export target. One of: 'qasm' (default, no deps), 'qiskit', 'pennylane',
        'cirq', 'tket', 'braket', 'qsharp', 'iqm'.
        Plugin exporters registered via :func:`register_exporter` are also accepted.
    ingester : ingester object, optional
        A modality ingester instance whose ``load(source)`` method is called
        before encoding. Use this for non-tabular data::

            qd.prepare("images/", encoding="angle", ingester=qd.ImageIngester())
            qd.prepare(texts, encoding="angle", ingester=qd.TextIngester())
            qd.prepare(adj, encoding="angle", ingester=qd.GraphIngester(n_features=8))

        When ``None`` (default) the pipeline auto-detects CSV, NumPy arrays,
        and DataFrames.
    **kwargs
        Extra keyword arguments forwarded to the encoder/exporter constructor.
        Common options: ``rotation`` ('ry'/'rx'/'rz'), ``pad`` (amplitude),
        ``threshold`` (basis), ``reps``, ``layers``, ``p``, ``connectivity``.

    Returns
    -------
    PipelineResult
        Object with `.circuits`, `.encoded`, `.dataset`, and `.circuit` (first sample).
    """
    from quprep.encode.amplitude import AmplitudeEncoder
    from quprep.encode.angle import AngleEncoder
    from quprep.encode.basis import BasisEncoder
    from quprep.encode.entangled_angle import EntangledAngleEncoder
    from quprep.encode.hamiltonian import HamiltonianEncoder
    from quprep.encode.iqp import IQPEncoder
    from quprep.encode.pauli_feature_map import PauliFeatureMapEncoder
    from quprep.encode.qaoa_problem import QAOAProblemEncoder
    from quprep.encode.random_fourier import RandomFourierEncoder
    from quprep.encode.reupload import ReUploadEncoder
    from quprep.encode.tensor_product import TensorProductEncoder
    from quprep.encode.zz_feature_map import ZZFeatureMapEncoder
    from quprep.export.qasm_export import QASMExporter

    _encoders = {
        "angle": lambda: AngleEncoder(rotation=kwargs.get("rotation", "ry")),
        "entangled_angle": lambda: EntangledAngleEncoder(
            rotation=kwargs.get("rotation", "ry"),
            layers=kwargs.get("layers", 1),
            entanglement=kwargs.get("entanglement", "linear"),
        ),
        "amplitude": lambda: AmplitudeEncoder(pad=kwargs.get("pad", True)),
        "basis": lambda: BasisEncoder(threshold=kwargs.get("threshold", 0.5)),
        "iqp": lambda: IQPEncoder(reps=kwargs.get("reps", 2)),
        "reupload": lambda: ReUploadEncoder(
            layers=kwargs.get("layers", 3),
            rotation=kwargs.get("rotation", "ry"),
        ),
        "hamiltonian": lambda: HamiltonianEncoder(
            evolution_time=kwargs.get("evolution_time", 1.0),
            trotter_steps=kwargs.get("trotter_steps", 4),
        ),
        "zz_feature_map": lambda: ZZFeatureMapEncoder(reps=kwargs.get("reps", 2)),
        "pauli_feature_map": lambda: PauliFeatureMapEncoder(
            paulis=kwargs.get("paulis", None),
            reps=kwargs.get("reps", 2),
        ),
        "random_fourier": lambda: RandomFourierEncoder(
            n_components=kwargs.get("n_components", 8),
            gamma=kwargs.get("gamma", 1.0),
            random_state=kwargs.get("random_state", None),
        ),
        "tensor_product": lambda: TensorProductEncoder(),
        "qaoa_problem": lambda: QAOAProblemEncoder(
            p=kwargs.get("p", 1),
            gamma=kwargs.get("gamma", 0.7853981633974483),   # π/4
            beta=kwargs.get("beta", 0.39269908169872414),    # π/8
            connectivity=kwargs.get("connectivity", "linear"),
        ),
    }

    # Check plugin registry if not in built-ins
    if encoding not in _encoders:
        plugin_cls = get_encoder_class(encoding)
        if plugin_cls is None:
            raise ValueError(
                f"Unknown encoding '{encoding}'. "
                f"Built-ins: {sorted(_encoders)}. "
                "Plugin encoders: use register_encoder() to add custom ones."
            )
        _encoders[encoding] = lambda cls=plugin_cls: cls()  # type: ignore[misc]

    _exporters = {
        "qasm": lambda: QASMExporter(),
        "qiskit": _lazy_qiskit_exporter,
        "pennylane": _lazy_pennylane_exporter,
        "cirq": _lazy_cirq_exporter,
        "tket": _lazy_tket_exporter,
        "braket": _lazy_braket_exporter,
        "qsharp": _lazy_qsharp_exporter,
        "iqm": _lazy_iqm_exporter,
    }

    # Check plugin registry if not in built-ins
    if framework not in _exporters:
        plugin_cls = get_exporter_class(framework)
        if plugin_cls is None:
            raise ValueError(
                f"Unknown framework '{framework}'. "
                f"Built-ins: {sorted(_exporters)}. "
                "Plugin exporters: use register_exporter() to add custom ones."
            )
        _exporters[framework] = lambda cls=plugin_cls: cls()  # type: ignore[misc]

    encoder = _encoders[encoding]()
    exporter = _exporters[framework]()
    return Pipeline(
        encoder=encoder, exporter=exporter, ingester=ingester, preprocessor=preprocessor,
    ).fit_transform(source)


def batch_export(
    source,
    directory: str,
    *,
    encoding: str = "angle",
    stem: str = "circuit",
    **kwargs,
) -> list:
    """
    Convert a dataset to QASM circuits and save each sample to a file.

    Combines :func:`prepare` with :meth:`QASMExporter.save_batch`. Output
    files are named ``{stem}_0000.qasm``, ``{stem}_0001.qasm``, etc. in
    the given directory.

    Parameters
    ----------
    source : str, Path, np.ndarray, or pd.DataFrame
        Input data.
    directory : str or Path
        Output directory (created if it does not exist).
    encoding : str
        Encoding method (default: ``'angle'``).
    stem : str
        Filename stem (default: ``'circuit'``).
    **kwargs
        Extra keyword arguments forwarded to :func:`prepare`.

    Returns
    -------
    list of Path
        Paths of the written files, in sample order.
    """
    from pathlib import Path as _Path

    from quprep.export.qasm_export import QASMExporter

    result = prepare(source, encoding=encoding, framework="qasm", **kwargs)
    exporter = QASMExporter()
    return exporter.save_batch(result.encoded, _Path(directory), stem=stem)


def _lazy_qiskit_exporter():
    from quprep.export.qiskit_export import QiskitExporter
    return QiskitExporter()


def _lazy_pennylane_exporter():
    from quprep.export.pennylane_export import PennyLaneExporter
    return PennyLaneExporter()


def _lazy_cirq_exporter():
    from quprep.export.cirq_export import CirqExporter
    return CirqExporter()


def _lazy_tket_exporter():
    from quprep.export.tket_export import TKETExporter
    return TKETExporter()


def _lazy_braket_exporter():
    from quprep.export.braket_export import BraketExporter
    return BraketExporter()


def _lazy_qsharp_exporter():
    from quprep.export.qsharp_export import QSharpExporter
    return QSharpExporter()


def _lazy_iqm_exporter():
    from quprep.export.iqm_export import IQMExporter
    return IQMExporter()
