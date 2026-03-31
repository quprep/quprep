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

__version__ = "0.5.0"
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
from quprep.encode.hamiltonian import HamiltonianEncoder
from quprep.encode.iqp import IQPEncoder
from quprep.encode.reupload import ReUploadEncoder

# Exporters
from quprep.export.qasm_export import QASMExporter
from quprep.export.visualize import draw_ascii, draw_matplotlib

# Normalizer
from quprep.normalize.scalers import Scaler

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
    "AmplitudeEncoder",
    "BasisEncoder",
    "IQPEncoder",
    "ReUploadEncoder",
    "HamiltonianEncoder",
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
    # Validation
    "DataSchema",
    "FeatureSpec",
    "SchemaViolationError",
    "CostEstimate",
    "estimate_cost",
    "QuPrepWarning",
]


def prepare(source, *, encoding: str = "angle", framework: str = "qasm", **kwargs):
    """
    Convert a dataset to quantum circuits in one call.

    Parameters
    ----------
    source : str, Path, np.ndarray, or pd.DataFrame
        Input data — file path or in-memory array/frame.
    encoding : str
        Encoding method: 'angle' (default), 'amplitude', 'basis'.
    framework : str
        Export target: 'qasm' (default, no deps), 'qiskit', 'pennylane', 'cirq', 'tket'.
    **kwargs
        Extra keyword arguments:
        - rotation : str — 'ry' (default), 'rx', 'rz'. Only for 'angle' encoding.
        - pad : bool — zero-pad to power of two. Only for 'amplitude' encoding.
        - threshold : float — binarization cutoff. Only for 'basis' encoding.

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
    from quprep.encode.reupload import ReUploadEncoder
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
    }
    if encoding not in _encoders:
        raise ValueError(
            f"Unknown encoding '{encoding}'. Choose from: {sorted(_encoders)}"
        )

    _exporters = {
        "qasm": lambda: QASMExporter(),
        "qiskit": _lazy_qiskit_exporter,
        "pennylane": _lazy_pennylane_exporter,
        "cirq": _lazy_cirq_exporter,
        "tket": _lazy_tket_exporter,
    }
    if framework not in _exporters:
        raise ValueError(
            f"Unknown framework '{framework}'. Choose from: {sorted(_exporters)}"
        )

    encoder = _encoders[encoding]()
    exporter = _exporters[framework]()
    return Pipeline(encoder=encoder, exporter=exporter).fit_transform(source)


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
