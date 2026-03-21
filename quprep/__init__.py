"""
QuPrep — Quantum Data Preparation.

The missing preprocessing layer between classical datasets and quantum computing frameworks.

    import quprep

    # One-liner
    circuit = quprep.prepare("data.csv", encoding="angle", framework="qiskit")

    # Pipeline
    pipeline = quprep.Pipeline(
        cleaner=quprep.Cleaner(impute="knn"),
        reducer=quprep.LDAReducer(n_components=4),
        encoder=quprep.AngleEncoder(),
        exporter=quprep.QiskitExporter(),
    )

    # Recommendation
    rec = quprep.recommend(df, task="classification", qubits=8)
"""

__version__ = "0.3.0"
__author__ = "Hasarindu Perera"
__license__ = "Apache-2.0"

from quprep.core.pipeline import Pipeline
from quprep.core.recommender import recommend
from quprep.export.visualize import draw_ascii, draw_matplotlib

__all__ = [
    "__version__",
    "Pipeline",
    "recommend",
    "prepare",
    "draw_ascii",
    "draw_matplotlib",
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
