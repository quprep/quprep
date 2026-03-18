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

__version__ = "0.1.0"
__author__ = "Hasarindu Perera"
__license__ = "Apache-2.0"

from quprep.core.pipeline import Pipeline
from quprep.core.recommender import recommend

__all__ = [
    "__version__",
    "Pipeline",
    "recommend",
    "prepare",
]


def prepare(source, *, encoding: str = "angle", framework: str = "qiskit", **kwargs):
    """
    Convert a dataset to a quantum circuit in one call.

    Parameters
    ----------
    source : str, Path, np.ndarray, or pd.DataFrame
        Input data — file path or in-memory array/frame.
    encoding : str
        Encoding method: 'angle', 'amplitude', 'basis', 'iqp', 'reupload', 'hamiltonian'.
    framework : str
        Export target: 'qiskit', 'pennylane', 'cirq', 'tket', 'qasm'.
    **kwargs
        Passed to the underlying Pipeline components.

    Returns
    -------
    ExportResult
        Object with a `.circuit` attribute and `.draw()` method.
    """
    raise NotImplementedError("prepare() — coming in v0.1.0")
