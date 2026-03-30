"""Encoding comparison — run all encoders on a dataset and return side-by-side cost stats."""

from __future__ import annotations

from dataclasses import dataclass, field

from quprep.validation.cost import CostEstimate, estimate_cost

_ALL_ENCODINGS = [
    "angle",
    "amplitude",
    "basis",
    "iqp",
    "reupload",
    "entangled_angle",
    "hamiltonian",
]


def _default_encoders() -> dict[str, object]:
    """Return one default-configured instance per encoder."""
    from quprep.encode.amplitude import AmplitudeEncoder
    from quprep.encode.angle import AngleEncoder
    from quprep.encode.basis import BasisEncoder
    from quprep.encode.entangled_angle import EntangledAngleEncoder
    from quprep.encode.hamiltonian import HamiltonianEncoder
    from quprep.encode.iqp import IQPEncoder
    from quprep.encode.reupload import ReUploadEncoder

    return {
        "angle": AngleEncoder(),
        "amplitude": AmplitudeEncoder(),
        "basis": BasisEncoder(),
        "iqp": IQPEncoder(),
        "reupload": ReUploadEncoder(),
        "entangled_angle": EntangledAngleEncoder(),
        "hamiltonian": HamiltonianEncoder(),
    }


@dataclass
class ComparisonResult:
    """
    Side-by-side cost estimates for multiple encoding methods.

    Attributes
    ----------
    rows : list[CostEstimate]
        One :class:`~quprep.CostEstimate` per encoding method, in the order they were
        evaluated.
    recommended : str or None
        Encoding name highlighted as the best match for the specified task/qubit budget.
        ``None`` if no *task* was passed to :func:`compare_encodings`.

    Examples
    --------
    >>> import quprep as qd
    >>> result = qd.compare_encodings("data.csv", task="classification")
    >>> print(result)
    >>> best = result.best(prefer="nisq")
    """

    rows: list[CostEstimate]
    recommended: str | None = field(default=None)

    def best(self, *, prefer: str = "nisq") -> CostEstimate:
        """
        Return the best row according to *prefer*.

        Parameters
        ----------
        prefer : {"nisq", "depth", "gates", "qubits"}
            ``"nisq"`` (default) — prefer NISQ-safe encodings, then minimise depth.
            ``"depth"`` — minimise circuit depth globally.
            ``"gates"`` — minimise total gate count.
            ``"qubits"`` — minimise qubit count.

        Returns
        -------
        CostEstimate
        """
        if prefer == "nisq":
            pool = [r for r in self.rows if r.nisq_safe] or self.rows
            return min(pool, key=lambda r: (r.circuit_depth, r.gate_count))
        if prefer == "depth":
            return min(self.rows, key=lambda r: r.circuit_depth)
        if prefer == "gates":
            return min(self.rows, key=lambda r: r.gate_count)
        if prefer == "qubits":
            return min(self.rows, key=lambda r: r.n_qubits)
        raise ValueError(
            f"Unknown prefer={prefer!r}. Choose: nisq, depth, gates, qubits"
        )

    def to_dict(self) -> list[dict]:
        """Return all rows as a list of plain dicts (JSON-serialisable)."""
        return [
            {
                "encoding": r.encoding,
                "n_qubits": r.n_qubits,
                "gate_count": r.gate_count,
                "circuit_depth": r.circuit_depth,
                "two_qubit_gates": r.two_qubit_gates,
                "nisq_safe": r.nisq_safe,
                "warning": r.warning,
            }
            for r in self.rows
        ]

    def __str__(self) -> str:
        col_w = [18, 8, 11, 7, 10, 10]
        headers = ["Encoding", "Qubits", "Gate Count", "Depth", "2Q Gates", "NISQ Safe"]
        sep = "  ".join("-" * w for w in col_w)
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_w))

        lines = [header_line, sep]
        for r in self.rows:
            marker = " *" if r.encoding == self.recommended else "  "
            row = "  ".join([
                (r.encoding + marker).ljust(col_w[0]),
                str(r.n_qubits).ljust(col_w[1]),
                str(r.gate_count).ljust(col_w[2]),
                str(r.circuit_depth).ljust(col_w[3]),
                str(r.two_qubit_gates).ljust(col_w[4]),
                ("Yes" if r.nisq_safe else "No").ljust(col_w[5]),
            ])
            lines.append(row)

        if self.recommended:
            lines.append("\n* recommended for the specified task/budget")

        warnings = [r for r in self.rows if r.warning]
        if warnings:
            lines.append("")
            for r in warnings:
                lines.append(f"  [{r.encoding}] {r.warning}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        rec = f", recommended={self.recommended!r}" if self.recommended else ""
        return f"ComparisonResult(rows={len(self.rows)}{rec})"


def compare_encodings(
    source,
    *,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    task: str | None = None,
    qubits: int | None = None,
) -> ComparisonResult:
    """
    Compare all (or selected) encoding methods on *source* and return side-by-side stats.

    No circuits are generated — costs are estimated analytically from the dataset shape,
    so this is fast even for large datasets.

    Parameters
    ----------
    source : str, numpy.ndarray, pandas.DataFrame, or Dataset
        Input data — same formats accepted by :class:`~quprep.Pipeline`.
    include : list[str] or None
        Encoder names to include. If ``None``, all 7 encoders are compared.
        Valid names: ``"angle"``, ``"amplitude"``, ``"basis"``, ``"iqp"``,
        ``"reupload"``, ``"entangled_angle"``, ``"hamiltonian"``.
    exclude : list[str] or None
        Encoder names to exclude. Applied after *include*.
    task : str or None
        If provided, the recommended encoder for this task is starred in the output
        table. Passed to :func:`~quprep.recommend`.
        Valid: ``"classification"``, ``"regression"``, ``"qaoa"``, ``"kernel"``,
        ``"simulation"``.
    qubits : int or None
        Maximum qubit budget. Encoders requiring more qubits have ``nisq_safe`` set to
        ``False`` and a budget warning added to their row.

    Returns
    -------
    ComparisonResult

    Examples
    --------
    Compare all encoders on a CSV, highlight the best for classification:

    >>> import quprep as qd
    >>> result = qd.compare_encodings("data.csv", task="classification", qubits=8)
    >>> print(result)
    >>> result.best(prefer="nisq")

    Compare a subset:

    >>> result = qd.compare_encodings(X, include=["angle", "iqp", "amplitude"])
    """
    ds = _ingest(source)
    n_features = ds.data.shape[1] if ds.data is not None and ds.data.ndim == 2 else 1

    encoders = _default_encoders()

    if include is not None:
        unknown = set(include) - set(encoders)
        if unknown:
            raise ValueError(
                f"Unknown encoder(s): {sorted(unknown)}. "
                f"Valid names: {sorted(encoders)}"
            )
        encoders = {k: v for k, v in encoders.items() if k in include}

    if exclude is not None:
        encoders = {k: v for k, v in encoders.items() if k not in exclude}

    rows: list[CostEstimate] = []
    for encoder in encoders.values():
        est = estimate_cost(encoder, n_features)
        if qubits is not None and est.n_qubits > qubits:
            budget_warning = (
                f"{est.encoding} requires {est.n_qubits} qubits "
                f"but budget is {qubits}."
            )
            est = CostEstimate(
                encoding=est.encoding,
                n_features=est.n_features,
                n_qubits=est.n_qubits,
                gate_count=est.gate_count,
                circuit_depth=est.circuit_depth,
                two_qubit_gates=est.two_qubit_gates,
                nisq_safe=False,
                warning=budget_warning,
            )
        rows.append(est)

    recommended: str | None = None
    if task is not None:
        from quprep.core.recommender import recommend
        rec = recommend(source, task=task, qubits=qubits)
        if rec.method in encoders:
            recommended = rec.method

    return ComparisonResult(rows=rows, recommended=recommended)


def _ingest(source):
    """Ingest *source* into a Dataset — mirrors Pipeline._ingest()."""
    import numpy as np

    from quprep.core.dataset import Dataset
    from quprep.ingest.csv_ingester import CSVIngester
    from quprep.ingest.numpy_ingester import NumpyIngester

    if isinstance(source, Dataset):
        return source
    if isinstance(source, str):
        return CSVIngester().load(source)
    try:
        import pandas as pd
        if isinstance(source, pd.DataFrame):
            return NumpyIngester().load(source)
    except ImportError:
        pass
    if isinstance(source, (np.ndarray, list)):
        return NumpyIngester().load(np.asarray(source, dtype=float))
    raise TypeError(
        f"Unsupported source type: {type(source).__name__}. "
        "Expected: str (file path), numpy.ndarray, pandas.DataFrame, or Dataset."
    )
