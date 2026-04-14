"""Reproducibility fingerprinting — deterministic hash of a pipeline configuration.

Captures the class and parameters of every pipeline stage plus key dependency
versions. The SHA-256 hash is stable across runs for the same configuration,
making it suitable for paper methods sections and experiment logs.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from quprep.core.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


class FingerprintResult:
    """
    Output of :func:`fingerprint_pipeline`.

    Attributes
    ----------
    config : dict
        Full pipeline configuration (stages + dependency versions). This is
        the dict that was hashed — no timestamp, fully deterministic.
    hash : str
        SHA-256 hex digest of the canonical JSON serialisation of ``config``.
    """

    def __init__(self, config: dict, hash_hex: str) -> None:
        self.config = config
        self.hash = hash_hex

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return the config augmented with the hash and a UTC timestamp."""
        from datetime import datetime, timezone

        return {
            "hash": f"sha256:{self.hash}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **self.config,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return a JSON string (hash + timestamp + config)."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_yaml(self) -> str:
        """Return a YAML string (requires ``pyyaml``)."""
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "pyyaml is required for YAML export. "
                "Install it with: pip install pyyaml"
            ) from exc
        return yaml.safe_dump(self.to_dict(), default_flow_style=False, allow_unicode=True)

    def save(self, path: str, format: str = "json") -> None:
        """
        Write the fingerprint to a file.

        Parameters
        ----------
        path : str
            Destination file path.
        format : {"json", "yaml"}
            Output format.
        """
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            p.write_text(self.to_json())
        elif format == "yaml":
            p.write_text(self.to_yaml())
        else:
            raise ValueError(f"format must be 'json' or 'yaml', got {format!r}")

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n = len(self.config.get("stages", {}))
        return f"FingerprintResult(hash=sha256:{self.hash[:12]}..., stages={n})"

    def __str__(self) -> str:
        return self.to_json()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def fingerprint_pipeline(pipeline: Pipeline) -> FingerprintResult:
    """
    Compute a reproducibility fingerprint for *pipeline*.

    The fingerprint captures the class name and constructor parameters of every
    configured stage (ingester, preprocessor, cleaner, reducer, normalizer,
    encoder, exporter, schema, drift_detector) plus the installed versions of
    key dependencies.  The resulting SHA-256 hash is deterministic: the same
    configuration always produces the same hash regardless of when or where the
    pipeline runs.

    Parameters
    ----------
    pipeline : Pipeline
        A ``Pipeline`` instance (fitted or unfitted).

    Returns
    -------
    FingerprintResult
        Contains ``config`` (serialisable dict) and ``hash`` (SHA-256 hex string).

    Examples
    --------
    >>> import quprep as qd
    >>> pipeline = qd.Pipeline(encoder=qd.AngleEncoder(), exporter=qd.QASMExporter())
    >>> fp = qd.fingerprint_pipeline(pipeline)
    >>> print(fp.hash)
    >>> fp.save("experiment.json")
    """
    config = _build_config(pipeline)
    hash_hex = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()
    return FingerprintResult(config=config, hash_hex=hash_hex)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_TRACKED_PACKAGES = [
    "numpy",
    "scikit-learn",
    "scipy",
    "pandas",
    "qiskit",
    "pennylane",
    "cirq-core",
    "pytket",
    "amazon-braket-sdk",
    "qsharp",
    "iqm-client",
    "datasets",
    "kaggle",
    "openml",
]


def _build_config(pipeline: Pipeline) -> dict:
    """Assemble the serialisable config dict (no timestamp — must be stable)."""
    import importlib.metadata
    import sys

    # Resolve normalizer: prefer the fitted one if available
    normalizer = getattr(pipeline, "_resolved_normalizer", None) or pipeline.normalizer

    raw_stages: dict = {
        "ingester": pipeline.ingester,
        "preprocessor": pipeline.preprocessor,
        "cleaner": pipeline.cleaner,
        "reducer": pipeline.reducer,
        "normalizer": normalizer,
        "encoder": pipeline.encoder,
        "exporter": pipeline.exporter,
        "schema": pipeline.schema,
        "drift_detector": pipeline.drift_detector,
    }

    stages: dict = {}
    for name, stage in raw_stages.items():
        if stage is None:
            continue
        if name == "preprocessor" and isinstance(stage, list):
            stages[name] = [_stage_config(s) for s in stage]
        else:
            stages[name] = _stage_config(stage)

    # Dependency versions
    dependencies: dict = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            dependencies[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    try:
        quprep_version = importlib.metadata.version("quprep")
    except importlib.metadata.PackageNotFoundError:
        import quprep as _qp
        quprep_version = getattr(_qp, "__version__", "unknown")

    return {
        "quprep_version": quprep_version,
        "python_version": sys.version.split()[0],
        "stages": stages,
        "dependencies": dependencies,
    }


def _stage_config(stage: object) -> dict:
    """Return a serialisable dict describing *stage*."""
    config: dict = {
        "class": type(stage).__name__,
        "module": type(stage).__module__,
    }

    if hasattr(stage, "get_params"):
        raw = stage.get_params()
    else:
        sig = inspect.signature(type(stage).__init__)
        obj_vars = vars(stage) if hasattr(stage, "__dict__") else {}
        raw = {
            k: obj_vars.get(k)
            for k in sig.parameters
            if k != "self"
        }

    config["params"] = _make_serializable(raw)
    return config


def _make_serializable(value: object) -> object:
    """Recursively convert *value* into a JSON-safe type."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, (list, tuple)):
        return [_make_serializable(v) for v in value]

    if isinstance(value, dict):
        return {str(k): _make_serializable(v) for k, v in value.items()}

    # numpy arrays / scalars
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except ImportError:
        pass

    # Objects with get_params (nested sklearn-style estimators)
    if hasattr(value, "get_params"):
        return {
            "class": type(value).__name__,
            "module": type(value).__module__,
            "params": _make_serializable(value.get_params()),
        }

    # Fallback: record class name only
    return f"<{type(value).__name__}>"
