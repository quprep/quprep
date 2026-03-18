"""Export encoded data as OpenQASM 3.0 strings or files.

OpenQASM 3.0 is a universal interchange format accepted by all major
quantum frameworks and hardware platforms. Use this exporter for maximum
portability when you don't need framework-specific features.
"""

from __future__ import annotations

from pathlib import Path


class QASMExporter:
    """
    Export EncodedResult objects to OpenQASM 3.0.

    No additional dependencies required.

    Parameters
    ----------
    version : str
        QASM version. Currently only '3.0' is supported.
    """

    def __init__(self, version: str = "3.0"):
        if version != "3.0":
            raise ValueError("Only OpenQASM 3.0 is supported.")
        self.version = version

    def export(self, encoded) -> str:
        """Return the OpenQASM 3.0 string for an EncodedResult."""
        raise NotImplementedError("QASMExporter.export() — coming in v0.1.0")

    def save(self, encoded, path: str | Path) -> None:
        """Write the QASM string to a .qasm file."""
        qasm_str = self.export(encoded)
        Path(path).write_text(qasm_str, encoding="utf-8")
