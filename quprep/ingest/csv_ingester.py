"""CSV and delimiter-separated file ingestion."""

from __future__ import annotations

from pathlib import Path

from quprep.core.dataset import Dataset


class CSVIngester:
    """
    Ingest CSV and TSV files into a Dataset.

    Supports automatic type detection (continuous, discrete, binary, categorical)
    and basic dataset profiling on load.

    Parameters
    ----------
    delimiter : str
        Field delimiter. Defaults to ',' (auto-detects TSV via file extension).
    encoding : str
        File encoding. Defaults to 'utf-8'.
    """

    def __init__(self, delimiter: str = ",", encoding: str = "utf-8"):
        self.delimiter = delimiter
        self.encoding = encoding

    def load(self, path: str | Path) -> Dataset:
        """Load a CSV file and return a Dataset."""
        raise NotImplementedError("CSVIngester.load() — coming in v0.1.0")
