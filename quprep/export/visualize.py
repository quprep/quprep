"""Circuit visualization — matplotlib and ASCII diagrams."""

from __future__ import annotations

from pathlib import Path


def draw_ascii(encoded, width: int = 80) -> str:
    """
    Return an ASCII circuit diagram for an EncodedResult.

    No additional dependencies required.

    Parameters
    ----------
    encoded : EncodedResult
    width : int
        Target line width. Default 80.
    """
    raise NotImplementedError("draw_ascii() — coming in v0.2.0")


def draw_matplotlib(encoded, filename: str | Path | None = None):
    """
    Draw a matplotlib circuit diagram.

    Requires: pip install quprep[viz]

    Parameters
    ----------
    encoded : EncodedResult
    filename : str or Path, optional
        Save to file if provided (PNG, PDF, SVG).
    """
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        raise ImportError(
            "matplotlib is not installed. Run: pip install quprep[viz]"
        ) from None
    raise NotImplementedError("draw_matplotlib() — coming in v0.2.0")
