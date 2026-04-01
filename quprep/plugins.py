"""Plugin registry — register custom encoders and exporters by name.

Usage
-----
Register a custom encoder so it works with :func:`quprep.prepare`
and the CLI::

    from quprep.plugins import register_encoder
    from quprep.encode.base import BaseEncoder, EncodedResult
    import numpy as np

    @register_encoder("my_encoder")
    class MyEncoder(BaseEncoder):
        @property
        def n_qubits(self):
            return None

        @property
        def depth(self):
            return 1

        def encode(self, x: np.ndarray) -> EncodedResult:
            return EncodedResult(
                parameters=x.copy(),
                metadata={"encoding": "my_encoder", "n_qubits": len(x), "depth": 1},
            )

    # Now usable everywhere:
    result = quprep.prepare("data.csv", encoding="my_encoder")

Register a custom exporter::

    from quprep.plugins import register_exporter

    @register_exporter("my_backend")
    class MyExporter:
        def export(self, encoded):
            ...
        def export_batch(self, encoded_list):
            return [self.export(e) for e in encoded_list]

    result = quprep.prepare("data.csv", framework="my_backend")

Inspect the registry::

    from quprep.plugins import list_encoders, list_exporters
    print(list_encoders())   # ['angle', 'amplitude', ..., 'my_encoder']
    print(list_exporters())  # ['qasm', 'qiskit', ..., 'my_backend']
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T")

# name → class (not instance) — instantiated on use
_encoder_registry: dict[str, type] = {}
_exporter_registry: dict[str, type] = {}


def register_encoder(name: str) -> Callable[[type[_T]], type[_T]]:
    """
    Register a custom encoder class under ``name``.

    Can be used as a class decorator or called directly:

    .. code-block:: python

        @register_encoder("my_enc")
        class MyEncoder(BaseEncoder): ...

        # or:
        register_encoder("my_enc")(MyEncoder)

    Parameters
    ----------
    name : str
        Encoding name used in :func:`quprep.prepare` and CLI.

    Returns
    -------
    Callable
        A decorator that registers and returns the class unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered (use :func:`unregister_encoder` first).
    """
    def decorator(cls: type[_T]) -> type[_T]:
        if name in _encoder_registry:
            raise ValueError(
                f"Encoder '{name}' is already registered. "
                "Call unregister_encoder('{name}') first."
            )
        _encoder_registry[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def register_exporter(name: str) -> Callable[[type[_T]], type[_T]]:
    """
    Register a custom exporter class under ``name``.

    Parameters
    ----------
    name : str
        Framework name used in :func:`quprep.prepare`.

    Returns
    -------
    Callable
        A decorator that registers and returns the class unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered.
    """
    def decorator(cls: type[_T]) -> type[_T]:
        if name in _exporter_registry:
            raise ValueError(
                f"Exporter '{name}' is already registered. "
                "Call unregister_exporter('{name}') first."
            )
        _exporter_registry[name] = cls  # type: ignore[assignment]
        return cls

    return decorator


def unregister_encoder(name: str) -> None:
    """Remove an encoder from the registry (useful for testing)."""
    _encoder_registry.pop(name, None)


def unregister_exporter(name: str) -> None:
    """Remove an exporter from the registry (useful for testing)."""
    _exporter_registry.pop(name, None)


def get_encoder_class(name: str) -> type | None:
    """Return the encoder class registered under ``name``, or ``None``."""
    return _encoder_registry.get(name)


def get_exporter_class(name: str) -> type | None:
    """Return the exporter class registered under ``name``, or ``None``."""
    return _exporter_registry.get(name)


def list_encoders() -> list[str]:
    """Return names of all registered plugin encoders."""
    return sorted(_encoder_registry)


def list_exporters() -> list[str]:
    """Return names of all registered plugin exporters."""
    return sorted(_exporter_registry)
