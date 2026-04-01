import warnings


def pytest_configure(config):
    # NumbaWarning about TBB threading layer is not reliably caught by
    # pyproject.toml filterwarnings because numba may issue it before
    # pytest installs its own warning filters. Suppress it here directly.
    try:
        from numba.core.errors import NumbaWarning

        warnings.filterwarnings("ignore", category=NumbaWarning)
    except ImportError:
        pass
