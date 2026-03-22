from pathlib import Path

import tomllib


def on_config(config):
    with open(Path(__file__).parent.parent / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    config.extra["pkg_version"] = data["project"]["version"]
    return config
