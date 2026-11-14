"""Load project configuration files."""

import yaml


def load_config(path):
    """Load settings from `data/data.yaml` and return selected or nested keys."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
