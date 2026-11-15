"""Load project configuration files."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

import yaml


def load_config(path):
    """Load settings from `data/data.yaml` and return selected or nested keys."""
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


@contextmanager
def project_dataset_yaml(project_root):
    """Provide a temporary dataset YAML with a root valid in this checkout."""
    data_yaml = Path(project_root) / "data" / "data.yaml"
    config = load_config(data_yaml)
    config["path"] = str(data_yaml.parent.resolve())

    with tempfile.TemporaryDirectory(prefix="corrosion-dataset-") as temp_dir:
        normalized_yaml = Path(temp_dir) / "data.yaml"
        with normalized_yaml.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        yield str(normalized_yaml)
