"""Load project configuration files."""

from contextlib import contextmanager
from pathlib import Path
import tempfile

import yaml


def load_config(path):
    """Load a YAML configuration file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        The parsed YAML value.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


@contextmanager
def project_dataset_yaml(project_root):
    """Yield a temporary dataset YAML rooted at the current checkout.

    Args:
        project_root: Root directory of the project checkout.

    Yields:
        Path to the temporary YAML file. The file is deleted when the context
        manager exits.
    """
    data_yaml = Path(project_root) / "data" / "data.yaml"
    config = load_config(data_yaml)
    config["path"] = str(data_yaml.parent.resolve())

    with tempfile.TemporaryDirectory(prefix="corrosion-dataset-") as temp_dir:
        normalized_yaml = Path(temp_dir) / "data.yaml"
        with normalized_yaml.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)
        yield str(normalized_yaml)
