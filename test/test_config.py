from pathlib import Path

import yaml

from helper.config import project_dataset_yaml


def test_project_dataset_yaml_uses_current_checkout_path(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "data.yaml").write_text(
        "path: /old/host/checkout/data\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "names:\n"
        "  0: corrosion\n",
        encoding="utf-8",
    )

    with project_dataset_yaml(tmp_path) as normalized_yaml:
        config = yaml.safe_load(Path(normalized_yaml).read_text(encoding="utf-8"))
        assert config["path"] == str(data_dir.resolve())
        assert config["train"] == "images/train"
        assert Path(normalized_yaml).is_file()

    assert not Path(normalized_yaml).exists()
