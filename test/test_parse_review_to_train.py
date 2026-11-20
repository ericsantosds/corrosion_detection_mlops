import os
import json
import subprocess
import pytest
from unittest.mock import patch, MagicMock

from src.data_automation.parse_curated_data_to_train import (
    main,
    trigger_airflow_phase2,
)


def test_parse_and_conversion_success(tmp_path):
    curated_dir = tmp_path / "curated_data"
    curated_dir.mkdir()
    review_dir = tmp_path / "review_data"
    review_dir.mkdir()

    train_images = tmp_path / "data" / "images" / "train"
    train_labels = tmp_path / "data" / "labels" / "train"

    # Simula um JSON legítimo exportado pelo Labelme (Retângulo de corrosão)
    fake_labelme_data = {
        "imagePath": "sample_corrosion.jpg",
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [
            {
                "shape_type": "rectangle",
                "label": "corrosion",
                "points": [[10, 10], [50, 50]]  # x1, y1, x2, y2
            }
        ]
    }

    production_prediction = dict(fake_labelme_data)
    production_prediction["shapes"] = [
        {
            "shape_type": "rectangle",
            "label": "corrosion",
            "points": [[0, 0], [20, 20]],
        }
    ]

    # Keeps the production prediction as an immutable source copy.
    with open(review_dir / "sample_corrosion.json", "w") as f:
        json.dump(production_prediction, f)
    (review_dir / "sample_corrosion.jpg").write_text("fake_binary_image")
    with open(curated_dir / "sample_corrosion.json", "w") as f:
        json.dump(fake_labelme_data, f)
    (curated_dir / "sample_corrosion.jpg").write_text("fake_binary_image")

    with patch("src.data_automation.parse_curated_data_to_train.CURATED_DIR", str(curated_dir)), \
            patch("src.data_automation.parse_curated_data_to_train.TRAIN_IMAGES_DIR", str(train_images)), \
            patch("src.data_automation.parse_curated_data_to_train.TRAIN_LABELS_DIR", str(train_labels)), \
            patch("src.data_automation.parse_curated_data_to_train.trigger_airflow_phase2") as mock_trigger:

        main()

        # Garante que o gatilho da CLI do Airflow foi acionado após o reprocessamento com sucesso
        mock_trigger.assert_called_once()

    # ---- VALIDAÇÃO DA CONVERSÃO MATEMÁTICA ----
    txt_output_path = os.path.join(train_labels, "sample_corrosion.txt")
    assert os.path.exists(txt_output_path)

    with open(txt_output_path, "r") as f:
        yolo_line = f.read().strip()

    # Coordenadas convertidas esperadas: Centro_X=0.3, Centro_Y=0.3, Largura=0.4, Altura=0.4
    assert yolo_line == "0 0.300000 0.300000 0.400000 0.400000"

    # Curated files are consumed; the production prediction remains untouched.
    assert os.path.exists(os.path.join(train_images, "sample_corrosion.jpg"))
    assert not os.path.exists(curated_dir / "sample_corrosion.json")
    assert not os.path.exists(curated_dir / "sample_corrosion.jpg")
    assert (review_dir / "sample_corrosion.json").exists()
    assert (review_dir / "sample_corrosion.jpg").read_text() == "fake_binary_image"
    with open(review_dir / "sample_corrosion.json", encoding="utf-8") as file:
        assert json.load(file) == production_prediction


def test_unpaired_curated_image_is_not_ignored(tmp_path):
    curated_dir = tmp_path / "curated_data"
    curated_dir.mkdir()
    (curated_dir / "unreviewed.jpg").write_text("image")

    with patch(
        "src.data_automation.parse_curated_data_to_train.CURATED_DIR",
        str(curated_dir),
    ):
        with pytest.raises(ValueError, match="Every curated image"):
            main()


def test_existing_training_sample_is_not_overwritten(tmp_path, monkeypatch):
    curated_dir = tmp_path / "curated_data"
    curated_dir.mkdir()
    train_images = tmp_path / "data" / "images" / "train"
    train_labels = tmp_path / "data" / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    (train_images / "sample.jpg").write_text("existing image")
    (train_labels / "sample.txt").write_text("existing label")

    annotation = {
        "imagePath": "sample.jpg",
        "imageWidth": 100,
        "imageHeight": 100,
        "shapes": [],
    }
    (curated_dir / "sample.json").write_text(json.dumps(annotation))
    (curated_dir / "sample.jpg").write_text("new image")
    monkeypatch.setenv("CORROSION_SKIP_AIRFLOW_TRIGGER", "1")

    with patch(
        "src.data_automation.parse_curated_data_to_train.CURATED_DIR",
        str(curated_dir),
    ), patch(
        "src.data_automation.parse_curated_data_to_train.TRAIN_IMAGES_DIR",
        str(train_images),
    ), patch(
        "src.data_automation.parse_curated_data_to_train.TRAIN_LABELS_DIR",
        str(train_labels),
    ), patch(
        "src.data_automation.parse_curated_data_to_train.trigger_airflow_phase2"
    ) as mock_trigger:
        main()

    mock_trigger.assert_not_called()
    assert (train_images / "sample.jpg").read_text() == "existing image"
    assert (train_labels / "sample.txt").read_text() == "existing label"
    assert (train_images / "sample_curated_1.jpg").read_text() == "new image"
    assert (train_labels / "sample_curated_1.txt").read_text() == ""


@patch("subprocess.run")
def test_trigger_airflow_cli(mock_run, monkeypatch, tmp_path):
    """Garante que a CLI local do Airflow é chamada com os argumentos corretos."""
    airflow_home = str(tmp_path / "airflow")
    monkeypatch.setenv("AIRFLOW_HOME", airflow_home)
    monkeypatch.setenv("CORROSION_AIRFLOW_TARGET", "local")

    trigger_airflow_phase2()

    mock_run.assert_called_once()
    args_called = mock_run.call_args[0][0]
    assert "dags" in args_called
    assert "trigger" in args_called
    assert "corrosion_phase2_retrain" in args_called
    assert mock_run.call_args.kwargs["env"]["AIRFLOW_HOME"] == airflow_home


@patch("subprocess.run")
def test_trigger_airflow_cli_in_docker(mock_run, monkeypatch):
    """Usa a CLI do serviço Airflow quando o alvo é o Compose."""
    monkeypatch.setenv("CORROSION_AIRFLOW_TARGET", "docker")

    trigger_airflow_phase2()

    mock_run.assert_called_once()
    command = mock_run.call_args.args[0]
    assert command[:5] == [
        "docker", "compose", "exec", "-T", "airflow-standalone",
    ]
    assert command[-3:] == [
        "dags", "trigger", "corrosion_phase2_retrain",
    ]


@patch("subprocess.run")
def test_trigger_airflow_auto_selects_running_docker(mock_run, monkeypatch):
    """Auto seleciona Docker Compose quando o serviço Airflow está ativo."""
    mock_run.side_effect = [
        MagicMock(stdout="airflow-standalone\n"),
        MagicMock(stdout="triggered\n"),
    ]
    monkeypatch.setenv("CORROSION_AIRFLOW_TARGET", "auto")

    trigger_airflow_phase2()

    assert mock_run.call_count == 2
    assert mock_run.call_args_list[1].args[0][:5] == [
        "docker", "compose", "exec", "-T", "airflow-standalone",
    ]


@patch("subprocess.run")
def test_trigger_airflow_auto_falls_back_to_local_when_docker_unavailable(
    mock_run, monkeypatch, capsys
):
    """Uses local Airflow if Docker Compose cannot be queried."""
    mock_run.side_effect = [
        subprocess.CalledProcessError(
            1,
            ["docker", "compose", "ps"],
            stderr="Docker daemon is unavailable",
        ),
        MagicMock(stdout="triggered\n"),
    ]
    monkeypatch.setenv("CORROSION_AIRFLOW_TARGET", "auto")

    trigger_airflow_phase2()

    assert mock_run.call_count == 2
    local_command = mock_run.call_args_list[1].args[0]
    assert local_command[0].endswith("/airflow")
    assert local_command[1:] == [
        "dags", "trigger", "corrosion_phase2_retrain",
    ]
    assert "Falling back to local Airflow" in capsys.readouterr().out
