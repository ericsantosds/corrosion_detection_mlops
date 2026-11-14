from types import SimpleNamespace
from unittest.mock import Mock

from src.serving import app as serving_app


def test_load_champion_model_downloads_weights_from_mlflow(monkeypatch, tmp_path):
    model_path = str(tmp_path / "best.pt")
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        run_id=None,
        source="file:///artifacts/model.pt",
    )
    download = Mock(return_value=model_path)
    monkeypatch.setattr(serving_app, "MlflowClient", lambda: client)
    monkeypatch.setattr(serving_app, "download_artifacts", download)
    monkeypatch.setattr(serving_app, "YOLO", lambda path: path)

    assert serving_app.load_champion_model() == model_path
    download.assert_called_once_with(artifact_uri="file:///artifacts/model.pt")
