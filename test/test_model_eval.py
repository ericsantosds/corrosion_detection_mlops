from unittest.mock import Mock

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS

from src.model_eval import yolo


def test_register_model_version_supports_existing_registry_model(monkeypatch):
    client = Mock()
    client.create_registered_model.side_effect = MlflowException(
        "Model already exists",
        error_code=RESOURCE_ALREADY_EXISTS,
    )
    expected_version = object()
    client.create_model_version.return_value = expected_version
    monkeypatch.setattr(
        yolo.RunsArtifactRepository,
        "get_underlying_uri",
        lambda uri: "file:///mlflow/artifacts/run/yolo_weights/best.pt",
    )

    version = yolo._register_model_version(
        client,
        "Corrosion_Detection_YOLO_Model",
        "runs:/run-id/yolo_weights/best.pt",
        "run-id",
    )

    assert version is expected_version
    client.create_model_version.assert_called_once_with(
        name="Corrosion_Detection_YOLO_Model",
        source="file:///mlflow/artifacts/run/yolo_weights/best.pt",
        run_id="run-id",
    )
