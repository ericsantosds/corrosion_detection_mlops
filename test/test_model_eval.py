from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
)

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


def test_missing_champion_alias_is_the_only_lookup_error_treated_as_absent():
    client = Mock()
    client.get_model_version_by_alias.side_effect = MlflowException(
        "Champion alias does not exist",
        error_code=RESOURCE_DOES_NOT_EXIST,
    )

    assert yolo._get_champion_version(
        client, "Corrosion_Detection_YOLO_Model") is None

    client.get_model_version_by_alias.side_effect = MlflowException(
        "Registry unavailable",
        error_code="INTERNAL_ERROR",
    )
    with pytest.raises(MlflowException, match="Registry unavailable"):
        yolo._get_champion_version(client, "Corrosion_Detection_YOLO_Model")


@pytest.mark.parametrize(
    ("evaluation_function", "expected_split"),
    [
        (yolo._evaluate_validation, "val"),
        (yolo._evaluate_final_test, "test"),
    ],
)
def test_evaluation_uses_the_requested_dataset_split(
    evaluation_function, expected_split,
):
    model = Mock()

    evaluation_function(model, "dataset.yaml")

    model.val.assert_called_once_with(
        data="dataset.yaml",
        split=expected_split,
        workers=0,
    )


def test_existing_champion_is_rescored_on_validation(monkeypatch):
    champion_version = SimpleNamespace(source="models:/champion/1")
    champion_model = Mock()
    monkeypatch.setattr(
        yolo, "download_artifacts", lambda artifact_uri: "champion.pt")
    monkeypatch.setattr(yolo, "YOLO", lambda _: champion_model)
    monkeypatch.setattr(
        yolo, "project_dataset_yaml", lambda _: nullcontext("dataset.yaml"))
    validation_metrics = SimpleNamespace(
        box=SimpleNamespace(map50=0.82))
    evaluate = Mock(return_value=validation_metrics)
    monkeypatch.setattr(yolo, "_evaluate_validation", evaluate)

    score = yolo._get_champion_validation_map50(champion_version)

    assert score == 0.82
    evaluate.assert_called_once_with(champion_model, "dataset.yaml")


@pytest.mark.parametrize("challenger_map50", [0.79, 0.8])
def test_challenger_not_better_keeps_current_champion(
    challenger_map50, monkeypatch,
):
    monkeypatch.setattr(yolo, "_get_champion_validation_map50", lambda _: 0.8)
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="1",
        run_id="champion-run",
    )
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={"metrics_mAP50": 0.8}),
    )
    candidate = SimpleNamespace(version="2")

    promoted = yolo._promote_if_better(
        client,
        "Corrosion_Detection_YOLO_Model",
        candidate,
        challenger_map50,
    )

    assert promoted is False
    client.set_registered_model_alias.assert_not_called()


def test_challenger_that_beats_champion_is_promoted(monkeypatch):
    monkeypatch.setattr(yolo, "_get_champion_validation_map50", lambda _: 0.8)
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="1",
        run_id="champion-run",
    )
    client.get_run.return_value = SimpleNamespace(
        data=SimpleNamespace(metrics={"metrics_mAP50": 0.8}),
    )
    candidate = SimpleNamespace(version="2")

    promoted = yolo._promote_if_better(
        client,
        "Corrosion_Detection_YOLO_Model",
        candidate,
        0.81,
    )

    assert promoted is True
    client.set_registered_model_alias.assert_called_once_with(
        "Corrosion_Detection_YOLO_Model",
        "champion",
        "2",
    )
