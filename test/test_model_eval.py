from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mlflow.exceptions import MlflowException, RestException
from mlflow.protos.databricks_pb2 import (
    INVALID_PARAMETER_VALUE,
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
        "Registered model alias champion not found.",
        error_code=INVALID_PARAMETER_VALUE,
    )
    assert yolo._get_champion_version(
        client, "Corrosion_Detection_YOLO_Model") is None

    client.get_model_version_by_alias.side_effect = RestException({
        "error_code": "INVALID_PARAMETER_VALUE",
        "message": "Registered model alias champion not found.",
    })
    assert yolo._get_champion_version(
        client, "Corrosion_Detection_YOLO_Model") is None

    client.get_model_version_by_alias.side_effect = MlflowException(
        "Registry unavailable",
        error_code="INTERNAL_ERROR",
    )
    with pytest.raises(MlflowException, match="Registry unavailable"):
        yolo._get_champion_version(client, "Corrosion_Detection_YOLO_Model")

    client.get_model_version_by_alias.side_effect = MlflowException(
        "Registered model name not found.",
        error_code=INVALID_PARAMETER_VALUE,
    )
    with pytest.raises(MlflowException, match="Registered model name not found"):
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


def test_existing_champion_is_rescored_on_test(monkeypatch):
    champion_version = SimpleNamespace(source="models:/champion/1")
    champion_model = Mock()
    monkeypatch.setattr(
        yolo, "download_artifacts", lambda artifact_uri: "champion.pt")
    monkeypatch.setattr(yolo, "YOLO", lambda _: champion_model)
    monkeypatch.setattr(
        yolo, "project_dataset_yaml", lambda _: nullcontext("dataset.yaml"))
    test_metrics = SimpleNamespace(
        box=SimpleNamespace(
            r=0.72,
            p=0.76,
            map50=0.82,
            map=0.62,
        ))
    evaluate = Mock(return_value=test_metrics)
    monkeypatch.setattr(yolo, "_evaluate_final_test", evaluate)

    score = yolo._get_champion_test_map50(champion_version)

    assert score == 0.82
    evaluate.assert_called_once_with(champion_model, "dataset.yaml")


@pytest.mark.parametrize("challenger_map50", [0.79, 0.8])
def test_challenger_not_better_keeps_current_champion(
    challenger_map50, monkeypatch,
):
    monkeypatch.setattr(yolo.mlflow, "set_tag", Mock())
    monkeypatch.setattr(yolo.mlflow, "log_param", Mock())
    monkeypatch.setattr(yolo.mlflow, "log_metrics", Mock())
    monkeypatch.setattr(
        yolo,
        "_get_champion_test_metrics",
        lambda _: {
            "metrics_recall": 0.7,
            "metrics_precision": 0.75,
            "metrics_mAP50": 0.8,
            "metrics_mAP50_95": 0.6,
        },
    )
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="1", source="models:/champion/1",
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
    monkeypatch.setattr(yolo.mlflow, "set_tag", Mock())
    monkeypatch.setattr(yolo.mlflow, "log_param", Mock())
    monkeypatch.setattr(yolo.mlflow, "log_metrics", Mock())
    monkeypatch.setattr(
        yolo,
        "_get_champion_test_metrics",
        lambda _: {
            "metrics_recall": 0.7,
            "metrics_precision": 0.75,
            "metrics_mAP50": 0.8,
            "metrics_mAP50_95": 0.6,
        },
    )
    client = Mock()
    client.get_model_version_by_alias.return_value = SimpleNamespace(
        version="1", source="models:/champion/1",
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


def test_governance_evaluation_uses_test_metrics_for_promotion(
    monkeypatch, tmp_path,
):
    """Use test metrics for promotion and avoid validation-based selection."""
    run = SimpleNamespace(info=SimpleNamespace(run_id="candidate-run"))
    candidate = SimpleNamespace(version="2")
    model = Mock()
    client = Mock()
    evaluate_test = Mock(return_value=object())
    evaluate_validation = Mock()
    metrics = {
        "metrics_mAP50": 0.8,
        "metrics_recall": 0.75,
        "metrics_precision": 0.7,
        "metrics_mAP50_95": 0.6,
    }

    monkeypatch.setattr(
        yolo.mlflow, "start_run", lambda **_: nullcontext(run))
    monkeypatch.setattr(yolo, "MlflowClient", lambda: client)
    monkeypatch.setattr(
        yolo, "project_dataset_yaml", lambda _: nullcontext("dataset.yaml"))
    monkeypatch.setattr(yolo, "_evaluate_final_test", evaluate_test)
    monkeypatch.setattr(yolo, "_evaluate_validation", evaluate_validation)
    monkeypatch.setattr(yolo, "_log_evaluation_metrics", lambda _: metrics)
    monkeypatch.setattr(yolo, "_register_model_version", lambda *args: candidate)
    monkeypatch.setattr(
        yolo, "load_config", lambda _: {"MIN_MAP50": 0.5, "MIN_RECALL": 0.5})
    promote = Mock(return_value=True)
    monkeypatch.setattr(yolo, "_promote_if_better", promote)
    monkeypatch.setattr(yolo, "_set_model_metadata", Mock())
    monkeypatch.setattr(yolo.mlflow, "set_tag", Mock())
    monkeypatch.setattr(yolo.mlflow, "log_param", Mock())
    monkeypatch.setattr(yolo.mlflow, "log_artifact", Mock())

    assert yolo._run_governance_evaluation(
        model, tmp_path / "best.pt") is True

    evaluate_test.assert_called_once_with(model, "dataset.yaml")
    evaluate_validation.assert_not_called()
    promote.assert_called_once_with(
        client,
        yolo.MODEL_NAME,
        candidate,
        metrics["metrics_mAP50"],
        challenger_metrics=metrics,
    )
