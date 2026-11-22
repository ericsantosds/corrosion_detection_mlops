"""Evaluate YOLO models and promote the best eligible model through MLflow."""

import os
from pathlib import Path
import sys
from typing import Any
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    INVALID_PARAMETER_VALUE,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from ultralytics import YOLO, settings
import yaml

from helper.config import load_config, project_dataset_yaml

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "Corrosion_Detection_YOLO_Model"
EXPERIMENT_NAME = "Corrosion_Detection_YOLO"
METRICS = (
    ("precision", "metrics_precision"),
    ("recall", "metrics_recall"),
    ("mAP50", "metrics_mAP50"),
    ("mAP50:95", "metrics_mAP50_95"),
)


def _register_model_version(
    client: MlflowClient, model_name: str, model_uri: str, run_id: str,
) -> Any:
    try:
        client.create_registered_model(model_name)
    except MlflowException as error:
        existing_codes = {
            ErrorCode.Name(RESOURCE_ALREADY_EXISTS),
            ErrorCode.Name(ALREADY_EXISTS),
        }
        if error.error_code not in existing_codes:
            raise

    return client.create_model_version(
        name=model_name,
        source=RunsArtifactRepository.get_underlying_uri(model_uri),
        run_id=run_id,
    )


def _get_champion_version(client: MlflowClient, model_name: str) -> Any | None:
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as error:
        missing_alias = "Registered model alias champion not found."
        missing_codes = {
            RESOURCE_DOES_NOT_EXIST,
            ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
        }
        invalid_codes = {
            INVALID_PARAMETER_VALUE,
            ErrorCode.Name(INVALID_PARAMETER_VALUE),
        }
        if error.error_code in missing_codes or (
            error.error_code in invalid_codes
            and error.message in {
                missing_alias,
                f"{ErrorCode.Name(INVALID_PARAMETER_VALUE)}: {missing_alias}",
            }
        ):
            return None
        raise


def _evaluate_test(model: YOLO, data_yaml_path: str) -> Any:
    return model.val(data=data_yaml_path, split="test", workers=0)


def _metric_value(value: Any) -> float:
    try:
        return float(value[0])
    except (IndexError, KeyError, TypeError):
        return float(value)


def _extract_metrics(metrics: Any) -> dict[str, float]:
    return {
        "metrics_precision": _metric_value(metrics.box.p),
        "metrics_recall": _metric_value(metrics.box.r),
        "metrics_mAP50": float(metrics.box.map50),
        "metrics_mAP50_95": float(metrics.box.map),
    }


def _get_champion_test_metrics(champion_version: Any) -> dict[str, float]:
    if not champion_version.source:
        raise ValueError("The champion model has no artifact source URI.")

    model = YOLO(download_artifacts(artifact_uri=champion_version.source))
    with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
        return _extract_metrics(_evaluate_test(model, data_yaml_path))


def _promote_if_better(
    client: MlflowClient,
    model_name: str,
    candidate: Any,
    challenger_metrics: dict[str, float],
) -> bool:
    champion = _get_champion_version(client, model_name)
    if champion is None:
        client.set_registered_model_alias(model_name, "champion", candidate.version)
        mlflow.set_tag("promotion_decision", "promoted_first_champion")
        mlflow.log_param("champion_version", "none")
        mlflow.log_param("challenger_version", str(candidate.version))
        return True

    champion_metrics = _get_champion_test_metrics(champion)
    for role, values in (
        ("champion", champion_metrics),
        ("challenger", challenger_metrics),
    ):
        mlflow.log_metrics({
            f"{role}_{name.removeprefix('metrics_')}": value
            for name, value in values.items()
        })

    print("Test metrics (precision, recall, mAP50, mAP50:95)")
    for role, values in (
        ("Champion", champion_metrics),
        ("Challenger", challenger_metrics),
    ):
        scores = " | ".join(f"{label}={values[key]:.4f}" for label, key in METRICS)
        print(f"{role}: {scores}")

    mlflow.log_param("champion_version", str(champion.version))
    mlflow.log_param("challenger_version", str(candidate.version))
    if challenger_metrics["metrics_mAP50"] <= champion_metrics["metrics_mAP50"]:
        mlflow.set_tag("promotion_decision", "kept_champion")
        return False

    client.set_registered_model_alias(model_name, "champion", candidate.version)
    mlflow.set_tag("promotion_decision", "promoted_challenger")
    return True


def _get_latest_model_path() -> Path:
    detect_dir = PROJECT_ROOT / "runs" / "detect"
    if not detect_dir.is_dir():
        raise FileNotFoundError(f"No training directory found at: {detect_dir}")

    training_runs = [
        path for path in detect_dir.iterdir()
        if path.is_dir() and not path.name.startswith(("predict", "val"))
    ]
    if not training_runs:
        raise FileNotFoundError(f"No valid training folder found at: {detect_dir}")

    model_path = max(training_runs, key=lambda path: path.stat().st_mtime)
    model_path = model_path / "weights" / "best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"The weights file 'best.pt' was not found at: {model_path}")
    return model_path


def _log_evaluation_metrics(metrics: Any) -> dict[str, float]:
    values = _extract_metrics(metrics)
    mlflow.log_metrics(values)
    return values


def _set_model_metadata(client: MlflowClient, model_name: str) -> None:
    for key, value in {
        "framework": "ultralytics-yolo",
        "task": "corrosion-detection",
    }.items():
        client.set_registered_model_tag(name=model_name, key=key, value=value)

    dvc_hash = "unknown"
    dvc_file = PROJECT_ROOT / "data.dvc"
    if dvc_file.is_file():
        with dvc_file.open(encoding="utf-8") as file:
            dvc_hash = (yaml.safe_load(file) or {}).get(
                "outs", [{}],
            )[0].get("md5", "unknown")
    client.set_registered_model_tag(
        name=model_name, key="dataset_version_md5", value=dvc_hash,
    )


def _run_governance_evaluation(model: YOLO, model_path: Path) -> bool:
    client = MlflowClient()
    with mlflow.start_run(run_name="champion_promotion_test") as run:
        mlflow.set_tag("pipeline_stage", "champion_promotion")
        mlflow.log_param("evaluated_model_path", str(model_path))
        with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
            metrics = _log_evaluation_metrics(_evaluate_test(model, data_yaml_path))

        model_uri = f"runs:/{run.info.run_id}/yolo_weights/{model_path.name}"
        mlflow.log_artifact(str(model_path), artifact_path="yolo_weights")
        candidate = _register_model_version(
            client, MODEL_NAME, model_uri, run.info.run_id,
        )

        business_config = load_config(PROJECT_ROOT / "business_args.yaml")
        if (
            metrics["metrics_mAP50"] < float(business_config["MIN_MAP50"])
            or metrics["metrics_recall"] < float(business_config["MIN_RECALL"])
        ):
            mlflow.set_tag("promotion_decision", "rejected_business_rules")
            print("Candidate rejected: business quality thresholds were not met.")
            return False

        promoted = _promote_if_better(client, MODEL_NAME, candidate, metrics)
        _set_model_metadata(client, MODEL_NAME)
        print(
            "Candidate promoted to champion."
            if promoted else "Candidate did not outperform the current champion."
        )
        return promoted


def main() -> None:
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
    )
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
    settings.update({"mlflow": True})

    model_path = _get_latest_model_path()
    print(f"Loading model weights from: {model_path}")
    promoted = _run_governance_evaluation(YOLO(str(model_path)), model_path)
    print(f"MODEL_PROMOTED={str(promoted).lower()}")


if __name__ == "__main__":
    main()
