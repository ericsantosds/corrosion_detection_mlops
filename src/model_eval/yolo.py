"""Evaluate YOLO models and govern champion promotion through MLflow."""

import argparse
import os
from pathlib import Path
import sys
from typing import Any
import warnings

import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    RESOURCE_ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    ErrorCode,
)
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from ultralytics import YOLO, settings
import yaml
from helper.config import load_config, project_dataset_yaml

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_NAME = "Corrosion_Detection_YOLO_Model"
EXPERIMENT_NAME = "Corrosion_Detection_YOLO"


def _register_model_version(
    client: MlflowClient,
    model_name: str,
    model_uri: str,
    run_id: str,
) -> Any:
    """Register a model artifact, creating the registry entry if necessary.

    Args:
        client: MLflow client used to access the model registry.
        model_name: Name of the registered model.
        model_uri: URI of the model artifact stored in the evaluation run.
        run_id: ID of the MLflow run that logged the artifact.

    Returns:
        The created MLflow model-version metadata.

    Raises:
        MlflowException: If registration fails for a reason other than the
            model already existing.
    """
    try:
        client.create_registered_model(model_name)
    except MlflowException as error:
        existing_model_codes = {
            ErrorCode.Name(RESOURCE_ALREADY_EXISTS),
            ErrorCode.Name(ALREADY_EXISTS),
        }
        if error.error_code not in existing_model_codes:
            raise

    return client.create_model_version(
        name=model_name,
        source=RunsArtifactRepository.get_underlying_uri(model_uri),
        run_id=run_id,
    )


def _get_champion_version(client: MlflowClient, model_name: str) -> Any | None:
    """Return the version with the champion alias, if one is registered.

    Args:
        client: MLflow client used to query the registry.
        model_name: Name of the registered model.

    Returns:
        The aliased model version, or ``None`` if the alias does not exist.

    Raises:
        MlflowException: If the registry lookup fails for a reason other than
            the missing alias.
    """
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as error:
        missing_alias_codes = {
            RESOURCE_DOES_NOT_EXIST,
            ErrorCode.Name(RESOURCE_DOES_NOT_EXIST),
        }
        if error.error_code not in missing_alias_codes:
            raise
        return None


def _evaluate_validation(model: YOLO, data_yaml_path: str) -> Any:
    """Evaluate a model on the validation split.

    Args:
        model: YOLO model to evaluate.
        data_yaml_path: Path to the dataset configuration.

    Returns:
        Ultralytics validation metrics. This split is not used for promotion.
    """
    return model.val(data=data_yaml_path, split="val", workers=0)


def _evaluate_final_test(model: YOLO, data_yaml_path: str) -> Any:
    """Evaluate a model on the test split.

    Args:
        model: YOLO model to evaluate.
        data_yaml_path: Path to the dataset configuration.

    Returns:
        Ultralytics test metrics.
    """
    return model.val(data=data_yaml_path, split="test", workers=0)


def _get_champion_test_metrics(champion_version: Any) -> dict[str, float]:
    """Rescore the current champion and return all test metrics.

    Args:
        champion_version: MLflow registry metadata for the current champion.

    Returns:
        The champion's test recall, precision, mAP50, and mAP50-95.

    Raises:
        ValueError: If the champion has no artifact source URI.
    """
    if not champion_version.source:
        raise ValueError("The champion model has no artifact source URI.")

    model_path = download_artifacts(artifact_uri=champion_version.source)
    model = YOLO(model_path)
    with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
        metrics = _evaluate_final_test(model, data_yaml_path)
    return {
        "metrics_recall": _metric_value(metrics.box.r),
        "metrics_precision": _metric_value(metrics.box.p),
        "metrics_mAP50": float(metrics.box.map50),
        "metrics_mAP50_95": float(metrics.box.map),
    }


def _get_champion_test_map50(champion_version: Any) -> float:
    """Return the rescored champion test mAP50 for promotion comparisons."""
    return _get_champion_test_metrics(champion_version)["metrics_mAP50"]


def _promote_if_better(
    client: MlflowClient,
    model_name: str,
    candidate: Any,
    challenger_map50: float,
    challenger_metrics: dict[str, float] | None = None,
) -> bool:
    """Promote a challenger only when it beats the champion on test data.

    Args:
        client: MLflow client used to query and update the registry.
        model_name: Name of the registered model.
        candidate: Newly registered candidate model version.
        challenger_map50: Candidate's test mAP50.

    Returns:
        Whether the candidate was promoted.
    """
    champion_version = _get_champion_version(client, model_name)
    if champion_version is None:
        client.set_registered_model_alias(
            model_name, "champion", candidate.version)
        mlflow.set_tag("promotion_decision", "promoted_first_champion")
        mlflow.log_param("champion_version", "none")
        mlflow.log_param("challenger_version", str(candidate.version))
        mlflow.log_param("tournament_result", "first_champion")
        return True

    champion_metrics = _get_champion_test_metrics(champion_version)
    champion_map50 = champion_metrics["metrics_mAP50"]
    mlflow.log_param("champion_version", str(champion_version.version))
    mlflow.log_param("challenger_version", str(candidate.version))
    mlflow.log_metrics({
        f"champion_{name}": value
        for name, value in champion_metrics.items()
    })
    if challenger_metrics:
        mlflow.log_metrics({
            f"challenger_{name}": value
            for name, value in challenger_metrics.items()
        })
        mlflow.log_metrics({
            "comparison_delta_recall": (
                challenger_metrics["metrics_recall"]
                - champion_metrics["metrics_recall"]
            ),
            "comparison_delta_precision": (
                challenger_metrics["metrics_precision"]
                - champion_metrics["metrics_precision"]
            ),
            "comparison_delta_mAP50": challenger_map50
            - champion_metrics["metrics_mAP50"],
            "comparison_delta_mAP50_95": (
                challenger_metrics["metrics_mAP50_95"]
                - champion_metrics["metrics_mAP50_95"]
            ),
        })

    if challenger_metrics:
        print(
            "Performance comparison - "
            f"Champion Recall={champion_metrics['metrics_recall']:.4f}, "
            f"Precision={champion_metrics['metrics_precision']:.4f}, "
            f"mAP50={champion_map50:.4f} | "
            f"Challenger Recall={challenger_metrics['metrics_recall']:.4f}, "
            f"Precision={challenger_metrics['metrics_precision']:.4f}, "
            f"mAP50={challenger_map50:.4f}"
        )
    else:
        print(
            f"Performance comparison - Champion mAP50={champion_map50:.4f} "
            f"| Challenger mAP50={challenger_map50:.4f}"
        )
    if challenger_map50 <= champion_map50:
        mlflow.set_tag("promotion_decision", "kept_champion")
        mlflow.log_param("tournament_result", "challenger_not_better")
        return False

    client.set_registered_model_alias(
        model_name, "champion", candidate.version)
    mlflow.set_tag("promotion_decision", "promoted_challenger")
    mlflow.log_param("tournament_result", "challenger_won")
    return True


def _metric_value(value: Any) -> float:
    """Convert a scalar or one-element metric array to a Python float.

    Args:
        value: Metric value returned by Ultralytics.

    Returns:
        The scalar metric value.
    """
    try:
        return float(value[0])
    except (IndexError, KeyError, TypeError):
        return float(value)


def _get_latest_model_path() -> Path:
    """Find the most recently modified YOLO training run's best weights.

    Returns:
        Path to the latest training run's ``best.pt`` weights.

    Raises:
        FileNotFoundError: If no training run or weights file is available.
    """
    detect_dir = PROJECT_ROOT / "runs" / "detect"
    if not detect_dir.is_dir():
        raise FileNotFoundError(
            f"No training directory found at: {detect_dir}")

    training_runs = [
        path for path in detect_dir.iterdir()
        if path.is_dir()
        and not path.name.startswith(("predict", "val"))
    ]
    if not training_runs:
        raise FileNotFoundError(
            f"No valid training folder found at: {detect_dir}")

    latest_run = max(training_runs, key=lambda path: path.stat().st_mtime)
    model_path = latest_run / "weights" / "best.pt"
    if not model_path.is_file():
        raise FileNotFoundError(
            f"The weights file 'best.pt' was not found at: {model_path}")
    return model_path


def _log_evaluation_metrics(metrics: Any, prefix: str = "") -> dict[str, float]:
    """Extract and log precision, recall, and mAP metrics to MLflow.

    Args:
        metrics: Ultralytics validation or test result.
        prefix: Optional metric-name prefix, such as ``"final_test_"``.

    Returns:
        A mapping of logged metric names to Python float values.
    """
    values = {
        "recall": _metric_value(metrics.box.r),
        "precision": _metric_value(metrics.box.p),
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
    }
    logged = {f"{prefix}metrics_{name}": value
              for name, value in values.items()}
    mlflow.log_metrics(logged)
    return logged


def _set_model_metadata(
    client: MlflowClient,
    model_name: str,
) -> None:
    """Attach framework, task, and dataset-version tags to a model.

    Args:
        client: MLflow client used to update model metadata.
        model_name: Name of the registered model.
    """
    client.set_registered_model_tag(
        name=model_name,
        key="framework",
        value="ultralytics-yolo",
    )
    client.set_registered_model_tag(
        name=model_name,
        key="task",
        value="corrosion-detection",
    )

    dvc_file_path = PROJECT_ROOT / "data.dvc"
    dvc_hash = "unknown"
    if dvc_file_path.is_file():
        with dvc_file_path.open(encoding="utf-8") as file:
            dvc_data = yaml.safe_load(file) or {}
        dvc_hash = dvc_data.get("outs", [{}])[0].get("md5", "unknown")

    client.set_registered_model_tag(
        name=model_name,
        key="dataset_version_md5",
        value=dvc_hash,
    )


def _run_final_test(model: YOLO, model_path: Path) -> None:
    """Record test metrics without registering or promoting a model.

    Args:
        model: Frozen YOLO model to evaluate.
        model_path: Path to the model weights.
    """
    with mlflow.start_run(run_name="final_test_evaluation"):
        mlflow.set_tag("pipeline_stage", "final_test")
        mlflow.log_param("evaluated_model_path", str(model_path))
        with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
            metrics = _evaluate_final_test(model, data_yaml_path)
        _log_evaluation_metrics(metrics, prefix="final_test_")


def _run_governance_evaluation(model: YOLO, model_path: Path) -> bool:
    """Evaluate a candidate on test data and apply promotion quality gates.

    Args:
        model: Candidate YOLO model to evaluate.
        model_path: Path to the candidate model weights.

    Returns:
        Whether the candidate became the registered champion.
    """
    client = MlflowClient()
    with mlflow.start_run(run_name="champion_promotion_test") as run:
        mlflow.set_tag("pipeline_stage", "champion_promotion")
        mlflow.log_param("evaluated_model_path", str(model_path))
        with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
            metrics = _evaluate_final_test(model, data_yaml_path)

        logged_metrics = _log_evaluation_metrics(metrics)
        model_uri = f"runs:/{run.info.run_id}/yolo_weights/{model_path.name}"
        mlflow.log_artifact(str(model_path), artifact_path="yolo_weights")
        candidate = _register_model_version(
            client,
            MODEL_NAME,
            model_uri,
            run.info.run_id,
        )

        business_config = load_config(PROJECT_ROOT / "business_args.yaml")
        min_map50 = float(business_config["MIN_MAP50"])
        min_recall = float(business_config["MIN_RECALL"])
        challenger_map50 = logged_metrics["metrics_mAP50"]
        challenger_recall = logged_metrics["metrics_recall"]
        mlflow.log_metrics({
            "business_min_mAP50": min_map50,
            "business_min_recall": min_recall,
            "business_challenger_mAP50": challenger_map50,
            "business_challenger_recall": challenger_recall,
        })
        mlflow.log_params({
            "business_map50_threshold_met": str(
                challenger_map50 >= min_map50).lower(),
            "business_recall_threshold_met": str(
                challenger_recall >= min_recall).lower(),
        })
        if challenger_map50 < min_map50 or challenger_recall < min_recall:
            mlflow.set_tag("promotion_decision", "rejected_business_rules")
            print("Candidate rejected: business quality thresholds were not met.")
            return False

        promoted = _promote_if_better(
            client,
            MODEL_NAME,
            candidate,
            challenger_map50,
            challenger_metrics=logged_metrics,
        )
        _set_model_metadata(client, MODEL_NAME)
        if promoted:
            print("Candidate promoted to champion.")
        else:
            print("Candidate did not outperform the current champion.")
        return promoted


def main(argv: list[str] | None = None) -> None:
    """Evaluate the latest model, or run a one-time final test.

    Args:
        argv: Optional command-line arguments, primarily for programmatic use.
            When omitted, arguments are read from the process command line.

    Raises:
        FileNotFoundError: If no trained model weights are available.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evaluate a frozen model on the final test split without promotion.",
    )
    args = parser.parse_args(argv)

    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}",
    )
    os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME
    settings.update({"mlflow": True})

    model_path = _get_latest_model_path()
    print(f"Loading model weights from: {model_path}")
    model = YOLO(str(model_path))

    if args.final_test:
        _run_final_test(model, model_path)
        return

    promoted = _run_governance_evaluation(model, model_path)
    print(f"MODEL_PROMOTED={str(promoted).lower()}")


if __name__ == "__main__":
    main()
