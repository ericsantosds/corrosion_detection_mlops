"""Evaluate and register the corrosion detection YOLO model."""

import argparse
from ultralytics import YOLO, settings
import os
import sys
import mlflow
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    ALREADY_EXISTS,
    RESOURCE_DOES_NOT_EXIST,
    RESOURCE_ALREADY_EXISTS,
    ErrorCode,
)
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from helper.config import load_config, project_dataset_yaml
import yaml
import warnings

# Silences MLflow API deprecation warnings from legacy modules
warnings.filterwarnings("ignore", category=FutureWarning)

# 1. Configures the paths before making any custom imports
current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _register_model_version(client, model_name, model_uri, run_id):
    """Register a run artifact as a model version, including on MLflow 3."""
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


def _get_champion_version(client, model_name):
    """Return None only when the model registry has no champion alias."""
    try:
        return client.get_model_version_by_alias(model_name, "champion")
    except MlflowException as error:
        if error.error_code != ErrorCode.Name(RESOURCE_DOES_NOT_EXIST):
            raise
        return None


def _evaluate_validation(model, data_yaml_path):
    """Evaluate a model on the split used for model selection."""
    return model.val(data=data_yaml_path, split="val", workers=0)


def _evaluate_final_test(model, data_yaml_path):
    """Evaluate a frozen champion on the independent final test split."""
    return model.val(data=data_yaml_path, split="test", workers=0)


def _get_champion_validation_map50(champion_version):
    """Score the current champion on the same validation split as the challenger."""
    if not champion_version.source:
        raise ValueError("The champion model version has no artifact source.")

    model_path = download_artifacts(artifact_uri=champion_version.source)
    champion_model = YOLO(model_path)
    with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
        metrics = _evaluate_validation(champion_model, data_yaml_path)
    return float(metrics.box.map50)


def _promote_if_better(client, model_name, model_details, current_map50):
    """Promote the challenger only when its validation mAP50 is higher."""
    champion_version = _get_champion_version(client, model_name)
    if champion_version is None:
        print("✨ No active Champion found. Promoting the first Champion!")
        client.set_registered_model_alias(
            model_name, "champion", model_details.version)
        return True

    champion_map50 = _get_champion_validation_map50(champion_version)
    print(
        f"👑 Active Champion found! Version: "
        f"{champion_version.version} | Champion mAP50: {champion_map50:.4f}"
    )

    if current_map50 > champion_map50:
        print("🚀 CHALLENGER WINS! Promoting the new model version.")
        client.set_registered_model_alias(
            model_name, "champion", model_details.version)
        return True

    print("📉 Challenger did not outperform the Champion. Keeping active model.")
    return False


def main():
    """Evaluate the latest model and promote it when it meets quality gates."""
    # Database configuration at the project root
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}",
    )
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"

    settings.update({"mlflow": True})

    # 2. AUTOMATIC AND DYNAMIC LOCATION OF THE LATEST TRAINED MODEL
    base_detect_dir = os.path.join(PROJECT_ROOT, "runs", "detect")

    # Lists ONLY actual training folders, ignoring validations (val) and predictions (predict)
    all_train_folders = [
        os.path.join(base_detect_dir, d)
        for d in os.listdir(base_detect_dir)
        if os.path.isdir(os.path.join(base_detect_dir, d))
        and not d.startswith("predict")
        and not d.startswith("val")
    ]

    if not all_train_folders:
        raise FileNotFoundError(
            f"❌ No valid training folder found at: {base_detect_dir}")

    # Finds the training folder modified last by the operating system
    latest_train_folder = max(all_train_folders, key=os.path.getmtime)

    # Builds the final path to the 'best.pt' weights file
    model_path = os.path.normpath(os.path.join(
        latest_train_folder, "weights", "best.pt"))

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"❌ The weights file 'best.pt' was not found at: {model_path}")

    print(f"📦 [Automation] Latest training located: {latest_train_folder}")
    print(
        f"🎯 [Automation] Loading weights for validation evaluation: {model_path}")

    # Loads the trained model dynamically
    model = YOLO(model_path)

    # Starts the exclusive run for testing
    with mlflow.start_run(run_name="validation_model_selection") as run:

        # Adds structural tags to the Run
        mlflow.set_tag("pipeline_stage", "model_selection")
        mlflow.set_tag("evaluation_split", "val")
        mlflow.log_param("evaluated_model_path", model_path)

        # Validation is used for model selection; the final test split stays untouched.
        with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
            metrics = _evaluate_validation(model, data_yaml_path)

        # Converts YOLO metrics to native float
        current_recall = float(metrics.box.r[0]) if hasattr(
            metrics.box.r, "__len__") else float(metrics.box.r)
        current_precision = float(metrics.box.p[0]) if hasattr(
            metrics.box.p, "__len__") else float(metrics.box.p)
        current_mAP50 = float(metrics.box.map50)
        current_mAP50_95 = float(metrics.box.map)

        # Logs to MLflow Tracking
        mlflow.log_metrics({
            "validation_recall": current_recall,
            "validation_precision": current_precision,
            "validation_mAP50": current_mAP50,
            "validation_mAP50_95": current_mAP50_95
        })

        # Saves the physical weights file as a Run artifact
        model_filename = os.path.basename(model_path)
        mlflow.log_artifact(model_path, artifact_path="yolo_weights")

        # Registers the YOLO weights artifact as a model version
        model_uri = f"runs:/{run.info.run_id}/yolo_weights/{model_filename}"
        client = MlflowClient()
        model_name = "Corrosion_Detection_YOLO_Model"
        model_details = _register_model_version(
            client, model_name, model_uri, run.info.run_id)

        # ---- MULTI-METRIC BUSINESS LOGIC (CHAMPION VS CHALLENGER) ----

        business_cfg = load_config(os.path.join(
            PROJECT_ROOT, 'business_args.yaml'))
        MIN_MAP50 = float(business_cfg['MIN_MAP50'])
        MIN_RECALL = float(business_cfg['MIN_RECALL'])

        print("\n--- BUSINESS RULES VALIDATION ---")
        print(
            f"Obtained Metrics -> mAP50: {current_mAP50:.4f} | Recall: {current_recall:.4f}")

        # 1. Check against baseline absolute thresholds
        if current_mAP50 < MIN_MAP50 or current_recall < MIN_RECALL:
            print("❌ REJECTED: Model did not meet the minimum business requirements.")
            return False

        print("✅ Passed baseline requirements. Checking tournament against current Champion...")

        promoted = _promote_if_better(
            client, model_name, model_details, current_mAP50)

        # ---- ADDITIONAL METADATA ATTRIBUTION (TAGS) ----
        client.set_registered_model_tag(
            name=model_name,
            key="framework",
            value="ultralytics-yolo"
        )
        client.set_registered_model_tag(
            name=model_name,
            key="task",
            value="corrosion-detection"
        )

        # Dataset version tracking via DVC
        dvc_file_path = os.path.join(PROJECT_ROOT, "data.dvc")
        dvc_hash = "unknown"

        if os.path.exists(dvc_file_path):
            with open(dvc_file_path, "r") as f:
                dvc_data = yaml.safe_load(f)
                dvc_hash = dvc_data.get("outs", [{}])[0].get("md5", "unknown")

        client.set_registered_model_tag(
            name=model_name,
            key="dataset_version_md5",
            value=dvc_hash
        )

    return promoted


def evaluate_final_test():
    """Evaluate the registered champion once on the independent test split."""
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}",
    )
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "Corrosion_Detection_YOLO"
    settings.update({"mlflow": True})

    client = MlflowClient()
    model_name = "Corrosion_Detection_YOLO_Model"
    champion_version = _get_champion_version(client, model_name)
    if champion_version is None:
        raise FileNotFoundError(
            "No champion model is registered; select a model before final testing.")
    if not champion_version.source:
        raise ValueError("The champion model version has no artifact source.")

    model_path = download_artifacts(artifact_uri=champion_version.source)
    model = YOLO(model_path)
    with project_dataset_yaml(PROJECT_ROOT) as data_yaml_path:
        with mlflow.start_run(run_name="final_independent_test") as run:
            mlflow.set_tag("pipeline_stage", "final_test")
            mlflow.set_tag("evaluation_split", "test")
            mlflow.log_params({
                "model_name": model_name,
                "model_version": champion_version.version,
                "model_run_id": champion_version.run_id,
            })

            metrics = _evaluate_final_test(model, data_yaml_path)
            final_recall = float(metrics.box.r[0]) if hasattr(
                metrics.box.r, "__len__") else float(metrics.box.r)
            final_precision = float(metrics.box.p[0]) if hasattr(
                metrics.box.p, "__len__") else float(metrics.box.p)
            final_map50 = float(metrics.box.map50)
            final_map50_95 = float(metrics.box.map)
            mlflow.log_metrics({
                "final_test_recall": final_recall,
                "final_test_precision": final_precision,
                "final_test_mAP50": final_map50,
                "final_test_mAP50_95": final_map50_95,
            })
            print(
                f"Final test metrics logged to MLflow run {run.info.run_id}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evaluate the frozen champion on the independent test split.",
    )
    args = parser.parse_args()
    if args.final_test:
        evaluate_final_test()
    else:
        was_promoted = main()
        print(f"MODEL_PROMOTED={str(was_promoted).lower()}")
