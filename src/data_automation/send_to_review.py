"""Generate Labelme review annotations for online images."""

import hashlib
import os
import json
import cv2
from datetime import datetime, timezone
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts
from ultralytics import YOLO
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
ONLINE_DATA_DIR = os.path.join(PROJECT_ROOT, "online_data")
REVIEW_DIR = os.path.join(PROJECT_ROOT, "review_data")
PRODUCTION_PREDICTIONS_DIR = os.path.join(
    PROJECT_ROOT, "production_predictions")

MODEL_NAME = "Corrosion_Detection_YOLO_Model"
CANDIDATE_CONFIDENCE = 0.001
PRODUCTION_CONFIDENCE = 0.30
NMS_IOU = 0.70


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_artifact(path):
    """Hash a model file or all files in a downloaded MLflow artifact."""
    if os.path.isfile(path):
        return _sha256_file(path)

    if not os.path.isdir(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")

    digest = hashlib.sha256()
    files = sorted(
        os.path.join(root, name)
        for root, _, names in os.walk(path)
        for name in names
    )
    if not files:
        raise ValueError(f"Model artifact directory is empty: {path}")

    for file_path in files:
        relative_path = os.path.relpath(file_path, path).replace(os.sep, "/")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        with open(file_path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, data):
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def run_active_learning_inference():
    """Annotate online images with the champion model for human review.

    Raises:
        ValueError: If the champion registry entry has no artifact source.
        FileNotFoundError: If an input image or model artifact is missing.
        MlflowException: If the champion model cannot be retrieved.
    """
    os.environ.setdefault(
        "MLFLOW_TRACKING_URI",
        f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}",
    )
    os.makedirs(REVIEW_DIR, exist_ok=True)

    # 1. Loads the Champion model inside the execution context
    client = MlflowClient()
    model_metadata = client.get_model_version_by_alias(
        MODEL_NAME, "champion")
    if not model_metadata.source:
        raise ValueError("The champion model has no artifact source URI.")
    model_path = download_artifacts(artifact_uri=model_metadata.source)
    print(f"🎯 [Automation] Loaded champion weights from: {model_path}")
    model = YOLO(model_path)
    model_record = {
        "name": MODEL_NAME,
        "alias": "champion",
        "version": str(model_metadata.version),
        "artifact_sha256": _sha256_artifact(model_path),
    }

    # 2. Scans online images
    valid_extensions = ('.jpg', '.jpeg', '.png')
    if not os.path.exists(ONLINE_DATA_DIR):
        print("⚠️ ONLINE_DATA_DIR does not exist.")
        return

    images = sorted(
        f for f in os.listdir(ONLINE_DATA_DIR)
        if f.lower().endswith(valid_extensions)
    )
    if not images:
        print("⚠️ ONLINE_DATA_DIR contains no supported images.")
        return
    image_stems = [os.path.splitext(image)[0] for image in images]
    if len(set(image_stems)) != len(image_stems):
        raise ValueError(
            "ONLINE_DATA_DIR contains images with duplicate filename stems.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prediction_run_dir = os.path.join(
        PRODUCTION_PREDICTIONS_DIR, run_id)
    os.makedirs(prediction_run_dir, exist_ok=False)
    prediction_files_dir = os.path.join(prediction_run_dir, "predictions")
    os.makedirs(prediction_files_dir)
    image_records = []
    inference_record = {
        "candidate_confidence": CANDIDATE_CONFIDENCE,
        "production_confidence": PRODUCTION_CONFIDENCE,
        "nms_iou": NMS_IOU,
        "image_size": 640,
        "max_detections": 300,
    }

    for img_name in images:
        img_path = os.path.join(ONLINE_DATA_DIR, img_name)

        results = model.predict(
            source=img_path,
            conf=CANDIDATE_CONFIDENCE,
            iou=NMS_IOU,
            imgsz=inference_record["image_size"],
            max_det=inference_record["max_detections"],
        )[0]

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise ValueError(f"Could not read online image: {img_path}")
        h, w, _ = img_bgr.shape

        labelme_json = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w
        }
        detections = []

        for box in results.boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item()) if hasattr(
                box.cls, "__len__") else int(box.cls.item())
            label_name = results.names[cls_id]
            score = float(box.conf[0].item())
            if not 0 <= score <= 1:
                raise ValueError(
                    f"Invalid detection score {score} in {img_name}.")

            detections.append({
                "class_id": cls_id,
                "label": label_name,
                "score": score,
                "bbox_xyxy": [float(value) for value in coords],
            })

            if score >= PRODUCTION_CONFIDENCE:
                shape = {
                    "label": label_name,
                    "points": [
                        [coords[0], coords[1]],
                        [coords[2], coords[3]]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                }
                labelme_json["shapes"].append(shape)

        image_sha256 = _sha256_file(img_path)
        sidecar_name = f"{os.path.splitext(img_name)[0]}.json"
        _write_json(
            os.path.join(prediction_files_dir, sidecar_name),
            {
                "image": {
                    "file_name": img_name,
                    "sha256": image_sha256,
                    "width": w,
                    "height": h,
                },
                "detections": detections,
            },
        )

        shutil.copy2(img_path, os.path.join(REVIEW_DIR, img_name))
        json_path = os.path.join(REVIEW_DIR, os.path.splitext(
            img_name)[0] + ".json")
        _write_json(json_path, labelme_json)

        image_records.append({
            "file_name": img_name,
            "prediction_file": os.path.join("predictions", sidecar_name),
            "image_sha256": image_sha256,
        })

        os.remove(img_path)

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "model": model_record,
        "inference": inference_record,
        "images": image_records,
    }
    _write_json(os.path.join(prediction_run_dir, "run.json"), manifest)

    print(
        f"📥 {len(images)} images sent with automatic pre-annotations to: "
        f"{REVIEW_DIR}")
    print(f"📊 Prediction sidecar: {prediction_run_dir}")


if __name__ == "__main__":
    run_active_learning_inference()
