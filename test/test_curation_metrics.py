import hashlib
import json

from PIL import Image
import pytest

from src.data_automation.curation_metrics import (
    CurationMetrics,
    _select_prediction_run,
    evaluate_curation,
)


def test_select_prediction_run_uses_latest_manifest_timestamp(tmp_path):
    predictions_dir = tmp_path / "production_predictions"
    older_run = predictions_dir / "zzz"
    newer_run = predictions_dir / "aaa"
    older_run.mkdir(parents=True)
    newer_run.mkdir()
    (older_run / "run.json").write_text(json.dumps({
        "run_id": "20261121T212949699809Z",
    }))
    (newer_run / "run.json").write_text(json.dumps({
        "run_id": "20261121T214530449751Z",
    }))

    assert _select_prediction_run(predictions_dir) == newer_run


def test_retraining_gate_requires_every_metric_below_threshold():
    below_threshold = CurationMetrics(0.89, 0.88, 0.87, 0.89, 0.86)
    at_threshold = CurationMetrics(0.89, 0.90, 0.87, 0.89, 0.86)

    assert below_threshold.all_below()
    assert not at_threshold.all_below()


def test_retraining_gate_rejects_undefined_metrics():
    metrics = CurationMetrics(0.8, 0.8, 0.8, 0.8, -1.0)

    with pytest.raises(ValueError, match="undefined or invalid"):
        metrics.all_below()


def test_evaluate_curation_uses_sidecar_scores_and_curated_boxes(tmp_path):
    review_dir = tmp_path / "review_data"
    curated_dir = tmp_path / "curated_data"
    predictions_dir = tmp_path / "production_predictions"
    run_dir = predictions_dir / "run-1"
    sidecar_dir = run_dir / "predictions"
    for folder in (review_dir, curated_dir, sidecar_dir):
        folder.mkdir(parents=True)

    image_path = review_dir / "frame.jpg"
    Image.new("RGB", (64, 64)).save(image_path)
    curated_image = curated_dir / image_path.name
    curated_image.write_bytes(image_path.read_bytes())
    shape = {
        "label": "corrosion",
        "points": [[10, 10], [30, 30]],
        "shape_type": "rectangle",
    }
    for folder in (review_dir, curated_dir):
        annotation = {
            "imagePath": image_path.name,
            "imageWidth": 64,
            "imageHeight": 64,
            "shapes": [shape],
        }
        (folder / "frame.json").write_text(json.dumps(annotation))

    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    (run_dir / "run.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "run-1",
        "inference": {
            "candidate_confidence": 0.001,
            "production_confidence": 0.3,
        },
        "images": [{
            "file_name": image_path.name,
            "prediction_file": "predictions/frame.json",
            "image_sha256": image_hash,
        }],
    }))
    (sidecar_dir / "frame.json").write_text(json.dumps({
        "image": {
            "file_name": image_path.name,
            "sha256": image_hash,
            "width": 64,
            "height": 64,
        },
        "detections": [
            {
                "label": "corrosion",
                "score": 0.95,
                "bbox_xyxy": [10, 10, 30, 30],
            },
            {
                "label": "corrosion",
                "score": 0.1,
                "bbox_xyxy": [35, 35, 35, 45],
            },
        ],
    }))

    result = evaluate_curation(
        review_dir=review_dir,
        curated_dir=curated_dir,
        predictions_dir=predictions_dir,
    )

    assert result.image_count == 1
    assert result.prediction_count == 2
    assert result.ground_truth_count == 1
    assert result.degenerate_prediction_count == 1
    assert result.metrics.map_50 == pytest.approx(1.0)


def test_evaluate_curation_marks_all_negative_review_as_undefined(tmp_path):
    review_dir = tmp_path / "review_data"
    curated_dir = tmp_path / "curated_data"
    predictions_dir = tmp_path / "production_predictions"
    run_dir = predictions_dir / "run-1"
    sidecar_dir = run_dir / "predictions"
    for folder in (review_dir, curated_dir, sidecar_dir):
        folder.mkdir(parents=True)

    image_path = review_dir / "frame.jpg"
    Image.new("RGB", (64, 64)).save(image_path)
    curated_image = curated_dir / image_path.name
    curated_image.write_bytes(image_path.read_bytes())
    image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    shape = {
        "label": "corrosion",
        "points": [[10, 10], [30, 30]],
        "shape_type": "rectangle",
    }
    (review_dir / "frame.json").write_text(json.dumps({
        "imagePath": image_path.name,
        "imageWidth": 64,
        "imageHeight": 64,
        "shapes": [shape],
    }))
    (curated_dir / "frame.json").write_text(json.dumps({
        "imagePath": image_path.name,
        "imageWidth": 64,
        "imageHeight": 64,
        "shapes": [],
    }))
    (run_dir / "run.json").write_text(json.dumps({
        "schema_version": 1,
        "run_id": "run-1",
        "inference": {
            "candidate_confidence": 0.001,
            "production_confidence": 0.3,
        },
        "images": [{
            "file_name": image_path.name,
            "prediction_file": "predictions/frame.json",
            "image_sha256": image_hash,
        }],
    }))
    (sidecar_dir / "frame.json").write_text(json.dumps({
        "image": {
            "file_name": image_path.name,
            "sha256": image_hash,
            "width": 64,
            "height": 64,
        },
        "detections": [{
            "label": "corrosion",
            "score": 0.9,
            "bbox_xyxy": [10, 10, 30, 30],
        }],
    }))

    result = evaluate_curation(
        review_dir=review_dir,
        curated_dir=curated_dir,
        predictions_dir=predictions_dir,
    )

    assert result.ground_truth_count == 0
    assert result.metrics is None
