"""Evaluate human-curated Labelme data against scored champion predictions."""

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "review_data"
CURATED_DIR = PROJECT_ROOT / "curated_data"
PREDICTIONS_DIR = PROJECT_ROOT / "production_predictions"
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
CATEGORY_IDS = {"corrosion": 1}
RETRAIN_THRESHOLD = 0.90


@dataclass(frozen=True)
class CurationMetrics:
    map_50_95: float
    map_50: float
    map_75: float
    precision: float
    recall: float

    def as_dict(self):
        return {
            "mAP@0.50:0.95": self.map_50_95,
            "mAP@0.50": self.map_50,
            "mAP@0.75": self.map_75,
            "Precision": self.precision,
            "Recall": self.recall,
        }

    def all_below(self, threshold=RETRAIN_THRESHOLD):
        values = tuple(self.as_dict().values())
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("The retraining threshold must be between 0 and 1.")
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("COCOeval returned an undefined or invalid metric.")
        return all(value < threshold for value in values)


@dataclass(frozen=True)
class CurationEvaluation:
    metrics: CurationMetrics | None
    run_id: str
    image_count: int
    prediction_count: int
    ground_truth_count: int
    degenerate_prediction_count: int
    coco_eval: COCOeval | None


def _collect_pairs(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Data directory not found: {folder}")

    images = [
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    image_by_stem = {path.stem: path for path in images}
    annotations = list(folder.glob("*.json"))
    annotation_by_stem = {path.stem: path for path in annotations}

    if len(image_by_stem) != len(images):
        raise ValueError(f"Duplicate image filename stems in {folder}.")
    if len(annotation_by_stem) != len(annotations):
        raise ValueError(f"Duplicate annotation filename stems in {folder}.")
    if image_by_stem.keys() != annotation_by_stem.keys():
        raise ValueError(
            f"Image/annotation pairs are incomplete in {folder}: "
            f"missing JSON={sorted(image_by_stem.keys() - annotation_by_stem.keys())}; "
            f"missing image={sorted(annotation_by_stem.keys() - image_by_stem.keys())}"
        )
    return {
        stem: (image_by_stem[stem], annotation_by_stem[stem])
        for stem in image_by_stem
    }


def _select_prediction_run(predictions_dir, prediction_run_dir=None):
    if prediction_run_dir is not None:
        run_dir = Path(prediction_run_dir).expanduser()
        return run_dir if run_dir.is_absolute() else PROJECT_ROOT / run_dir

    manifests = sorted(predictions_dir.glob("*/run.json"))
    if not manifests:
        raise FileNotFoundError(f"No prediction runs found in {predictions_dir}.")
    if len(manifests) == 1:
        return manifests[0].parent

    def run_timestamp(manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid prediction manifest: {manifest_path}")
        try:
            return datetime.strptime(
                manifest["run_id"], "%Y%m%dT%H%M%S%fZ"
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid UTC timestamp in prediction manifest: {manifest_path}"
            ) from error

    return max(manifests, key=run_timestamp).parent


def _sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyxy_to_coco(points, width, height, source, check_bounds, allow_degenerate):
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        raise ValueError(f"Invalid bounding box in {source}.")
    try:
        x1, y1, x2, y2 = map(float, points)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid coordinates in {source}.") from error

    x_min, x_max = sorted((x1, x2))
    y_min, y_max = sorted((y1, y2))
    coordinates = (x_min, y_min, x_max, y_max)
    outside_image = x_min < 0 or y_min < 0 or x_max > width or y_max > height
    if (
        not all(math.isfinite(value) for value in coordinates)
        or (not allow_degenerate and (x_min == x_max or y_min == y_max))
        or (check_bounds and outside_image)
    ):
        raise ValueError(f"Invalid or out-of-image bounding box in {source}.")

    return [x_min, y_min, x_max, y_max], [
        x_min, y_min, x_max - x_min, y_max - y_min
    ]


def _read_labelme_boxes(annotation_path, image_path, check_bounds, allow_degenerate):
    labelme = json.loads(annotation_path.read_text(encoding="utf-8"))
    with Image.open(image_path) as image:
        width, height = image.size

    if labelme.get("imagePath") != image_path.name:
        raise ValueError(f"Invalid imagePath in {annotation_path}.")
    if (labelme.get("imageWidth"), labelme.get("imageHeight")) != (width, height):
        raise ValueError(f"Invalid image dimensions in {annotation_path}.")

    shapes = labelme.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"Invalid shapes list in {annotation_path}.")

    boxes = []
    for shape in shapes:
        label = shape.get("label")
        if label not in CATEGORY_IDS or shape.get("shape_type") != "rectangle":
            raise ValueError(f"Unsupported label or shape in {annotation_path}.")
        points = shape.get("points", [])
        if (
            not isinstance(points, list)
            or len(points) != 2
            or any(
                not isinstance(point, (list, tuple)) or len(point) != 2
                for point in points
            )
        ):
            raise ValueError(f"Invalid rectangle in {annotation_path}.")

        xyxy, bbox = _xyxy_to_coco(
            [*points[0], *points[1]],
            width,
            height,
            annotation_path,
            check_bounds,
            allow_degenerate,
        )
        boxes.append({
            "label": label,
            "category_id": CATEGORY_IDS[label],
            "bbox_xyxy": xyxy,
            "bbox": bbox,
        })
    return width, height, boxes


def _box_signature(box):
    return box["label"], *(round(value, 6) for value in box["bbox_xyxy"])


def evaluate_curation(
    review_dir=REVIEW_DIR,
    curated_dir=CURATED_DIR,
    predictions_dir=PREDICTIONS_DIR,
    prediction_run_dir=None,
):
    """Run COCOeval using scored production sidecars and curated Labelme data."""
    review_pairs = _collect_pairs(Path(review_dir))
    curated_pairs = _collect_pairs(Path(curated_dir))
    run_dir = _select_prediction_run(
        Path(predictions_dir), prediction_run_dir
    )
    manifest_path = run_dir / "run.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prediction manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported prediction schema: {manifest.get('schema_version')}"
        )

    inference = manifest["inference"]
    production_confidence = float(inference["production_confidence"])
    candidate_confidence = float(inference["candidate_confidence"])
    if (
        not math.isfinite(production_confidence)
        or not math.isfinite(candidate_confidence)
        or not 0 <= candidate_confidence <= production_confidence <= 1
    ):
        raise ValueError("Invalid confidence thresholds in the prediction manifest.")

    image_records = manifest["images"]
    sidecar_by_stem = {
        Path(record["file_name"]).stem: record for record in image_records
    }
    if len(sidecar_by_stem) != len(image_records):
        raise ValueError("The prediction manifest contains duplicate image stems.")
    if (
        review_pairs.keys() != curated_pairs.keys()
        or review_pairs.keys() != sidecar_by_stem.keys()
    ):
        raise ValueError(
            "review_data, curated_data, and the prediction manifest "
            "must contain the same frames."
        )

    images = []
    ground_truths = []
    detections = []
    degenerate_predictions = 0
    predictions_subdir = (run_dir / "predictions").resolve()

    for image_id, stem in enumerate(sorted(review_pairs), start=1):
        review_image, review_json = review_pairs[stem]
        curated_image, curated_json = curated_pairs[stem]
        record = sidecar_by_stem[stem]
        if (
            review_image.name != curated_image.name
            or review_image.name != record["file_name"]
        ):
            raise ValueError(f"Image filename mismatch for {stem}.")

        image_hash = _sha256_file(review_image)
        if (
            image_hash != record["image_sha256"]
            or _sha256_file(curated_image) != image_hash
        ):
            raise ValueError(f"Image does not match the prediction sidecar: {stem}.")

        width, height, review_boxes = _read_labelme_boxes(
            review_json, review_image, check_bounds=False, allow_degenerate=True
        )
        curated_width, curated_height, human_boxes = _read_labelme_boxes(
            curated_json, curated_image, check_bounds=True, allow_degenerate=False
        )
        if (width, height) != (curated_width, curated_height):
            raise ValueError(f"Image dimensions differ for {stem}.")

        sidecar_path = (run_dir / record["prediction_file"]).resolve()
        if sidecar_path.parent != predictions_subdir or not sidecar_path.is_file():
            raise FileNotFoundError(f"Invalid prediction sidecar: {sidecar_path}")
        prediction_data = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar_image = prediction_data["image"]
        if (
            sidecar_image["file_name"] != review_image.name
            or sidecar_image["sha256"] != image_hash
            or (sidecar_image["width"], sidecar_image["height"]) != (width, height)
        ):
            raise ValueError(f"Sidecar metadata does not match image {stem}.")

        sidecar_boxes = []
        for detection in prediction_data.get("detections", []):
            label = detection.get("label")
            if label not in CATEGORY_IDS:
                raise ValueError(f"Unsupported sidecar label: {label}")
            score = float(detection["score"])
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid confidence score in {sidecar_path}.")
            xyxy, bbox = _xyxy_to_coco(
                detection["bbox_xyxy"],
                width,
                height,
                sidecar_path,
                check_bounds=False,
                allow_degenerate=True,
            )
            if bbox[2] == 0 or bbox[3] == 0:
                degenerate_predictions += 1
            sidecar_boxes.append({
                "label": label,
                "category_id": CATEGORY_IDS[label],
                "bbox_xyxy": xyxy,
                "bbox": bbox,
                "score": score,
            })

        expected_review = Counter(
            _box_signature(box) for box in sidecar_boxes
            if box["score"] >= production_confidence
        )
        if Counter(_box_signature(box) for box in review_boxes) != expected_review:
            raise ValueError(f"review_data differs from the scored sidecar for {stem}.")

        images.append({
            "id": image_id,
            "file_name": review_image.name,
            "width": width,
            "height": height,
        })
        for box in human_boxes:
            ground_truths.append({
                "id": len(ground_truths) + 1,
                "image_id": image_id,
                "category_id": box["category_id"],
                "bbox": box["bbox"],
                "area": box["bbox"][2] * box["bbox"][3],
                "iscrowd": 0,
            })
        for box in sidecar_boxes:
            detections.append({
                "image_id": image_id,
                "category_id": box["category_id"],
                "bbox": box["bbox"],
                "score": box["score"],
            })

    if not ground_truths:
        return CurationEvaluation(
            metrics=None,
            run_id=manifest["run_id"],
            image_count=len(images),
            prediction_count=len(detections),
            ground_truth_count=0,
            degenerate_prediction_count=degenerate_predictions,
            coco_eval=None,
        )

    categories = [
        {"id": category_id, "name": name}
        for name, category_id in CATEGORY_IDS.items()
    ]
    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "categories": categories,
        "annotations": ground_truths,
    }
    coco_gt.createIndex()
    if detections:
        coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = COCO()
        coco_dt.dataset = {
            "images": images,
            "categories": categories,
            "annotations": [],
        }
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluator.params.catIds = list(CATEGORY_IDS.values())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    stats = evaluator.stats
    metrics = CurationMetrics(
        map_50_95=float(stats[0]),
        map_50=float(stats[1]),
        map_75=float(stats[2]),
        # Preserve the precision value emitted by the supplied audit notebook.
        precision=float(stats[0]),
        recall=float(stats[8]),
    )
    metrics.all_below()

    return CurationEvaluation(
        metrics=metrics,
        run_id=manifest["run_id"],
        image_count=len(images),
        prediction_count=len(detections),
        ground_truth_count=len(ground_truths),
        degenerate_prediction_count=degenerate_predictions,
        coco_eval=evaluator,
    )


def main():
    evaluation = evaluate_curation()
    print(f"Execution: {evaluation.run_id}")
    print(f"Images: {evaluation.image_count}")
    print(f"Scored predictions: {evaluation.prediction_count}")
    print(f"Human boxes: {evaluation.ground_truth_count}")
    print(
        "Degenerate predictions included: "
        f"{evaluation.degenerate_prediction_count}"
    )
    if evaluation.metrics is None:
        print("COCO metrics are undefined because curated_data has no boxes.")
        print("RETRAIN_REQUIRED=false")
        return
    for name, value in evaluation.metrics.as_dict().items():
        print(f"{name}: {value}")
    print(f"RETRAIN_REQUIRED={str(evaluation.metrics.all_below()).lower()}")


if __name__ == "__main__":
    main()
