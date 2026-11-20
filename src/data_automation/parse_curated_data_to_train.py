"""Convert human-curated Labelme annotations to YOLO training data."""

import json
import os
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
CURATED_DIR = os.path.join(PROJECT_ROOT, "curated_data")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
TRAIN_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "train")

CLASS_MAPPING = {"corrosion": 0}
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def trigger_airflow_phase2():
    """Trigger phase-two retraining in Docker Compose or local Airflow."""
    target = os.getenv("CORROSION_AIRFLOW_TARGET", "auto").lower()
    if target not in {"auto", "docker", "local"}:
        raise ValueError(
            "CORROSION_AIRFLOW_TARGET must be 'auto', 'docker', or 'local'."
        )

    env = os.environ.copy()
    if target == "auto":
        try:
            result = subprocess.run(
                [
                    "docker", "compose", "ps", "--services", "--status",
                    "running", "airflow-standalone",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            details = (
                error.stderr.strip() or error.stdout.strip()
                if isinstance(error, subprocess.CalledProcessError)
                else str(error)
            )
            print(
                "⚠️ Could not detect the Docker Airflow service"
                + (f": {details}" if details else "")
                + ". Falling back to local Airflow."
            )
            target = "local"
        else:
            target = (
                "docker"
                if "airflow-standalone" in result.stdout.split()
                else "local"
            )

    if target == "docker":
        command = [
            "docker", "compose", "exec", "-T", "airflow-standalone",
            "airflow", "dags", "trigger", "corrosion_phase2_retrain",
        ]
    else:
        airflow_binary = os.path.join(os.path.dirname(sys.executable), "airflow")
        command = [
            airflow_binary, "dags", "trigger", "corrosion_phase2_retrain",
        ]

    print(f"📡 [CLI Trigger] Invoking {target} Airflow: {' '.join(command)}")
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(
            f"Failed to trigger corrosion_phase2_retrain via {target} Airflow"
            + (f": {details}" if details else "")
        ) from error

    print("🚀 [MLOps] Airflow trigger output:")
    print(result.stdout.strip())


def _convert_annotation(annotation_path):
    """Validate a Labelme annotation and return its image and YOLO labels."""
    with open(annotation_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    if image_width <= 0 or image_height <= 0:
        raise ValueError(
            f"Invalid image dimensions in annotation: {annotation_path}")

    image_name = os.path.basename(data["imagePath"])
    if os.path.splitext(image_name)[0] != os.path.splitext(
            os.path.basename(annotation_path))[0]:
        raise ValueError(
            f"Image and annotation names do not match: {annotation_path}")
    image_path = os.path.join(CURATED_DIR, image_name)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Curated image referenced by {annotation_path} not found: "
            f"{image_path}")
    if os.path.splitext(image_name)[1].lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type in annotation: {image_name}")

    yolo_lines = []
    for shape in data["shapes"]:
        if shape["shape_type"] != "rectangle":
            continue
        label = shape["label"]
        if label not in CLASS_MAPPING:
            raise ValueError(
                f"Unknown label {label!r} in annotation: {annotation_path}")

        (x1, y1), (x2, y2) = shape["points"]
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        box_width = abs(x2 - x1) / image_width
        box_height = abs(y2 - y1) / image_height
        yolo_lines.append(
            f"{CLASS_MAPPING[label]} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}")

    return image_name, yolo_lines


def _collect_annotation_paths():
    image_paths = [
        os.path.join(CURATED_DIR, name)
        for name in os.listdir(CURATED_DIR)
        if os.path.isfile(os.path.join(CURATED_DIR, name))
        and os.path.splitext(name)[1].lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    annotation_paths = [
        os.path.join(CURATED_DIR, name)
        for name in os.listdir(CURATED_DIR)
        if name.lower().endswith(".json")
    ]
    images_by_stem = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in image_paths
    }
    annotations_by_stem = {
        os.path.splitext(os.path.basename(path))[0]: path
        for path in annotation_paths
    }

    if len(images_by_stem) != len(image_paths):
        raise ValueError("Curated data contains duplicate image filename stems.")
    if len(annotations_by_stem) != len(annotation_paths):
        raise ValueError(
            "Curated data contains duplicate annotation filename stems.")
    if images_by_stem.keys() != annotations_by_stem.keys():
        raise ValueError(
            "Every curated image must have exactly one Labelme JSON: "
            f"missing JSON={sorted(images_by_stem.keys() - annotations_by_stem.keys())}; "
            f"missing image={sorted(annotations_by_stem.keys() - images_by_stem.keys())}"
        )
    return [annotations_by_stem[stem] for stem in sorted(annotations_by_stem)]


def _training_image_name(image_name):
    base_name, extension = os.path.splitext(image_name)
    candidate = base_name
    suffix = 1
    existing_image_stems = {
        os.path.splitext(name)[0]
        for name in os.listdir(TRAIN_IMAGES_DIR)
    }
    while (
        candidate in existing_image_stems
        or os.path.exists(os.path.join(TRAIN_LABELS_DIR, f"{candidate}.txt"))
    ):
        candidate = f"{base_name}_curated_{suffix}"
        suffix += 1
    return f"{candidate}{extension}", candidate


def main():
    """Convert curated annotations and, outside Airflow, trigger phase two."""
    print("🔄 Starting data conversion from curated Labelme data to YOLO...")

    if not os.path.isdir(CURATED_DIR):
        raise FileNotFoundError(f"Curated data directory not found: {CURATED_DIR}")

    annotation_paths = _collect_annotation_paths()
    if not annotation_paths:
        raise ValueError(
            f"No curated Labelme annotations found in {CURATED_DIR}.")

    converted = [
        (annotation_path, *_convert_annotation(annotation_path))
        for annotation_path in annotation_paths
    ]

    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)

    for annotation_path, image_name, yolo_lines in converted:
        training_image_name, base_name = _training_image_name(image_name)
        image_path = os.path.join(CURATED_DIR, image_name)
        label_path = os.path.join(TRAIN_LABELS_DIR, f"{base_name}.txt")
        with open(label_path, "w", encoding="utf-8") as file:
            file.write("\n".join(yolo_lines))
        shutil.move(
            image_path,
            os.path.join(TRAIN_IMAGES_DIR, training_image_name),
        )
        os.remove(annotation_path)

    print(
        f"✅ Curation completed! {len(converted)} images converted and "
        "merged into the training dataset.")

    if os.getenv("CORROSION_SKIP_AIRFLOW_TRIGGER") != "1":
        trigger_airflow_phase2()


if __name__ == "__main__":
    main()
