"""Convert reviewed Labelme annotations to YOLO training data."""

import os
import json
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
REVIEW_DIR = os.path.join(PROJECT_ROOT, "review_data")
TRAIN_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "images", "train")
TRAIN_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "labels", "train")

# Class mapping from your config.yaml/data.yaml
CLASS_MAPPING = {"corrosion": 0}


def trigger_airflow_phase2():
    """Trigger phase-two retraining in Docker Compose or local Airflow.

    Raises:
        ValueError: If ``CORROSION_AIRFLOW_TARGET`` is unsupported.
        RuntimeError: If the Airflow CLI fails to trigger the DAG.
        subprocess.CalledProcessError: If automatic Docker detection fails.
    """
    target = os.getenv("CORROSION_AIRFLOW_TARGET", "auto").lower()
    if target not in {"auto", "docker", "local"}:
        raise ValueError(
            "CORROSION_AIRFLOW_TARGET must be 'auto', 'docker', or 'local'."
        )

    env = os.environ.copy()
    if target == "auto":
        result = subprocess.run(
            [
                "docker", "compose", "ps", "--services", "--status", "running",
                "airflow-standalone",
            ],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
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


def main():
    """Convert reviewed Labelme annotations and request retraining.

    The Airflow trigger is skipped when there are no reviewed annotations or
    when ``CORROSION_SKIP_AIRFLOW_TRIGGER`` is set to ``"1"``.
    """
    print("🔄 Starting data conversion from Labelme to YOLO format...")

    if not os.path.exists(REVIEW_DIR) or not os.listdir(REVIEW_DIR):
        print(
            "⚠️ 'review_data' folder is empty or non-existent. No curation data to process.")
        return

    # Ensures that destination directories exist
    os.makedirs(TRAIN_IMAGES_DIR, exist_ok=True)
    os.makedirs(TRAIN_LABELS_DIR, exist_ok=True)

    processed_count = 0

    for file in os.listdir(REVIEW_DIR):
        if file.endswith(".json"):
            json_path = os.path.join(REVIEW_DIR, file)
            base_name = os.path.splitext(file)[0]

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_w = data["imageWidth"]
            img_h = data["imageHeight"]
            img_name = data["imagePath"]

            txt_lines = []
            for shape in data["shapes"]:
                if shape["shape_type"] == "rectangle":
                    label = shape["label"]
                    cls_id = CLASS_MAPPING.get(label, 0)

                    # Labelme coordinates [[x1, y1], [x2, y2]]
                    p1, p2 = shape["points"]
                    x1, y1 = p1[0], p1[1]
                    x2, y2 = p2[0], p2[1]

                    # Converts to YOLO format (X_Center, Y_Center, Width, Height) normalized (0 to 1)
                    x_center = ((x1 + x2) / 2) / img_w
                    y_center = ((y1 + y2) / 2) / img_h
                    bbox_w = abs(x2 - x1) / img_w
                    bbox_h = abs(y2 - y1) / img_h

                    txt_lines.append(
                        f"{cls_id} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")

            # 1. Saves the official TXT file into the training folder
            with open(os.path.join(TRAIN_LABELS_DIR, f"{base_name}.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(txt_lines))

            # 2. Moves the original image to the training folder
            shutil.move(os.path.join(REVIEW_DIR, img_name),
                        os.path.join(TRAIN_IMAGES_DIR, img_name))

            # 3. Deletes the JSON file to clear the review queue
            os.remove(json_path)
            processed_count += 1

    print(
        f"✅ Curation completed! {processed_count} images converted and merged into the training dataset.")

    # TRIGGER DISPATCH: Only runs after all files above are completely written to disk
    if (
        processed_count > 0
        and os.getenv("CORROSION_SKIP_AIRFLOW_TRIGGER") != "1"
    ):
        trigger_airflow_phase2()


if __name__ == "__main__":
    main()
