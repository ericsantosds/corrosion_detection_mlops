import hashlib
import json
import os
import pytest
from unittest.mock import patch, MagicMock

mock_client = MagicMock()
mock_version = MagicMock()
mock_version.source = "file:///home/eric/fake_champion/best.pt"
mock_version.version = "7"
mock_client.get_model_version_by_alias.return_value = mock_version

mock_yolo_instance = MagicMock()
mock_box = MagicMock()

# CORREÇÃO CRÍTICA: Simula o retorno de uma lista de floats puros (x1, y1, x2, y2)
# compatíveis com o serializador JSON nativo do Python
mock_tensor_xyxy = MagicMock()
mock_tensor_xyxy.tolist.return_value = [10.0, 20.0, 50.0, 60.0]
# Bate com o índice [0] mapeado no script original
mock_box.xyxy = [mock_tensor_xyxy]

# Simula o ID de classe e o item numérico do PyTorch de forma limpa
mock_cls_tensor = MagicMock()
mock_cls_tensor.item.return_value = 0
mock_box.cls = [mock_cls_tensor]
mock_conf_tensor = MagicMock()
mock_conf_tensor.item.return_value = 0.87
mock_box.conf = [mock_conf_tensor]

mock_low_conf_box = MagicMock()
mock_low_conf_box.xyxy = [mock_tensor_xyxy]
mock_low_conf_box.cls = [mock_cls_tensor]
mock_low_conf_tensor = MagicMock()
mock_low_conf_tensor.item.return_value = 0.12
mock_low_conf_box.conf = [mock_low_conf_tensor]

mock_results = MagicMock()
mock_results.boxes = [mock_box, mock_low_conf_box]
mock_results.names = {0: "corrosion"}
mock_yolo_instance.predict.return_value = [mock_results]


@patch("src.data_automation.send_to_review.MlflowClient", return_value=mock_client)
@patch("src.data_automation.send_to_review.YOLO", return_value=mock_yolo_instance)
@patch("cv2.imread")
def test_send_to_review_generation(mock_imread, mock_yolo, mock_mlflow, tmp_path):
    online_dir = tmp_path / "online_data"
    online_dir.mkdir()
    review_dir = tmp_path / "review_data"
    review_dir.mkdir()
    predictions_dir = tmp_path / "production_predictions"
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"fake model weights")

    (online_dir / "frame_01.jpg").write_text("fake_pixels")
    mock_imread.return_value = MagicMock(shape=(100, 100, 3))

    with patch("src.data_automation.send_to_review.ONLINE_DATA_DIR", str(online_dir)), \
            patch("src.data_automation.send_to_review.REVIEW_DIR", str(review_dir)), \
            patch(
                "src.data_automation.send_to_review.PRODUCTION_PREDICTIONS_DIR",
                str(predictions_dir),
            ), \
            patch(
                "src.data_automation.send_to_review.download_artifacts",
                return_value=str(model_path),
            ) as mock_download:

        from src.data_automation.send_to_review import run_active_learning_inference
        run_active_learning_inference()

        assert os.path.exists(review_dir / "frame_01.jpg")
        assert os.path.exists(review_dir / "frame_01.json")
        labelme = json.loads((review_dir / "frame_01.json").read_text())
        assert len(labelme["shapes"]) == 1
        assert all("score" not in shape for shape in labelme["shapes"])

        run_dirs = list(predictions_dir.iterdir())
        assert len(run_dirs) == 1
        manifest = json.loads((run_dirs[0] / "run.json").read_text())
        assert manifest["model"]["version"] == "7"
        assert manifest["model"]["artifact_sha256"] == hashlib.sha256(
            b"fake model weights"
        ).hexdigest()
        assert manifest["inference"]["candidate_confidence"] == 0.001
        assert manifest["inference"]["production_confidence"] == 0.30
        assert manifest["images"][0]["prediction_file"] == (
            "predictions/frame_01.json")
        prediction = json.loads(
            (run_dirs[0] / "predictions" / "frame_01.json").read_text())
        assert prediction["image"]["file_name"] == "frame_01.jpg"
        assert prediction["image"]["sha256"] == hashlib.sha256(
            b"fake_pixels"
        ).hexdigest()
        assert [detection["score"] for detection in prediction["detections"]] == [
            0.87, 0.12
        ]
        assert prediction["detections"][0]["bbox_xyxy"] == [
            10.0, 20.0, 50.0, 60.0
        ]
        mock_yolo_instance.predict.assert_called_once_with(
            source=str(online_dir / "frame_01.jpg"),
            conf=0.001,
            iou=0.70,
            imgsz=640,
            max_det=300,
        )
        mock_download.assert_called_once_with(
            artifact_uri=mock_version.source,
        )
