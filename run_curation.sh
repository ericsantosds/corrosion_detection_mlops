#!/bin/bash
set -e

source .venv/bin/activate

echo "🎨 Opening Labelme for Human-in-the-Loop curation..."
QT_QPA_PLATFORM=xcb python - <<'PY'
import os
import sys

import cv2
from PyQt5.QtCore import QLibraryInfo

os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
    QLibraryInfo.location(QLibraryInfo.PluginsPath), "platforms"
)

from labelme.__main__ import main

sys.argv = ["labelme", "review_data"]
main()
PY

echo "🚀 Labelme closed! Launching automated conversion and Airflow Webhook..."
python -m src.data_automation.parse_review_to_train
