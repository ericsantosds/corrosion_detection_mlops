#!/bin/bash
set -euo pipefail

echo "🎨 Opening Labelme for Human-in-the-Loop curation..."

if [ ! -d review_data ] || ! find review_data -maxdepth 1 -type f -name '*.json' -print -quit | grep -q .; then
    echo "No model predictions found in review_data." >&2
    exit 1
fi

mkdir -p curated_data
cp -an review_data/. curated_data/

QT_QPA_PLATFORM=xcb uv run labelme curated_data

if ! find curated_data -maxdepth 1 -type f -name '*.json' -print -quit | grep -q .; then
    echo "No curated annotations found in curated_data; Phase 2 was not triggered." >&2
    exit 1
fi

echo "📊 Labelme closed! Evaluating the curated annotations with COCOeval..."
METRICS_OUTPUT="$(uv run python -m src.data_automation.curation_metrics)"
printf '%s\n' "$METRICS_OUTPUT"

if grep -qx "RETRAIN_REQUIRED=true" <<< "$METRICS_OUTPUT"; then
    echo "🚀 All metrics are below 0.90. Triggering Phase 2 retraining."
    uv run python -c "from src.data_automation.parse_curated_data_to_train import trigger_airflow_phase2; trigger_airflow_phase2()"
else
    echo "✅ The degradation gate was not met. Phase 2 will not be triggered."
    echo "📦 Adding all reviewed data to the training set without retraining."
    CORROSION_SKIP_AIRFLOW_TRIGGER=1 uv run python -m src.data_automation.parse_curated_data_to_train
fi
