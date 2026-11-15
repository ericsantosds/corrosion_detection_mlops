# corrosion_detection_mlops

Corrosion detection with YOLO and an MLOps workflow orchestrated by Apache Airflow.

This guide describes two ways to reproduce the project:

1. **Local test:** run Python commands from the activated `.venv`, without Docker.
2. **Containerized test:** run the services and Airflow DAGs with Docker Compose.

Both modes need the dataset restored from DVC. They use different MLflow tracking
stores: local testing uses `mlflow.db` in the repository; Compose uses the
`mlflow-server` service and persists its state in the `mlflow-data` volume. Do
not mix the two tracking URIs.

## Shared prerequisites

- Python 3.11.5 or newer, `uv`, Git, and access to the configured DVC remote.
- An NVIDIA GPU and compatible CUDA setup for GPU training/inference.
- For Docker GPU use, Docker Compose and NVIDIA Container Toolkit.
- On Ubuntu/WSL, install the GUI libraries if using Labelme or OpenCV windows:

  ```bash
  sudo apt update
  sudo apt install -y libxcb-xinerama0 libqt5gui5 libgles2-mesa-dev
  ```

The dataset is DVC-managed and is not included in the Git checkout. From the
repository root, restore it before training or monitoring:

```bash
dvc pull
```

If `dvc pull` cannot find the remote, configure the DVC remote for the current
machine and its SharePoint/OneDrive mount before continuing. Do not commit
machine-specific credentials or paths.

---

## Test LOCAL — Python `.venv`, without Docker

### 1. Create and activate the environment

Run these commands from the repository root:

```bash
uv sync --frozen
source .venv/bin/activate
python --version
```

Set the project root and local MLflow database in **each terminal** used for
this local workflow:

```bash
export CORROSION_PROJECT_ROOT="$PWD"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
```

The URI above points to `mlflow.db` in the repository. This is separate from
the registry used by Docker Compose.

### 2. Start the local MLflow UI

In Terminal 1, with `.venv` activated and the local tracking URI set:

```bash
mlflow ui \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --host 127.0.0.1 \
  --port 5000
```

Open `http://127.0.0.1:5000` to view runs and registered models.

### 3. Run a direct training/evaluation smoke test

In Terminal 2, activate `.venv`, set the same local tracking URI, and run from
the repository root:

```bash
source .venv/bin/activate
export CORROSION_PROJECT_ROOT="$PWD"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
dvc pull
python -m src.model_train.yolo
```

Wait for training to finish successfully and produce
`runs/detect/<run>/weights/best.pt`. Then evaluate and register the model:

```bash
python -m src.model_eval.yolo
```

This direct run is useful for checking the Python environment and model code. It
does **not** exercise Airflow orchestration.

### 4. Run local serving (optional)

After a model has been registered with the `champion` alias, start the API in a
new terminal:

```bash
source .venv/bin/activate
export CORROSION_PROJECT_ROOT="$PWD"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
python src/serving/app.py
```

To collect webcam frames, use another terminal with `.venv` activated:

```bash
python src/serving/client_webcam.py
```

The API is available at `http://127.0.0.1:8000/docs`. The webcam client writes
captured images to `online_data/`.

### 5. Run the full workflow with local Airflow

The direct commands above are smoke tests. To let Airflow orchestrate the
workflow locally, start a dedicated local Airflow instance. In Terminal 3:

```bash
source .venv/bin/activate
export CORROSION_PROJECT_ROOT="$PWD"
export AIRFLOW_HOME="$HOME/airflow-corrosion"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
export CORROSION_API_URL="http://127.0.0.1:8000"
airflow standalone
```

`airflow standalone` initializes and starts the all-in-one local Airflow
instance. Use the credentials printed in its startup output to sign in at
`http://127.0.0.1:8080`. Keep the same `AIRFLOW_HOME` and environment in any
terminal that triggers a local DAG.

Trigger the monitoring DAG from the Airflow UI or, in a terminal with the
environment above:

```bash
airflow dags trigger corrosion_phase1_monitoring
```

If drift is detected, the DAG creates pseudo-labels in `review_data/`. Review
them with Labelme. Then run the curation script using the environment below.
It converts reviewed annotations and triggers `corrosion_phase2_retrain` in the
**local** Airflow instance. That DAG versions the dataset with DVC, trains and
evaluates a model, and asks the local API to reload the champion.

In a new terminal used to run `./run_curation.sh`, activate `.venv` and export
the same settings so the script triggers this local Airflow instance:

```bash
source .venv/bin/activate
export CORROSION_PROJECT_ROOT="$PWD"
export AIRFLOW_HOME="$HOME/airflow-corrosion"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
export CORROSION_API_URL="http://127.0.0.1:8000"
./run_curation.sh
```

---

## Test CONTAINERIZADO — Docker Compose

### 1. Restore the dataset and start the services

Run `dvc pull` on the host from the repository root before starting Compose; the
repository is mounted into the containers at `/app`.

```bash
dvc pull
docker compose up --build -d
docker compose ps
```

The services and interfaces are:

| Service | Address |
| --- | --- |
| FastAPI serving | `http://127.0.0.1:8000/docs` |
| MLflow | `http://127.0.0.1:5000` |
| Airflow | `http://127.0.0.1:8080` |

Compose stores MLflow data in the `mlflow-data` volume and shares the Airflow
home between its Airflow services. The containers use
`MLFLOW_TRACKING_URI=http://mlflow-server:5000`; do not point them at the
host's local `mlflow.db`.

The Phase 2 DAG runs `dvc add` and `dvc push` inside the container. Make sure
the DVC remote is accessible from the container: a remote configured as a
host-only filesystem path must also be mounted into the required services, or
reconfigured to a location reachable from Docker.

### 2. Run container commands directly (smoke test)

These commands execute Python inside the Airflow container. They are useful for
debugging, but running them directly does **not** create an Airflow DAG run:

```bash
# Start training and wait for successful completion.
docker compose exec airflow-standalone \
  python -m src.model_train.yolo

# Run only after training has produced runs/detect/<run>/weights/best.pt.
docker compose exec airflow-standalone \
  python -m src.model_eval.yolo
```

### 3. Let Airflow orchestrate the containerized workflow

Trigger the monitoring DAG from the Airflow UI or with the Airflow CLI inside
the container:

```bash
docker compose exec airflow-standalone \
  airflow dags trigger corrosion_phase1_monitoring
```

For a first Airflow login, retrieve the generated credentials from the
standalone service logs:

```bash
docker compose logs airflow-standalone 2>&1 | grep -iE 'username|password'
```

The monitoring DAG runs observability and, if drift is detected, sends images
to `review_data/` for curation. The `review_data/` directory is shared with the
host through the `/app` bind mount.

For human review, run the curation script on the host with the local `.venv`.
The script opens Labelme; after you save and close it, the script converts the
annotations and automatically triggers Phase 2 in the running Docker Airflow
service using `docker compose exec`:

```bash
source .venv/bin/activate
./run_curation.sh
```

Keep the Compose stack running while using the script. It detects the active
`airflow-standalone` service and triggers the containerized DAG; if no Compose
Airflow service is running, it uses local Airflow instead. The containerized
Phase 2 DAG performs annotation conversion, DVC versioning, training, model
evaluation/governance, and the API reload in order.

Configure the `gmail_connection` connection in the containerized Airflow UI if
email alerts are needed. This is separate from any connection configured in
local Airflow. For either mode, create the connection in that mode's Airflow UI:
use connection ID `gmail_connection`, type `Generic`, host `smtp.gmail.com`,
port `587`, your Gmail address as the login, and a Google App Password as the
password. Set the Extra JSON to:

```json
{
  "starttls": true,
  "disable_ssl": true,
  "use_ssl": false,
  "from_email": "your_email@gmail.com"
}
```

Never put the real App Password in this file or commit it to Git.

The model must pass the configured minimums (`mAP50 >= 0.40` and
`recall >= 0.50`) and outperform the current champion to be promoted. The API
returns `503` until a champion is registered. To request a reload manually:

```bash
curl --fail --show-error -X POST http://127.0.0.1:8000/reload
```

### 4. Resource guidance

Do not stop Airflow while a DAG is running; doing so interrupts orchestration.
The current training settings use `batch: 4` and `workers: 0`. If training runs
out of memory, try `batch: 2`; lowering `imgsz` from `640` to `512` can also
reduce GPU memory use, but compare evaluation metrics because it may affect
accuracy. Stop `fastapi-serving` only when inference is not needed, and start it
again before a DAG task that reloads the API.

To stop the stack without deleting persistent MLflow/Airflow volumes:

```bash
docker compose stop
```

Resume it with `docker compose start`. Avoid `docker compose down -v` unless you
intend to delete the persisted volumes.

---

## Dataset and experiment reproducibility

`data.dvc` tracks the dataset version. Restore a specific version by checking
out the matching Git revision and pulling the data:

```bash
git checkout <commit-hash> -- data.dvc
dvc pull
```

The local workflow stores tracking data in `mlflow.db`; the containerized
workflow stores it in the Compose `mlflow-data` volume. Keep the corresponding
tracking store with the experiments and model registry you need to reproduce.
