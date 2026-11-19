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
`runs/detect/<run>/weights/best.pt`. Then evaluate and register the model. The
test split is used for quality gates and challenger-versus-champion promotion;
the validation split does not determine promotion:

```bash
python -m src.model_eval.yolo
```

This direct run is useful for checking the Python environment and model code. It
does **not** exercise Airflow orchestration.

The `--final-test` option records test-split metrics in a separate MLflow run
without promoting a model:

```bash
python -m src.model_eval.yolo --final-test
```

This records `final_test_*` metrics and does not promote models. Because the
test split is used for champion promotion, it is not an independent holdout;
do not report its results as an unbiased final estimate. Create a fresh,
untouched test split if an independent final estimate is required.

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
workflow locally, use Airflow's default home directory (`$HOME/airflow`). This
is the same directory mounted by Docker Compose, so the local and containerized
Airflow use the same metadata database, configuration, and connections.
Run only one Airflow mode at a time; the shared SQLite metadata database is not
intended for simultaneous local and Compose Airflow instances.

Create the directory and initialize the Airflow metadata database once:

```bash
mkdir -p "$HOME/airflow"
source .venv/bin/activate
export CORROSION_PROJECT_ROOT="$PWD"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
export CORROSION_API_URL="http://127.0.0.1:8000"
airflow db migrate
```

In Terminal 3, start the local Airflow services:

```bash
source .venv/bin/activate
export CORROSION_PROJECT_ROOT="$PWD"
export AIRFLOW__CORE__DAGS_FOLDER="$PWD/dags"
export MLFLOW_TRACKING_URI="sqlite:///$PWD/mlflow.db"
export CORROSION_API_URL="http://127.0.0.1:8000"
airflow standalone
```

`airflow standalone` initializes and starts the all-in-one local Airflow
instance. It uses `$HOME/airflow` by default. Use the credentials printed in
its startup output to sign in at `http://127.0.0.1:8080`. In other terminals,
activate `.venv` and export the project settings before triggering a local DAG;
Airflow will use the same default home directory.

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
mkdir -p "$HOME/airflow"
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

Compose stores MLflow data in the `mlflow-data` volume. Both Airflow services
mount the host's `$HOME/airflow` directory at `/opt/airflow`, matching Airflow's
default home on the host. This keeps the metadata database and connections
available to both local Airflow and Docker Airflow. Initialize or upgrade the
metadata database with:

```bash
docker compose exec airflow-standalone airflow db migrate
```

If you previously ran the Compose setup, its Airflow state is still in the old
named volume `airflow-home`; it is not copied into `$HOME/airflow` or deleted
by this change. Migrate any connections or history you need before switching.

The containers use
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

The automated evaluation uses the test split for candidate quality gates and
champion promotion; it does not use the validation split for promotion. The
`--final-test` command below only records metrics and does not promote a model:

```bash
docker compose exec airflow-standalone \
  python -m src.model_eval.yolo --final-test
```

Because the automated workflow already uses the test split for promotion,
these metrics are not an independent final estimate. Use a separate untouched
test split if an unbiased final estimate is required.

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

### Configure Gmail email alerts

### Step 3.1: Configure SMTP Connection in Airflow UI (Anti-SSL & Sender Rule)
To ensure alert delivery via Gmail and bypass strict Airflow 3.0 `Email/SMTP` hooks that cause connection errors (`[SSL: WRONG_VERSION_NUMBER]`), you must define the connection explicitly using the **Generic** type.

1. Open the Airflow Web Dashboard (`http://localhost:8080`).
2. Navigate to **Admin** -> **Connections** and click the **+** (Add a new record) button.
3. Configure the connection fields exactly as defined below:
   * **Connection Id:** `smtp_default` (or `gmail_connection`, corresponding to the `conn_id` string declared in your DAGs).
   * **Connection Type:** `Generic`
   * **Host:** `smtp.gmail.com`
   * **Login:** `your_email@gmail.com`
   * **Password:** *Your 16-digit Google App Password* (Generated under Google Account Security -> 2-Step Verification -> App Passwords).
   * **Port:** `587`
4. In the **Extra** textbox field, paste the unmapped unified JSON configuration exactly as follows:
   ```json
   {
     "starttls": true,
     "disable_ssl": true,
     "use_ssl": false,
     "from_email": "your_email@gmail.com"
   }
   ```
   *💡 Note: Merging security parameters with `from_email` inside a single JSON dictionary fulfills required server parameters and eliminates 'unset from_email' execution exceptions.*
5. Click **Save**.

### 4. Resource guidance

Do not stop Airflow while a DAG is running; doing so interrupts orchestration.
The current training settings use `batch: 4` and `workers: 0`. If training runs
out of memory, try `batch: 2`; lowering `imgsz` from `640` to `512` can also
reduce GPU memory use, but compare evaluation metrics because it may affect
accuracy. Stop `fastapi-serving` only when inference is not needed, and start it
again before a DAG task that reloads the API.

Stop the stack without deleting the persistent Airflow directory or MLflow
volume:

```bash
docker compose stop
```

Resume it with `docker compose start`. `docker compose down -v` deletes the
named `mlflow-data` volume; the Airflow data in `$HOME/airflow` is a host
directory and is not removed by that command.

---

## Dataset and experiment reproducibility

### Corporate SharePoint/OneDrive DVC remote (WSL2)

For corporate SharePoint libraries protected by MFA/Azure AD, use the Windows
host's native OneDrive sync and access its local mirror from WSL2. Configure
DVC to use the mounted directory as its default remote (the path below is
specific to this workstation):

```bash
dvc remote add --local -f -d sharepoint \
  "/mnt/c/Users/myuser/My Enterprise /Documents - folder"
dvc add data
dvc push
```

`dvc add data` recalculates the dataset checksum and updates `data.dvc`.
Dataset contents are stored by content hash in the DVC remote (for example,
under `files/md5/`); Git tracks the small `data.dvc` pointer so a dataset
version can be tied to a Git revision. The `--local` option keeps this
machine-specific path out of the shared DVC configuration. Docker containers
cannot automatically access a WSL2 host path; mount the remote into the
services that run DVC or configure a remote reachable from those containers.

`data.dvc` tracks the dataset version. Restore a specific version by checking
out the matching Git revision and pulling the data:

```bash
git checkout <commit-hash> -- data.dvc
dvc pull
```

### Model governance and live reload

For a direct local run, use the same local MLflow tracking URI configured above.
After training, evaluation logs the test metrics and registers a model
candidate. It must meet the configured business thresholds (`mAP50 >= 0.50`
and `Recall >= 0.51`); an eligible candidate becomes the initial `champion` or
replaces it only when its mAP50 is higher. The API must already be running
with this same tracking URI before requesting a reload:

```bash
python -m src.model_train.yolo
python -m src.model_eval.yolo
curl --fail --show-error -X POST http://127.0.0.1:8000/reload
```

The reload endpoint loads the current `champion` into the running API without
restarting the process. To inspect experiment runs and registered model
versions, start the local MLflow UI as described above:

```bash
mlflow ui --backend-store-uri "$MLFLOW_TRACKING_URI" --host 127.0.0.1 --port 5000
```

The local workflow stores tracking data in `mlflow.db`; the containerized
workflow stores it in the Compose `mlflow-data` volume. Keep the corresponding
tracking store with the experiments and model registry you need to reproduce.
