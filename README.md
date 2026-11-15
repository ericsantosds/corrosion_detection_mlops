# 🚀 corrosion_detection_mlops

A corrosion detection solution using AI and MLOps.

---

## 🛠️ Virtual Environment & Dedicated Setup (`uv`)

To guarantee total compatibility between Apache Airflow components, graphical tools, and Computer Vision libraries, this project utilizes the high-performance package manager `uv`, locking the environment to a stable Python 3.11 interpreter.

```bash
# 1. Install required Linux system graphics and X11 dependencies for WSL2 UI rendering
sudo apt update && sudo apt install -y libxcb-xinerama0 libqt5gui5 libgles2-mesa-dev

# 2. Ensure the working context is in the project root directory and create an isolated venv in Python 3.11 based pyproject.toml
uv sync

# 3. Activate the created virtual environment
source .venv/bin/activate
```

---

## 🐳 Containerized Stack

The Compose stack builds the project with Python 3.11 from `uv`, starts the
FastAPI service, an MLflow tracking server, and Airflow. Docker Compose with
NVIDIA Container Toolkit is required for inference and retraining.

```bash
docker compose up --build
```

The local interfaces are available at `http://localhost:8000/docs` (API),
`http://localhost:5000` (MLflow), and `http://localhost:8080` (Airflow).
Compose persists MLflow data in the `mlflow-data` volume and shares the Airflow
home between its services. The repository is mounted at `/app` so DAG tasks can
read and update project data. The webcam client remains a host-side process;
the API container does not require a camera device.

In Compose, all training and serving processes use the MLflow server at
`http://mlflow-server:5000`. The API returns `503` until a model has been
registered and assigned the `champion` alias.

---

## 📹 Phase 1: Real-Time Data Collection & Dynamic Serving

This initial phase boots up the production application. The FastAPI server
locates the active `@champion` model in MLflow (local SQLite in a local setup,
or the MLflow tracking service when using Compose). The OpenCV client captures
the webcam stream and uses a background thread pool to dispatch frames,
filtering and saving captures into the `online_data/` directory only when
detections score above a **0.30 confidence threshold**.

To test inference serving, open **two separate terminals** at the root of the project:

### Terminal 1
*Initializes the FastAPI server and loads the live `@champion` model:*
```bash
python src/serving/app.py
```

### Terminal 2
*Starts the webcam stream capture powered by asynchronous background processing:*
```bash
python src/serving/client_webcam.py
```
*💡 Hint: Point the webcam to focus on metallic corrosion test scenarios. Once the terminal indicates that frames have been generated and flushed to `online_data/`, press the **'q'** key on the video window to quit.*

---

## 🧭 Phase 2: Monitoring & Automated Orchestration (Apache Airflow 3.0+)

With live production data saved inside the `online_data/` folder, Apache Airflow steps in as the main orchestrator. The monitoring phase runs statistical evaluations to check for brightness, contrast, and color shifts across new frames using **Evidently AI**.

### Step 1: Sync DAGs Folder Configuration
Ensure your `airflow.cfg` has the `dags_folder` pointing exactly to your absolute repository directory (e.g., `dags_folder = /home/eric/projects/corrosion_detection_mlops/dags`) or dispatch files to your central path. If choosing the central path, copy the pipeline scripts:
```bash
mkdir -p ~/airflow/dags
cp dags/*.py ~/airflow/dags/
```
When Airflow loads the Phase 2 DAG from a separate `AIRFLOW_HOME`, it locates the checkout from the Python environment path. If that cannot identify the checkout, set `CORROSION_PROJECT_ROOT` in the Airflow environment to the repository path.

### Step 2: Initialize the Metadata Database Backend
Before starting the triggers, migrate the database schema to generate all mandatory internal metadata tables (such as the `job` table):
```bash
airflow db migrate
```

```bash
airflow users create \
    --username admin \
    --firstname MyName \
    --lastname MyName \
    --role Admin \
    --email your@gmail.com \
    --password admin
```

### Step 3: Initialize Web UI and Motors (Optimized 2-Terminal Architecture)
Airflow 3.0+ strictly isolates the DAG reading worker loop into a decoupled processor. Instead of opening 4 messy screens, run the ecosystem using only **two parallel terminals** with the virtual environment active (`source .venv/bin/activate`):

#### Terminal A (Infrastructure Hub)
*Boots the webserver, internal scheduler, and API motors simultaneously:*
```bash
airflow standalone
```
*💡 Access the modern web interface at `http://localhost:8080/dags`. Use the username `admin` and the temporary password auto-generated inside this terminal's startup logs.*

To find the generated password when Airflow is running with Docker Compose, run this from the repository root:
```bash
sudo docker compose logs airflow-standalone 2>&1 | grep -iE 'username|password'
```
Omit `sudo` if your user already has permission to access Docker.

#### Terminal B (Code Parsing Engine)
*Launches the decoupled loop that reads and refreshes your Python MLOps DAG scripts in real-time:*
```bash
airflow dag-processor
```

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

### Step 4: Run DAG 1 (Monitoring & Observability)
The first pipeline (`corrosion_phase1_monitoring`) computes data drift metrics and dynamically branches out decisions:
1. Triggers the observability job: `python -m src.observability.monitor` (Injecting the absolute repository path inside the `PYTHONPATH` system environment variable).
2. **Modern SMTP Import:** The email alerting subsystem imports directly from `airflow.providers.smtp.operators.smtp` to fit modern decoupled architecture.
3. If the calculated shift breaks past the critical threshold, a `BranchPythonOperator` fires an Active Learning script (`python -m src.data_automation.send_to_review`), creating pseudo-annotations automatically and pushing files to `review_data/`.
4. If data is healthy and under control, the execution pipeline routes to `stop_pipeline` and finishes cleanly to save compute cycles.

---

## 🎨 Phase 3: Human Curation & Active Learning (Human-in-the-Loop)

If a Data Drift alert was triggered during the previous phase, the flagged flawed images will be sitting inside the `review_data/` directory, enriched with high-quality AI pseudo-boxes pre-labeled by the current champion model.

To streamline the human annotation workflow, a unified execution script is used:
```bash
./run_curation.sh
```
*💥 MLOps MAGIC:* The execution of `./run_curation.sh` launches the Labelme UI over the native XCB platform. Once the operator updates the boxes, saves, and closes the window, the `parse_review_to_train.py` segment automatically triggers. It translates JSON files into normalized YOLO `.txt` files, incorporates them into the local `data/images/train` split, purges staging queues, and **invokes the local Airflow CLI wrapper (`airflow dags trigger`)**, instantly initiating the Phase 2 training pipeline without needing network HTTP credentials or generating authentication faults. The CLI inherits the caller's environment, including `AIRFLOW_HOME`, so configure it for the same Airflow instance that runs the DAG.

---

## 🏆 Phase 4: Syncing, Retraining & Live Hot Reloads (Apache Airflow 3.0+ - Part II)

The secondary pipeline (`corrosion_phase2_retrain`) wakes up immediately after receiving the incoming local CLI trigger kicked off by the curation stage. It automates heavy computing tasks, strict asset governance, and continuous deployment:

### 4.1 Corporate Cloud Sincronization & Governance (DVC via SharePoint Mount)
To bypass corporate tenant MFA and Azure AD OAuth2 restrictions, the system utilizes a secure file-system mirror strategy. The host Windows machine syncs the SharePoint document library via the native OneDrive sync engine, which the WSL2 Linux environment mounts under `/mnt/c/`.

Before the first training run, restore the dataset from DVC on the host, from the repository root. The Compose stack mounts this checkout at `/app`, so the restored `data/` directory will also be available to the containers:

```bash
dvc pull
```

The training configuration uses the repository-relative path `data/data.yaml`; do not replace it with a host-specific absolute path. Training and evaluation normalize the dataset root from `data/data.yaml` to the current checkout at runtime, so no host-specific edit to the DVC-managed dataset is needed.

```bash
# 1. Link the local DVC repository tracking configuration directly to the mapped corporate SharePoint mount point
dvc remote add -f -d sharepoint_{your_enterprise} "/mnt/c/Users/username/{share point path to} - mlops_for_computervision"

# 2. Re-compute immutable dataset hashes and generate clean MD5 checksum pointers
dvc add data

# 3. Synchronize dataset blobs directly into the Microsoft local mount
dvc push
```
*💡 Data Lineage Rule:* Dataset variants are committed as Content-Addressable Storage hashes under `files/md5/` inside SharePoint to maintain historic immutability, while the small metadata file `data.dvc` is tracked inside Git to maintain lineage records.

### 4.2 Training, Governance & Continuous Deployment (MLflow)
Training and evaluation must run in Compose so both use the same MLflow server and model registry. First, from the repository root on the host, restore the DVC-managed dataset if it is not already present:

```bash
dvc pull
```

The normal workflow keeps the Compose services running. Ensure the stack is up, then train and evaluate in order:

```bash
docker compose up -d

# Train using model_args.yaml. Wait for the command to finish successfully.
docker compose exec airflow-standalone python -m src.model_train.yolo

# Evaluate the completed run and register its weights in MLflow.
docker compose exec airflow-standalone python -m src.model_eval.yolo
```

Do not start evaluation until training has completed and produced
`runs/detect/<run>/weights/best.pt`. Evaluation uses the test split and registers
the resulting weights. Promotion to `champion` requires meeting the minimum
business metrics (`mAP50 >= 0.40` and `recall >= 0.50`) and outperforming the
current champion.

#### If Docker runs out of memory

On machines with limited RAM, pause the Airflow and serving services while
training and evaluation run. Keep `mlflow-server` running so both commands
continue to use the Compose registry:

```bash
docker compose stop airflow-processor fastapi-serving airflow-standalone

# Run training in a one-off container. Wait for successful completion.
docker compose run --rm --no-deps airflow-standalone \
  python -m src.model_train.yolo

# Evaluate and register the completed training run.
docker compose run --rm --no-deps airflow-standalone \
  python -m src.model_eval.yolo

# Resume the paused services after evaluation (also run this if a command fails).
docker compose start airflow-standalone airflow-processor fastapi-serving
```

Check that training produced `best.pt` before running evaluation. If training or
evaluation fails, skip the next step, restart the paused services, and resolve
the error before retrying.

#### Reload the serving API

After a new model is promoted to `champion`, reload the live API:

```bash
curl --fail --show-error -X POST http://127.0.0.1:8000/reload
```

Use `http://127.0.0.1:5000` to review runs, evaluation metrics, and the model
registry. The local Compose MLflow UI is already served there; do not start a
second local `mlflow ui` process for the Compose registry.

---

## 🔄 Dataset Reproducibility & Version Checkout

To reproduce an experiment or roll back the dataset to an exact state tracked by a specific MLflow run, use the DVC tracking file state via Git:

```bash
git checkout <commit-hash> data.dvc
dvc pull
```
