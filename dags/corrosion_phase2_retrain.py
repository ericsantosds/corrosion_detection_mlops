"""Airflow DAG for curating data, retraining, and deploying the model."""

from datetime import datetime, timedelta
import os
from pathlib import Path
import sys
import pendulum
from airflow import DAG
from airflow.utils.email import send_email_smtp
from airflow.providers.standard.operators.python import BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator

PYTHON_EXECUTABLE = sys.executable
API_URL = os.getenv("CORROSION_API_URL", "http://127.0.0.1:8000")


def _find_project_root(dag_file: Path, python_executable: Path) -> Path:
    """Locate the checkout containing the phase-two curation script.

    Args:
        dag_file: Path to this DAG file, which may be copied into Airflow home.
        python_executable: Python executable path used by Airflow.

    Returns:
        The project root containing
        ``src/data_automation/parse_curated_data_to_train.py``.

    Raises:
        RuntimeError: If no project root can be found.
    """
    candidates = (dag_file.resolve().parent.parent, *
                  python_executable.absolute().parents)
    for candidate in candidates:
        if (candidate / "src" / "data_automation" / "parse_curated_data_to_train.py").is_file():
            return candidate
    raise RuntimeError(
        "Could not locate the project root; set CORROSION_PROJECT_ROOT to the checkout path."
    )


configured_project_root = os.getenv("CORROSION_PROJECT_ROOT")
if configured_project_root:
    PROJECT_ROOT = str(Path(configured_project_root).expanduser().resolve())
else:
    PROJECT_ROOT = str(_find_project_root(
        Path(__file__), Path(PYTHON_EXECUTABLE)))

LOCAL_TZ = pendulum.timezone("America/Sao_Paulo")


def gmail_fail_alert(context):
    """Send an email alert containing context for a failed Airflow task.

    Args:
        context: Airflow task context with the task instance and exception.
    """
    ti = context["task_instance"]
    dag_id = ti.dag_id
    task_id = ti.task_id
    run_id = getattr(ti, "run_id", "unknown")
    execution_time = getattr(ti, "start_date", "unknown")
    error = str(context.get("exception", "Nenhuma exceção capturada"))

    base_url = os.getenv("AIRFLOW__WEBSERVER__BASE_URL",
                         "http://localhost:8080")
    log_url = f"{base_url}/dags/{dag_id}/runs/{run_id}"

    subject = f"🚨 Tarefa do Airflow Falha: {dag_id}. {task_id}"
    html_content = f"""
    <h3>🚨 Fall on Airflow Task!</h3>
    <ul>
      <li><b>DAG:</b> {dag_id}</li>
      <li><b>Task:</b> {task_id}</li>
      <li><b>ID of execution:</b> {run_id}</li>
      <li><b>Tempo of execution:</b> {execution_time}</li>
      <li><b>Error:</b> {error}</li>
    </ul>
    <p><a href="{log_url}">🔗 Show Logs on Airflow</a></p>
    """

    send_email_smtp(
        to="eric.oliveira.pro@gmail.com",
        subject=subject,
        html_content=html_content,
        conn_id="gmail_connection",
    )


def choose_deployment_path(ti):
    """Select the serving reload branch from the evaluation task result.

    Args:
        ti: Airflow task instance used to retrieve the evaluation output.

    Returns:
        The API reload task when promoted, otherwise the no-reload task.

    Raises:
        ValueError: If the evaluation task did not emit a valid promotion
            result.
    """
    evaluation_output = ti.xcom_pull(
        task_ids="model_governance_evaluation")
    if not isinstance(evaluation_output, str):
        raise ValueError("Model evaluation did not return a promotion result.")

    result_lines = evaluation_output.strip().splitlines()
    if not result_lines or result_lines[-1] not in {
        "MODEL_PROMOTED=true",
        "MODEL_PROMOTED=false",
    }:
        raise ValueError("Model evaluation returned an invalid promotion result.")

    if result_lines[-1] == "MODEL_PROMOTED=true":
        return "api_live_reload"
    return "stop_before_live_reload"


default_args = {
    'owner': 'ai_team',
    'depends_on_past': False,
    'start_date': datetime(2026, 9, 30, tzinfo=LOCAL_TZ),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'on_failure_callback': gmail_fail_alert,
}

with DAG(
    'corrosion_phase2_retrain',
    default_args=default_args,
    description='Phase 2 MLOps: Converts curated data, retrains, and deploys the model',
    schedule=None,  # IMPORTANT: Disabled! Awaits external trigger exclusively via HTTP API
    catchup=False,
    tags=['mlops', 'retrain', 'deployment']
) as dag:

    # 1. Converts human-corrected predictions into normalized YOLO labels
    task_merge_curated_data = BashOperator(
        task_id='merge_curated_data',
        bash_command=f"{PYTHON_EXECUTABLE} -m src.data_automation.parse_curated_data_to_train",
        env={
            "PYTHONPATH": PROJECT_ROOT,
            "CORROSION_SKIP_AIRFLOW_TRIGGER": "1",
        },
        append_env=True,
        cwd=PROJECT_ROOT
    )

    # 2. Versions the expanded dataset in DVC and synchronizes it with Sharepoint
    task_dvc_versioning = BashOperator(
        task_id='dvc_add_and_push',
        bash_command=f"""
            cd {PROJECT_ROOT};
            {PYTHON_EXECUTABLE} -m dvc add data;
            {PYTHON_EXECUTABLE} -m dvc push;
        """,
        env={"PYTHONPATH": PROJECT_ROOT},
        append_env=True,
        cwd=PROJECT_ROOT
    )

    # 3. Triggers YOLO training using the augmentations and hyperparameters
    task_train_yolo = BashOperator(
        task_id='train_new_yolo_model',
        bash_command=f"{PYTHON_EXECUTABLE} -m src.model_train.yolo",
        env={"PYTHONPATH": PROJECT_ROOT},
        append_env=True,
        cwd=PROJECT_ROOT
    )

    # 4. Runs the strict governance evaluation (mAP50 + Recall vs Champion)
    task_evaluate_and_gate = BashOperator(
        task_id='model_governance_evaluation',
        bash_command=f"cd {PROJECT_ROOT} && {PYTHON_EXECUTABLE} -m src.model_eval.yolo",
        env={"PYTHONPATH": PROJECT_ROOT},
        append_env=True,
        cwd=PROJECT_ROOT,
        do_xcom_push=True,
    )

    task_choose_deployment_path = BranchPythonOperator(
        task_id='choose_deployment_path',
        python_callable=choose_deployment_path,
    )

    # 5. CD (Continuous Deployment): Notifies the live API endpoint to dynamic reload the new champion model
    task_live_reload = BashOperator(
        task_id='api_live_reload',
        bash_command=f"curl --fail --show-error -X POST {API_URL}/reload",
        env={"PYTHONPATH": PROJECT_ROOT},
        append_env=True,
        cwd=PROJECT_ROOT
    )

    task_stop_before_live_reload = BashOperator(
        task_id='stop_before_live_reload',
        bash_command='echo "Challenger was not promoted; keeping the serving model unchanged."',
    )

    task_merge_curated_data >> task_dvc_versioning >> task_train_yolo >> task_evaluate_and_gate
    task_evaluate_and_gate >> task_choose_deployment_path
    task_choose_deployment_path >> [task_live_reload, task_stop_before_live_reload]
