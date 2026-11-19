from pathlib import Path
from unittest.mock import Mock

import pytest

from dags.corrosion_phase2_retrain import (
    _find_project_root,
    choose_deployment_path,
    dag,
)


def test_find_project_root_when_dag_is_copied_to_airflow_home(tmp_path):
    project_root = tmp_path / "checkout"
    module_path = project_root / "src" / \
        "data_automation" / "parse_review_to_train.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()

    airflow_dag = tmp_path / "airflow" / "dags" / "corrosion_phase2_retrain.py"
    airflow_dag.parent.mkdir(parents=True)
    airflow_dag.touch()

    python_executable = project_root / ".venv" / "bin" / "python"

    assert _find_project_root(airflow_dag, python_executable) == project_root


@pytest.mark.parametrize(
    ("evaluation_result", "expected_task"),
    [
        ("MODEL_PROMOTED=true", "api_live_reload"),
        ("MODEL_PROMOTED=false", "stop_before_live_reload"),
    ],
)
def test_choose_deployment_path(evaluation_result, expected_task):
    task_instance = Mock()
    task_instance.xcom_pull.return_value = (
        f"evaluation logs\n{evaluation_result}")

    assert choose_deployment_path(task_instance) == expected_task


def test_choose_deployment_path_rejects_missing_result():
    task_instance = Mock()
    task_instance.xcom_pull.return_value = "evaluation logs without result"

    with pytest.raises(ValueError, match="invalid promotion result"):
        choose_deployment_path(task_instance)


def test_live_reload_is_gated_by_deployment_branch():
    branch_task = dag.get_task("choose_deployment_path")

    assert branch_task.downstream_task_ids == {
        "api_live_reload",
        "stop_before_live_reload",
    }
