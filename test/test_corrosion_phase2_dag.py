from pathlib import Path

from dags.corrosion_phase2_retrain import _find_project_root


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
