import sys

from dags.corrosion_phase1_monitoring import _find_project_root


def test_find_project_root_when_dag_is_copied_to_airflow_home(tmp_path):
    project_root = tmp_path / "checkout"
    (project_root / "src" / "observability").mkdir(parents=True)
    (project_root / "src" / "observability" / "monitor.py").touch()
    python_executable = project_root / ".venv" / "bin" / "python"
    python_executable.parent.mkdir(parents=True)
    python_executable.touch()

    copied_dag = tmp_path / "airflow" / "dags" / "corrosion_phase1_monitoring.py"
    copied_dag.parent.mkdir(parents=True)
    copied_dag.touch()

    assert _find_project_root(copied_dag, python_executable) == project_root
