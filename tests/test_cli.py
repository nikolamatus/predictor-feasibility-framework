"""CLI against the synthetic toy study."""

import subprocess
import sys
from pathlib import Path

from feasibility_framework.cli import main
from examples.toy_study import ToyStudy


ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy_study.py"


def test_validate_does_not_need_a_lock():
    assert main(["validate", str(TOY)]) == 0


def test_run_without_lock_exits_nonzero(tmp_path):
    code = main(["run", str(TOY), "--lock-dir", str(tmp_path), "--output", str(tmp_path / "out.json")])
    assert code == 1
    assert not (tmp_path / "out.json").exists()


def test_cli_run_and_report(tmp_path):
    study = ToyStudy(lock_dir=tmp_path)
    study.lock_preregistration()
    output = tmp_path / "result.json"
    assert main(["run", str(TOY), "--lock-dir", str(tmp_path), "--output", str(output)]) == 0
    assert output.is_file()
    assert main(["report", str(output)]) == 0


def test_installed_entrypoint_reports_help():
    exe = Path(sys.executable).with_name("feasibility.exe")
    if not exe.is_file():
        exe = Path(sys.executable).parent / "feasibility"
    completed = subprocess.run(
        [str(exe), "validate", str(TOY)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "config ok" in completed.stdout
