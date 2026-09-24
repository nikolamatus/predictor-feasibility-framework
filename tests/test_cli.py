"""CLI against synthetic studies. No real observations."""

import json
import os
import subprocess
import sys
from pathlib import Path

from feasibility_framework.cli import main
from examples.toy_study import ToyStudy


ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy_study.py"


def test_validate_does_not_need_a_lock():
    assert main(["validate", str(TOY)]) == 0


def test_run_without_lock_exits_nonzero(tmp_path, capsys):
    code = main(["run", str(TOY), "--lock-dir", str(tmp_path), "--output", str(tmp_path / "out.json")])
    assert code == 1
    assert not (tmp_path / "out.json").exists()
    assert "feasibility lock" in capsys.readouterr().err


def _write_tiny_study(directory: Path, floor: str = "1000.0") -> Path:
    path = directory / "tiny_study.py"
    path.write_text(
        f'''
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from feasibility_framework import FeasibilityStudy, Predictor, PredictorBlock, StudyConfig


class TinyStudy(FeasibilityStudy):
    """Synthetic rows only."""

    def define_config(self):
        return StudyConfig(
            study_id="tiny-synthetic",
            outcome="event",
            outcome_date=date(2022, 1, 1),
            cutoff=date(2021, 12, 31),
            id_key="person_id",
            blocks=(
                PredictorBlock(
                    name="history",
                    confirmed=True,
                    predictors=(
                        Predictor(
                            name="prior_count",
                            treatment="numeric",
                            confirmed=True,
                            availability=date(2021, 12, 31),
                            description="synthetic",
                        ),
                    ),
                ),
                PredictorBlock(
                    name="extra",
                    confirmed=True,
                    predictors=(
                        Predictor(
                            name="signal",
                            treatment="numeric",
                            confirmed=True,
                            availability=date(2021, 6, 30),
                            description="synthetic",
                        ),
                    ),
                ),
            ),
            baseline_blocks=("history",),
            full_blocks=("history", "extra"),
            sources=("cohort",),
            primary_source="cohort",
            epv_floor={floor},
            min_usable_n=20,
            min_outcome_events=5,
            min_outcome_prevalence=0.05,
            min_baseline_roc_auc=0.55,
            min_incremental_roc_lift=0.02,
            min_incremental_pr_lift=0.01,
        )

    def load_data(self, path_map):
        del path_map
        rng = np.random.default_rng(0)
        n = 80
        return {{
            "cohort": pd.DataFrame(
                {{
                    "person_id": np.arange(n),
                    "prior_count": rng.normal(size=n),
                    "signal": rng.normal(size=n),
                    "event": rng.integers(0, 2, size=n),
                }}
            )
        }}
''',
        encoding="utf-8",
    )
    return path


def test_lock_then_run_succeeds(tmp_path, capsys):
    study = _write_tiny_study(tmp_path)
    output = tmp_path / "result.json"
    assert main(["lock", str(study)]) == 0
    locked = capsys.readouterr().out
    lock_path = tmp_path / "locks" / "tiny-synthetic.lock.json"
    assert lock_path.is_file()
    assert str(lock_path.resolve()) in locked
    record = json.loads(lock_path.read_text(encoding="utf-8"))
    assert record["config_hash"] in locked
    assert record["locked_at"] in locked
    assert "already locked" not in locked
    assert main(["run", str(study), "--output", str(output)]) == 0
    assert output.is_file()


def test_lock_twice_is_idempotent(tmp_path, capsys):
    study = _write_tiny_study(tmp_path)
    assert main(["lock", str(study)]) == 0
    capsys.readouterr()
    lock_path = tmp_path / "locks" / "tiny-synthetic.lock.json"
    before = lock_path.read_bytes()
    assert main(["lock", str(study)]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "already locked, hash unchanged" in captured.out
    assert lock_path.read_bytes() == before


def test_lock_after_config_change_refuses(tmp_path, capsys):
    study = _write_tiny_study(tmp_path, floor="1000.0")
    assert main(["lock", str(study)]) == 0
    capsys.readouterr()
    lock_path = tmp_path / "locks" / "tiny-synthetic.lock.json"
    before = lock_path.read_bytes()
    _write_tiny_study(tmp_path, floor="1001.0")
    assert main(["lock", str(study)]) != 0
    assert lock_path.read_bytes() == before
    assert not (tmp_path / "locks" / "history").exists()
    assert "amend" in capsys.readouterr().err


def test_amend_archives_original_timestamp(tmp_path, capsys):
    study = _write_tiny_study(tmp_path, floor="1000.0")
    assert main(["lock", str(study)]) == 0
    capsys.readouterr()
    lock_path = tmp_path / "locks" / "tiny-synthetic.lock.json"
    original = json.loads(lock_path.read_text(encoding="utf-8"))
    _write_tiny_study(tmp_path, floor="1001.0")
    assert main(["lock", str(study), "--amend"]) == 0
    archived = list((tmp_path / "locks" / "history").glob("*.lock.json"))
    assert len(archived) == 1
    safe_time = original["locked_at"].replace(":", "-")
    assert safe_time in archived[0].name
    moved = json.loads(archived[0].read_text(encoding="utf-8"))
    assert moved["locked_at"] == original["locked_at"]
    assert moved["config_hash"] == original["config_hash"]
    current = json.loads(lock_path.read_text(encoding="utf-8"))
    assert current["config_hash"] != original["config_hash"]
    assert current["locked_at"] != original["locked_at"]


def test_run_from_another_cwd_uses_the_study_lock(tmp_path, capsys):
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    study = _write_tiny_study(study_dir)
    assert main(["lock", str(study)]) == 0
    capsys.readouterr()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    output = elsewhere / "result.json"
    previous = Path.cwd()
    try:
        os.chdir(elsewhere)
        code = main(["run", str(study), "--output", str(output)])
    finally:
        os.chdir(previous)
    assert code == 0
    assert output.is_file()
    assert not (elsewhere / "locks").exists()
    assert (study_dir / "locks" / "tiny-synthetic.lock.json").is_file()


def test_gitignore_does_not_exclude_locks():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "locks/" not in text
    assert "locks/history/" not in text


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


def test_validate_does_not_import_heavy_stack():
    script = f"""
import sys
from feasibility_framework.cli import main

code = main(["validate", r"{TOY}"])
assert code == 0, code
heavy = ("sklearn", "scipy", "statsmodels")
loaded = sorted(
    name
    for name in sys.modules
    if name in heavy or name.startswith(tuple(h + "." for h in heavy))
)
assert not loaded, loaded
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
