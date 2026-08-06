from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import eoslab


ROOT = Path(__file__).resolve().parents[1]


def test_doctor_reports_missing_dependencies_without_import_crash():
    completed = subprocess.run(
        [sys.executable, "-S", str(ROOT / "eoslab.py"), "doctor"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    assert "NumPy" in completed.stdout
    assert "Numba" in completed.stdout
    assert "FAIL" in completed.stdout
    assert "Traceback" not in completed.stderr


def test_validate_prints_professional_pair_boundary(capsys):
    code = eoslab.main(
        ["validate", str(ROOT / "configs" / "apr1_cfl4_reproduction.toml")]
    )
    output = capsys.readouterr()
    assert code == 0
    assert "Bag constant B" in output.out
    assert "Deformation centre epsilon_0" in output.out
    assert "Expected curves" in output.out
    assert "Classification: disabled" in output.out
    assert "universal matter-phase classification" in output.out


def test_pair_run_refuses_family_configuration(capsys):
    code = eoslab.main(["run", str(ROOT / "configs" / "family_classification.toml")])
    output = capsys.readouterr()
    assert code == 2
    assert "family classification" in output.err
    assert "family develop" in output.err


def test_family_status_reports_one_shot_lock(capsys):
    code = eoslab.main(
        ["family", "status", str(ROOT / "configs" / "family_classification.toml")]
    )
    output = capsys.readouterr()
    assert code == 0
    assert "Configured split-family groups" in output.out
    assert "Family-level partitions" in output.out
    assert "Recorded development safeguards" in output.out
    assert "LOCKED_TEST_OPENED" in output.out
    assert "Final-test open count" in output.out
    assert "False" in output.out
    assert "Recorded one-time final-test result" in output.out
    assert "Exact held-out-family results" in output.out
    assert "MDI-2" in output.out
    assert "CFL14" in output.out
    assert "not physical probabilities of quark matter" in output.out
