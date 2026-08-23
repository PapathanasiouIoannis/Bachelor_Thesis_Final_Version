from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from src.family_workflow import family_workflow_status


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = (
    "data",
    "models",
    "outputs",
    "outputs_perturb",
    "plots",
    "plots_perturb",
    "runs",
)
LOCK_MARKER = "data/family_pilot_v1/family_ml/LOCKED_TEST_OPENED.json"
DEPLOYMENT_MANIFEST = ROOT / "deployment" / "streamlit-artifacts.sha256"


def _deployment_artifacts() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in DEPLOYMENT_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, path = line.split("  ", maxsplit=1)
        entries[path] = digest
    return entries


def _git(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_only_declared_deployment_artifacts_and_lock_marker_are_tracked():
    completed = _git("ls-files", "--", *RUNTIME_ROOTS)
    assert completed.returncode == 0, completed.stderr
    tracked = {line for line in completed.stdout.splitlines() if line}

    assert tracked == {LOCK_MARKER, *_deployment_artifacts()}


def test_deployment_artifact_hashes_match_the_reviewed_manifest():
    for path, expected_digest in _deployment_artifacts().items():
        payload = (ROOT / path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected_digest, path


def test_runtime_outputs_are_ignored_but_the_lock_marker_is_explicitly_allowed():
    for generated_path in (
        "data/example/generated.parquet",
        "outputs/example/model.bin",
        "outputs_perturb/example/probabilities.npy",
        "plots/example/figure.png",
        "plots_perturb/example/figure.png",
        "runs/example/run_manifest.json",
    ):
        completed = _git("check-ignore", "--no-index", "--quiet", "--", generated_path)
        assert completed.returncode == 0, generated_path

    marker = _git("check-ignore", "--no-index", "--quiet", "--", LOCK_MARKER)
    assert marker.returncode == 1


def test_checked_in_family_evidence_chain_remains_valid():
    final = family_workflow_status()["final_test"]

    assert final["state"] == "LOCKED_TEST_OPENED"
    assert final["open_count"] == 1
    assert final["integrity"] == "valid"
    assert final["integrity_errors"] == []
    assert final["rerun_permitted"] is False
    assert Path(final["marker_path"]).resolve() == (ROOT / LOCK_MARKER).resolve()
