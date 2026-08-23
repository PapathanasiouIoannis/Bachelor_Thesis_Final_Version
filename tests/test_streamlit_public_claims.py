from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
APP_PATHS = (ROOT / "app_ref.py", ROOT / "perturb_app_ref.py")


@pytest.mark.parametrize("app_path", APP_PATHS)
def test_public_demonstrations_state_their_scientific_boundary(app_path: Path) -> None:
    source = app_path.read_text(encoding="utf-8")
    normalized_source = " ".join(source.replace('"', "").split())

    for required in (
        "restricted synthetic",
        "not observational measurements",
        "not a confidence interval",
        "Charalampos Moustakidis",
        "Theodoros Diakonidis",
        "Codex assisted with",
        "Fixed_CFL4_Model_Score",
        "Reference XGBoost score explanation",
    ):
        assert required in normalized_source


@pytest.mark.parametrize("app_path", APP_PATHS)
def test_public_demonstrations_do_not_restore_overstated_labels(app_path: Path) -> None:
    source = app_path.read_text(encoding="utf-8")

    for forbidden in (
        "Inference Engine",
        "Observatory Mode",
        "Telemetry Input",
        "Quark_Probability",
        "Inference Reliability",
        "95% CI",
        "Quark Phase %",
        "Telescope Measurement",
        "rigorous Monte Carlo",
        "robust Monte Carlo",
    ):
        assert forbidden not in source
