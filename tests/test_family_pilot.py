import json

import pandas as pd
import pytest

from framework.family_pilot import (
    FAMILY_METADATA_COLUMNS,
    load_family_pilot_profile,
    profile_entries,
    profile_sweep_points,
    validate_family_pilot_dataset,
)
from src.config import CONFIG


def test_locked_family_profile_is_balanced_and_scan_verified():
    profile = load_family_pilot_profile()
    entries = profile_entries(profile)
    points = profile_sweep_points(profile)

    assert len(profile["hadronic_eos_ids"]) == 9
    assert len(profile["quark_eos_ids"]) == 9
    assert len(entries) == 18
    assert [point.amplitude for point in points] == pytest.approx(
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
    )
    assert "PS" not in profile["hadronic_eos_ids"]
    assert "MDI-3" not in profile["hadronic_eos_ids"]


def test_profile_rejects_an_unverified_amplitude(tmp_path):
    profile = json.loads(
        open("framework/family_pilot_profile.json", encoding="utf-8").read()
    )
    profile["deformation"]["amplitudes"] = [0.0, 0.01, 0.09]
    path = tmp_path / "invalid_profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="rejected combinations"):
        load_family_pilot_profile(path)


def test_family_dataset_validation_rejects_incomplete_amplitude_grid():
    profile = load_family_pilot_profile()
    columns = [*CONFIG["COLUMN_SCHEMA"], *FAMILY_METADATA_COLUMNS]
    frame = pd.DataFrame(columns=columns)
    frame.loc[0, "EoS_ID"] = "APR-1"
    frame.loc[0, "Sweep_ID"] = "A00000"
    frame.loc[0, "Curve_ID"] = "curve"
    frame.loc[0, "Label"] = 0
    frame.loc[0, "Generation_Profile"] = profile["profile_id"]
    frame.loc[0, "Mass"] = 1.0
    frame.loc[0, "Perturb_A"] = 0.0
    frame.loc[0, "Variant_ID"] = "APR-1:A00000"
    frame.loc[0, "Family_Group_ID"] = "H_APR"
    frame.loc[0, "Parameter_Block_ID"] = ""
    frame.loc[0, "Model_Superfamily_ID"] = "H_REPOSITORY_SURROGATES"
    frame.loc[0, "Gaussian_Grid_Max_Weight"] = 1.0
    frame.loc[0, "Primary_Citation_Available"] = True
    frame.loc[0, "Exact_Formula_Primary_Verified"] = False

    with pytest.raises(ValueError, match="EoS IDs"):
        validate_family_pilot_dataset(frame, profile)
