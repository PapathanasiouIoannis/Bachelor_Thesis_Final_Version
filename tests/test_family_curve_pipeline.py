import pandas as pd
import pytest

from src.config import CONFIG
from src.ml.family_dataset import (
    build_curve_samples,
    curve_feature_columns,
)
from src.ml.family_splitting import (
    attach_family_split_assignments,
    build_family_split_manifest,
    load_family_split_profile,
)


def _physics_curve(curve_id: str, label: int) -> list[dict]:
    eos_id = "APR-1" if label == 0 else "CFL4"
    family_id = "H_APR" if label == 0 else "Q_CFL4"
    rows = []
    for mass in (0.8, 1.0, 1.5, 2.0, 2.2):
        rows.append(
            {
                "Curve_ID": curve_id,
                "EoS_ID": eos_id,
                "Variant_ID": f"{eos_id}:A00000",
                "Sweep_ID": "A00000",
                "Perturb_A": 0.0,
                "Perturb_eps0": 220.0,
                "Perturb_sigma": 50.0,
                "Baseline_Name": eos_id,
                "Family_Group_ID": family_id,
                "Parameter_Block_ID": "" if label == 0 else "Q_CFL_B060",
                "Model_Superfamily_ID": "H_TEST" if label == 0 else "Q_TEST",
                "Generation_Profile": "family_pilot_v1_2p0",
                "Gaussian_Grid_Max_Weight": 1.0,
                "Primary_Citation_Available": True,
                "Exact_Formula_Primary_Verified": bool(label),
                "Mass": mass,
                "Radius": 12.5 - label - 0.2 * mass,
                "Lambda": 10 ** (3.2 - mass - 0.1 * label),
                "Label": label,
            }
        )
    return rows


def test_complete_curve_becomes_one_observable_only_sample():
    physics = pd.DataFrame.from_records(
        [*_physics_curve("H_curve", 0), *_physics_curve("Q_curve", 1)]
    )

    samples = build_curve_samples(physics)

    assert len(samples) == 2
    assert samples["Sample_ID"].nunique() == 2
    assert len(curve_feature_columns("MR")) == CONFIG["ML_MASS_GRID_POINTS"]
    assert len(curve_feature_columns("MRL")) == 2 * CONFIG["ML_MASS_GRID_POINTS"]
    assert samples[curve_feature_columns("MRL")].notna().all().all()
    assert "Perturb_A" not in curve_feature_columns("MRL")


def _split_samples() -> pd.DataFrame:
    profile = load_family_split_profile()
    records = []
    for split, group_ids in profile["splits"].items():
        del split
        for group_id in group_ids:
            label = int(group_id.startswith("Q_"))
            records.append(
                {
                    "Sample_ID": f"{group_id}:0",
                    "EoS_ID": group_id,
                    "Family_Group_ID": group_id,
                    "Parameter_Block_ID": (
                        "Q_CFL_B100"
                        if group_id == "Q_CFL14"
                        else ("Q_CFL_B060" if label else "")
                    ),
                    "Generation_Profile": "family_pilot_v1_2p0",
                    "Label": label,
                }
            )
            if group_id == "H_SKYRME":
                records.append(
                    {
                        **records[-1],
                        "Sample_ID": "H_SKYRME:1",
                        "EoS_ID": "H_SKYRME_SECOND",
                    }
                )
    return pd.DataFrame.from_records(records)


def test_locked_family_split_is_balanced_and_disjoint():
    samples = _split_samples()
    profile = load_family_split_profile()
    manifest = build_family_split_manifest(samples, profile)
    assigned = attach_family_split_assignments(samples, manifest)

    assert assigned.groupby("Family_Group_ID")["Split"].nunique().max() == 1
    balance = assigned.groupby(["Split", "Label"]).size().unstack()
    assert balance[0].eq(balance[1]).all()
    assert set(
        assigned.loc[
            (assigned["Split"] == "test") & (assigned["Label"] == 1),
            "Parameter_Block_ID",
        ]
    ) == {"Q_CFL_B100"}


def test_locked_family_split_rejects_missing_group():
    samples = _split_samples()
    samples = samples[samples["Family_Group_ID"] != "H_MDI"]

    with pytest.raises(ValueError, match="do not match"):
        build_family_split_manifest(samples, load_family_split_profile())
