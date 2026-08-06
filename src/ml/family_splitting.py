"""Locked physical-family splits for the one-week classification pilot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FAMILY_SPLIT_PROFILE = (
    PROJECT_ROOT / "framework" / "family_split_profile.json"
)
FAMILY_SPLITS = ("train", "val", "test")


def load_family_split_profile(path: Path | None = None) -> dict:
    profile_path = (path or DEFAULT_FAMILY_SPLIT_PROFILE).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    required = {
        "profile_id",
        "expected_generation_profile",
        "locked_before_model_fit",
        "selection_rule",
        "splits",
        "secondary_quark_block_ood",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Family split profile is missing fields: {missing}")
    if not profile["locked_before_model_fit"]:
        raise ValueError("Family test split must be locked before any model fit.")
    if set(profile["splits"]) != set(FAMILY_SPLITS):
        raise ValueError("Family split profile must define train, val, and test.")

    assignments = [
        (group_id, split)
        for split, group_ids in profile["splits"].items()
        for group_id in group_ids
    ]
    group_ids = [group_id for group_id, _ in assignments]
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("A physical family appears in multiple locked splits.")
    if any(not group_ids for group_ids in profile["splits"].values()):
        raise ValueError("Every locked split must contain at least one family.")
    profile["profile_path"] = str(profile_path)
    profile["profile_sha256"] = hashlib.sha256(profile_path.read_bytes()).hexdigest()
    return profile


def build_family_split_manifest(samples: pd.DataFrame, profile: dict) -> pd.DataFrame:
    required = {
        "Sample_ID",
        "EoS_ID",
        "Family_Group_ID",
        "Parameter_Block_ID",
        "Generation_Profile",
        "Label",
    }
    missing = sorted(required - set(samples.columns))
    if missing:
        raise KeyError(f"Curve samples are missing split metadata: {missing}")
    if samples["Sample_ID"].duplicated().any():
        raise ValueError("Sample_ID must be unique before family splitting.")
    if set(samples["Generation_Profile"].unique()) != {
        profile["expected_generation_profile"]
    }:
        raise ValueError("Curve samples do not match the locked generation profile.")
    if samples.groupby("Family_Group_ID")["Label"].nunique().max() != 1:
        raise ValueError("A physical family crosses matter-class labels.")

    assignment = {
        group_id: split
        for split, group_ids in profile["splits"].items()
        for group_id in group_ids
    }
    observed_groups = set(samples["Family_Group_ID"].astype(str))
    expected_groups = set(assignment)
    if observed_groups != expected_groups:
        raise ValueError(
            "Locked split families do not match the dataset: "
            f"missing={sorted(expected_groups - observed_groups)}, "
            f"unexpected={sorted(observed_groups - expected_groups)}"
        )

    manifest = (
        samples.assign(
            Group_ID=samples["Family_Group_ID"].astype(str),
            Split=samples["Family_Group_ID"].astype(str).map(assignment),
        )
        .groupby(["Group_ID", "Split"], as_index=False)
        .agg(
            Label=("Label", "first"),
            EoS_Count=("EoS_ID", "nunique"),
            Curve_Count=("Sample_ID", "nunique"),
            Parameter_Blocks=(
                "Parameter_Block_ID",
                lambda values: ",".join(sorted({str(value) for value in values if str(value)})),
            ),
        )
    )
    manifest["Split_Profile"] = profile["profile_id"]
    if manifest["Group_ID"].duplicated().any():
        raise RuntimeError("A family has more than one manifest assignment.")
    for split in FAMILY_SPLITS:
        labels = set(manifest.loc[manifest["Split"] == split, "Label"].astype(int))
        if labels != {0, 1}:
            raise ValueError(f"The {split} split does not contain both matter classes.")

    assigned = attach_family_split_assignments(samples, manifest)
    curve_balance = assigned.groupby(["Split", "Label"])["Sample_ID"].nunique().unstack()
    if not curve_balance[0].eq(curve_balance[1]).all():
        raise ValueError(
            "Locked split is not curve-balanced within each split: "
            f"{curve_balance.to_dict()}"
        )

    ood = profile["secondary_quark_block_ood"]
    test_block = str(ood["test_parameter_block_id"])
    quark = assigned[assigned["Label"].astype(int) == 1]
    train_blocks = set(
        quark.loc[quark["Split"] == "train", "Parameter_Block_ID"].astype(str)
    )
    test_blocks = set(
        quark.loc[quark["Split"] == "test", "Parameter_Block_ID"].astype(str)
    )
    if test_block in train_blocks or test_blocks != {test_block}:
        raise ValueError(
            "Secondary quark block-OOD lock failed: "
            f"train={sorted(train_blocks)}, test={sorted(test_blocks)}"
        )
    return manifest.sort_values(["Split", "Label", "Group_ID"]).reset_index(drop=True)


def attach_family_split_assignments(
    samples: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    split_map = manifest.set_index("Group_ID")["Split"]
    assigned = samples.copy()
    assigned["Group_ID"] = assigned["Family_Group_ID"].astype(str)
    assigned["Split"] = assigned["Group_ID"].map(split_map)
    if assigned["Split"].isna().any():
        missing = sorted(assigned.loc[assigned["Split"].isna(), "Group_ID"].unique())
        raise RuntimeError(f"Curve samples have no family split assignment: {missing}")
    if assigned.groupby("Group_ID")["Split"].nunique().max() != 1:
        raise RuntimeError("Physical-family leakage detected across outer splits.")
    if assigned.groupby("EoS_ID")["Split"].nunique().max() != 1:
        raise RuntimeError("A fixed EoS baseline crosses outer splits.")
    return assigned


def family_manifest_fingerprint(manifest: pd.DataFrame) -> str:
    canonical = manifest.sort_values("Group_ID").reset_index(drop=True)
    hashes = pd.util.hash_pandas_object(canonical, index=False).to_numpy()
    return hashlib.sha256(hashes.tobytes()).hexdigest()
