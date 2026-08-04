"""Generate the locked multi-family one-week physics pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from framework.family_pilot import (
    DEFAULT_PROFILE_PATH,
    generate_family_curve,
    load_family_pilot_profile,
    profile_entries,
    profile_sweep_points,
    validate_family_pilot_dataset,
)
from src.runtime import runtime_paths, write_run_manifest


def _save_ready_files(frame: pd.DataFrame, data_root: Path) -> None:
    paths = runtime_paths(data_root)
    paths.hadronic_ready_dir.mkdir(parents=True, exist_ok=True)
    paths.quark_ready_dir.mkdir(parents=True, exist_ok=True)
    for directory in (paths.hadronic_ready_dir, paths.quark_ready_dir):
        for path in directory.glob("*.parquet"):
            path.unlink()
    for eos_id, subset in frame.groupby("EoS_ID", sort=True):
        destination = (
            paths.hadronic_ready_dir if int(subset["Label"].iloc[0]) == 0 else paths.quark_ready_dir
        )
        subset.to_parquet(destination / f"dataset_{eos_id}.parquet", index=False)


def generate_family_pilot(
    *,
    data_root: Path,
    profile_path: Path,
    jobs: int,
    force: bool,
) -> tuple[pd.DataFrame, Path]:
    profile = load_family_pilot_profile(profile_path)
    paths = runtime_paths(data_root)
    paths.data_root.mkdir(parents=True, exist_ok=True)
    if paths.physics_dataset.exists() and not force:
        frame = pd.read_parquet(paths.physics_dataset)
        validate_family_pilot_dataset(frame, profile)
        return frame, paths.physics_dataset

    entries = profile_entries(profile)
    points = profile_sweep_points(profile)
    frames = Parallel(n_jobs=jobs, verbose=10)(
        delayed(generate_family_curve)(entry, point, profile)
        for entry in entries
        for point in points
    )
    frame = pd.concat(frames, ignore_index=True)
    frame["LogLambda"] = np.log10(frame["Lambda"])
    validate_family_pilot_dataset(frame, profile)
    frame = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
    frame.to_parquet(paths.physics_dataset, index=False)
    _save_ready_files(frame, paths.data_root)

    catalog = (
        frame[
            [
                "EoS_ID",
                "Label",
                "Family_Group_ID",
                "Parameter_Block_ID",
                "Model_Superfamily_ID",
                "Primary_Citation_Available",
                "Exact_Formula_Primary_Verified",
            ]
        ]
        .drop_duplicates()
        .sort_values(["Label", "EoS_ID"])
    )
    catalog.to_parquet(paths.data_root / "family_catalog.parquet", index=False)
    manifest_metadata = {
        "experiment_scope": profile["claim_boundary"],
        "profile_id": profile["profile_id"],
        "profile_sha256": profile["profile_sha256"],
        "profile_path": profile["profile_path"],
        "minimum_maximum_mass_msun": profile["screens"][
            "minimum_maximum_mass_msun"
        ],
        "strict_2p08_sensitivity_required": True,
        "hadronic_eos_ids": profile["hadronic_eos_ids"],
        "quark_eos_ids": profile["quark_eos_ids"],
        "amplitudes": profile["deformation"]["amplitudes"],
        "epsilon0_mev_fm3": profile["deformation"]["epsilon0_mev_fm3"],
        "sigma_mev_fm3": profile["deformation"]["sigma_mev_fm3"],
        "curves": int(frame["Curve_ID"].nunique()),
        "curves_per_class": {
            str(label): int(count)
            for label, count in frame.groupby("Label")["Curve_ID"].nunique().items()
        },
        "rows": int(len(frame)),
        "family_groups": int(frame["Family_Group_ID"].nunique()),
        "quark_parameter_blocks": int(
            frame.loc[frame["Label"] == 1, "Parameter_Block_ID"].nunique()
        ),
    }
    manifest_path = write_run_manifest(
        paths.data_root,
        "family_physics_generation",
        paths.data_root,
        manifest_metadata,
    )
    return frame, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--force-regenerate", action="store_true")
    args = parser.parse_args()
    frame, artifact = generate_family_pilot(
        data_root=args.data_root,
        profile_path=args.profile,
        jobs=args.n_jobs,
        force=args.force_regenerate,
    )
    summary = {
        "artifact": str(artifact),
        "rows": len(frame),
        "curves": frame["Curve_ID"].nunique(),
        "curves_by_label": frame.groupby("Label")["Curve_ID"].nunique().to_dict(),
        "eos_ids": frame["EoS_ID"].nunique(),
        "family_groups": frame["Family_Group_ID"].nunique(),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
