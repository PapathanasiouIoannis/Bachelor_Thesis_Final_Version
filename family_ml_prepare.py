"""Prepare leakage-safe one-sample-per-curve family classification data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from framework.family_pilot import (
    DEFAULT_PROFILE_PATH,
    load_family_pilot_profile,
    validate_family_pilot_dataset,
)
from src.ml.family_dataset import (
    CURVE_METADATA_COLUMNS,
    build_curve_samples,
    curve_dataset_fingerprint,
    curve_feature_columns,
    observable_mass_grid,
)
from src.ml.family_splitting import (
    DEFAULT_FAMILY_SPLIT_PROFILE,
    attach_family_split_assignments,
    build_family_split_manifest,
    family_manifest_fingerprint,
    load_family_split_profile,
)
from src.runtime import runtime_paths, write_run_manifest


def prepare_family_ml_data(
    *,
    data_root: Path,
    generation_profile_path: Path = DEFAULT_PROFILE_PATH,
    split_profile_path: Path = DEFAULT_FAMILY_SPLIT_PROFILE,
) -> dict:
    paths = runtime_paths(data_root)
    if not paths.physics_dataset.is_file():
        raise FileNotFoundError(
            f"Family physics dataset is missing: {paths.physics_dataset}. "
            "Run family_physics_main.py first."
        )
    generation_profile = load_family_pilot_profile(generation_profile_path)
    split_profile = load_family_split_profile(split_profile_path)
    physics = pd.read_parquet(paths.physics_dataset)
    validate_family_pilot_dataset(physics, generation_profile)

    samples = build_curve_samples(physics)
    manifest = build_family_split_manifest(samples, split_profile)
    assigned = attach_family_split_assignments(samples, manifest)
    features_mr = curve_feature_columns("MR")
    features_mrl = curve_feature_columns("MRL")

    output_dir = paths.data_root / "family_ml"
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(output_dir / "curve_samples.parquet", index=False)
    manifest.to_parquet(output_dir / "split_manifest.parquet", index=False)
    audit_columns = [
        "Sample_ID",
        *CURVE_METADATA_COLUMNS,
        "Group_ID",
        "Split",
    ]
    assigned[audit_columns].to_parquet(
        output_dir / "sample_audit.parquet", index=False
    )
    for split in ("train", "val", "test"):
        subset = assigned[assigned["Split"] == split].sort_values("Sample_ID")
        subset[["Sample_ID", *features_mrl, "Label"]].to_parquet(
            output_dir / f"{split}.parquet", index=False
        )

    feature_manifest = {
        "representation": "one complete curve per sample",
        "mass_is_implicit": True,
        "mass_grid_msun": observable_mass_grid().tolist(),
        "feature_sets": {"MR": features_mr, "MRL": features_mrl},
        "forbidden_model_inputs": [
            column
            for column in CURVE_METADATA_COLUMNS
            if column not in {"Label"}
        ],
        "scaling_policy": "Fit preprocessing on outer training only and refit inside every family-held-out CV fold.",
    }
    (output_dir / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2) + "\n", encoding="utf-8"
    )

    split_counts = {
        split: {
            "curves": int((assigned["Split"] == split).sum()),
            "curves_by_label": {
                str(label): int(count)
                for label, count in assigned[assigned["Split"] == split][
                    "Label"
                ].value_counts().sort_index().items()
            },
            "family_groups": int(
                assigned.loc[assigned["Split"] == split, "Group_ID"].nunique()
            ),
        }
        for split in ("train", "val", "test")
    }
    metadata = {
        "experiment_scope": generation_profile["claim_boundary"],
        "generation_profile_id": generation_profile["profile_id"],
        "generation_profile_sha256": generation_profile["profile_sha256"],
        "split_profile_id": split_profile["profile_id"],
        "split_profile_sha256": split_profile["profile_sha256"],
        "curve_dataset_sha256": curve_dataset_fingerprint(samples),
        "split_manifest_sha256": family_manifest_fingerprint(manifest),
        "samples": int(len(samples)),
        "features_mr": len(features_mr),
        "features_mrl": len(features_mrl),
        "split_counts": split_counts,
        "test_status": "LOCKED_NOT_EVALUATED",
        "test_use_policy": split_profile["test_use_policy"],
    }
    run_manifest_path = write_run_manifest(
        output_dir,
        "family_curve_level_preparation",
        paths.data_root,
        metadata,
    )
    return {
        "output_dir": str(output_dir),
        "run_manifest": str(run_manifest_path),
        **metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--generation-profile", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--split-profile", type=Path, default=DEFAULT_FAMILY_SPLIT_PROFILE)
    args = parser.parse_args()
    summary = prepare_family_ml_data(
        data_root=args.data_root,
        generation_profile_path=args.generation_profile,
        split_profile_path=args.split_profile,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
