"""Run pre-model shortcut probes without opening the locked family test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ml.family_shortcuts import (
    audit_family_shortcuts,
    load_development_shortcut_inputs,
    write_shortcut_report,
)
from src.runtime import runtime_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    paths = runtime_paths(args.data_root)
    ml_dir = paths.data_root / "family_ml"
    required = [
        paths.physics_dataset,
        ml_dir / "curve_samples.parquet",
        ml_dir / "sample_audit.parquet",
        ml_dir / "split_manifest.parquet",
        ml_dir / "feature_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Shortcut audit is missing artifacts: {missing}")

    inputs = load_development_shortcut_inputs(
        samples_path=ml_dir / "curve_samples.parquet",
        sample_audit_path=ml_dir / "sample_audit.parquet",
        physics_path=paths.physics_dataset,
        split_manifest_path=ml_dir / "split_manifest.parquet",
    )
    report = audit_family_shortcuts(
        samples=inputs.samples,
        sample_audit=inputs.sample_audit,
        physics=inputs.physics,
        locked_test_family_ids=inputs.locked_test_family_ids,
        feature_manifest=json.loads(
            (ml_dir / "feature_manifest.json").read_text(encoding="utf-8")
        ),
    )
    write_shortcut_report(report, args.output_dir)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Family shortcut audit failed one or more gates.")


if __name__ == "__main__":
    main()
