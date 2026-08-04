"""Run pre-model shortcut probes without opening the locked family test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.ml.family_shortcuts import audit_family_shortcuts, write_shortcut_report
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
        ml_dir / "feature_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Shortcut audit is missing artifacts: {missing}")

    report = audit_family_shortcuts(
        samples=pd.read_parquet(ml_dir / "curve_samples.parquet"),
        sample_audit=pd.read_parquet(ml_dir / "sample_audit.parquet"),
        physics=pd.read_parquet(paths.physics_dataset),
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
