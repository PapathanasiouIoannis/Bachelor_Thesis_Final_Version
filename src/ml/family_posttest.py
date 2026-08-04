"""Tamper checks and predeclared post-test sensitivity helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.ml.family_final import file_sha256
from src.ml.family_model_selection import (
    Candidate,
    build_family_pair_folds,
    evaluate_candidate_cv,
)


def load_completed_final_result(result_path: Path, marker_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "COMPLETED":
        raise RuntimeError("Locked test marker is not completed.")
    if marker.get("result_sha256") != file_sha256(result_path):
        raise RuntimeError("Locked final result hash does not match its one-shot marker.")
    if result.get("test_open_count") != 1:
        raise RuntimeError("Locked final result does not record exactly one test opening.")
    if result.get("locked_git_commit") != marker.get("locked_git_commit"):
        raise RuntimeError("Locked result and marker commit identities differ.")
    if result.get("model_profile_sha256") != marker.get("model_profile_sha256"):
        raise RuntimeError("Locked result and marker model-profile identities differ.")
    return result


def strict_2p08_development_sensitivity(
    *,
    samples: pd.DataFrame,
    sample_audit: pd.DataFrame,
    physics: pd.DataFrame,
    c_value: float,
    expected_variants: int = 6,
) -> tuple[dict, pd.DataFrame]:
    """Run OOF family CV on complete development EoSs with Mmax>=2.08 for every A."""

    curve_mmax = (
        physics.groupby(["EoS_ID", "Curve_ID"], as_index=False)["M_Max"].first()
    )
    eos_screen = (
        curve_mmax.groupby("EoS_ID", as_index=False)
        .agg(minimum_mmax_msun=("M_Max", "min"), variants=("Curve_ID", "nunique"))
    )
    strict_ids = set(
        eos_screen.loc[
            (eos_screen["minimum_mmax_msun"] >= 2.08)
            & (eos_screen["variants"] == expected_variants),
            "EoS_ID",
        ]
    )
    development_audit = sample_audit[
        sample_audit["Split"].isin(["train", "val"])
    ][["Sample_ID", "Group_ID", "Split"]]
    development = samples.merge(
        development_audit,
        on="Sample_ID",
        how="inner",
        validate="one_to_one",
    )
    strict = development[development["EoS_ID"].isin(strict_ids)].copy()
    if strict.empty or set(strict["Label"].astype(int)) != {0, 1}:
        raise RuntimeError("Strict-2.08 development sensitivity lost a matter class.")
    if strict.groupby("EoS_ID")["Sample_ID"].nunique().min() != expected_variants:
        raise RuntimeError("Strict-2.08 sensitivity contains an incomplete A grid.")
    candidate = Candidate(
        "logistic_mr_c0p1",
        "logistic",
        "MR",
        {"C": float(c_value)},
        10,
    )
    metrics, predictions, fold_metrics = evaluate_candidate_cv(
        strict,
        candidate,
        build_family_pair_folds(strict),
    )
    report = {
        "scope": "development-family out-of-fold sensitivity; not an independent test",
        "maximum_mass_floor_msun": 2.08,
        "complete_shared_A_grid_required": True,
        "eos_ids": sorted(strict_ids),
        "eos_count": int(strict["EoS_ID"].nunique()),
        "family_groups": int(strict["Group_ID"].nunique()),
        "curves": int(len(strict)),
        "curves_by_label": {
            str(label): int(count)
            for label, count in strict["Label"].value_counts().sort_index().items()
        },
        "metrics": metrics,
        "fold_metrics": fold_metrics,
        "minimum_mmax_by_eos": {
            row["EoS_ID"]: float(row["minimum_mmax_msun"])
            for _, row in eos_screen[eos_screen["EoS_ID"].isin(strict_ids)].iterrows()
        },
        "test_rows_used": 0,
    }
    return report, predictions
