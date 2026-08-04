"""Open the immutable family test pair once and write the final result."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.ml.family_final import (
    DEFAULT_MODEL_PROFILE,
    file_sha256,
    fit_locked_estimator,
    load_locked_model_profile,
    predict_locked_estimator,
    verify_locked_evidence,
)
from src.ml.family_model_selection import classification_metrics
from src.runtime import runtime_paths, write_run_manifest


PROJECT_ROOT = Path(__file__).resolve().parent
LOCKED_SOURCE_PATHS = (
    "framework/family_model_profile.json",
    "src/ml/family_final.py",
    "family_final_test.py",
)


def _require_committed_lock() -> str:
    for path in LOCKED_SOURCE_PATHS:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
    for staged in (False, True):
        command = ["git", "diff", "--quiet"]
        if staged:
            command.append("--cached")
        command.extend(["--", *LOCKED_SOURCE_PATHS])
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                "Model profile and final-test code must be committed and unchanged "
                "before the locked test is opened."
            )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _load_split(ml_dir: Path, split: str) -> pd.DataFrame:
    features = pd.read_parquet(ml_dir / f"{split}.parquet")
    audit = pd.read_parquet(
        ml_dir / "sample_audit.parquet", filters=[("Split", "==", split)]
    )
    metadata = audit[
        ["Sample_ID", "EoS_ID", "Group_ID", "Perturb_A", "Label"]
    ]
    merged = features.merge(
        metadata.drop(columns="Label"),
        on="Sample_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise RuntimeError(f"The {split} tensor and audit sidecar are not aligned.")
    return merged


def _plot_predictions(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    for eos_id, subset in predictions.groupby("EoS_ID", sort=True):
        subset = subset.sort_values("Perturb_A")
        axis.plot(
            subset["Perturb_A"],
            subset["Probability_Quark"],
            marker="o",
            linewidth=2,
            label=eos_id,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="decision threshold")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Gaussian amplitude A")
    axis.set_ylabel("Locked-model probability of CFL class")
    axis.set_title("One-time held-out family test")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(report: dict, output_path: Path) -> None:
    metrics = report["test_metrics"]
    family_rows = [
        "| "
        + " | ".join(
            [
                row["EoS_ID"],
                str(row["Label"]),
                f"{row['accuracy']:.3f}",
                f"{row['mean_probability_quark']:.4f}",
                f"{row['probability_range_across_A']:.4f}",
                f"{row['minimum_mmax_msun']:.4f}",
            ]
        )
        + " |"
        for row in report["per_eos"]
    ]
    markdown = "\n".join(
        [
            "# One-time locked family test",
            "",
            f"Locked commit: `{report['locked_git_commit']}`.",
            "",
            f"The frozen radius-only logistic model classified {metrics['samples']} of "
            f"{metrics['samples']} curve variants with balanced accuracy "
            f"{metrics['balanced_accuracy']:.3f} and ROC AUC {metrics['roc_auc']:.3f}.",
            "",
            "| EoS | Label | Accuracy | Mean P(CFL) | P range across A | Minimum Mmax |",
            "|---|---:|---:|---:|---:|---:|",
            *family_rows,
            "",
            "Only two independent physical families are present in this final test; the",
            "six A variants per EoS are correlated sensitivity variants, not twelve",
            "independent validation objects. No family-level confidence interval is",
            "claimed. The CFL test family is also an unseen B=100 parameter block.",
            "",
            "Neither test EoS reaches the 2.08 M_sun sensitivity threshold over the",
            "shared A grid. Consequently this is a 2.0 M_sun-screen result; strict-2.08",
            "performance must be reported separately as development-family OOF only.",
        ]
    )
    output_path.write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument("--profile", type=Path, default=DEFAULT_MODEL_PROFILE)
    args = parser.parse_args()
    final_json = args.output_dir / "family_final_test.json"
    if final_json.exists():
        raise RuntimeError(f"Locked final result already exists; refusing to rerun: {final_json}")

    profile = load_locked_model_profile(args.profile)
    paths = runtime_paths(args.data_root)
    ml_dir = paths.data_root / "family_ml"
    verify_locked_evidence(profile, ml_dir)
    locked_commit = _require_committed_lock()

    development = pd.concat(
        [_load_split(ml_dir, "train"), _load_split(ml_dir, "val")],
        ignore_index=True,
    )
    estimator = fit_locked_estimator(development, profile)
    marker_path = ml_dir / "LOCKED_TEST_OPENED.json"
    marker_payload = {
        "status": "OPENED",
        "opened_utc": datetime.now(timezone.utc).isoformat(),
        "locked_git_commit": locked_commit,
        "model_profile_sha256": profile["profile_sha256"],
    }
    try:
        with marker_path.open("x", encoding="utf-8") as handle:
            json.dump(marker_payload, handle, indent=2)
    except FileExistsError as exc:
        raise RuntimeError(
            f"Locked test marker already exists; refusing to open test again: {marker_path}"
        ) from exc

    test = _load_split(ml_dir, "test")
    expected = profile["final_test"]
    if len(test) != expected["expected_samples"]:
        raise RuntimeError("Locked test sample count changed.")
    if set(test["Group_ID"]) != set(expected["expected_family_groups"]):
        raise RuntimeError("Locked test family identities changed.")
    if set(test["EoS_ID"]) != set(expected["expected_eos_ids"]):
        raise RuntimeError("Locked test EoS identities changed.")

    predictions = predict_locked_estimator(estimator, test, profile)
    metric_input = predictions[
        ["Sample_ID", "EoS_ID", "Group_ID", "Perturb_A", "Label", "Probability_Quark"]
    ]
    metrics = classification_metrics(metric_input)
    physics = pd.read_parquet(paths.physics_dataset)
    test_mmax = (
        physics[physics["Curve_ID"].isin(test["Sample_ID"])]
        .groupby("EoS_ID")["M_Max"]
        .min()
        .to_dict()
    )
    per_eos = []
    for eos_id, subset in predictions.groupby("EoS_ID", sort=True):
        per_eos.append(
            {
                "EoS_ID": eos_id,
                "Label": int(subset["Label"].iloc[0]),
                "curves": int(len(subset)),
                "accuracy": float(subset["Correct"].mean()),
                "mean_probability_quark": float(subset["Probability_Quark"].mean()),
                "probability_range_across_A": float(
                    subset["Probability_Quark"].max()
                    - subset["Probability_Quark"].min()
                ),
                "minimum_mmax_msun": float(test_mmax[eos_id]),
            }
        )
    report = {
        "model_profile_id": profile["profile_id"],
        "model_profile_sha256": profile["profile_sha256"],
        "locked_git_commit": locked_commit,
        "opened_utc": marker_payload["opened_utc"],
        "test_open_count": 1,
        "test_metrics": metrics,
        "per_eos": per_eos,
        "independent_test_family_units": int(test["Group_ID"].nunique()),
        "correlated_A_variants": int(len(test)),
        "strict_2p08_test_applicable": bool(
            all(value >= 2.08 for value in test_mmax.values())
        ),
        "claim_boundary": profile["claim_boundary"],
        "predictions": predictions.to_dict(orient="records"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    predictions.to_parquet(ml_dir / "final_test_predictions.parquet", index=False)
    joblib.dump(estimator, ml_dir / "locked_logistic_mr.joblib")
    _plot_predictions(predictions, args.output_dir / "FAMILY_FINAL_TEST.png")
    _write_markdown(report, args.output_dir / "FAMILY_FINAL_TEST.md")
    marker_payload["status"] = "COMPLETED"
    marker_payload["result_sha256"] = file_sha256(final_json)
    marker_path.write_text(json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8")
    write_run_manifest(
        ml_dir / "final_test",
        "family_locked_final_test",
        paths.data_root,
        {
            "model_profile_sha256": profile["profile_sha256"],
            "locked_git_commit": locked_commit,
            "result_sha256": marker_payload["result_sha256"],
            "test_open_count": 1,
        },
    )
    print(json.dumps(report["test_metrics"], indent=2))


if __name__ == "__main__":
    main()
