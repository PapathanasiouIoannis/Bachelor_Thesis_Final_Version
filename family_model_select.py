"""Select a family-held-out curve classifier without opening the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiment_config import FamilyClassificationSpec, load_experiment_config
from src.ml.family_model_selection import run_development_selection
from src.runtime import runtime_paths


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "family_classification.toml"


def _development_frames(ml_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(ml_dir / "train.parquet")
    validation = pd.read_parquet(ml_dir / "val.parquet")
    audit = pd.read_parquet(
        ml_dir / "sample_audit.parquet",
        filters=[("Split", "in", ["train", "val"])],
    )
    metadata = audit[["Sample_ID", "EoS_ID", "Group_ID", "Perturb_A", "Label", "Split"]]
    train = train.merge(
        metadata[metadata["Split"] == "train"].drop(columns=["Label", "Split"]),
        on="Sample_ID",
        how="inner",
        validate="one_to_one",
    )
    validation = validation.merge(
        metadata[metadata["Split"] == "val"].drop(columns=["Label", "Split"]),
        on="Sample_ID",
        how="inner",
        validate="one_to_one",
    )
    if len(train) != 84 or len(validation) != 12:
        raise RuntimeError(
            f"Development split size changed: train={len(train)}, val={len(validation)}"
        )
    return train, validation


def _write_plot(report: dict, output_path: Path) -> None:
    finalists = report["finalists"]
    labels = [record["candidate_id"] for record in finalists]
    inner = [record["cv_metrics"]["family_balanced_accuracy"] for record in finalists]
    validation = [
        record["validation_metrics"]["family_balanced_accuracy"] for record in finalists
    ]
    x = np.arange(len(finalists))
    width = 0.36
    fig, axis = plt.subplots(figsize=(10, 5.5))
    bars_inner = axis.bar(x - width / 2, inner, width, label="training family OOF")
    bars_val = axis.bar(x + width / 2, validation, width, label="validation families")
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Family-balanced accuracy")
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
    axis.set_title("Development-only family-held-out model selection")
    axis.legend(frameon=False)
    for bars in (bars_inner, bars_val):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(report: dict, output_path: Path) -> None:
    rows = []
    for record in report["finalists"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    record["candidate_id"],
                    f"{record['cv_metrics']['family_balanced_accuracy']:.3f}",
                    f"{record['validation_metrics']['family_balanced_accuracy']:.3f}",
                    f"{record['validation_metrics']['family_weighted_brier']:.4f}",
                    f"{record['validation_metrics']['maximum_probability_range_across_A']:.4f}",
                ]
            )
            + " |"
        )
    selected = report["selected_candidate"]
    markdown = "\n".join(
        [
            "# Family-held-out development model selection",
            "",
            "No locked-test row or metric was used. Hyperparameters were tuned with",
            "exhaustive out-of-family predictions on the 13 training groups, followed",
            "by one comparison of the dummy baseline and two logistic-regression",
            "specifications on the two validation groups.",
            "",
            "All primary metrics give equal total weight to each physical EoS family",
            "within each matter class.",
            "",
            "| Reporting candidate | Inner family accuracy | Validation family accuracy | Family-weighted validation Brier | Max score range across A |",
            "|---|---:|---:|---:|---:|",
            *rows,
            "",
            f"Selected candidate: `{selected['candidate_id']}`.",
            "",
            report["selection_rule"],
            "",
            "Random forest is retained only as an exploratory cross-validation",
            "diagnostic. XGBoost and the neural network remain separate exploratory",
            "workflows. None can enter reporting-grade selection.",
            "",
            "The selected specification must be committed as an immutable model profile",
            "before the final test pair is opened exactly once.",
        ]
    )
    output_path.write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="audited family-classification TOML profile",
    )
    args = parser.parse_args()
    specification = load_experiment_config(args.config)
    if not isinstance(specification, FamilyClassificationSpec):
        raise TypeError("Model selection requires a family-classification profile.")
    paths = runtime_paths(args.data_root)
    ml_dir = paths.data_root / "family_ml"
    train, validation = _development_frames(ml_dir)
    report, predictions = run_development_selection(
        train,
        validation,
        primary_models=specification.models.primary,
        exploratory_models=specification.models.exploratory,
    )
    report["configuration"] = {
        "path": str(args.config.resolve()),
        "primary_models": list(specification.models.primary),
        "exploratory_models": list(specification.models.exploratory),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "family_model_selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    predictions.to_parquet(
        paths.data_root / "family_ml" / "development_predictions.parquet",
        index=False,
    )
    _write_plot(report, args.output_dir / "FAMILY_MODEL_SELECTION.png")
    _write_markdown(report, args.output_dir / "FAMILY_MODEL_SELECTION.md")
    print(json.dumps(report["selected_candidate"], indent=2))


if __name__ == "__main__":
    main()
