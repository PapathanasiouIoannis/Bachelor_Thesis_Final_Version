"""Run development-only ablations and the physical-family permutation null."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from family_model_select import _development_frames
from src.ml.family_dataset import radius_feature_columns
from src.ml.family_robustness import (
    family_label_permutation_null,
    single_mass_ablation,
)
from src.runtime import runtime_paths


def _plot(report: dict, output_path: Path) -> None:
    ablation = report["single_mass_ablation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for observable, color in (("radius", "#4C78A8"), ("tidal", "#F58518")):
        rows = [row for row in ablation if row["observable"] == observable]
        masses = [row["mass_msun"] for row in rows]
        axes[0].plot(
            masses,
            [row["inner_oof_balanced_accuracy"] for row in rows],
            marker="o",
            markersize=3,
            color=color,
            label=f"{observable}: training OOF",
        )
        axes[0].plot(
            masses,
            [row["validation_balanced_accuracy"] for row in rows],
            linestyle="--",
            color=color,
            label=f"{observable}: validation",
        )
    axes[0].axhline(0.5, color="black", linestyle=":", linewidth=1)
    axes[0].set_xlabel(r"Single observable mass [$M_\odot$]")
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Single-mass feature ablation")
    axes[0].legend(frameon=False, fontsize=8)

    null = report["family_label_permutation_null"]
    axes[1].hist(null["null_scores"], bins=np.linspace(0.0, 1.0, 21), color="#B8B8B8")
    axes[1].axvline(
        null["observed_inner_oof_balanced_accuracy"],
        color="#E45756",
        linewidth=2,
        label="observed",
    )
    axes[1].axvline(0.5, color="black", linestyle=":", linewidth=1, label="chance")
    axes[1].set_xlabel("Family-held-out balanced accuracy")
    axes[1].set_ylabel("Permutations")
    axes[1].set_title("Whole-family label permutation null")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_markdown(report: dict, output_path: Path) -> None:
    radius = [
        row for row in report["single_mass_ablation"] if row["observable"] == "radius"
    ]
    tidal = [
        row for row in report["single_mass_ablation"] if row["observable"] == "tidal"
    ]
    best_radius = max(radius, key=lambda row: row["inner_oof_balanced_accuracy"])
    best_tidal = max(tidal, key=lambda row: row["inner_oof_balanced_accuracy"])
    null = report["family_label_permutation_null"]
    markdown = "\n".join(
        [
            "# Development robustness and ablation",
            "",
            "This analysis uses only training and validation families; the locked test",
            "pair remains unopened.",
            "",
            f"- Best single radius: {best_radius['mass_msun']:.2f} M_sun, inner "
            f"accuracy {best_radius['inner_oof_balanced_accuracy']:.3f}, validation "
            f"accuracy {best_radius['validation_balanced_accuracy']:.3f}.",
            f"- Best single tidal feature: {best_tidal['mass_msun']:.2f} M_sun, inner "
            f"accuracy {best_tidal['inner_oof_balanced_accuracy']:.3f}, validation "
            f"accuracy {best_tidal['validation_balanced_accuracy']:.3f}.",
            f"- Whole-family permutation null: observed {null['observed_inner_oof_balanced_accuracy']:.3f}, "
            f"null mean {null['null_mean']:.3f}, maximum {null['null_maximum']:.3f}, "
            f"empirical p={null['empirical_p_value']:.4f} ({null['permutations']} permutations).",
            "",
            "The low-mass radius separation is already sufficient in this selected",
            "surrogate/CFL catalog. Therefore the defensible claim is model-set",
            "discrimination driven mainly by low-mass radius topology, not a universal",
            "or opaque machine-learning discovery.",
        ]
    )
    output_path.write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--permutations",
        type=int,
        default=0,
        help="0 exhausts all family-label assignments; otherwise use this many draws.",
    )
    args = parser.parse_args()
    paths = runtime_paths(args.data_root)
    training, validation = _development_frames(paths.data_root / "family_ml")
    report = {
        "scope": "training and validation only; locked test not opened",
        "selected_development_specification": {
            "architecture": "logistic",
            "feature_set": "MR",
            "C": 0.1,
        },
        "single_mass_ablation": single_mass_ablation(
            training, validation, c_value=0.1
        ),
        "family_label_permutation_null": family_label_permutation_null(
            training,
            radius_feature_columns(),
            c_value=0.1,
            permutations=args.permutations or None,
        ),
        "test_rows_used": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "family_development_robustness.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(report, args.output_dir / "FAMILY_DEVELOPMENT_ROBUSTNESS.md")
    _plot(report, args.output_dir / "FAMILY_DEVELOPMENT_ROBUSTNESS.png")
    print(json.dumps(report["family_label_permutation_null"], indent=2))


if __name__ == "__main__":
    main()
