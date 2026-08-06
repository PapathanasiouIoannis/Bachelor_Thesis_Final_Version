"""Generate post-test sensitivity, interpretation plots, and final report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.ml.family_dataset import observable_mass_grid, radius_feature_columns
from src.ml.family_final import load_locked_model_profile
from src.ml.family_posttest import (
    load_completed_final_result,
    strict_2p08_development_sensitivity,
)
from src.runtime import runtime_paths


def _plot_final_test(predictions: pd.DataFrame, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    for eos_id, subset in predictions.groupby("EoS_ID", sort=True):
        subset = subset.sort_values("Perturb_A")
        axis.plot(
            subset["Perturb_A"],
            subset["Probability_Quark"],
            marker="o",
            linewidth=2,
            label=eos_id,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="threshold")
    axis.set_ylim(-0.03, 1.03)
    axis.set_xlabel("Gaussian amplitude A")
    axis.set_ylabel("Locked-model probability of CFL class")
    axis.set_title("One-time held-out family test")
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(
    samples: pd.DataFrame,
    predictions: pd.DataFrame,
    estimator,
    output_path: Path,
) -> None:
    masses = observable_mass_grid()
    radius_columns = radius_feature_columns()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {0: "#4C78A8", 1: "#E45756"}
    names = {0: "hadronic surrogate", 1: "analytic CFL"}
    for label in (0, 1):
        values = samples.loc[samples["Label"] == label, radius_columns].to_numpy(
            dtype=float
        )
        median = np.median(values, axis=0)
        lower, upper = np.quantile(values, [0.1, 0.9], axis=0)
        axes[0].plot(masses, median, color=colors[label], label=names[label])
        axes[0].fill_between(masses, lower, upper, color=colors[label], alpha=0.2)
    axes[0].set_xlabel(r"Mass [$M_\odot$]")
    axes[0].set_ylabel("Radius [km]")
    axes[0].set_title("Catalog geometry (10–90% bands)")
    axes[0].legend(frameon=False, fontsize=8)

    coefficients = estimator.named_steps["logisticregression"].coef_[0]
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].plot(masses, coefficients, marker="o", markersize=3, color="#54A24B")
    axes[1].set_xlabel(r"Mass [$M_\odot$]")
    axes[1].set_ylabel("Standardized logistic coefficient")
    axes[1].set_title("Frozen radius-only decision weights")

    for eos_id, subset in predictions.groupby("EoS_ID", sort=True):
        subset = subset.sort_values("Perturb_A")
        axes[2].plot(
            subset["Perturb_A"],
            subset["Probability_Quark"],
            marker="o",
            label=eos_id,
        )
    axes[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_xlabel("Gaussian amplitude A")
    axes[2].set_ylabel("P(CFL)")
    axes[2].set_title("Locked test stability across A")
    axes[2].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_strict_markdown(report: dict, output_path: Path) -> None:
    metrics = report["metrics"]
    markdown = "\n".join(
        [
            "# Strict 2.08-M_sun development sensitivity",
            "",
            "This is an out-of-family development sensitivity, not another independent",
            "test. Both locked test EoSs fail the 2.08-M_sun screen, so their labels and",
            "metrics are not reused here.",
            "",
            f"The complete strict subset contains {report['eos_count']} EoSs, "
            f"{report['family_groups']} physical families, and {report['curves']} curves.",
            f"Family-held-out balanced accuracy is {metrics['family_balanced_accuracy']:.3f}, "
            f"ROC AUC is {metrics['roc_auc']:.3f}, and Brier score is {metrics['brier']:.4f}.",
        ]
    )
    output_path.write_text(markdown + "\n", encoding="utf-8")


def _write_final_report(
    final: dict,
    strict: dict,
    output_path: Path,
) -> None:
    test = final["test_metrics"]
    strict_metrics = strict["metrics"]
    markdown = "\n".join(
        [
            "# Family-pilot classification report",
            "",
            "## Outcome",
            "",
            "The frozen model is L2-regularized logistic regression on 21 radii sampled",
            "from 1.0 to 2.0 M_sun. Mass is implicit and no generation, provenance,",
            "central-density, surface-density, maximum-mass, or quark-parameter metadata",
            "is exposed to the classifier.",
            "",
            "| Evaluation | Independent families | Curve variants | Balanced accuracy | ROC AUC | Brier |",
            "|---|---:|---:|---:|---:|---:|",
            "| Training-family OOF | 13 | 84 | 1.000 | 1.000 | 0.0153 |",
            "| Validation families | 2 | 12 | 1.000 | 1.000 | 0.0013 |",
            f"| One-time locked test | {final['independent_test_family_units']} | {test['samples']} | "
            f"{test['family_balanced_accuracy']:.3f} | {test['roc_auc']:.3f} | {test['brier']:.4f} |",
            f"| Strict-2.08 development OOF | {strict['family_groups']} | {strict['curves']} | "
            f"{strict_metrics['family_balanced_accuracy']:.3f} | {strict_metrics['roc_auc']:.3f} | "
            f"{strict_metrics['brier']:.4f} |",
            "",
            "## Integrity findings",
            "",
            "- A alone scores 0.50, and every A value is class-balanced in development.",
            "- Raw sequence geometry, global physics summaries, provenance flags, and",
            "  quark-parameter presence each score 1.00 and are therefore hard-forbidden.",
            "- All A variants and related EoSs remain within one physical-family split.",
            "- The fixed-specification exhaustive family-label null gives p=1/1716.",
            "- The final test was opened once at the committed pre-test lock recorded in",
            "  `docs/family_final_test.json`.",
            "",
            "## Interpretation and limitations",
            "",
            "A single low-mass radius already separates the development catalog; tidal",
            "features alone are weaker. The result therefore reflects low-mass radius",
            "topology of these repository surrogates versus this analytic CFL model, not",
            "an opaque or universal matter-phase classifier. The final test has only two",
            "independent families; its six A variants per family are correlated. All CFL",
            "families share one analytic MIT-bag superfamily, exact hadronic fit",
            "coefficients remain verified through a secondary thesis source, and this",
            "theoretical full-curve input is not a direct observational deployment setup.",
            "The independent test is a 2.0-M_sun-screen result; strict 2.08-M_sun evidence",
            "is development OOF only.",
        ]
    )
    output_path.write_text(markdown + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/family_pilot_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    args = parser.parse_args()
    paths = runtime_paths(args.data_root)
    ml_dir = paths.data_root / "family_ml"
    final_path = args.output_dir / "family_final_test.json"
    final = load_completed_final_result(
        final_path, ml_dir / "LOCKED_TEST_OPENED.json"
    )
    profile = load_locked_model_profile()
    samples = pd.read_parquet(ml_dir / "curve_samples.parquet")
    audit = pd.read_parquet(ml_dir / "sample_audit.parquet")
    physics = pd.read_parquet(paths.physics_dataset)
    strict, strict_predictions = strict_2p08_development_sensitivity(
        samples=samples,
        sample_audit=audit,
        physics=physics,
        c_value=float(profile["model"]["C"]),
    )
    predictions = pd.read_parquet(ml_dir / "final_test_predictions.parquet")
    estimator = joblib.load(ml_dir / "locked_logistic_mr.joblib")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "family_strict_2p08_sensitivity.json").write_text(
        json.dumps(strict, indent=2) + "\n", encoding="utf-8"
    )
    strict_predictions.to_parquet(
        ml_dir / "strict_2p08_development_oof_predictions.parquet", index=False
    )
    _write_strict_markdown(
        strict, args.output_dir / "FAMILY_STRICT_2P08_SENSITIVITY.md"
    )
    _plot_final_test(predictions, args.output_dir / "FAMILY_FINAL_TEST.png")
    _plot_summary(
        samples,
        predictions,
        estimator,
        args.output_dir / "FAMILY_CLASSIFICATION_SUMMARY.png",
    )
    _write_final_report(
        final,
        strict,
        args.output_dir / "FAMILY_CLASSIFICATION_FINAL_REPORT.md",
    )
    print(json.dumps(strict["metrics"], indent=2))


if __name__ == "__main__":
    main()
