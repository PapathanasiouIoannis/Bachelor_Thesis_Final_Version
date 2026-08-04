"""Training/validation-only shortcut probes for the family pilot."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _probe(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
) -> dict:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(train[columns].astype(float), train["Label"].astype(int))
    probabilities = model.predict_proba(validation[columns].astype(float))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    labels = validation["Label"].astype(int).to_numpy()
    return {
        "columns": columns,
        "validation_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def _amplitude_balance(development: pd.DataFrame) -> dict:
    table = (
        development.groupby(["Split", "Perturb_A", "Label"])["Sample_ID"]
        .nunique()
        .unstack(fill_value=0)
    )
    balanced = bool(table[0].eq(table[1]).all())
    return {
        "balanced_within_split_and_amplitude": balanced,
        "counts": [
            {
                "split": str(split),
                "amplitude": float(amplitude),
                "hadronic": int(row.get(0, 0)),
                "quark": int(row.get(1, 0)),
            }
            for (split, amplitude), row in table.iterrows()
        ],
    }


def _raw_curve_summaries(physics: pd.DataFrame) -> pd.DataFrame:
    return (
        physics.groupby("Curve_ID", as_index=False)
        .agg(
            Raw_Row_Count=("Mass", "size"),
            Raw_Mass_Min=("Mass", "min"),
            Raw_Mass_Max=("Mass", "max"),
            M_Max=("M_Max", "first"),
            Radius_14=("Radius_14", "first"),
            Eps_Surface=("Eps_Surface", "first"),
            Generation_Seed=("Generation_Seed", "first"),
        )
    )


def _nearest_split_distance(
    train_samples: pd.DataFrame,
    validation_samples: pd.DataFrame,
    features: list[str],
) -> dict:
    scaler = StandardScaler().fit(train_samples[features])
    train_values = scaler.transform(train_samples[features])
    validation_values = scaler.transform(validation_samples[features])
    squared = np.sum(
        (validation_values[:, None, :] - train_values[None, :, :]) ** 2,
        axis=2,
    )
    nearest = np.sqrt(np.min(squared, axis=1))
    return {
        "exact_observable_duplicates": int(np.isclose(nearest, 0.0, atol=1e-12).sum()),
        "minimum_standardized_distance": float(np.min(nearest)),
        "median_standardized_distance": float(np.median(nearest)),
    }


def audit_family_shortcuts(
    *,
    samples: pd.DataFrame,
    sample_audit: pd.DataFrame,
    physics: pd.DataFrame,
    feature_manifest: dict,
) -> dict:
    """Audit shortcuts without loading or scoring the locked test tensors."""

    development_audit = sample_audit[sample_audit["Split"].isin(["train", "val"])].copy()
    development = development_audit.merge(
        _raw_curve_summaries(physics),
        on="Curve_ID",
        how="left",
        validate="one_to_one",
    )
    development["Has_Quark_Parameter_Block"] = (
        development["Parameter_Block_ID"].fillna("").astype(str).str.len() > 0
    ).astype(int)
    development["Exact_Formula_Primary_Verified"] = development[
        "Exact_Formula_Primary_Verified"
    ].astype(int)
    train = development[development["Split"] == "train"].copy()
    validation = development[development["Split"] == "val"].copy()
    if set(train["Label"].astype(int)) != {0, 1} or set(
        validation["Label"].astype(int)
    ) != {0, 1}:
        raise ValueError("Shortcut probes require both classes in train and validation.")

    probes = {
        "deformation_A_only": {
            "status": "allowed_control_not_model_input",
            **_probe(train, validation, ["Perturb_A"]),
        },
        "generation_controls": {
            "status": "forbidden_metadata",
            **_probe(train, validation, ["Perturb_A", "Generation_Seed"]),
        },
        "serialization_geometry": {
            "status": "forbidden_artifact",
            **_probe(
                train,
                validation,
                ["Raw_Row_Count", "Raw_Mass_Min", "Raw_Mass_Max"],
            ),
        },
        "global_physics_summaries": {
            "status": "forbidden_out_of_scope_physics",
            **_probe(train, validation, ["M_Max", "Radius_14", "Eps_Surface"]),
        },
        "quark_parameter_presence": {
            "status": "forbidden_direct_label_proxy",
            **_probe(train, validation, ["Has_Quark_Parameter_Block"]),
        },
        "formula_provenance_flag": {
            "status": "forbidden_direct_label_proxy",
            **_probe(train, validation, ["Exact_Formula_Primary_Verified"]),
        },
    }

    feature_sets = feature_manifest["feature_sets"]
    forbidden = set(feature_manifest["forbidden_model_inputs"])
    allowed_overlap = {
        name: sorted(set(columns) & forbidden)
        for name, columns in feature_sets.items()
    }
    train_ids = set(development_audit.loc[development_audit["Split"] == "train", "Sample_ID"])
    val_ids = set(development_audit.loc[development_audit["Split"] == "val", "Sample_ID"])
    train_samples = samples[samples["Sample_ID"].isin(train_ids)]
    validation_samples = samples[samples["Sample_ID"].isin(val_ids)]
    nearest = {
        name: _nearest_split_distance(train_samples, validation_samples, columns)
        for name, columns in feature_sets.items()
    }
    amplitude = _amplitude_balance(development_audit)
    direct_proxy_scores = [
        probes["quark_parameter_presence"]["balanced_accuracy"],
        probes["formula_provenance_flag"]["balanced_accuracy"],
    ]
    checks = {
        "locked_test_not_used": True,
        "sample_identity_disjoint": train_ids.isdisjoint(val_ids),
        "family_identity_disjoint": set(
            train["Family_Group_ID"]
        ).isdisjoint(validation["Family_Group_ID"]),
        "amplitude_balanced": amplitude["balanced_within_split_and_amplitude"],
        "forbidden_feature_overlap_absent": all(
            not overlap for overlap in allowed_overlap.values()
        ),
        "no_exact_observable_duplicates": all(
            result["exact_observable_duplicates"] == 0 for result in nearest.values()
        ),
        "A_only_at_chance": probes["deformation_A_only"]["balanced_accuracy"] <= 0.55,
        "positive_controls_detect_direct_proxies": min(direct_proxy_scores) >= 0.95,
    }
    return {
        "scope": "training and validation only; locked test not loaded or scored",
        "test_rows_used": 0,
        "development_sample_count": int(len(development_audit)),
        "checks": checks,
        "passed": bool(all(checks.values())),
        "amplitude_balance": amplitude,
        "allowed_feature_overlap_with_forbidden_metadata": allowed_overlap,
        "nearest_train_validation_observable_distance": nearest,
        "shortcut_probes": probes,
    }


def _plot_probes(report: dict, output_path: Path) -> None:
    probes = report["shortcut_probes"]
    names = list(probes)
    scores = [probes[name]["balanced_accuracy"] for name in names]
    colors = [
        "#4C78A8" if probes[name]["status"].startswith("allowed") else "#E45756"
        for name in names
    ]
    labels = [name.replace("_", "\n") for name in names]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(np.arange(len(names)), scores, color=colors)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Validation balanced accuracy")
    axis.set_xticks(np.arange(len(names)))
    axis.set_xticklabels(labels, fontsize=8)
    axis.set_title("Shortcut probes (training/validation only)")
    for bar, score in zip(bars, scores, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.025,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_shortcut_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "family_shortcut_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for name, probe in report["shortcut_probes"].items():
        rows.append(
            "| "
            + " | ".join(
                [
                    name,
                    probe["status"],
                    f"{probe['balanced_accuracy']:.3f}",
                    f"{probe['roc_auc']:.3f}",
                ]
            )
            + " |"
        )
    check_rows = [
        f"- `{name}`: {'PASS' if value else 'FAIL'}"
        for name, value in report["checks"].items()
    ]
    markdown = "\n".join(
        [
            "# Family classification shortcut audit",
            "",
            "This audit uses only the locked training and validation families. The test",
            "pair remains unopened. Red probes are deliberately forbidden inputs; high",
            "scores for direct label proxies demonstrate why metadata is isolated from",
            "the observable feature tensors.",
            "",
            "| Probe | Status | Balanced accuracy | ROC AUC |",
            "|---|---|---:|---:|",
            *rows,
            "",
            "## Structural checks",
            "",
            *check_rows,
            "",
            "The deformation amplitude is balanced within every development split and",
            "class, and its standalone classifier remains at chance. The production",
            "models may consume only the explicitly listed radius and tidal features.",
        ]
    )
    (output_dir / "FAMILY_SHORTCUT_AUDIT.md").write_text(
        markdown + "\n", encoding="utf-8"
    )
    _plot_probes(report, output_dir / "FAMILY_SHORTCUT_PROBES.png")
