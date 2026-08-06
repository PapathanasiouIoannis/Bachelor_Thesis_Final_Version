"""Training/validation-only shortcut probes for the family pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Collection

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.family_model_selection import (
    family_weighted_binary_metrics,
    inverse_family_class_weights,
)


DEVELOPMENT_SPLITS = ("train", "val")


@dataclass(frozen=True)
class DevelopmentShortcutInputs:
    """Development-only tables plus the locked-family denylist."""

    samples: pd.DataFrame
    sample_audit: pd.DataFrame
    physics: pd.DataFrame
    locked_test_family_ids: frozenset[str]


def _read_development_parquet(
    path: Path,
    *,
    filter_column: str,
    permitted_values: Collection[str],
) -> pd.DataFrame:
    values = sorted({str(value) for value in permitted_values})
    if not values:
        raise ValueError(f"No permitted development IDs were resolved for {path}.")
    return pd.read_parquet(
        path,
        engine="pyarrow",
        filters=[(filter_column, "in", values)],
    )


def load_development_shortcut_inputs(
    *,
    samples_path: Path,
    sample_audit_path: Path,
    physics_path: Path,
    split_manifest_path: Path,
) -> DevelopmentShortcutInputs:
    """Load only rows assigned to the locked training and validation families.

    The small split manifest contains family identities, not observables.  It is
    read first so the locked test families form a denylist.  The sample audit is
    then filtered at the Parquet scan, and its development sample/curve IDs are
    used to filter both larger data files at their Parquet scans.
    """

    split_manifest = pd.read_parquet(
        split_manifest_path,
        engine="pyarrow",
        columns=["Group_ID", "Split"],
    )
    missing_splits = sorted(
        set((*DEVELOPMENT_SPLITS, "test")) - set(split_manifest["Split"])
    )
    if missing_splits:
        raise ValueError(
            f"Family split manifest is missing required partitions: {missing_splits}."
        )
    if split_manifest["Group_ID"].isna().any():
        raise ValueError("Family split manifest contains a missing Group_ID.")
    if split_manifest.groupby("Group_ID")["Split"].nunique().max() != 1:
        raise ValueError("A physical family appears in multiple locked splits.")

    development_family_ids = frozenset(
        split_manifest.loc[
            split_manifest["Split"].isin(DEVELOPMENT_SPLITS), "Group_ID"
        ].astype(str)
    )
    locked_test_family_ids = frozenset(
        split_manifest.loc[split_manifest["Split"] == "test", "Group_ID"].astype(str)
    )
    if not development_family_ids.isdisjoint(locked_test_family_ids):
        raise ValueError("Development and locked-test family identities overlap.")

    sample_audit = pd.read_parquet(
        sample_audit_path,
        engine="pyarrow",
        filters=[
            ("Split", "in", list(DEVELOPMENT_SPLITS)),
            ("Group_ID", "in", sorted(development_family_ids)),
        ],
    )
    observed_development_families = set(sample_audit["Group_ID"].astype(str))
    if observed_development_families != set(development_family_ids):
        raise ValueError(
            "Development sample audit does not match the locked split manifest: "
            f"missing={sorted(development_family_ids - observed_development_families)}, "
            f"unexpected={sorted(observed_development_families - development_family_ids)}."
        )

    development_sample_ids = frozenset(sample_audit["Sample_ID"].astype(str))
    development_curve_ids = frozenset(sample_audit["Curve_ID"].astype(str))
    samples = _read_development_parquet(
        samples_path,
        filter_column="Sample_ID",
        permitted_values=development_sample_ids,
    )
    physics = _read_development_parquet(
        physics_path,
        filter_column="Curve_ID",
        permitted_values=development_curve_ids,
    )
    _development_scope(
        samples=samples,
        sample_audit=sample_audit,
        physics=physics,
        locked_test_family_ids=locked_test_family_ids,
    )
    return DevelopmentShortcutInputs(
        samples=samples,
        sample_audit=sample_audit,
        physics=physics,
        locked_test_family_ids=locked_test_family_ids,
    )


def _required_string_ids(
    frame: pd.DataFrame,
    column: str,
    *,
    table_name: str,
) -> set[str]:
    if column not in frame:
        raise KeyError(f"{table_name} is missing required identity column {column!r}.")
    if frame[column].isna().any():
        raise ValueError(f"{table_name}.{column} contains missing identities.")
    return set(frame[column].astype(str))


def _development_scope(
    *,
    samples: pd.DataFrame,
    sample_audit: pd.DataFrame,
    physics: pd.DataFrame,
    locked_test_family_ids: Collection[str],
) -> dict[str, bool | int]:
    """Validate loaded identities before any shortcut statistic is computed."""

    locked_families = {str(value) for value in locked_test_family_ids}
    if not locked_families:
        raise ValueError("The locked-test family denylist is empty.")
    if "Split" not in sample_audit:
        raise KeyError("sample_audit is missing required identity column 'Split'.")

    audit_sample_ids = _required_string_ids(
        sample_audit, "Sample_ID", table_name="sample_audit"
    )
    audit_curve_ids = _required_string_ids(
        sample_audit, "Curve_ID", table_name="sample_audit"
    )
    sample_ids = _required_string_ids(samples, "Sample_ID", table_name="samples")
    sample_curve_ids = _required_string_ids(samples, "Curve_ID", table_name="samples")
    physics_curve_ids = _required_string_ids(physics, "Curve_ID", table_name="physics")
    audit_families = _required_string_ids(
        sample_audit, "Family_Group_ID", table_name="sample_audit"
    )
    audit_group_ids = _required_string_ids(
        sample_audit, "Group_ID", table_name="sample_audit"
    )
    sample_families = _required_string_ids(
        samples, "Family_Group_ID", table_name="samples"
    )
    physics_families = _required_string_ids(
        physics, "Family_Group_ID", table_name="physics"
    )
    observed_families = (
        audit_families | audit_group_ids | sample_families | physics_families
    )
    audit_family_ids_aligned = bool(
        sample_audit["Group_ID"]
        .astype(str)
        .eq(sample_audit["Family_Group_ID"].astype(str))
        .all()
    )
    development_splits_only = set(sample_audit["Split"].astype(str)) <= set(
        DEVELOPMENT_SPLITS
    )
    family_ids_disjoint = observed_families.isdisjoint(locked_families)
    samples_match = sample_ids == audit_sample_ids
    sample_curves_match = sample_curve_ids == audit_curve_ids
    physics_curves_match = physics_curve_ids == audit_curve_ids
    locked_test_not_used = bool(
        development_splits_only
        and family_ids_disjoint
        and audit_family_ids_aligned
        and samples_match
        and sample_curves_match
        and physics_curves_match
    )

    contaminated_rows = int(
        (~sample_audit["Split"].astype(str).isin(DEVELOPMENT_SPLITS)).sum()
        + sample_audit["Family_Group_ID"].astype(str).isin(locked_families).sum()
        + samples["Family_Group_ID"].astype(str).isin(locked_families).sum()
        + physics["Family_Group_ID"].astype(str).isin(locked_families).sum()
    )
    scope = {
        "development_splits_only": bool(development_splits_only),
        "locked_family_ids_disjoint": bool(family_ids_disjoint),
        "audit_family_ids_aligned": audit_family_ids_aligned,
        "development_sample_ids_exact": bool(samples_match),
        "development_sample_curve_ids_exact": bool(sample_curves_match),
        "development_physics_curve_ids_exact": bool(physics_curves_match),
        "locked_test_not_used": locked_test_not_used,
        "test_rows_used": contaminated_rows,
    }
    if not locked_test_not_used:
        failed = sorted(name for name, passed in scope.items() if passed is False)
        raise ValueError(
            "Shortcut audit input escaped the locked development ID scope; "
            f"failed checks={failed}, detected locked-test rows={contaminated_rows}."
        )
    return scope


def _probe(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    columns: list[str],
) -> dict:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight=None,
            max_iter=2000,
            random_state=42,
        ),
    )
    model.fit(
        train[columns].astype(float),
        train["Label"].astype(int),
        logisticregression__sample_weight=inverse_family_class_weights(train),
    )
    probabilities = model.predict_proba(validation[columns].astype(float))[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    labels = validation["Label"].astype(int).to_numpy()
    family_metrics = family_weighted_binary_metrics(
        labels,
        probabilities,
        validation["Group_ID"].astype(str),
    )
    return {
        "columns": columns,
        "validation_samples": int(len(labels)),
        "validation_families": int(validation["Group_ID"].nunique()),
        "curve_accuracy": float(accuracy_score(labels, predictions)),
        "curve_balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "curve_roc_auc": float(roc_auc_score(labels, probabilities)),
        **family_metrics,
        # Compatibility aliases now point to the reporting-grade family metrics.
        "accuracy": family_metrics["family_weighted_accuracy"],
        "balanced_accuracy": family_metrics["family_balanced_accuracy"],
        "roc_auc": family_metrics["family_weighted_roc_auc"],
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
    return physics.groupby("Curve_ID", as_index=False).agg(
        Raw_Row_Count=("Mass", "size"),
        Raw_Mass_Min=("Mass", "min"),
        Raw_Mass_Max=("Mass", "max"),
        M_Max=("M_Max", "first"),
        Radius_14=("Radius_14", "first"),
        Eps_Surface=("Eps_Surface", "first"),
        Generation_Seed=("Generation_Seed", "first"),
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
    locked_test_family_ids: Collection[str],
) -> dict:
    """Audit shortcuts without loading or scoring the locked test tensors."""

    input_scope = _development_scope(
        samples=samples,
        sample_audit=sample_audit,
        physics=physics,
        locked_test_family_ids=locked_test_family_ids,
    )
    development_audit = sample_audit.copy()
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
        raise ValueError(
            "Shortcut probes require both classes in train and validation."
        )

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
        name: sorted(set(columns) & forbidden) for name, columns in feature_sets.items()
    }
    train_ids = set(
        development_audit.loc[development_audit["Split"] == "train", "Sample_ID"]
    )
    val_ids = set(
        development_audit.loc[development_audit["Split"] == "val", "Sample_ID"]
    )
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
        "locked_test_not_used": bool(input_scope["locked_test_not_used"]),
        "sample_identity_disjoint": train_ids.isdisjoint(val_ids),
        "family_identity_disjoint": set(train["Family_Group_ID"]).isdisjoint(
            validation["Family_Group_ID"]
        ),
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
        "test_rows_used": int(input_scope["test_rows_used"]),
        "development_sample_count": int(len(development_audit)),
        "development_input_scope": input_scope,
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
    scores = [probes[name]["family_balanced_accuracy"] for name in names]
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
                    f"{probe['family_balanced_accuracy']:.3f}",
                    f"{probe['family_weighted_roc_auc']:.3f}",
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
            "pair is not opened or scored by this audit. Red probes are deliberately",
            "forbidden inputs; high",
            "scores for direct label proxies demonstrate why metadata is isolated from",
            "the observable feature tensors.",
            "",
            "Each metric gives equal total weight to every physical EoS family within",
            "each matter class.",
            "",
            "| Probe | Status | Family-balanced accuracy | Family-weighted ROC AUC |",
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
