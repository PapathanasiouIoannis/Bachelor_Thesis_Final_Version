"""Development-only feature ablations and family-label null tests."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.family_dataset import (
    observable_mass_grid,
    radius_feature_columns,
    tidal_feature_columns,
)
from src.ml.family_model_selection import (
    build_family_pair_folds,
    family_weighted_binary_metrics,
    inverse_family_class_weights,
)


def _fit_logistic(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    features: list[str],
    c_value: float,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=5000,
            random_state=42,
            solver="lbfgs",
        ),
    )
    model.fit(
        fit[features],
        fit["Label"].astype(int),
        logisticregression__sample_weight=inverse_family_class_weights(fit),
    )
    return model.predict_proba(score[features])[:, 1]


def _inner_oof_family_metrics(
    training: pd.DataFrame,
    features: list[str],
    c_value: float,
) -> dict[str, float]:
    folds = build_family_pair_folds(training)
    oof_labels = []
    oof_probabilities = []
    oof_groups = []
    for fold in folds:
        fit = training[training["Group_ID"].isin(fold["fit_groups"])]
        score = training[training["Group_ID"].isin(fold["score_groups"])]
        oof_labels.extend(score["Label"].astype(int))
        oof_probabilities.extend(_fit_logistic(fit, score, features, c_value))
        oof_groups.extend(score["Group_ID"].astype(str))
    return family_weighted_binary_metrics(
        oof_labels,
        oof_probabilities,
        oof_groups,
    )


def _inner_oof_balanced_accuracy(
    training: pd.DataFrame,
    features: list[str],
    c_value: float,
) -> float:
    """Compatibility helper returning equal-family-weighted OOF accuracy."""

    return _inner_oof_family_metrics(training, features, c_value)[
        "family_balanced_accuracy"
    ]


def development_feature_score(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    *,
    c_value: float,
) -> dict:
    validation_probabilities = _fit_logistic(training, validation, features, c_value)
    inner_metrics = _inner_oof_family_metrics(training, features, c_value)
    validation_metrics = family_weighted_binary_metrics(
        validation["Label"].astype(int),
        validation_probabilities,
        validation["Group_ID"].astype(str),
    )
    return {
        "features": features,
        "inner_oof_family_metrics": inner_metrics,
        "validation_family_metrics": validation_metrics,
        "inner_oof_family_balanced_accuracy": inner_metrics["family_balanced_accuracy"],
        "validation_family_balanced_accuracy": validation_metrics[
            "family_balanced_accuracy"
        ],
        # Compatibility aliases; both now contain equal-family-weighted values.
        "inner_oof_balanced_accuracy": inner_metrics["family_balanced_accuracy"],
        "validation_balanced_accuracy": validation_metrics["family_balanced_accuracy"],
    }


def single_mass_ablation(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    c_value: float,
) -> list[dict]:
    records = []
    for family, columns in (
        ("radius", radius_feature_columns()),
        ("tidal", tidal_feature_columns()),
    ):
        for mass, column in zip(observable_mass_grid(), columns, strict=True):
            records.append(
                {
                    "observable": family,
                    "mass_msun": float(mass),
                    **development_feature_score(
                        training,
                        validation,
                        [column],
                        c_value=c_value,
                    ),
                }
            )
    return records


def family_label_permutation_null(
    training: pd.DataFrame,
    features: list[str],
    *,
    c_value: float,
    permutations: int | None = None,
    seed: int = 2026,
) -> dict:
    """Shuffle labels only between whole families, never between A variants."""

    if permutations is not None and permutations < 20:
        raise ValueError("Use at least 20 family-label permutations.")
    group_labels = training.groupby("Group_ID")["Label"].first().astype(int)
    group_ids = list(group_labels.index)
    zero_count = int((group_labels == 0).sum())
    if permutations is None:
        label_maps = (
            {group_id: int(group_id not in zero_groups) for group_id in group_ids}
            for zero_groups in (
                set(values) for values in combinations(group_ids, zero_count)
            )
        )
        sampling = "exhaustive"
    else:
        rng = np.random.default_rng(seed)
        label_maps = (
            dict(
                zip(
                    group_ids,
                    rng.permutation(group_labels.to_numpy()),
                    strict=True,
                )
            )
            for _ in range(permutations)
        )
        sampling = "monte_carlo"
    scores = []
    for label_map in label_maps:
        permuted = training.copy()
        permuted["Label"] = permuted["Group_ID"].map(label_map).astype(int)
        folds = build_family_pair_folds(permuted)
        labels = []
        probabilities = []
        groups = []
        for fold in folds:
            fit = permuted[permuted["Group_ID"].isin(fold["fit_groups"])]
            score = permuted[permuted["Group_ID"].isin(fold["score_groups"])]
            labels.extend(score["Label"].astype(int))
            probabilities.extend(_fit_logistic(fit, score, features, c_value))
            groups.extend(score["Group_ID"].astype(str))
        scores.append(
            family_weighted_binary_metrics(labels, probabilities, groups)[
                "family_balanced_accuracy"
            ]
        )
    observed = _inner_oof_balanced_accuracy(training, features, c_value)
    null = np.asarray(scores, dtype=float)
    p_value = (
        np.sum(null >= observed) / len(null)
        if sampling == "exhaustive"
        else (1 + np.sum(null >= observed)) / (len(null) + 1)
    )
    return {
        "unit": "physical family",
        "sampling": sampling,
        "permutations": int(len(null)),
        "seed": seed,
        "metric": "equal-family-weighted balanced accuracy",
        "observed_inner_oof_family_balanced_accuracy": float(observed),
        "observed_inner_oof_balanced_accuracy": float(observed),
        "null_mean": float(null.mean()),
        "null_std": float(null.std(ddof=1)),
        "null_maximum": float(null.max()),
        "empirical_p_value": float(p_value),
        "null_scores": scores,
    }
