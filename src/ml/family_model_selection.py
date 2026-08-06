"""Low-capacity, family-held-out model selection for curve samples."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.family_dataset import curve_feature_columns


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    architecture: str
    feature_set: str
    parameters: dict
    simplicity_rank: int
    selection_role: str

    @property
    def eligible_for_reporting_selection(self) -> bool:
        return self.selection_role == "reporting"


REPORTING_MODEL_POLICY = ("dummy", "logistic_regression")
EXPLORATORY_MODEL_POLICY = ("xgboost", "mlp")
REPORTING_ARCHITECTURES = frozenset({"dummy", "logistic"})
EXPLORATORY_ARCHITECTURES = frozenset({"forest", "xgboost", "mlp"})


def candidate_grid() -> list[Candidate]:
    candidates = [
        Candidate("dummy_prior", "dummy", "MR", {}, 0, "reporting"),
    ]
    for feature_set, feature_rank in (("MR", 0), ("MRL", 1)):
        for c_value in (0.001, 0.01, 0.1, 1.0, 10.0):
            token = str(c_value).replace(".", "p")
            candidates.append(
                Candidate(
                    f"logistic_{feature_set.lower()}_c{token}",
                    "logistic",
                    feature_set,
                    {"C": c_value},
                    10 + feature_rank,
                    "reporting",
                )
            )
        for depth in (2, 4, None):
            for leaf in (3, 6):
                depth_token = "none" if depth is None else str(depth)
                candidates.append(
                    Candidate(
                        f"forest_{feature_set.lower()}_d{depth_token}_l{leaf}",
                        "forest",
                        feature_set,
                        {"max_depth": depth, "min_samples_leaf": leaf},
                        20 + feature_rank,
                        "exploratory",
                    )
                )
    return candidates


def validate_model_policy(
    primary_models: Iterable[str], exploratory_models: Iterable[str]
) -> None:
    """Require the audited reporting/exploratory policy.

    Random forest is retained as a development diagnostic in this module, while
    XGBoost and the neural network remain separate exploratory workflows.  None
    of those architectures is eligible to become the reporting model.
    """

    primary = tuple(primary_models)
    exploratory = tuple(exploratory_models)
    if primary != REPORTING_MODEL_POLICY:
        raise ValueError(
            "Reporting-grade model selection is locked to dummy and logistic "
            "regression, in that order."
        )
    if exploratory != EXPLORATORY_MODEL_POLICY:
        raise ValueError(
            "Exploratory models are locked to XGBoost and the multilayer "
            "perceptron; they cannot enter reporting-grade selection."
        )


def build_family_pair_folds(metadata: pd.DataFrame, seed: int = 42) -> list[dict]:
    """Cover every training family once with both labels present per score fold."""

    group_labels = metadata.groupby("Group_ID")["Label"].agg(
        lambda values: set(values.astype(int))
    )
    if any(len(labels) != 1 for labels in group_labels):
        raise ValueError("A training family contains multiple class labels.")
    hadronic = sorted(
        group_labels[group_labels.map(lambda values: values == {0})].index
    )
    quark = sorted(group_labels[group_labels.map(lambda values: values == {1})].index)
    if len(hadronic) < 3 or len(quark) < 3:
        raise ValueError("Family-pair CV requires at least three families per class.")

    rng_h = np.random.default_rng(seed)
    rng_q = np.random.default_rng(seed)
    hadronic = list(rng_h.permutation(hadronic))
    quark = list(rng_q.permutation(quark))
    fold_count = min(len(hadronic), len(quark))
    hadronic_chunks = np.array_split(np.asarray(hadronic, dtype=object), fold_count)
    quark_chunks = np.array_split(np.asarray(quark, dtype=object), fold_count)
    folds = []
    all_groups = set(group_labels.index.astype(str))
    scored_groups: list[str] = []
    for index, (hadronic_chunk, quark_chunk) in enumerate(
        zip(hadronic_chunks, quark_chunks, strict=True)
    ):
        score_groups = {
            *(str(value) for value in hadronic_chunk),
            *(str(value) for value in quark_chunk),
        }
        fit_groups = all_groups - score_groups
        if not fit_groups or not score_groups:
            raise RuntimeError("Family-pair CV produced an empty fold partition.")
        score_labels = set(
            metadata.loc[metadata["Group_ID"].isin(score_groups), "Label"].astype(int)
        )
        fit_labels = set(
            metadata.loc[metadata["Group_ID"].isin(fit_groups), "Label"].astype(int)
        )
        if score_labels != {0, 1} or fit_labels != {0, 1}:
            raise RuntimeError("Family-pair CV fold lost a matter class.")
        folds.append(
            {
                "fold": index,
                "fit_groups": sorted(fit_groups),
                "score_groups": sorted(score_groups),
            }
        )
        scored_groups.extend(score_groups)
    if sorted(scored_groups) != sorted(all_groups):
        raise RuntimeError("Family-pair CV does not score every family exactly once.")
    return folds


def inverse_family_class_weights(metadata: pd.DataFrame) -> np.ndarray:
    """Give every family equal weight and both classes equal total weight."""

    table = metadata[["Group_ID", "Label"]].copy()
    group_sizes = table.groupby("Group_ID").size()
    group_labels = table.groupby("Group_ID")["Label"].first().astype(int)
    groups_per_label = group_labels.value_counts()
    if set(groups_per_label.index) != {0, 1}:
        raise ValueError("Family weights require both classes.")
    raw = np.asarray(
        [
            1.0 / (group_sizes[group_id] * groups_per_label[group_labels[group_id]])
            for group_id in table["Group_ID"]
        ],
        dtype=float,
    )
    return raw * (len(raw) / raw.sum())


def family_weighted_binary_metrics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    group_ids: Iterable[str],
) -> dict[str, float]:
    """Score binary predictions with equal physical-family and class weight."""

    table = pd.DataFrame(
        {
            "Group_ID": list(group_ids),
            "Label": np.asarray(list(labels), dtype=int),
        }
    )
    probability_array = np.clip(
        np.asarray(list(probabilities), dtype=float), 1e-8, 1.0 - 1e-8
    )
    if len(table) == 0 or len(table) != len(probability_array):
        raise ValueError("Labels, probabilities, and family identifiers must align.")
    if not np.isfinite(probability_array).all():
        raise ValueError("Predicted probabilities must be finite.")
    group_labels = table.groupby("Group_ID")["Label"].nunique(dropna=False)
    if (group_labels != 1).any():
        raise ValueError("A physical family contains multiple class labels.")

    labels_array = table["Label"].to_numpy(dtype=int)
    classes = (probability_array >= 0.5).astype(int)
    weights = inverse_family_class_weights(table)
    return {
        "family_weighted_accuracy": float(
            accuracy_score(labels_array, classes, sample_weight=weights)
        ),
        "family_balanced_accuracy": float(
            balanced_accuracy_score(labels_array, classes, sample_weight=weights)
        ),
        "family_weighted_roc_auc": float(
            roc_auc_score(labels_array, probability_array, sample_weight=weights)
        ),
        "family_weighted_brier": float(
            brier_score_loss(labels_array, probability_array, sample_weight=weights)
        ),
        "family_weighted_log_loss": float(
            log_loss(
                labels_array,
                probability_array,
                labels=[0, 1],
                sample_weight=weights,
            )
        ),
    }


def _build_estimator(candidate: Candidate):
    if candidate.architecture == "dummy":
        return DummyClassifier(strategy="prior", random_state=42), "sample_weight"
    if candidate.architecture == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(candidate.parameters["C"]),
                class_weight=None,
                max_iter=5000,
                random_state=42,
                solver="lbfgs",
            ),
        )
        return model, "logisticregression__sample_weight"
    if candidate.architecture == "forest":
        return (
            RandomForestClassifier(
                n_estimators=500,
                max_depth=candidate.parameters["max_depth"],
                min_samples_leaf=int(candidate.parameters["min_samples_leaf"]),
                max_features="sqrt",
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            ),
            "sample_weight",
        )
    raise ValueError(f"Unknown candidate architecture: {candidate.architecture}")


def classification_metrics(predictions: pd.DataFrame) -> dict:
    labels = predictions["Label"].astype(int).to_numpy()
    probabilities = np.clip(
        predictions["Probability_Quark"].to_numpy(dtype=float), 1e-8, 1.0 - 1e-8
    )
    classes = (probabilities >= 0.5).astype(int)
    family_metrics = family_weighted_binary_metrics(
        labels,
        probabilities,
        predictions["Group_ID"].astype(str),
    )
    probability_ranges = predictions.groupby("EoS_ID")["Probability_Quark"].agg(
        lambda values: float(values.max() - values.min())
    )
    return {
        "samples": int(len(predictions)),
        "families": int(predictions["Group_ID"].nunique()),
        "curve_accuracy": float(accuracy_score(labels, classes)),
        "curve_balanced_accuracy": float(balanced_accuracy_score(labels, classes)),
        "curve_roc_auc": float(roc_auc_score(labels, probabilities)),
        "curve_brier": float(brier_score_loss(labels, probabilities)),
        "curve_log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        **family_metrics,
        # Compatibility aliases for the already-frozen one-time result. New
        # development reports use the explicitly named curve/family fields.
        "accuracy": float(accuracy_score(labels, classes)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, classes)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "mean_probability_range_across_A": float(probability_ranges.mean()),
        "maximum_probability_range_across_A": float(probability_ranges.max()),
    }


def _fit_predict(
    fit: pd.DataFrame,
    score: pd.DataFrame,
    candidate: Candidate,
) -> np.ndarray:
    features = curve_feature_columns(candidate.feature_set)
    model, weight_argument = _build_estimator(candidate)
    weights = inverse_family_class_weights(fit)
    model.fit(
        fit[features],
        fit["Label"].astype(int),
        **{weight_argument: weights},
    )
    return model.predict_proba(score[features])[:, 1]


def evaluate_candidate_cv(
    training: pd.DataFrame,
    candidate: Candidate,
    folds: list[dict],
) -> tuple[dict, pd.DataFrame, list[dict]]:
    predictions = []
    fold_metrics = []
    for fold in folds:
        fit_groups = set(fold["fit_groups"])
        score_groups = set(fold["score_groups"])
        fit = training[training["Group_ID"].isin(fit_groups)].copy()
        score = training[training["Group_ID"].isin(score_groups)].copy()
        if set(fit["Group_ID"]) & set(score["Group_ID"]):
            raise RuntimeError("A family leaked across an inner CV fold.")
        probabilities = _fit_predict(fit, score, candidate)
        scored = score[["Sample_ID", "EoS_ID", "Group_ID", "Perturb_A", "Label"]].copy()
        scored["Probability_Quark"] = probabilities
        scored["Fold"] = int(fold["fold"])
        predictions.append(scored)
        fold_metrics.append(
            {"fold": int(fold["fold"]), **classification_metrics(scored)}
        )
    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != len(training) or not oof["Sample_ID"].is_unique:
        raise RuntimeError("Inner CV did not produce one OOF prediction per sample.")
    return classification_metrics(oof), oof, fold_metrics


def _hyperparameter_simplicity_key(record: dict) -> tuple:
    if record["architecture"] == "logistic":
        return (float(record["parameters"]["C"]),)
    if record["architecture"] == "forest":
        depth = record["parameters"]["max_depth"]
        depth_rank = float("inf") if depth is None else int(depth)
        return (depth_rank, -int(record["parameters"]["min_samples_leaf"]))
    return (0,)


def _best_by_architecture_feature(records: Iterable[dict]) -> list[dict]:
    winners = []
    table: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        if record["architecture"] == "dummy":
            continue
        table.setdefault((record["architecture"], record["feature_set"]), []).append(
            record
        )
    for candidates in table.values():
        best_accuracy = max(
            record["cv_metrics"]["family_balanced_accuracy"] for record in candidates
        )
        near_best = [
            record
            for record in candidates
            if record["cv_metrics"]["family_balanced_accuracy"] >= best_accuracy - 0.02
        ]
        winners.append(
            sorted(
                near_best,
                key=lambda record: (
                    _hyperparameter_simplicity_key(record),
                    record["cv_metrics"]["family_weighted_brier"],
                    record["candidate_id"],
                ),
            )[0]
        )
    return sorted(winners, key=lambda record: record["candidate_id"])


def run_development_selection(
    training: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    primary_models: Iterable[str] = REPORTING_MODEL_POLICY,
    exploratory_models: Iterable[str] = EXPLORATORY_MODEL_POLICY,
) -> tuple[dict, pd.DataFrame]:
    """Select only dummy/logistic models; audit larger models separately."""

    validate_model_policy(primary_models, exploratory_models)
    if set(training["Group_ID"]) & set(validation["Group_ID"]):
        raise ValueError("Outer training and validation families overlap.")
    folds = build_family_pair_folds(training)
    cv_records = []
    oof_frames = []
    for candidate in candidate_grid():
        metrics, predictions, per_fold = evaluate_candidate_cv(
            training, candidate, folds
        )
        record = {
            "candidate_id": candidate.candidate_id,
            "architecture": candidate.architecture,
            "feature_set": candidate.feature_set,
            "parameters": candidate.parameters,
            "simplicity_rank": candidate.simplicity_rank,
            "selection_role": candidate.selection_role,
            "eligible_for_reporting_selection": (
                candidate.eligible_for_reporting_selection
            ),
            "cv_metrics": metrics,
            "cv_fold_metrics": per_fold,
            "cv_fold_family_accuracy_mean": float(
                np.mean([fold["family_balanced_accuracy"] for fold in per_fold])
            ),
            "cv_fold_family_accuracy_std": float(
                np.std([fold["family_balanced_accuracy"] for fold in per_fold], ddof=1)
            ),
        }
        cv_records.append(record)
        predictions["Candidate_ID"] = candidate.candidate_id
        oof_frames.append(predictions)

    reporting_records = [
        record
        for record in cv_records
        if record["architecture"] in REPORTING_ARCHITECTURES
    ]
    exploratory_records = [
        record
        for record in cv_records
        if record["architecture"] in EXPLORATORY_ARCHITECTURES
    ]
    dummy = next(
        record for record in reporting_records if record["architecture"] == "dummy"
    )
    finalists = [
        dummy,
        *_best_by_architecture_feature(
            record
            for record in reporting_records
            if record["architecture"] == "logistic"
        ),
    ]
    exploratory_cv_finalists = _best_by_architecture_feature(exploratory_records)
    validation_predictions = []
    for record in finalists:
        candidate = next(
            candidate
            for candidate in candidate_grid()
            if candidate.candidate_id == record["candidate_id"]
        )
        probabilities = _fit_predict(training, validation, candidate)
        scored = validation[
            ["Sample_ID", "EoS_ID", "Group_ID", "Perturb_A", "Label"]
        ].copy()
        scored["Probability_Quark"] = probabilities
        scored["Candidate_ID"] = candidate.candidate_id
        validation_predictions.append(scored)
        record["validation_metrics"] = classification_metrics(scored)

    best_validation = max(
        record["validation_metrics"]["family_balanced_accuracy"] for record in finalists
    )
    tolerance = 0.02
    eligible = [
        record
        for record in finalists
        if record["validation_metrics"]["family_balanced_accuracy"]
        >= best_validation - tolerance - 1e-12
    ]
    best_inner = max(
        record["cv_metrics"]["family_balanced_accuracy"] for record in eligible
    )
    near_inner = [
        record
        for record in eligible
        if record["cv_metrics"]["family_balanced_accuracy"] >= best_inner - 0.02
    ]
    winner = sorted(
        near_inner,
        key=lambda record: (
            record["simplicity_rank"],
            record["validation_metrics"]["family_weighted_brier"],
            record["candidate_id"],
        ),
    )[0]
    if winner["architecture"] not in REPORTING_ARCHITECTURES:
        raise RuntimeError("An exploratory model entered reporting-grade selection.")
    report = {
        "scope": "outer training and validation only; locked test not opened",
        "model_policy": {
            "reporting_grade": list(REPORTING_MODEL_POLICY),
            "selection_eligible_architectures": sorted(REPORTING_ARCHITECTURES),
            "exploratory": [
                "random_forest",
                *EXPLORATORY_MODEL_POLICY,
            ],
            "exploratory_models_can_win_reporting_selection": False,
        },
        "selection_rule": (
            "Tune within training by exhaustive family-pair OOF CV; retain the best "
            "logistic hyperparameters within 0.02 of the best equal-family-weighted "
            "accuracy using the strongest regularization. Compare those logistic "
            "candidates with the dummy baseline on validation families. Among reporting "
            "candidates within 0.02 validation and inner-CV family-balanced accuracy, "
            "choose the simpler model and then lower family-weighted Brier score. Random "
            "forest, XGBoost, and neural-network results are exploratory and cannot win."
        ),
        "training_samples": int(len(training)),
        "validation_samples": int(len(validation)),
        "training_families": int(training["Group_ID"].nunique()),
        "validation_families": int(validation["Group_ID"].nunique()),
        "inner_folds": folds,
        "dummy_baseline": dummy,
        "all_cv_candidates": cv_records,
        "reporting_candidates": finalists,
        "finalists": finalists,
        "exploratory_cv_finalists": exploratory_cv_finalists,
        "selected_candidate": winner,
        "test_rows_used": 0,
    }
    prediction_table = pd.concat(
        [
            pd.concat(oof_frames, ignore_index=True).assign(Stage="inner_oof"),
            pd.concat(validation_predictions, ignore_index=True).assign(
                Stage="outer_validation"
            ),
        ],
        ignore_index=True,
    )
    return report, prediction_table
