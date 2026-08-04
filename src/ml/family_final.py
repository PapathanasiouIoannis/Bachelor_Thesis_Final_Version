"""Immutable model-profile validation and final family-test helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ml.family_dataset import curve_feature_columns
from src.ml.family_model_selection import inverse_family_class_weights


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PROFILE = PROJECT_ROOT / "framework" / "family_model_profile.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_model_profile(path: Path | None = None) -> dict:
    profile_path = (path or DEFAULT_MODEL_PROFILE).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    required = {
        "profile_id",
        "lock_state",
        "claim_boundary",
        "data_identity",
        "development_evidence",
        "model",
        "final_test",
        "predeclared_sensitivities",
    }
    missing = sorted(required - set(profile))
    if missing:
        raise ValueError(f"Locked model profile is missing fields: {missing}")
    if profile["lock_state"] != "PRE_TEST_LOCKED":
        raise ValueError("Final evaluation requires the PRE_TEST_LOCKED model state.")
    if profile["model"]["feature_columns"] != curve_feature_columns("MR"):
        raise ValueError("Locked model columns differ from the MR observable allowlist.")
    if profile["model"]["decision_threshold"] != 0.5:
        raise ValueError("Locked decision threshold must remain 0.5.")
    profile["profile_path"] = str(profile_path)
    profile["profile_sha256"] = file_sha256(profile_path)
    return profile


def verify_locked_evidence(profile: dict, family_ml_dir: Path) -> None:
    for stem in ("shortcut_audit", "model_selection", "robustness"):
        path = PROJECT_ROOT / profile["development_evidence"][f"{stem}_path"]
        expected = profile["development_evidence"][f"{stem}_sha256"]
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"Locked development evidence changed: {path}")

    manifest_path = family_ml_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "curve_dataset_sha256",
        "split_manifest_sha256",
        "split_profile_sha256",
    ):
        if manifest.get(key) != profile["data_identity"][key]:
            raise RuntimeError(f"Locked dataset identity mismatch in {key}.")
    selection = json.loads(
        (PROJECT_ROOT / profile["development_evidence"]["model_selection_path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        selection["selected_candidate"]["candidate_id"]
        != profile["development_evidence"]["selected_candidate_id"]
    ):
        raise RuntimeError("Locked candidate differs from the development selection.")


def build_locked_estimator(profile: dict):
    model = profile["model"]
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(model["C"]),
            solver=str(model["solver"]),
            max_iter=int(model["maximum_iterations"]),
            random_state=int(model["random_state"]),
        ),
    )


def fit_locked_estimator(development: pd.DataFrame, profile: dict):
    features = profile["model"]["feature_columns"]
    estimator = build_locked_estimator(profile)
    weights = inverse_family_class_weights(development)
    estimator.fit(
        development[features],
        development["Label"].astype(int),
        logisticregression__sample_weight=weights,
    )
    return estimator


def predict_locked_estimator(estimator, frame: pd.DataFrame, profile: dict) -> pd.DataFrame:
    probabilities = estimator.predict_proba(frame[profile["model"]["feature_columns"]])[:, 1]
    result = frame[
        ["Sample_ID", "EoS_ID", "Group_ID", "Perturb_A", "Label"]
    ].copy()
    result["Probability_Quark"] = probabilities
    result["Prediction"] = (
        probabilities >= float(profile["model"]["decision_threshold"])
    ).astype("int8")
    result["Correct"] = (
        result["Prediction"].astype(int) == result["Label"].astype(int)
    ).astype("int8")
    if not np.isfinite(result["Probability_Quark"]).all():
        raise RuntimeError("Locked estimator produced non-finite probabilities.")
    return result
