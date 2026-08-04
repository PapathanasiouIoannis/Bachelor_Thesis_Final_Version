import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.runtime import add_runtime_args, configure_runtime_from_args, runtime_paths, write_run_manifest


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PERTURB_DATA_PIPELINE")


def inject_observational_noise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject synthetic Gaussian noise mimicking LIGO/NICER observational uncertainties."""
    rng = np.random.default_rng(seed)
    logger.info("Injecting synthetic observational noise into macroscopic features...")

    noisy_df = df.copy()
    noisy_df["Mass"] = noisy_df["Mass"] + rng.normal(0, 0.05 * noisy_df["Mass"])
    noisy_df["Radius"] = noisy_df["Radius"] + rng.normal(0, 0.10 * noisy_df["Radius"])

    if "Lambda" in noisy_df.columns:
        lambda_noise = rng.normal(0, 0.20 * noisy_df["Lambda"])
        noisy_df["Lambda"] = np.abs(noisy_df["Lambda"] + lambda_noise)
        noisy_df["log10_Lambda"] = np.log10(np.clip(noisy_df["Lambda"], a_min=1e-10, a_max=None))
    elif "LogLambda" in noisy_df.columns:
        noisy_df["log10_Lambda"] = noisy_df["LogLambda"] + rng.normal(0, 0.079, size=len(noisy_df))
    elif "log10_Lambda" in noisy_df.columns:
        noisy_df["log10_Lambda"] = noisy_df["log10_Lambda"] + rng.normal(0, 0.079, size=len(noisy_df))
    else:
        raise KeyError("Missing Lambda/LogLambda/log10_Lambda feature for perturbation.")

    return noisy_df


def _load_parquet_dir(directory: str, label: int, class_name: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(directory, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No {class_name} parquet files found in {directory}")
    df = pd.concat([pd.read_parquet(path, engine="pyarrow") for path in files], ignore_index=True)
    df["Label"] = label
    return df


def load_and_preprocess(hadronic_dir: str, quark_dir: str) -> pd.DataFrame:
    logger.info(f"Loading hadronic data from {hadronic_dir}...")
    df_hadronic = _load_parquet_dir(hadronic_dir, 0, "hadronic")

    logger.info(f"Loading quark data from {quark_dir}...")
    df_quark = _load_parquet_dir(quark_dir, 1, "quark")

    df = pd.concat([df_hadronic, df_quark], ignore_index=True)
    logger.info(f"Combined clean dataset shape: {df.shape}")

    df = df[df["Radius"] > 0].copy()
    if "Lambda" in df.columns:
        df = df[df["Lambda"] > 0]
    elif "LogLambda" in df.columns and "log10_Lambda" not in df.columns:
        df.rename(columns={"LogLambda": "log10_Lambda"}, inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Mass", "Radius", "Curve_ID", "Label"], inplace=True)
    df = inject_observational_noise(df)
    df.dropna(subset=["Mass", "Radius", "log10_Lambda"], inplace=True)

    logger.info(f"Data shape after noise injection and cleaning: {df.shape}")
    return df


def _write_split_sidecar(output_dir: str, x_train, x_val, x_test) -> None:
    logger.info("Saving split audit sidecar (Curve_ID -> split assignment)...")
    sidecar_full = pd.concat(
        [
            x_train[["Curve_ID"]].assign(Split="train"),
            x_val[["Curve_ID"]].assign(Split="val"),
            x_test[["Curve_ID"]].assign(Split="test"),
        ],
        ignore_index=True,
    )
    sidecar = sidecar_full.drop_duplicates(subset=["Curve_ID", "Split"])
    split_counts = sidecar.groupby("Curve_ID")["Split"].nunique()
    leaked_ids = split_counts[split_counts > 1]
    if not leaked_ids.empty:
        raise RuntimeError(f"Curve_ID split leakage detected before perturb tensor save: {len(leaked_ids)} curves span multiple splits.")
    sidecar.to_parquet(os.path.join(output_dir, "split_audit.parquet"), engine="pyarrow", index=False)
    logger.info(f"Saved split_audit.parquet with {len(sidecar)} Curve_ID/split rows.")


def run_pipeline(data_root=None):
    paths = runtime_paths(data_root)
    output_dir = str(paths.perturb_tensor_dir)
    os.makedirs(output_dir, exist_ok=True)

    df = load_and_preprocess(str(paths.hadronic_ready_dir), str(paths.quark_ready_dir))

    physical_features = ["Mass", "Radius", "log10_Lambda"]
    x = df[physical_features + ["Curve_ID"]]
    y = df["Label"]
    groups = df["Curve_ID"]

    logger.info("Performing Grouped Train/Val/Test Split (80/10/10) on Noisy Data...")
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_val_idx, test_idx = next(gss1.split(x, y, groups))

    x_train_val = x.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]

    x_test = x.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=42)
    train_idx, val_idx = next(gss2.split(x_train_val, y_train_val, groups_train_val))

    x_train = x_train_val.iloc[train_idx].copy()
    y_train = y_train_val.iloc[train_idx].copy()
    x_val = x_train_val.iloc[val_idx].copy()
    y_val = y_train_val.iloc[val_idx].copy()

    _write_split_sidecar(output_dir, x_train, x_val, x_test)

    x_train.drop(columns=["Curve_ID"], inplace=True)
    x_val.drop(columns=["Curve_ID"], inplace=True)
    x_test.drop(columns=["Curve_ID"], inplace=True)

    logger.info(f"Train size: {len(x_train)} (Hadronic: {sum(y_train == 0)}, Quark: {sum(y_train == 1)})")
    logger.info(f"Val size:   {len(x_val)} (Hadronic: {sum(y_val == 0)}, Quark: {sum(y_val == 1)})")
    logger.info(f"Test size:  {len(x_test)} (Hadronic: {sum(y_test == 0)}, Quark: {sum(y_test == 1)})")

    logger.info("Standardizing perturbed features based on training distribution...")
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=physical_features, index=x_train.index)
    x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=physical_features, index=x_val.index)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test), columns=physical_features, index=x_test.index)

    scaler_path = os.path.join(output_dir, "scaler_perturb.joblib")
    dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to {scaler_path}")

    pd.concat([x_train_scaled, y_train], axis=1).to_parquet(os.path.join(output_dir, "train.parquet"), engine="pyarrow", index=False)
    pd.concat([x_val_scaled, y_val], axis=1).to_parquet(os.path.join(output_dir, "val.parquet"), engine="pyarrow", index=False)
    pd.concat([x_test_scaled, y_test], axis=1).to_parquet(os.path.join(output_dir, "test.parquet"), engine="pyarrow", index=False)
    write_run_manifest(output_dir, "perturbed_data_pipeline", paths.data_root)

    logger.info("Perturbed Data Pipeline Complete. Noisy ML Tensors generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build perturbed ML tensors.")
    add_runtime_args(parser)
    args = parser.parse_args()
    configure_runtime_from_args(args)
    run_pipeline(args.data_root)
