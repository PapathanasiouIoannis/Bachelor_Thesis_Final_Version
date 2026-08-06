"""Build noisy tensors using the same latent rows and split manifest as clean ML."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from src.ml.dataset import load_and_preprocess
from src.ml.tensor_pipeline import build_tensor_artifacts
from src.runtime import add_runtime_args, configure_runtime_from_args, runtime_paths


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("PERTURB_DATA_PIPELINE")


_SPLIT_NOISE_SEEDS = {"train": 1042, "val": 2042, "test": 3042}


def inject_observational_noise(
    frame: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """Apply the declared synthetic error law without changing latent provenance."""

    rng = np.random.default_rng(seed)
    noisy = frame.copy()
    noisy["Mass"] += rng.normal(0.0, 0.05 * noisy["Mass"].to_numpy())
    noisy["Radius"] += rng.normal(0.0, 0.10 * noisy["Radius"].to_numpy())
    noisy["log10_Lambda"] += rng.normal(0.0, 0.079, size=len(noisy))
    if (noisy[["Mass", "Radius"]] <= 0.0).any(axis=None):
        raise ValueError("Synthetic noise produced a non-positive mass or radius.")
    return noisy


def _noise_by_split(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    return inject_observational_noise(frame, seed=_SPLIT_NOISE_SEEDS[split])


def run_pipeline(data_root=None) -> None:
    paths = runtime_paths(data_root)
    logger.info("Loading the same latent common-mass curves used by the clean pipeline...")
    frame = load_and_preprocess(
        str(paths.hadronic_ready_dir),
        str(paths.quark_ready_dir),
    )
    build_tensor_artifacts(
        frame,
        output_dir=paths.perturb_tensor_dir,
        data_root=paths.data_root,
        component="perturbed_data_pipeline",
        scaler_filename="scaler_perturb.joblib",
        split_transform=_noise_by_split,
        split_transform_metadata={
            "name": "independent_zero_mean_gaussian",
            "seeds_by_split": _SPLIT_NOISE_SEEDS,
            "Mass_relative_sigma": 0.05,
            "Radius_relative_sigma": 0.10,
            "log10_Lambda_absolute_sigma": 0.079,
        },
    )
    logger.info("Perturbed data pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build perturbed ML tensors.")
    add_runtime_args(parser)
    args = parser.parse_args()
    configure_runtime_from_args(args)
    run_pipeline(args.data_root)
