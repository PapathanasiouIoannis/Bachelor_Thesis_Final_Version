"""Build leakage-resistant clean tensors from controlled physics curves."""

from __future__ import annotations

import argparse
import logging

from src.ml.dataset import load_and_preprocess
from src.ml.tensor_pipeline import build_tensor_artifacts
from src.runtime import add_runtime_args, configure_runtime_from_args, runtime_paths


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("DATA_PIPELINE")


def run_pipeline(data_root=None) -> None:
    paths = runtime_paths(data_root)
    logger.info("Loading and resampling controlled physics curves...")
    frame = load_and_preprocess(
        str(paths.hadronic_ready_dir),
        str(paths.quark_ready_dir),
    )
    build_tensor_artifacts(
        frame,
        output_dir=paths.clean_tensor_dir,
        data_root=paths.data_root,
        component="clean_data_pipeline",
        scaler_filename="scaler.joblib",
    )
    logger.info("Clean data pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build clean ML tensors.")
    add_runtime_args(parser)
    args = parser.parse_args()
    configure_runtime_from_args(args)
    run_pipeline(args.data_root)
