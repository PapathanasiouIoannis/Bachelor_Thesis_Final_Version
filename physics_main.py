# physics_main.py

"""
  Isolated test-bed orchestrator for the Physics Generation and Visualization
  pipeline. This script is used to strictly validate the Equation of State (EoS)
  generation, Tolman-Oppenheimer-Volkoff (TOV) solver stability, and
  thermodynamic constraints.

Refactored:
  - PARQUET note: Synced with the main pipeline to use Apache Parquet
    via PyArrow for fast file I/O and strict type preservation.
"""

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from tqdm import tqdm

from src.config import CONFIG
from src.utils.exceptions import PhysicsSimulationError, ConfigurationError
from src.utils.logger import get_logger
from src.physics.run_worker_wrapper import run_worker_wrapper
from src.runtime import add_runtime_args, configure_runtime_from_args, write_run_manifest
from src.visualize.plot_core_physics import plot_core_physics
from src.visualize.plot_3d_interactive import plot_interactive_3d


def parse_args():
    parser = argparse.ArgumentParser(description="physics generation and visualization orchestrator")
    add_runtime_args(parser)
    parser.add_argument("--smoke-test", action="store_true", help="generate a tiny physics profile for readiness checks")
    parser.add_argument("--force-regenerate", action="store_true", help="regenerate physics data even if the profile dataset exists")
    parser.add_argument("--fast", action="store_true", help="alias for readiness-sized defaults")
    parser.add_argument("--total-curves", type=int, default=None, help="number of EoS curves to request from the physics workers")
    parser.add_argument("--curves-per-batch", type=int, default=None, help="number of curves each worker task should request")
    parser.add_argument("--n-jobs", type=int, default=None, help="joblib worker count; default is 1 for smoke/fast and all cores otherwise")
    return parser.parse_args()


def save_ml_ready_splits(df, paths, logger):
    paths.hadronic_ready_dir.mkdir(parents=True, exist_ok=True)
    paths.quark_ready_dir.mkdir(parents=True, exist_ok=True)

    for split_dir in (paths.hadronic_ready_dir, paths.quark_ready_dir):
        for stale_file in split_dir.glob("*.parquet"):
            stale_file.unlink()

    df_hadronic = df[df["Label"] == 0]
    if not df_hadronic.empty and "Baseline_Name" in df_hadronic.columns:
        for baseline_name, group in df_hadronic.groupby("Baseline_Name"):
            split_path = paths.hadronic_ready_dir / f"dataset_{baseline_name}.parquet"
            group.to_parquet(split_path, engine="pyarrow", index=False)
        logger.info(f"done Saved split hadronic datasets to {paths.hadronic_ready_dir}.")

    df_quark = df[df["Label"] == 1]
    if not df_quark.empty and "Baseline_Name" in df_quark.columns:
        for baseline_name, group in df_quark.groupby("Baseline_Name"):
            split_path = paths.quark_ready_dir / f"dataset_{baseline_name}.parquet"
            group.to_parquet(split_path, engine="pyarrow", index=False)
        logger.info(f"done Saved split quark datasets to {paths.quark_ready_dir}.")


def summarize_dataset(df: pd.DataFrame) -> dict:
    if df.empty or "Curve_ID" not in df.columns:
        return {
            "rows": int(len(df)),
            "unique_curves": 0,
            "hadronic_curves": 0,
            "quark_curves": 0,
            "hadronic_rows": 0,
            "quark_rows": 0,
        }

    by_label_curves = df.groupby("Label")["Curve_ID"].nunique()
    by_label_rows = df["Label"].value_counts()
    return {
        "rows": int(len(df)),
        "unique_curves": int(df["Curve_ID"].nunique()),
        "hadronic_curves": int(by_label_curves.get(0, 0)),
        "quark_curves": int(by_label_curves.get(1, 0)),
        "hadronic_rows": int(by_label_rows.get(0, 0)),
        "quark_rows": int(by_label_rows.get(1, 0)),
    }


def log_dataset_summary(logger, summary: dict, prefix: str = "Dataset") -> None:
    logger.info(
        f"{prefix}: {summary['unique_curves']} curves "
        f"({summary['hadronic_curves']} hadronic, {summary['quark_curves']} quark), "
        f"{summary['rows']} rows "
        f"({summary['hadronic_rows']} hadronic, {summary['quark_rows']} quark)."
    )


def main():
    args = parse_args()
    paths = configure_runtime_from_args(args)
    logger = get_logger("PHYSICS_TESTBED")

    try:
        logger.info("===============================================================")
        logger.info("      PHYSICS ISOLATION ENVIRONMENT: EOS & TOV PIPELINE        ")
        logger.info("===============================================================")

        # 1. Directory Setup
        paths.data_root.mkdir(parents=True, exist_ok=True)
        paths.plots_root.mkdir(parents=True, exist_ok=True)

        # 2. Physics Initialization
        logger.info("\n[Step 1] Initializing Physics Environment...")

        # 3. Parallel Generation
        logger.info("\n[Step 2] Generating Physics Test Data (Parallel) ...")

        generation_performed = False
        generation_manifest = None

        if paths.physics_dataset.exists() and not args.force_regenerate:
            logger.info(
                f"Reusing existing physics dataset at {paths.physics_dataset}. "
                "Pass --force-regenerate to create a fresh dataset."
            )
            # load the existing dataset so downstream tasks have the required 'df'
            df = pd.read_parquet(paths.physics_dataset)
            log_dataset_summary(logger, summarize_dataset(df), prefix="Reused physics dataset")
        else:
            generation_performed = True
            # keep totals moderate for rapid physics debugging
            default_total_curves = 20 if args.smoke_test or args.fast else 10000
            default_curves_per_batch = 5 if args.smoke_test or args.fast else 100
            TOTAL_CURVES = args.total_curves if args.total_curves is not None else default_total_curves
            CURVES_PER_BATCH = args.curves_per_batch if args.curves_per_batch is not None else default_curves_per_batch
            N_JOBS = args.n_jobs if args.n_jobs is not None else (1 if args.smoke_test or args.fast else -1)

            if TOTAL_CURVES <= 0:
                raise ValueError("--total-curves must be a positive integer.")
            if CURVES_PER_BATCH <= 0:
                raise ValueError("--curves-per-batch must be a positive integer.")

            tasks = []
            remaining_curves = TOTAL_CURVES
            batch_idx = 0
            while remaining_curves > 0:
                curves_this_batch = min(CURVES_PER_BATCH, remaining_curves)
                # interleave Hadronic and Quark tasks for load balancing across cores
                t_type = "hadronic" if batch_idx % 2 == 0 else "quark"
                tasks.append((t_type, curves_this_batch, batch_idx, batch_idx))
                remaining_curves -= curves_this_batch
                batch_idx += 1

            logger.info(f"Spawning {len(tasks)} tasks to generate {TOTAL_CURVES} EoS curves...")

            temp_dir = str(paths.physics_temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            for f in os.listdir(temp_dir):
                if f.endswith(".parquet"):
                    os.remove(os.path.join(temp_dir, f))

            def process_and_save(t, idx):
                mode, requested_curves, _, batch_id = t
                sublist = run_worker_wrapper(t)
                result = {
                    "batch_idx": int(batch_id),
                    "mode": mode,
                    "requested_curves": int(requested_curves),
                    "actual_curves": 0,
                    "rows": 0,
                    "path": None,
                }
                if sublist is not None and not sublist.empty:
                    result["actual_curves"] = int(sublist["Curve_ID"].nunique()) if "Curve_ID" in sublist.columns else 0
                    result["rows"] = int(len(sublist))

                    # ensure LogLambda is calculated
                    if "Lambda" in sublist.columns and "LogLambda" not in sublist.columns:
                        sublist["LogLambda"] = np.log10(sublist["Lambda"].replace(0, np.nan))

                    # validate the strictly required columns footprint
                    strict_fields = {
                        "Mass": pa.float32(),
                        "Radius": pa.float32(),
                        "LogLambda": pa.float32(),
                        "Curve_ID": pa.string(),
                        "Label": pa.int32(),
                        "Baseline_Name": pa.string(),
                    }

                    # convert to table and enforce schema cast (this guarantees dtypes)
                    table = pa.Table.from_pandas(sublist)
                    new_fields = []
                    for field in table.schema:
                        if field.name in strict_fields:
                            new_fields.append(pa.field(field.name, strict_fields[field.name]))
                        else:
                            new_fields.append(field)

                    strict_schema = pa.schema(new_fields)
                    table = table.cast(strict_schema)

                    chunk_path = os.path.join(temp_dir, f"chunk_{idx}.parquet")
                    pq.write_table(table, chunk_path)
                    result["path"] = chunk_path
                return result

            # execute Parallel Workers
            logger.info(
                f"Running {len(tasks)} physics batches with n_jobs={N_JOBS}. "
                "Progress below tracks completed batches, not submitted work."
            )
            results_iter = Parallel(n_jobs=N_JOBS, return_as="generator_unordered")(
                delayed(process_and_save)(t, idx) for idx, t in enumerate(tasks)
            )
            res = []
            for item in tqdm(results_iter, total=len(tasks), desc="Completed physics batches"):
                res.append(item)
                logger.info(
                    f"Completed batch {item['batch_idx']} ({item['mode']}): "
                    f"{item['actual_curves']}/{item['requested_curves']} curves, "
                    f"{item['rows']} rows."
                )

            # flatten DataFrames
            cols = CONFIG["COLUMN_SCHEMA"]
            valid_paths = [item["path"] for item in res if item.get("path")]

            if valid_paths:
                dataset = ds.dataset(temp_dir, format="parquet")
                
                # using PyArrow batch reader to stream safely without aggressively filtering here
                # so the full morphology down to M=0.1 can be plotted.
                filtered_dfs = []
                for batch in dataset.to_batches():
                    batch_df = batch.to_pandas()
                    filtered_dfs.append(batch_df)
                
                if filtered_dfs:
                    df = pd.concat(filtered_dfs, ignore_index=True)
                else:
                    df = pd.DataFrame(columns=cols)
            else:
                df = pd.DataFrame(columns=cols)

            # boundary Invariant Assertions
            initial_len = len(df)
            df = df.dropna()
            
            # filter out non-physical masses and infinite values silently
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            df = df[df["Mass"] >= 0.1]
            
            dropped_count = initial_len - len(df)
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows containing NaNs or violating bounds (Mass < 0.1).")
                
            if not df.empty:
                logger.info(f"Physical Bounds Validated:")
                logger.info(f"  > Mass Range:   [{df['Mass'].min():.3f}, {df['Mass'].max():.3f}] M_sun")
                logger.info(f"  > Radius Range: [{df['Radius'].min():.3f}, {df['Radius'].max():.3f}] km")
                if "Lambda" in df.columns:
                    logger.info(f"  > Lambda Range: [{df['Lambda'].min():.3e}, {df['Lambda'].max():.3e}]")
                elif "LogLambda" in df.columns:
                    logger.info(f"  > LogLambda:    [{df['LogLambda'].min():.3f}, {df['LogLambda'].max():.3f}]")

            # 4. Balancing & Saving
            logger.info("\n[Data Integrity] Checking Class Distribution...")
            counts = df["Label"].value_counts()
            logger.info(f"\n{counts}")
            summary = summarize_dataset(df)
            log_dataset_summary(logger, summary, prefix="Generated physics dataset")
            if summary["unique_curves"] != TOTAL_CURVES:
                raise RuntimeError(
                    f"Requested {TOTAL_CURVES} curves but generated {summary['unique_curves']} usable curves. "
                    "Generation aborted so stale or incomplete data is not promoted."
                )

            # shuffle to thoroughly mix classes
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # note: Save to Parquet
            df.to_parquet(paths.physics_dataset, engine="pyarrow", index=False)
            logger.info(f"done Saved physics dataset ({len(df)} samples) to {paths.physics_dataset}.")
            generation_manifest = {
                "requested_curves": int(TOTAL_CURVES),
                "curves_per_batch": int(CURVES_PER_BATCH),
                "n_jobs": int(N_JOBS),
                **summary,
                "batch_results": sorted(res, key=lambda item: item["batch_idx"]),
            }

        save_ml_ready_splits(df, paths, logger)
        if generation_performed:
            manifest_path = write_run_manifest(
                paths.data_root,
                "physics_generation",
                paths.data_root,
                generation_manifest,
            )
            logger.info(f"done Wrote fresh generation manifest to {manifest_path}.")
        else:
            logger.info("Skipped generation manifest update because no fresh physics generation was performed.")

        # 5. Physics Visualization Suite
        logger.info("\n[Step 3] Running Core Physics Visualization...")
        plot_core_physics(df)
        plot_interactive_3d(df)

        logger.info("\n===============================================================")
        logger.info("             PHYSICS PIPELINE COMPLETED SUCCESSFULLY           ")
    except PhysicsSimulationError as e:
        logger.exception("Physics engine failed.")
        raise
    except ConfigurationError as e:
        logger.exception("Configuration error.")
        raise
    except Exception as e:
        logger.exception("Unexpected error occurred during pipeline execution.")
        raise
    logger.info("===============================================================")


if __name__ == "__main__":
    main()
