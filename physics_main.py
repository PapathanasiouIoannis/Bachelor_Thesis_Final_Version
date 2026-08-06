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

from framework.eos_sweep import (
    admissible_amplitude_interval,
    amplitude_grid,
    cfl_baseline_grids,
    hadronic_baseline_grids,
    validate_sweep_within_interval,
)
from src.config import CONFIG
from src.utils.exceptions import PhysicsSimulationError, ConfigurationError
from src.utils.logger import get_logger
from src.physics.run_worker_wrapper import run_worker_wrapper
from src.physics.worker_quark_gen import controlled_quark_parameters
from src.runtime import add_runtime_args, configure_runtime_from_args, write_run_manifest
from src.visualize.plot_core_physics import plot_core_physics
from src.visualize.plot_3d_interactive import plot_interactive_3d


def parse_args():
    parser = argparse.ArgumentParser(description="physics generation and visualization orchestrator")
    add_runtime_args(parser)
    parser.add_argument("--smoke-test", action="store_true", help="generate a tiny physics profile for readiness checks")
    parser.add_argument("--force-regenerate", action="store_true", help="regenerate physics data even if the profile dataset exists")
    parser.add_argument("--fast", action="store_true", help="alias for readiness-sized defaults")
    parser.add_argument(
        "--a-min",
        type=float,
        default=CONFIG["CONTROLLED_A_MIN"],
        help="minimum shared Gaussian amplitude A",
    )
    parser.add_argument(
        "--a-max",
        type=float,
        default=CONFIG["CONTROLLED_A_MAX"],
        help="maximum shared Gaussian amplitude A",
    )
    parser.add_argument(
        "--a-points",
        type=int,
        default=None,
        help="number of shared A values (each creates one APR-1 and one CFL4 EoS)",
    )
    parser.add_argument(
        "--total-curves",
        type=int,
        default=None,
        help="deprecated compatibility alias; must be even and maps to 2 * --a-points",
    )
    parser.add_argument(
        "--curves-per-batch",
        type=int,
        default=None,
        help="number of A values per class in each worker task",
    )
    parser.add_argument("--n-jobs", type=int, default=None, help="joblib worker count; default is 1 for smoke/fast and all cores otherwise")
    return parser.parse_args()


def _controlled_sweep(args):
    if args.a_points is not None and args.total_curves is not None:
        raise ValueError("Use --a-points or the deprecated --total-curves alias, not both.")
    if args.total_curves is not None:
        if args.total_curves <= 0 or args.total_curves % 2:
            raise ValueError("--total-curves must be a positive even integer.")
        a_points = args.total_curves // 2
    elif args.a_points is not None:
        a_points = args.a_points
    else:
        a_points = 5 if args.smoke_test or args.fast else CONFIG["CONTROLLED_A_POINTS"]

    if a_points < 5:
        raise ValueError(
            "The paired train/validation/test design requires at least five A values."
        )

    sweep_points = amplitude_grid(args.a_min, args.a_max, a_points)
    eps0 = CONFIG["CONTROLLED_PERTURB_EPS0"]
    sigma = CONFIG["CONTROLLED_PERTURB_SIGMA"]

    hadronic_eps, hadronic_cs2, _, _, _ = hadronic_baseline_grids(
        CONFIG["CONTROLLED_HADRONIC_BASELINE"]
    )
    hadronic_interval = admissible_amplitude_interval(
        hadronic_eps, hadronic_cs2, eps0, sigma
    )
    quark_parameters = controlled_quark_parameters()
    _, quark_eps, quark_cs2, _ = cfl_baseline_grids(quark_parameters)
    quark_interval = admissible_amplitude_interval(quark_eps, quark_cs2, eps0, sigma)
    validate_sweep_within_interval(sweep_points, hadronic_interval, "APR-1")
    validate_sweep_within_interval(sweep_points, quark_interval, "CFL4")
    return sweep_points, hadronic_interval, quark_interval


def _validate_controlled_dataset(df: pd.DataFrame, sweep_points) -> None:
    required = set(CONFIG["COLUMN_SCHEMA"])
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(
            "Existing dataset is not a controlled-sweep artifact; missing columns: "
            f"{missing}. Regenerate it with --force-regenerate."
        )
    expected_sweeps = {point.sweep_id: point.amplitude for point in sweep_points}
    actual_sweeps = set(df["Sweep_ID"].astype(str).unique())
    if actual_sweeps != set(expected_sweeps):
        raise RuntimeError(
            "Existing dataset uses a different A grid. Regenerate with --force-regenerate."
        )
    per_sweep_labels = df.groupby("Sweep_ID")["Label"].agg(lambda values: set(values))
    if not all(labels == {0, 1} for labels in per_sweep_labels):
        raise RuntimeError("Every Sweep_ID must contain exactly the hadronic and quark labels.")
    curves_per_pair = df.groupby(["Sweep_ID", "Label"])["Curve_ID"].nunique()
    if not curves_per_pair.eq(1).all():
        raise RuntimeError(
            "Every (Sweep_ID, Label) pair must contain exactly one Curve_ID."
        )
    curve_membership = df.groupby("Curve_ID").agg(
        sweep_count=("Sweep_ID", "nunique"),
        label_count=("Label", "nunique"),
    )
    if not (
        curve_membership["sweep_count"].eq(1)
        & curve_membership["label_count"].eq(1)
    ).all():
        raise RuntimeError("A Curve_ID is reused across sweep points or class labels.")
    amplitudes = df.groupby("Sweep_ID")["Perturb_A"].agg(["min", "max"])
    if not np.allclose(amplitudes["min"], amplitudes["max"], atol=1e-7):
        raise RuntimeError("Hadronic and quark members of a Sweep_ID do not share A.")
    for sweep_id, amplitude in expected_sweeps.items():
        actual = float(amplitudes.loc[sweep_id, "min"])
        if not np.isclose(actual, amplitude, atol=1e-7):
            raise RuntimeError(
                f"Sweep {sweep_id} stores A={actual}, expected A={amplitude}."
            )
    fixed_deformation = {
        "Perturb_eps0": CONFIG["CONTROLLED_PERTURB_EPS0"],
        "Perturb_sigma": CONFIG["CONTROLLED_PERTURB_SIGMA"],
    }
    for column, expected in fixed_deformation.items():
        if not np.allclose(df[column], expected, atol=1e-6):
            raise RuntimeError(
                f"Controlled deformation column {column} is not fixed at {expected}."
            )
    if set(df.loc[df["Label"] == 0, "Baseline_Name"]) != {
        CONFIG["CONTROLLED_HADRONIC_BASELINE"]
    }:
        raise RuntimeError("Hadronic data contains a baseline other than controlled APR-1.")
    quark_parameters = controlled_quark_parameters()
    quark = df[df["Label"] == 1]
    fixed_columns = {
        "Bag_B": quark_parameters.bag_b,
        "Gap_Delta": quark_parameters.gap_delta,
        "Mass_Strange": quark_parameters.strange_mass,
    }
    for column, expected in fixed_columns.items():
        if not np.allclose(quark[column], expected, atol=1e-6):
            raise RuntimeError(f"Quark column {column} is not fixed at {expected}.")


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
        sweep_points, hadronic_interval, quark_interval = _controlled_sweep(args)
        total_curves = 2 * len(sweep_points)
        logger.info(
            "Controlled sweep: APR-1 versus fixed CFL4 at "
            f"{len(sweep_points)} shared A values "
            f"[{sweep_points[0].amplitude:.6g}, {sweep_points[-1].amplitude:.6g}], "
            f"epsilon0={CONFIG['CONTROLLED_PERTURB_EPS0']} MeV/fm^3, "
            f"sigma={CONFIG['CONTROLLED_PERTURB_SIGMA']} MeV/fm^3."
        )

        if paths.physics_dataset.exists() and not args.force_regenerate:
            logger.info(
                f"Reusing existing physics dataset at {paths.physics_dataset}. "
                "Pass --force-regenerate to create a fresh dataset."
            )
            # load the existing dataset so downstream tasks have the required 'df'
            df = pd.read_parquet(paths.physics_dataset)
            _validate_controlled_dataset(df, sweep_points)
            log_dataset_summary(logger, summarize_dataset(df), prefix="Reused physics dataset")
        else:
            generation_performed = True
            default_curves_per_batch = 2 if args.smoke_test or args.fast else 5
            curves_per_batch = (
                args.curves_per_batch
                if args.curves_per_batch is not None
                else default_curves_per_batch
            )
            n_jobs = (
                args.n_jobs
                if args.n_jobs is not None
                else (1 if args.smoke_test or args.fast else -1)
            )

            if curves_per_batch <= 0:
                raise ValueError("--curves-per-batch must be a positive integer.")

            tasks = []
            batch_idx = 0
            for start in range(0, len(sweep_points), curves_per_batch):
                batch_points = sweep_points[start : start + curves_per_batch]
                for mode in ("hadronic", "quark"):
                    tasks.append((mode, batch_points, batch_idx))
                    batch_idx += 1

            logger.info(
                f"Spawning {len(tasks)} tasks to generate {total_curves} paired EoS curves..."
            )

            temp_dir = str(paths.physics_temp_dir)
            os.makedirs(temp_dir, exist_ok=True)
            for f in os.listdir(temp_dir):
                if f.endswith(".parquet"):
                    os.remove(os.path.join(temp_dir, f))

            def process_and_save(t, idx):
                mode, requested_points, batch_id = t
                requested_curves = len(requested_points)
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
                        "Sweep_ID": pa.string(),
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
                f"Running {len(tasks)} physics batches with n_jobs={n_jobs}. "
                "Progress below tracks completed batches, not submitted work."
            )
            results_iter = Parallel(n_jobs=n_jobs, return_as="generator_unordered")(
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
                logger.info("Physical Bounds Validated:")
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
            if summary["unique_curves"] != total_curves:
                raise RuntimeError(
                    f"Requested {total_curves} curves but generated {summary['unique_curves']} usable curves. "
                    "Generation aborted so stale or incomplete data is not promoted."
                )
            _validate_controlled_dataset(df, sweep_points)

            # shuffle to thoroughly mix classes
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # note: Save to Parquet
            df.to_parquet(paths.physics_dataset, engine="pyarrow", index=False)
            logger.info(f"done Saved physics dataset ({len(df)} samples) to {paths.physics_dataset}.")
            stale_split_manifest = paths.data_root / "split_manifest.parquet"
            if stale_split_manifest.exists():
                stale_split_manifest.unlink()
                logger.info(
                    "Invalidated the prior split manifest after fresh physics generation."
                )
            generation_manifest = {
                "experiment_scope": "APR-1 versus fixed-CFL4 model-pair discrimination",
                "requested_curves": int(total_curves),
                "sweep_points_per_class": len(sweep_points),
                "curves_per_batch": int(curves_per_batch),
                "n_jobs": int(n_jobs),
                "hadronic_baseline": CONFIG["CONTROLLED_HADRONIC_BASELINE"],
                "quark_parameters": {
                    "B_MeV_per_fm3": CONFIG["CONTROLLED_QUARK_B"],
                    "Delta_MeV": CONFIG["CONTROLLED_QUARK_DELTA"],
                    "m_s_MeV": CONFIG["CONTROLLED_QUARK_MS"],
                },
                "gaussian_deformation": {
                    "epsilon0_MeV_per_fm3": CONFIG["CONTROLLED_PERTURB_EPS0"],
                    "sigma_MeV_per_fm3": CONFIG["CONTROLLED_PERTURB_SIGMA"],
                    "amplitudes": [point.amplitude for point in sweep_points],
                    "hadronic_admissible_interval": list(hadronic_interval),
                    "quark_admissible_interval": list(quark_interval),
                },
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
    except PhysicsSimulationError:
        logger.exception("Physics engine failed.")
        raise
    except ConfigurationError:
        logger.exception("Configuration error.")
        raise
    except Exception:
        logger.exception("Unexpected error occurred during pipeline execution.")
        raise
    logger.info("===============================================================")


if __name__ == "__main__":
    main()
