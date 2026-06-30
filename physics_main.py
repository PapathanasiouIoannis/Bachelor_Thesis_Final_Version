# physics_main.py

"""
  Isolated test-bed orchestrator for the Physics Generation and Visualization
  pipeline. This script is used to strictly validate the Equation of State (EoS)
  generation, Tolman-Oppenheimer-Volkoff (TOV) solver stability, and
  thermodynamic constraints.

Refactored:
  - PARQUET note: Synced with the main pipeline to use Apache Parquet
    via PyArrow for lightning-fast file I/O and strict type preservation.
"""

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
from src.visualize.plot_core_physics import plot_core_physics
from src.visualize.plot_3d_interactive import plot_interactive_3d

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = "data"
# note: Synced to Parquet format
DATA_FILE = os.path.join(DATA_DIR, "physics_test_dataset.parquet")


def main():
    logger = get_logger("PHYSICS_TESTBED")

    try:
        logger.info("===============================================================")
        logger.info("      PHYSICS ISOLATION ENVIRONMENT: EOS & TOV PIPELINE        ")
        logger.info("===============================================================")

        # 1. Directory Setup
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs("plots", exist_ok=True)

        # 2. Physics Initialization
        logger.info("\n[Step 1] Initializing Physics Environment...")

        # 3. Parallel Generation
        logger.info("\n[Step 2] Generating Physics Test Data (Parallel) ...")

        if os.path.exists(DATA_FILE):
            logger.info("[INFO] Existing dataset found. Skipping physics generation phase.")
            # load the existing dataset so downstream tasks have the required 'df'
            df = pd.read_parquet(DATA_FILE)
        else:
            # keep totals moderate for rapid physics debugging
            TOTAL_CURVES = 1000
            CURVES_PER_BATCH = 100
            N_JOBS = -1  # use all available CPU cores

            tasks = []
            num_batches = max(1, TOTAL_CURVES // CURVES_PER_BATCH)

            for i in range(num_batches):
                # interleave Hadronic and Quark tasks for load balancing across cores
                t_type = "hadronic" if i % 2 == 0 else "quark"
                tasks.append((t_type, CURVES_PER_BATCH, i, i))

            logger.info(f"Spawning {len(tasks)} tasks to generate {TOTAL_CURVES} EoS curves...")

            temp_dir = os.path.join(DATA_DIR, "physics_temp")
            os.makedirs(temp_dir, exist_ok=True)
            for f in os.listdir(temp_dir):
                if f.endswith(".parquet"):
                    os.remove(os.path.join(temp_dir, f))

            def process_and_save(t, idx):
                sublist = run_worker_wrapper(t)
                if sublist is not None and not sublist.empty:
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
                    return chunk_path
                return None

            # execute Parallel Workers
            res = Parallel(n_jobs=N_JOBS)(
                delayed(process_and_save)(t, idx) for idx, t in enumerate(tqdm(tasks))
            )

            # flatten DataFrames
            cols = CONFIG["COLUMN_SCHEMA"]
            valid_paths = [p for p in res if p is not None]

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

            # shuffle to thoroughly mix classes
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)

            # note: Save to Parquet
            df.to_parquet(DATA_FILE, engine="pyarrow", index=False)
            logger.info(f"done Saved physics dataset ({len(df)} samples) to {DATA_FILE}.")

            # --- HADRONIC SPLIT ---
            ml_ready_dir = os.path.join(DATA_DIR, "ml_ready_hadronic")
            os.makedirs(ml_ready_dir, exist_ok=True)
            
            df_hadronic = df[df['Label'] == 0]
            if not df_hadronic.empty and 'Baseline_Name' in df_hadronic.columns:
                for baseline_name, group in df_hadronic.groupby('Baseline_Name'):
                    split_path = os.path.join(ml_ready_dir, f"dataset_{baseline_name}.parquet")
                    group.to_parquet(split_path, engine="pyarrow", index=False)
                logger.info(f"done Saved split hadronic datasets to {ml_ready_dir}.")
            # ----------------------

            # --- QUARK SPLIT ---
            ml_ready_quark_dir = os.path.join(DATA_DIR, "ml_ready_quark")
            os.makedirs(ml_ready_quark_dir, exist_ok=True)
            
            df_quark = df[df['Label'] == 1]
            if not df_quark.empty and 'Baseline_Name' in df_quark.columns:
                for baseline_name, group in df_quark.groupby('Baseline_Name'):
                    split_path = os.path.join(ml_ready_quark_dir, f"dataset_{baseline_name}.parquet")
                    group.to_parquet(split_path, engine="pyarrow", index=False)
                logger.info(f"done Saved split quark datasets to {ml_ready_quark_dir}.")
            # ----------------------

        # 5. Physics Visualization Suite
        logger.info("\n[Step 3] Running Core Physics Visualization...")
        plot_core_physics(df)
        plot_interactive_3d(df)

        logger.info("\n===============================================================")
        logger.info("             PHYSICS PIPELINE COMPLETED SUCCESSFULLY           ")
    except PhysicsSimulationError as e:
        logger.exception("Physics engine failed.")
    except ConfigurationError as e:
        logger.exception("Configuration error.")
    except Exception as e:
        logger.exception("Unexpected error occurred during pipeline execution.")
    logger.info("===============================================================")


if __name__ == "__main__":
    main()
