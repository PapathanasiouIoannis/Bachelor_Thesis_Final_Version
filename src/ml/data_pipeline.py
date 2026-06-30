import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from joblib import dump
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DATA_PIPELINE")

import glob

def load_and_preprocess(hadronic_dir: str, quark_dir: str) -> pd.DataFrame:
    logger.info(f"Loading hadronic data from {hadronic_dir}...")
    hadronic_files = glob.glob(os.path.join(hadronic_dir, "*.parquet"))
    df_hadronic = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in hadronic_files], ignore_index=True)
    df_hadronic['Label'] = 0

    logger.info(f"Loading quark data from {quark_dir}...")
    quark_files = glob.glob(os.path.join(quark_dir, "*.parquet"))
    df_quark = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in quark_files], ignore_index=True)
    df_quark['Label'] = 1

    df = pd.concat([df_hadronic, df_quark], ignore_index=True)
    logger.info(f"Combined dataset shape: {df.shape}")

    # feature Engineering
    logger.info("Performing feature engineering...")
    
    # filter non-physical Radii to prevent zero division
    initial_len = len(df)
    df = df[df['Radius'] > 0].copy()
    dropped = initial_len - len(df)
    if dropped > 0:
        logger.warning(f"Dropped {dropped} rows with Radius <= 0.")

    # calculate Compactness C = M/R (Keeping it in the dataframe for physics analysis, but removing from ML features)
    df['C'] = df['Mass'] / df['Radius']

    # handle Lambda if present (or use LogLambda if it was precalculated)
    if 'Lambda' in df.columns:
        # filter Lambda <= 0 to prevent log10(0) or log10(negative)
        len_before_lambda = len(df)
        df = df[df['Lambda'] > 0]
        if len_before_lambda - len(df) > 0:
             logger.warning(f"Dropped {len_before_lambda - len(df)} rows with Lambda <= 0.")
        df['log10_Lambda'] = np.log10(df['Lambda'])
    elif 'LogLambda' in df.columns:
        logger.info("LogLambda is already present in dataset. Renaming to log10_Lambda for consistency.")
        df.rename(columns={'LogLambda': 'log10_Lambda'}, inplace=True)
    else:
        logger.error("Neither 'Lambda' nor 'LogLambda' found in the dataset.")
        raise KeyError("Missing Lambda feature.")
        
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Mass', 'Radius', 'log10_Lambda'], inplace=True)
    
    # strictly drop macroscopic feature duplicates to prevent Train/Val/Test data leakage
    initial_clean_len = len(df)
    
    # use rounding to 5 decimal places to catch near-duplicates caused by float32/64 precision scaling collisions
    duplicate_mask = df[['Mass', 'Radius', 'log10_Lambda']].round(5).duplicated()
    df = df[~duplicate_mask].copy()
    
    if initial_clean_len - len(df) > 0:
        logger.warning(f"Dropped {initial_clean_len - len(df)} duplicate macroscopic rows (with precision threshold) to strictly prevent data leakage.")

    
    logger.info(f"Data shape after feature engineering and cleaning: {df.shape}")
    return df

def run_pipeline():
    DATA_DIR = "data"
    HADRONIC_DIR = os.path.join(DATA_DIR, "ml_ready_hadronic")
    QUARK_DIR = os.path.join(DATA_DIR, "ml_ready_quark")
    OUTPUT_DIR = os.path.join(DATA_DIR, "ml_tensors")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_preprocess(HADRONIC_DIR, QUARK_DIR)

    physical_features = ['Mass', 'Radius', 'log10_Lambda']
    X = df[physical_features + ['Curve_ID']]
    y = df['Label']
    groups = df['Curve_ID']

    logger.info("Performing Grouped Train/Val/Test Split (80/10/10)...")
    
    # 1. Split out Test set (10%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_val_idx, test_idx = next(gss1.split(X, y, groups))
    
    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()
    
    # 2. Split remaining 90% into Train (80%) and Validation (10%)
    # 10 / 90 = 0.11111111
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))
    
    X_train = X_train_val.iloc[train_idx].copy()
    y_train = y_train_val.iloc[train_idx].copy()
    X_val = X_train_val.iloc[val_idx].copy()
    y_val = y_train_val.iloc[val_idx].copy()

    # Feature Matrix Cleanup (Crucial)
    X_train.drop(columns=['Curve_ID'], inplace=True)
    X_val.drop(columns=['Curve_ID'], inplace=True)
    X_test.drop(columns=['Curve_ID'], inplace=True)
    
    features = physical_features

    logger.info(f"Train size: {len(X_train)} (Hadronic: {sum(y_train==0)}, Quark: {sum(y_train==1)})")
    logger.info(f"Val size:   {len(X_val)} (Hadronic: {sum(y_val==0)}, Quark: {sum(y_val==1)})")
    logger.info(f"Test size:  {len(X_test)} (Hadronic: {sum(y_test==0)}, Quark: {sum(y_test==1)})")

    logger.info("Standardizing features based on training distribution...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=features, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)
    
    # save Scaler for downstream ML use
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
    dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to {scaler_path}")

    # save to Parquet
    logger.info("Saving standardized matrices to data/ml_tensors/ ...")
    
    # re-attach labels for easier loading
    train_out = pd.concat([X_train_scaled, y_train], axis=1)
    val_out = pd.concat([X_val_scaled, y_val], axis=1)
    test_out = pd.concat([X_test_scaled, y_test], axis=1)

    train_out.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), engine='pyarrow', index=False)
    val_out.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), engine='pyarrow', index=False)
    test_out.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), engine='pyarrow', index=False)
    
    logger.info("Data Pipeline Complete. ML Tensors generated successfully.")

if __name__ == "__main__":
    run_pipeline()
