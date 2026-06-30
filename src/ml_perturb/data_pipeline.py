import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from joblib import dump
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PERTURB_DATA_PIPELINE")

def inject_observational_noise(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Inject synthetic Gaussian noise mimicking LIGO/NICER observational uncertainties."""
    np.random.seed(seed)
    logger.info("Injecting synthetic observational noise into macroscopic features...")
    
    noisy_df = df.copy()
    
    # Simulate ~5% uncertainty on Mass
    mass_noise = np.random.normal(0, 0.05 * noisy_df['Mass'])
    noisy_df['Mass'] = noisy_df['Mass'] + mass_noise
    
    # Simulate ~10% uncertainty on Radius
    radius_noise = np.random.normal(0, 0.10 * noisy_df['Radius'])
    noisy_df['Radius'] = noisy_df['Radius'] + radius_noise
    
    # Simulate ~20% uncertainty on Lambda (Tidal Deformability)
    if 'Lambda' in noisy_df.columns:
        lambda_noise = np.random.normal(0, 0.20 * noisy_df['Lambda'])
        noisy_df['Lambda'] = np.abs(noisy_df['Lambda'] + lambda_noise)  # keep positive
        noisy_df['log10_Lambda'] = np.log10(np.clip(noisy_df['Lambda'], a_min=1e-10, a_max=None))
    elif 'LogLambda' in noisy_df.columns:
        # 20% on Lambda translates to approx 0.086 on log10_Lambda
        log_lambda_noise = np.random.normal(0, 0.086)
        noisy_df['log10_Lambda'] = noisy_df['LogLambda'] + log_lambda_noise
        
    return noisy_df

def load_and_preprocess(hadronic_dir: str, quark_dir: str) -> pd.DataFrame:
    logger.info(f"Loading hadronic data from {hadronic_dir}...")
    hadronic_files = glob.glob(os.path.join(hadronic_dir, "*.parquet"))
    df_hadronic = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in hadronic_files], ignore_index=True) if hadronic_files else pd.DataFrame()
    if not df_hadronic.empty:
        df_hadronic['Label'] = 0

    logger.info(f"Loading quark data from {quark_dir}...")
    quark_files = glob.glob(os.path.join(quark_dir, "*.parquet"))
    df_quark = pd.concat([pd.read_parquet(f, engine='pyarrow') for f in quark_files], ignore_index=True) if quark_files else pd.DataFrame()
    if not df_quark.empty:
        df_quark['Label'] = 1

    df = pd.concat([df_hadronic, df_quark], ignore_index=True)
    if df.empty:
        logger.error("No data found! Check if ml_ready_hadronic and ml_ready_quark contain the big dataset.")
        return df
        
    logger.info(f"Combined clean dataset shape: {df.shape}")

    # feature Engineering & Cleaning
    df = df[df['Radius'] > 0].copy()
    
    if 'Lambda' in df.columns:
        df = df[df['Lambda'] > 0]
    elif 'LogLambda' in df.columns and 'log10_Lambda' not in df.columns:
        df.rename(columns={'LogLambda': 'log10_Lambda'}, inplace=True)
    
    # drop infinite
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Mass', 'Radius'], inplace=True)
    
    # inject Noise
    df = inject_observational_noise(df)
    
    # Note: Compactness (C) is explicitly omitted from this pipeline!
    
    logger.info(f"Data shape after noise injection and cleaning: {df.shape}")
    return df

def run_pipeline():
    DATA_DIR = "data"
    HADRONIC_DIR = os.path.join(DATA_DIR, "ml_ready_hadronic")
    QUARK_DIR = os.path.join(DATA_DIR, "ml_ready_quark")
    OUTPUT_DIR = os.path.join(DATA_DIR, "ml_tensors_perturb")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = load_and_preprocess(HADRONIC_DIR, QUARK_DIR)
    if df.empty:
        return

    # Using ONLY Mass, Radius, and log10_Lambda. 'C' is excluded.
    physical_features = ['Mass', 'Radius', 'log10_Lambda']
    X = df[physical_features + ['Curve_ID']]
    y = df['Label']
    groups = df['Curve_ID']

    logger.info("Performing Grouped Train/Val/Test Split (80/10/10) on Noisy Data...")
    
    # 1. Split out Test set (10%)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_val_idx, test_idx = next(gss1.split(X, y, groups))
    
    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    
    X_test = X.iloc[test_idx].copy()
    y_test = y.iloc[test_idx].copy()
    
    # 2. Split remaining 90% into Train (80%) and Validation (10%)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1/9, random_state=42)
    train_idx, val_idx = next(gss2.split(X_train_val, y_train_val, groups_train_val))
    
    X_train = X_train_val.iloc[train_idx].copy()
    y_train = y_train_val.iloc[train_idx].copy()
    X_val = X_train_val.iloc[val_idx].copy()
    y_val = y_train_val.iloc[val_idx].copy()

    # Drop Curve_ID
    X_train.drop(columns=['Curve_ID'], inplace=True)
    X_val.drop(columns=['Curve_ID'], inplace=True)
    X_test.drop(columns=['Curve_ID'], inplace=True)

    logger.info(f"Train size: {len(X_train)} (Hadronic: {sum(y_train==0)}, Quark: {sum(y_train==1)})")
    logger.info(f"Val size:   {len(X_val)} (Hadronic: {sum(y_val==0)}, Quark: {sum(y_val==1)})")
    logger.info(f"Test size:  {len(X_test)} (Hadronic: {sum(y_test==0)}, Quark: {sum(y_test==1)})")

    logger.info("Standardizing perturbed features based on training distribution...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=physical_features, index=X_train.index)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=physical_features, index=X_val.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=physical_features, index=X_test.index)
    
    scaler_path = os.path.join(OUTPUT_DIR, "scaler_perturb.joblib")
    dump(scaler, scaler_path)
    logger.info(f"Saved feature scaler to {scaler_path}")

    # Save outputs
    train_out = pd.concat([X_train_scaled, y_train], axis=1)
    val_out = pd.concat([X_val_scaled, y_val], axis=1)
    test_out = pd.concat([X_test_scaled, y_test], axis=1)

    train_out.to_parquet(os.path.join(OUTPUT_DIR, "train.parquet"), engine='pyarrow', index=False)
    val_out.to_parquet(os.path.join(OUTPUT_DIR, "val.parquet"), engine='pyarrow', index=False)
    test_out.to_parquet(os.path.join(OUTPUT_DIR, "test.parquet"), engine='pyarrow', index=False)
    
    logger.info("Perturbed Data Pipeline Complete. Noisy ML Tensors generated successfully.")

if __name__ == "__main__":
    run_pipeline()
