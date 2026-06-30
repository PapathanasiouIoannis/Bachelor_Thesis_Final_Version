import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import logging
from sklearn.metrics import precision_recall_curve, auc
import optuna.visualization as vis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PERTURB_OPT_XGBOOST")

def load_data(feature_set='MR'):
    TENSOR_DIR = os.path.join("data", "ml_tensors_perturb")
    logger.info(f"Loading noisy tensors from {TENSOR_DIR} for Feature Set: {feature_set}...")
    
    train_df = pd.read_parquet(os.path.join(TENSOR_DIR, "train.parquet"), engine='pyarrow')
    val_df = pd.read_parquet(os.path.join(TENSOR_DIR, "val.parquet"), engine='pyarrow')
    
    if feature_set == 'MR':
        features = ['Mass', 'Radius']
    elif feature_set == 'MRL':
        features = ['Mass', 'Radius', 'log10_Lambda']
    else:
        raise ValueError("Invalid feature set.")
    
    X_train = train_df[features]
    y_train = train_df['Label']
    
    X_val = val_df[features]
    y_val = val_df['Label']
    
    return X_train, y_train, X_val, y_val

def objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight):
    param = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_estimators": 1000,
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
    }

    model = xgb.XGBClassifier(
        **param,
        use_label_encoder=False,
        early_stopping_rounds=50,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    return pr_auc

def run_optimization(feature_set):
    X_train, y_train, X_val, y_val = load_data(feature_set)
    
    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0

    logger.info(f"Initializing Optuna Study for XGBoost [{feature_set}] PR-AUC maximization...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight), n_trials=30)

    logger.info(f"[{feature_set}] Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"[{feature_set}] Best Params: {study.best_params}")

    os.makedirs("outputs_perturb", exist_ok=True)
    out_path = os.path.join("outputs_perturb", f"xgboost_{feature_set}_best_params.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Saved {feature_set} optimized hyperparameters to {out_path}")

    # Visualizations
    plots_dir = os.path.join("plots_perturb", "ml_optimization")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        fig_hist = vis.plot_optimization_history(study)
        fig_hist.write_image(os.path.join(plots_dir, f"xgboost_opt_history_{feature_set}.pdf"))
        
        fig_para = vis.plot_parallel_coordinate(study)
        fig_para.write_image(os.path.join(plots_dir, f"xgboost_parallel_coordinate_{feature_set}.pdf"))
    except Exception as e:
        logger.warning(f"Could not save visualizations: {e}")

if __name__ == "__main__":
    logger.info("=== Optimizing XGBoost for M-R (Mass, Radius) ===")
    run_optimization('MR')
    logger.info("=== Optimizing XGBoost for M-R-L (Mass, Radius, Lambda) ===")
    run_optimization('MRL')
