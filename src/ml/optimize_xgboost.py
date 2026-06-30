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
logger = logging.getLogger("OPT_XGBOOST")

def load_data():
    TENSOR_DIR = os.path.join("data", "ml_tensors")
    logger.info(f"Loading tensors from {TENSOR_DIR}...")
    
    train_df = pd.read_parquet(os.path.join(TENSOR_DIR, "train.parquet"), engine='pyarrow')
    val_df = pd.read_parquet(os.path.join(TENSOR_DIR, "val.parquet"), engine='pyarrow')
    
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']
    
    X_val = val_df.drop(columns=['Label'])
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

def run_optimization():
    X_train, y_train, X_val, y_val = load_data()
    
    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0

    logger.info("Initializing Optuna Study for XGBoost PR-AUC maximization...")
    study = optuna.create_study(direction="maximize")
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, scale_pos_weight), n_trials=50)

    logger.info(f"Best Trial PR-AUC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    # Save Params
    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "xgboost_best_params.json")
    with open(out_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
    logger.info(f"Saved optimized hyperparameters to {out_path}")

    # Visualizations
    plots_dir = os.path.join("plots", "ml_optimization")
    os.makedirs(plots_dir, exist_ok=True)

    try:
        fig_hist = vis.plot_optimization_history(study)
        fig_hist.write_image(os.path.join(plots_dir, "xgboost_opt_history.pdf"))
        
        fig_para = vis.plot_parallel_coordinate(study)
        fig_para.write_image(os.path.join(plots_dir, "xgboost_parallel_coordinate.pdf"))
        logger.info(f"Saved Optuna visualizations to {plots_dir}")
    except Exception as e:
        logger.warning(f"Could not save visualizations (make sure kaleido/plotly are installed): {e}")

if __name__ == "__main__":
    run_optimization()
