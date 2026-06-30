import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import logging
from sklearn.metrics import classification_report, precision_recall_curve, auc, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PERTURB_RUN_XGBOOST")

def load_data(feature_set='MR'):
    TENSOR_DIR = os.path.join("data", "ml_tensors_perturb")
    
    train_df = pd.read_parquet(os.path.join(TENSOR_DIR, "train.parquet"), engine='pyarrow')
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    
    if feature_set == 'MR':
        features = ['Mass', 'Radius']
    elif feature_set == 'MRL':
        features = ['Mass', 'Radius', 'log10_Lambda']
        
    X_train = train_df[features]
    y_train = train_df['Label']
    
    X_test = test_df[features]
    y_test = test_df['Label']
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(feature_set):
    X_train, y_train, X_test, y_test = load_data(feature_set)
    
    param_path = os.path.join("outputs_perturb", f"xgboost_{feature_set}_best_params.json")
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            best_params = json.load(f)
        logger.info(f"Loaded {feature_set} optimized params: {best_params}")
    else:
        logger.warning(f"No optimized params found for {feature_set}, using defaults.")
        best_params = {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05}

    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0

    model_params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
        "use_label_encoder": False,
        "n_jobs": -1
    }
    model_params.update(best_params)
    
    model = xgb.XGBClassifier(**model_params)
    
    logger.info(f"Training XGBoost Final Model [{feature_set}] on Noisy Tensors...")
    model.fit(X_train, y_train, verbose=False)
    
    # Save Model
    out_dir = os.path.join("outputs_perturb", f"xgboost_{feature_set}")
    os.makedirs(out_dir, exist_ok=True)
    model.save_model(os.path.join(out_dir, "xgboost_weights.json"))
    
    # Evaluate
    logger.info(f"Evaluating XGBoost Final Model [{feature_set}] on Test Set...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    rep = classification_report(y_test, y_pred, target_names=["Hadronic", "Quark"], output_dict=True)
    
    metrics = {
        "PR-AUC": pr_auc,
        "F1-Score": rep["macro avg"]["f1-score"]
    }
    
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"[{feature_set}] Metrics saved to {out_dir}")
    logger.info(f"[{feature_set}] PR-AUC: {pr_auc:.4f}")
    
    # Save predictions for ensemble/plotting
    np.save(os.path.join(out_dir, "test_probs.npy"), y_pred_proba)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test.values)

if __name__ == "__main__":
    logger.info("=== Running Final XGBoost for M-R ===")
    train_and_evaluate('MR')
    logger.info("=== Running Final XGBoost for M-R-L ===")
    train_and_evaluate('MRL')
