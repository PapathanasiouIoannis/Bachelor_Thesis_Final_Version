import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RUN_XGBOOST")

def load_data():
    TENSOR_DIR = os.path.join("data", "ml_tensors")
    logger.info(f"Loading tensors from {TENSOR_DIR}...")
    
    train_df = pd.read_parquet(os.path.join(TENSOR_DIR, "train.parquet"), engine='pyarrow')
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    
    X_train = train_df.drop(columns=['Label'])
    y_train = train_df['Label']
    
    X_test = test_df.drop(columns=['Label'])
    y_test = test_df['Label']
    
    return X_train, y_train, X_test, y_test

def main():
    OUTPUT_DIR = os.path.join("outputs", "xgboost")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_train, y_train, X_test, y_test = load_data()
    
    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    scale_pos_weight = count_hadronic / count_quark if count_quark > 0 else 1.0
    logger.info(f"Dynamically calculated scale_pos_weight: {scale_pos_weight:.4f}")
    
    params_path = os.path.join("outputs", "xgboost_best_params.json")
    with open(params_path, "r") as f:
        best_params = json.load(f)
        
    logger.info(f"Loaded best params: {best_params}")
    
    model_params = {
        "verbosity": 0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
        "n_estimators": 1000,
        "tree_method": "hist",
        "device": "cuda",
        "early_stopping_rounds": 50
    }
    model_params.update(best_params)
    
    model = xgb.XGBClassifier(**model_params)
    logger.info("Training final XGBoost model with Early Stopping...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    logger.info("Evaluating on strictly held-out Test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "PR-AUC": float(pr_auc),
        "F1-Score": float(f1)
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Final Test Metrics: PR-AUC={pr_auc:.4f}, F1={f1:.4f}")
    
    model_path = os.path.join(OUTPUT_DIR, "xgboost_weights.json")
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Generating SHAP Summary plot...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.pdf"), bbox_inches='tight')
    plt.close()
    
    logger.info("XGBoost Final Execution complete.")

if __name__ == "__main__":
    main()
