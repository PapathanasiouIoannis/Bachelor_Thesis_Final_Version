import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.metrics import classification_report, precision_recall_curve, auc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PERTURB_RUN_MLP")

class DynamicMLP(nn.Module):
    def __init__(self, input_dim, hidden_sizes, dropout_rate):
        super(DynamicMLP, self).__init__()
        layers = []
        in_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(in_dim, size))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            in_dim = size
            
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def load_data(feature_set='MR'):
    TENSOR_DIR = os.path.join("data", "ml_tensors_perturb")
    
    train_df = pd.read_parquet(os.path.join(TENSOR_DIR, "train.parquet"), engine='pyarrow')
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    
    if feature_set == 'MR':
        features = ['Mass', 'Radius']
    elif feature_set == 'MRL':
        features = ['Mass', 'Radius', 'log10_Lambda']
        
    X_train = train_df[features].values
    y_train = train_df['Label'].values
    
    X_test = test_df[features].values
    y_test = test_df['Label'].values
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(feature_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{feature_set}] Using device: {device}")
    
    X_train, y_train, X_test, y_test = load_data(feature_set)
    
    param_path = os.path.join("outputs_perturb", f"mlp_{feature_set}_best_params.json")
    if os.path.exists(param_path):
        with open(param_path, "r") as f:
            best_params = json.load(f)
        logger.info(f"Loaded {feature_set} optimized params: {best_params}")
    else:
        logger.warning(f"No optimized params found for {feature_set}, using defaults.")
        best_params = {"hidden_layer_sizes": "[128, 64]", "dropout_rate": 0.2, "learning_rate": 0.001}

    hidden_sizes = eval(best_params["hidden_layer_sizes"])
    
    model = DynamicMLP(
        input_dim=X_train.shape[1], 
        hidden_sizes=hidden_sizes, 
        dropout_rate=best_params["dropout_rate"]
    ).to(device)
    
    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    pos_weight = torch.tensor([count_hadronic / count_quark if count_quark > 0 else 1.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=best_params["learning_rate"])

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train).unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    epochs = 150
    
    logger.info(f"Training Final MLP [{feature_set}] on Noisy Tensors...")
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Save Model
    out_dir = os.path.join("outputs_perturb", f"mlp_{feature_set}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "mlp_weights.pth"))
    
    # Evaluate
    logger.info(f"Evaluating Final MLP [{feature_set}] on Test Set...")
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy().squeeze()
        
    y_pred = (probs > 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(y_test, probs)
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
    
    np.save(os.path.join(out_dir, "test_probs.npy"), probs)
    np.save(os.path.join(out_dir, "test_labels.npy"), y_test)

if __name__ == "__main__":
    logger.info("=== Running Final MLP for M-R ===")
    train_and_evaluate('MR')
    logger.info("=== Running Final MLP for M-R-L ===")
    train_and_evaluate('MRL')
