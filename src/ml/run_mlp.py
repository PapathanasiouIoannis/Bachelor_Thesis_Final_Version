import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RUN_MLP")

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
        # Removed Sigmoid here because we will use BCEWithLogitsLoss which is numerically safer
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def load_and_preprocess_data():
    TENSOR_DIR = os.path.join("data", "ml_tensors")
    logger.info(f"Loading tensors from {TENSOR_DIR}...")
    
    train_df = pd.read_parquet(os.path.join(TENSOR_DIR, "train.parquet"), engine='pyarrow')
    val_df = pd.read_parquet(os.path.join(TENSOR_DIR, "val.parquet"), engine='pyarrow')
    test_df = pd.read_parquet(os.path.join(TENSOR_DIR, "test.parquet"), engine='pyarrow')
    
    features = ['Mass', 'Radius', 'log10_Lambda']
    
    X_train_raw = train_df[features].values
    y_train = train_df['Label'].values
    
    X_val_raw = val_df[features].values
    y_val = val_df['Label'].values

    X_test_raw = test_df[features].values
    y_test = test_df['Label'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, len(features)

def main():
    OUTPUT_DIR = os.path.join("outputs", "mlp")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    X_train_scaled, y_train, X_val, y_val, X_test, y_test, input_dim = load_and_preprocess_data()
    
    params_path = os.path.join("outputs", "mlp_best_params.json")
    with open(params_path, "r") as f:
        best_params = json.load(f)
        
    logger.info(f"Loaded best params: {best_params}")
    
    hidden_sizes = eval(best_params['hidden_layer_sizes'])
    dropout_rate = best_params['dropout_rate']
    learning_rate = best_params['learning_rate']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DynamicMLP(input_dim=input_dim, hidden_sizes=hidden_sizes, dropout_rate=dropout_rate).to(device)
    
    count_hadronic = np.sum(y_train == 0)
    count_quark = np.sum(y_train == 1)
    pos_weight = torch.tensor([count_hadronic / count_quark if count_quark > 0 else 1.0]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled.copy()), torch.FloatTensor(y_train.copy()).unsqueeze(1))
    val_dataset = TensorDataset(torch.FloatTensor(X_val.copy()), torch.FloatTensor(y_val.copy()).unsqueeze(1))
    test_dataset = TensorDataset(torch.FloatTensor(X_test.copy()), torch.FloatTensor(y_test.copy()).unsqueeze(1))

    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    epochs = 100
    patience = 15
    epochs_no_improve = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    history = {'train_loss': [], 'val_loss': [], 'val_pr_auc': []}
    
    logger.info("Training final MLP with Early Stopping on Validation set...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                probs = torch.sigmoid(outputs)
                val_preds.extend(probs.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        precision, recall, _ = precision_recall_curve(y_val, val_preds)
        val_pr_auc = auc(recall, precision)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pr_auc'].append(val_pr_auc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
            
    logger.info("Restoring best model weights...")
    model.load_state_dict(best_model_state)
    
    logger.info("Evaluating on strictly held-out Test set...")
    model.eval()
    test_preds_proba = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            outputs = model(X_batch.to(device))
            probs = torch.sigmoid(outputs)
            test_preds_proba.extend(probs.cpu().numpy())
            
    y_pred_proba = np.array(test_preds_proba).flatten()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    precision_t, recall_t, _ = precision_recall_curve(y_test, y_pred_proba)
    test_pr_auc = auc(recall_t, precision_t)
    f1 = f1_score(y_test, y_pred)
    
    metrics = {
        "PR-AUC": float(test_pr_auc),
        "F1-Score": float(f1)
    }
    
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Final Test Metrics: PR-AUC={test_pr_auc:.4f}, F1={f1:.4f}")
    
    model_path = os.path.join(OUTPUT_DIR, "mlp_weights.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    logger.info("Generating Training History plot...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(history['train_loss'], color=color, label='Train Loss')
    ax1.plot(history['val_loss'], color='darkred', linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='center right')

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('PR-AUC', color=color)  
    ax2.plot(history['val_pr_auc'], color=color, label='Val PR-AUC')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    fig.tight_layout()  
    plt.title('MLP Final Training History (Loss & PR-AUC)')
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.pdf"), bbox_inches='tight')
    plt.close()
    
    logger.info("MLP Final Execution complete.")

if __name__ == "__main__":
    main()
