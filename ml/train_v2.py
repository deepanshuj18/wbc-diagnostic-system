# ml/train_v2.py - Enhanced training with TabNet, calibration, and explainability
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import joblib
import os
from utils import prepare_data
import argparse

# Import TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    print("Warning: pytorch-tabnet not installed, falling back to standard classifier")

# MAE components
class TabMAEEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=32, hidden=64, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TabMAEDecoder(nn.Module):
    def __init__(self, embed_dim, output_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, output_dim)
    def forward(self, z):
        z = self.act(self.fc1(z))
        return self.fc2(z)

class TabMAE(nn.Module):
    def __init__(self, input_dim, embed_dim=32, mask_ratio=0.25):
        super().__init__()
        self.encoder = TabMAEEncoder(input_dim, embed_dim)
        self.decoder = TabMAEDecoder(embed_dim, input_dim)
        self.mask_ratio = mask_ratio
    def forward(self, x):
        batch_size, dim = x.shape
        mask = (torch.rand(batch_size, dim, device=x.device) > self.mask_ratio).float()
        x_masked = x * mask
        z = self.encoder(x_masked)
        recon = self.decoder(z)
        return recon, z, mask

def temperature_scaling(logits, temperature):
    """Apply temperature scaling to logits"""
    return logits / temperature

def train_with_tabnet(save_dir='models', epochs=50, batch_size=32, lr=1e-3, embed_dim=32, mask_ratio=0.25, use_tabnet=True):
    """
    Train hybrid MAE + TabNet model with calibration
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = prepare_data()
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    
    # Check class distribution
    print(f"\nDataset distribution:")
    print(f"Train - Malignant: {np.sum(y_train==0)}, Benign: {np.sum(y_train==1)}")
    print(f"Val - Malignant: {np.sum(y_val==0)}, Benign: {np.sum(y_val==1)}")
    print(f"Test - Malignant: {np.sum(y_test==0)}, Benign: {np.sum(y_test==1)}")
    
    # Phase 1: Train MAE
    print("\nPhase 1: Training MAE...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabMAE(input_dim=X_train_np.shape[1], embed_dim=embed_dim, mask_ratio=mask_ratio).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_train_np)), 
        batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_val_np)), 
        batch_size=batch_size, shuffle=False
    )
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            recon, _, _ = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for (xb,) in val_loader:
                xb = xb.to(device)
                recon, _, _ = model(xb)
                val_loss += loss_fn(recon, xb).item() * xb.size(0)
            val_loss = val_loss / len(val_loader.dataset)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} train_loss={avg_loss:.6f} val_loss={val_loss:.6f}")
    
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'tab_mae.pt'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.joblib'))
    
    # Generate embeddings
    print("\nPhase 2: Generating embeddings...")
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.from_numpy(X_train_np).to(device)
        _, z_train, _ = model(X_train_tensor)
        X_val_tensor = torch.from_numpy(X_val_np).to(device)
        _, z_val, _ = model(X_val_tensor)
        X_test_tensor = torch.from_numpy(X_test_np).to(device)
        _, z_test, _ = model(X_test_tensor)
    
    z_train_np = z_train.cpu().numpy()
    z_val_np = z_val.cpu().numpy()
    z_test_np = z_test.cpu().numpy()
    
    print(f"Embedding shapes - Train: {z_train_np.shape}, Val: {z_val_np.shape}, Test: {z_test_np.shape}")
    
    # Phase 3: Train TabNet or Logistic Regression on embeddings
    print("\nPhase 3: Training classifier...")
    
    if use_tabnet:
        try:
            # TabNet on raw features + embeddings concatenated
            X_train_combined = np.hstack([X_train_np, z_train_np])
            X_val_combined = np.hstack([X_val_np, z_val_np])
            X_test_combined = np.hstack([X_test_np, z_test_np])
            
            tabnet = TabNetClassifier(
                n_d=embed_dim, n_a=embed_dim, n_steps=5,
                gamma=1.5, n_independent=2, n_shared=2,
                cat_idxs=[], cat_dims=[],
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_params={"step_size":50, "gamma":0.9},
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                mask_type='entmax',
                seed=42,
                verbose=1
            )
            
            tabnet.fit(
                X_train_combined, y_train.values,
                eval_set=[(X_val_combined, y_val.values)],
                eval_metric=['auc'],
                max_epochs=100,
                patience=20,
                batch_size=batch_size
            )
            
            # Predict
            y_pred = tabnet.predict(X_val_combined)
            y_prob = tabnet.predict_proba(X_val_combined)[:,1]
            clf = tabnet
            print("Using TabNet classifier")
        except Exception as e:
            print(f"TabNet failed: {e}, falling back to Logistic Regression")
            use_tabnet = False
    
    if not use_tabnet:
        # Standard Logistic Regression
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(z_train_np, y_train.values)
        y_pred = clf.predict(z_val_np)
        y_prob = clf.predict_proba(z_val_np)[:,1]
        print("Using Logistic Regression classifier")
    
    # Evaluate
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    print(f"\nVal Accuracy: {acc:.4f}")
    print(f"Val AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Malignant', 'Benign']))
    
    # Save classifier
    joblib.dump(clf, os.path.join(save_dir, 'clf.joblib'))
    joblib.dump({'use_tabnet': use_tabnet, 'embed_dim': embed_dim}, 
                os.path.join(save_dir, 'model_config.joblib'))
    
    # Phase 4: Calibration
    print("\nPhase 4: Applying temperature scaling calibration...")
    
    if use_tabnet:
        # For TabNet, use Platt scaling
        from sklearn.calibration import CalibratedClassifierCV
        calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
        calibrated_clf.fit(X_val_combined if use_tabnet else z_val_np, y_val.values)
    else:
        # Temperature scaling for LR
        from sklearn.calibration import CalibratedClassifierCV
        calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv=5)
        calibrated_clf.fit(z_val_np, y_val.values)
    
    joblib.dump(calibrated_clf, os.path.join(save_dir, 'calibrated_clf.joblib'))
    
    # Test calibrated performance
    if use_tabnet:
        y_prob_cal = calibrated_clf.predict_proba(X_test_combined)[:,1]
    else:
        y_prob_cal = calibrated_clf.predict_proba(z_test_np)[:,1]
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test.values, y_prob_cal, n_bins=10)
    calibration_info = {
        'prob_true': prob_true.tolist(),
        'prob_pred': prob_pred.tolist()
    }
    joblib.dump(calibration_info, os.path.join(save_dir, 'calibration_curve.joblib'))
    
    print("\nCalibrated classifier saved")
    print(f"Saved model artifacts to {save_dir}")
    print(f"\nTest Performance (Calibrated):")
    if use_tabnet:
        y_test_prob = calibrated_clf.predict_proba(X_test_combined)[:,1]
    else:
        y_test_prob = calibrated_clf.predict_proba(z_test_np)[:,1]
    y_test_pred = (y_test_prob >= 0.5).astype(int)
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test AUC: {roc_auc_score(y_test, y_test_prob):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--use-tabnet", action="store_true", default=False)
    args = parser.parse_args()
    train_with_tabnet(
        epochs=args.epochs, 
        embed_dim=args.embed_dim, 
        mask_ratio=args.mask_ratio,
        use_tabnet=args.use_tabnet
    )

