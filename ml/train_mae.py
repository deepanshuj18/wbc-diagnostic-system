# ml/train_mae.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import os
from utils import prepare_data
import argparse

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
        # x is batch_size x input_dim
        batch_size, dim = x.shape
        mask = (torch.rand(batch_size, dim, device=x.device) > self.mask_ratio).float()
        x_masked = x * mask
        z = self.encoder(x_masked)
        recon = self.decoder(z)
        return recon, z, mask

def train_mae(save_dir='models', epochs=50, batch_size=32, lr=1e-3, embed_dim=32, mask_ratio=0.25):
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = prepare_data()
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TabMAE(input_dim=X_train_np.shape[1], embed_dim=embed_dim, mask_ratio=mask_ratio).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train_np)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_np)), batch_size=batch_size, shuffle=False)

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
        print(f"Epoch {epoch+1}/{epochs} train_loss={avg_loss:.6f} val_loss={val_loss:.6f}")

    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, 'tab_mae.pt'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.joblib'))
    joblib.dump(feature_names, os.path.join(save_dir, 'feature_names.joblib'))
    # Create embeddings for classifier
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.from_numpy(X_train_np).to(device)
        _, z_train, _ = model(X_train_tensor)
        X_val_tensor = torch.from_numpy(X_val_np).to(device)
        _, z_val, _ = model(X_val_tensor)
    z_train_np = z_train.cpu().numpy()
    z_val_np = z_val.cpu().numpy()
    # Logistic regression on embeddings
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z_train_np, y_train.values)
    y_pred = clf.predict(z_val_np)
    y_prob = clf.predict_proba(z_val_np)[:,1]
    print("Classifier val acc:", accuracy_score(y_val, y_pred))
    print("Classifier val auc:", roc_auc_score(y_val, y_prob))
    joblib.dump(clf, os.path.join(save_dir, 'clf.joblib'))
    print("Saved model artifacts to", save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--embed_dim", type=int, default=32)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    args = parser.parse_args()
    train_mae(epochs=args.epochs, embed_dim=args.embed_dim, mask_ratio=args.mask_ratio)
