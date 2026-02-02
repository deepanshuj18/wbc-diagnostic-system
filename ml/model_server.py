# ml/model_server.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import joblib
import torch
import torch.nn.functional as F
from typing import List
import io, csv, time
from utils import load_wbc
from shap_utils import compute_shap_for_embedding, map_embedding_shap_to_features
import os

app = FastAPI()
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_VERSION = "1.0.0"
start_time = time.time()

# Reconstruct model class definitions (same as training)
class TabMAEEncoder(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=32, hidden=64, dropout=0.1):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden, embed_dim)
        self.dropout = torch.nn.Dropout(dropout)
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

class TabMAEDecoder(torch.nn.Module):
    def __init__(self, embed_dim, output_dim, hidden=64):
        super().__init__()
        self.fc1 = torch.nn.Linear(embed_dim, hidden)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden, output_dim)
    def forward(self, z):
        z = self.act(self.fc1(z))
        return self.fc2(z)

class TabMAE(torch.nn.Module):
    def __init__(self, input_dim, embed_dim=32, mask_ratio=0.25, dropout=0.1):
        super().__init__()
        self.encoder = TabMAEEncoder(input_dim, embed_dim, dropout=dropout)
        self.decoder = TabMAEDecoder(embed_dim, input_dim)
        self.mask_ratio = mask_ratio
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

# load artifacts
try:
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    clf = joblib.load(f"{MODEL_DIR}/clf.joblib")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.joblib")
    input_dim = len(feature_names)
    embed_dim = clf.coef_.shape[1] if hasattr(clf, "coef_") else 32
    
    device = torch.device("cpu")
    model = TabMAE(input_dim=input_dim, embed_dim=embed_dim)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/tab_mae.pt", map_location=device))
    model.eval()
    
    # get decoder weight matrix
    with torch.no_grad():
        dec_fc1_w = model.decoder.fc1.weight.data.numpy()
        dec_fc2_w = model.decoder.fc2.weight.data.numpy()
        decoder_lin = np.dot(dec_fc2_w, dec_fc1_w)
    
    models_loaded = True
except Exception as e:
    print(f"Warning: Models not loaded yet. Run train_mae.py first. Error: {e}")
    models_loaded = False
    scaler = clf = feature_names = model = decoder_lin = None
    input_dim = embed_dim = 30  # WBC default

class PredictRequest(BaseModel):
    features: List[float]  # in original feature space

@app.get("/model_status")
def model_status():
    return {
        "model_version": MODEL_VERSION,
        "uptime_seconds": time.time() - start_time,
        "input_dim": input_dim,
        "embed_dim": embed_dim
    }

def preprocess_array(arr):
    # arr is 1D list length input_dim
    import numpy as np
    x = np.array(arr, dtype=float).reshape(1, -1)
    x_s = scaler.transform(x)
    return x_s

@app.post("/predict")
def predict(req: PredictRequest, mc_iters: int = 10):
    if not models_loaded:
        return {"error": "Models not loaded. Please train the model first."}
    x_s = preprocess_array(req.features)
    x_t = torch.from_numpy(x_s.astype(np.float32))
    zs = []
    recon = None
    model.train()  # enable dropout for MC
    with torch.no_grad():
        for i in range(mc_iters):
            recon_i, z_i = model(x_t)
            zs.append(z_i.numpy())
        # back to eval
    model.eval()
    z_mean = np.mean(np.vstack(zs), axis=0)
    z_std = np.std(np.vstack(zs), axis=0)
    # classifier probability
    prob = float(clf.predict_proba(z_mean.reshape(1, -1))[:,1])
    pred = int(prob >= 0.5)
    result = "Benign" if pred == 1 else "Malignant"
    # uncertainty: use entropy-like measure or use z_std magnitude
    uncertainty = float(np.linalg.norm(z_std))
    # SHAP on embedding: for single sample, use background from small sample of training (approx)
    # For speed, we use KernelExplainer on the classifier with background = zeros
    try:
        shap_vals_embed = compute_shap_for_embedding(clf, z_mean.reshape(1, -1), background=None)
        # shap returns list for binary (may be array) — normalize
        if isinstance(shap_vals_embed, list):
            shap_vals_embed = np.array(shap_vals_embed)[0]
        feat_shap = map_embedding_shap_to_features(shap_vals_embed.reshape(1, -1), decoder_lin)
        feat_shap = feat_shap.flatten().tolist()
    except Exception as e:
        print(f"SHAP computation error: {e}")
        feat_shap = [0.0]*input_dim

    resp = {
        "prediction": pred,
        "result": result,
        "probability": prob,
        "uncertainty": uncertainty,
        "shap_values": feat_shap,
        "feature_names": feature_names,
        "embedding_mean": z_mean.flatten().tolist(),
        "model_version": MODEL_VERSION
    }
    return resp

@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    if not models_loaded:
        return {"error": "Models not loaded. Please train the model first."}
    # Expect CSV with columns=feature_names
    content = await file.read()
    s = content.decode()
    import pandas as pd
    df = pd.read_csv(io.StringIO(s))
    # validate columns
    if not set(feature_names).issubset(set(df.columns)):
        return {"error": "CSV must include feature columns: " + ",".join(feature_names)}
    X = df[feature_names].values
    Xs = scaler.transform(X)
    Xt = torch.from_numpy(Xs.astype(np.float32))
    model.eval()
    with torch.no_grad():
        recon, z = model(Xt)
    z_np = z.numpy()
    probs = clf.predict_proba(z_np)[:,1].tolist()
    preds = (np.array(probs) >= 0.5).astype(int).tolist()
    # Compute shap for first N rows for speed
    shap_vals = []
    try:
        shap_vals_embed = compute_shap_for_embedding(clf, z_np[:50])
        for i in range(len(z_np)):
            idx = min(i, 49)
            emb_shap = shap_vals_embed[idx] if isinstance(shap_vals_embed, (list, np.ndarray)) else np.zeros(z_np.shape[1])
            feat_shap = map_embedding_shap_to_features(np.array(emb_shap).reshape(1, -1), decoder_lin).flatten().tolist()
            shap_vals.append(feat_shap)
    except Exception as e:
        shap_vals = [[0.0]*input_dim for _ in range(len(z_np))]
    return {
        "predictions": preds,
        "probabilities": probs,
        "shap_values": shap_vals,
        "feature_names": feature_names
    }
