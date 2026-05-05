# ml/model_server_v2.py - Enhanced FastAPI server with multi-explainer, calibration, and fairness
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import torch
from typing import List, Optional, Dict, Literal
import os
import time
from utils import prepare_data
from explainers import UnifiedExplainer, compute_fairness_metrics, compute_embeddings_tsne

app = FastAPI(title="WBC Diagnostic System v2", version="2.0.0")
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
start_time = time.time()

# Reconstruct model class definitions
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
        return recon, z, None

# Load models
models_loaded = False
scaler = clf = calibrated_clf = feature_names = model = decoder_lin = None
model_config = {}
input_dim = embed_dim = 30

try:
    scaler = joblib.load(f"{MODEL_DIR}/scaler.joblib")
    clf = joblib.load(f"{MODEL_DIR}/clf.joblib")
    feature_names = joblib.load(f"{MODEL_DIR}/feature_names.joblib")
    input_dim = len(feature_names)
    
    # Try to load calibrated classifier
    try:
        calibrated_clf = joblib.load(f"{MODEL_DIR}/calibrated_clf.joblib")
    except:
        calibrated_clf = clf
    
    # Load model config
    try:
        model_config = joblib.load(f"{MODEL_DIR}/model_config.joblib")
        embed_dim = model_config.get('embed_dim', 32)
    except:
        embed_dim = clf.coef_.shape[1] if hasattr(clf, "coef_") else 32
    
    device = torch.device("cpu")
    model = TabMAE(input_dim=input_dim, embed_dim=embed_dim)
    model.load_state_dict(torch.load(f"{MODEL_DIR}/tab_mae.pt", map_location=device))
    model.eval()
    
    # Get decoder weight matrix
    # NOTE: This is a linear approximation (ignores ReLU). Used as fast fallback.
    # For accurate explainability, we compute per-sample Jacobians in the explainer.
    with torch.no_grad():
        dec_fc1_w = model.decoder.fc1.weight.data.numpy()
        dec_fc2_w = model.decoder.fc2.weight.data.numpy()
        decoder_lin = np.dot(dec_fc2_w, dec_fc1_w)
    
    models_loaded = True
    print(f"✓ Models loaded successfully (v2.0.0)")
    print(f"  - Input dim: {input_dim}, Embed dim: {embed_dim}")
    print(f"  - Calibrated: {calibrated_clf is not clf}")
except Exception as e:
    print(f"⚠ Warning: Models not loaded. Run train_v2.py first. Error: {e}")
    models_loaded = False

# Initialize explainer
explainer = None
if models_loaded:
    explainer = UnifiedExplainer(model, scaler, feature_names, clf, decoder_lin)

# Request models
class PredictRequest(BaseModel):
    features: Optional[List[float]] = None
    named_features: Optional[Dict[str, float]] = None  # FIX #1: Accept named features
    age: Optional[int] = None
    gender: Optional[str] = None
    method: Optional[str] = "shap"

class RetrainRequest(BaseModel):
    verified_only: bool = True

# Endpoints
@app.get("/")
def root():
    return {"service": "WBC Diagnostic System v2", "version": "2.0.0", "status": "operational"}

@app.get("/model_status")
def model_status():
    return {
        "model_version": "2.0.0",
        "uptime_seconds": time.time() - start_time,
        "input_dim": input_dim,
        "embed_dim": embed_dim,
        "calibrated": calibrated_clf is not clf if models_loaded else False,
        "models_loaded": models_loaded,
        "config": model_config,
        "feature_names": feature_names
    }

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Enhanced predict endpoint with calibration, explainability, and input validation.
    Accepts either named_features (dict) or features (list).
    """
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded. Please train first.")
    
    warnings = []
    
    # FIX #1: Accept named features with guaranteed order
    if req.named_features:
        # Build feature array in the correct training order
        missing_features = [f for f in feature_names if f not in req.named_features]
        if missing_features:
            raise HTTPException(status_code=400, detail=f"Missing features: {missing_features}")
        raw_features = [float(req.named_features[f]) for f in feature_names]
    elif req.features is not None:
        if len(req.features) != input_dim:
            raise HTTPException(status_code=400, detail=f"Expected {input_dim} features, got {len(req.features)}")
        raw_features = req.features
        warnings.append("Using positional features array — feature order not verified. Use named_features for safety.")
    else:
        raise HTTPException(status_code=400, detail="Provide either 'features' (list) or 'named_features' (dict)")
    
    # Preprocess
    x = np.array(raw_features, dtype=float).reshape(1, -1)
    
    # FIX #2: Input validation
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        raise HTTPException(status_code=400, detail="Features contain NaN or Infinity values")
    
    x_s = scaler.transform(x)
    
    # Check for out-of-distribution features (> 5 std devs from training mean)
    extreme_mask = np.abs(x_s[0]) > 5.0
    if np.any(extreme_mask):
        ood_features = [feature_names[i] for i in np.where(extreme_mask)[0]]
        warnings.append(f"Unusual values detected for: {', '.join(ood_features)}. Prediction may be unreliable.")
    
    x_t = torch.from_numpy(x_s.astype(np.float32))
    
    # FIX #4: Use encoder directly — consistent with training embedding generation
    model.eval()
    with torch.no_grad():
        z = model.encoder(x_t)
        z_np = z.cpu().numpy()
    
    # Prediction with calibrated classifier
    if model_config.get('use_tabnet', False):
        X_combined = np.hstack([x_s, z_np])
        prob_cal = float(calibrated_clf.predict_proba(X_combined)[:, 1][0])
        prob_raw = float(clf.predict_proba(X_combined)[:, 1][0])
    else:
        prob_cal = float(calibrated_clf.predict_proba(z_np)[:, 1][0])
        prob_raw = float(clf.predict_proba(z_np)[:, 1][0])
    
    pred = int(prob_cal >= 0.5)
    result = "Benign" if pred == 1 else "Malignant"
    
    # FIX #5: Improved uncertainty estimation
    # Use probability-space variance with 30 MC dropout passes
    mc_probs = []
    model.train()  # Enable dropout
    with torch.no_grad():
        for _ in range(30):
            z_i = model.encoder(x_t)
            z_i_np = z_i.cpu().numpy()
            if model_config.get('use_tabnet', False):
                X_i = np.hstack([x_s, z_i_np])
                p_i = float(calibrated_clf.predict_proba(X_i)[:, 1][0])
            else:
                p_i = float(calibrated_clf.predict_proba(z_i_np)[:, 1][0])
            mc_probs.append(p_i)
    model.eval()
    uncertainty = float(np.std(mc_probs))  # Probability-space uncertainty
    z_mean = z_np.flatten()  # Use the clean embedding, not MC mean
    
    # Explainability
    explanation_method = req.method.lower() if req.method else "shap"
    shap_values = []
    
    if explainer:
        try:
            feat_shap, emb_shap = explainer.explain(x_s, method=explanation_method)
            shap_values = feat_shap.flatten().tolist()
        except Exception as e:
            print(f"Explanation error ({explanation_method}): {e}")
            import traceback
            traceback.print_exc()
            shap_values = [0.0] * input_dim
            warnings.append(f"Explanation method '{explanation_method}' failed: {str(e)}")
    
    return {
        "prediction": pred,
        "result": result,
        "probability": prob_raw,
        "calibrated_probability": prob_cal,
        "uncertainty": uncertainty,
        "explanation_method": explanation_method,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "embedding_mean": z_mean.tolist(),
        "model_version": "2.0.0",
        "warnings": warnings,
        "demographics": {
            "age": req.age,
            "gender": req.gender
        } if req.age or req.gender else None
    }

@app.post("/explain")
def explain(
    features: List[float],
    method: Literal["shap", "lime", "ig"] = Query("shap", description="Explanation method"),
    age: Optional[int] = None,
    gender: Optional[str] = None
):
    """
    Dedicated explainability endpoint
    """
    if not models_loaded or not explainer:
        raise HTTPException(status_code=500, detail="Models or explainer not loaded")
    
    if len(features) != input_dim:
        raise HTTPException(status_code=400, detail=f"Expected {input_dim} features")
    
    x = np.array(features, dtype=float).reshape(1, -1)
    x_s = scaler.transform(x)
    
    try:
        if model_config.get('use_tabnet', False):
            feat_importance, emb_importance = explainer.explain(x_s, method=method)
        else:
            feat_importance, emb_importance = explainer.explain(x_s, method=method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
    
    return {
        "method": method,
        "feature_importance": feat_importance.flatten().tolist(),
        "embedding_importance": emb_importance.flatten().tolist() if len(emb_importance.shape) == 1 else emb_importance.tolist(),
        "feature_names": feature_names,
        "top_features": [
            {"name": feature_names[i], "importance": float(feat_importance.flatten()[i])}
            for i in np.argsort(np.abs(feat_importance.flatten()))[-10:][::-1]
        ]
    }

@app.get("/model_fairness")
def model_fairness():
    """
    Compute fairness metrics on test data.
    FIX #6: Demographics are simulated and clearly labeled as such.
    """
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Load test data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, _ = prepare_data()
    X_test_np = X_test.values.astype(np.float32)
    
    # FIX #4: Use encoder directly for consistent embeddings
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test_np)
        z_test = model.encoder(X_test_tensor)
        z_test_np = z_test.cpu().numpy()
    
    # Predict
    if model_config.get('use_tabnet', False):
        X_test_combined = np.hstack([X_test_np, z_test_np])
        y_pred = calibrated_clf.predict(X_test_combined)
    else:
        y_pred = calibrated_clf.predict(z_test_np)
    
    # FIX #6: Simulated demographics — clearly labeled
    # The WBC dataset does not contain real age/gender data.
    # These are generated for demonstration of the fairness pipeline only.
    np.random.seed(42)
    demographics = {
        'age': np.random.randint(20, 80, size=len(y_test)),
        'gender': np.random.choice(['M', 'F', 'Other'], size=len(y_test), p=[0.3, 0.65, 0.05])
    }
    
    # Compute fairness metrics
    fairness_metrics = compute_fairness_metrics(y_test.values, y_pred, demographics)
    
    return {
        "fairness_metrics": fairness_metrics,
        "model_version": "2.0.0",
        "test_samples": len(y_test),
        "simulated_demographics": True,
        "demographics_note": "Demographics are simulated. The Wisconsin Breast Cancer dataset does not contain real age/gender data. These metrics demonstrate the fairness monitoring pipeline."
    }

@app.post("/embeddings_visualization")
def embeddings_visualization():
    """
    Generate t-SNE visualization data
    """
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, _ = prepare_data()
    X_all = np.vstack([X_train.values, X_val.values, X_test.values]).astype(np.float32)
    y_all = np.concatenate([y_train.values, y_val.values, y_test.values])
    
    # FIX #4: Use encoder directly for consistent embeddings
    model.eval()
    with torch.no_grad():
        X_all_tensor = torch.from_numpy(X_all)
        z_all = model.encoder(X_all_tensor)
        z_all_np = z_all.cpu().numpy()
    
    # t-SNE
    embeddings_2d = compute_embeddings_tsne(z_all_np, y_all)
    
    return {
        "embeddings": embeddings_2d,
        "labels": y_all.tolist(),
        "label_names": ['Benign', 'Malignant']
    }

@app.post("/retrain")
async def retrain(req: RetrainRequest = RetrainRequest()):
    """
    Retrain model with verified cases (placeholder - full implementation requires database)
    """
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # TODO: Load verified cases from database
    # For now, just use existing test data
    return {
        "status": "Retraining requires database integration",
        "message": "This endpoint will retrain on verified cases from MongoDB",
        "model_version": "2.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

