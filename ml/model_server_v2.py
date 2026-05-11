# ml/model_server_v2.py - Enhanced FastAPI server with multi-explainer, calibration, and evaluation
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import torch
from typing import List, Optional, Dict, Literal
import os
import time
from utils import prepare_data
from explainers import UnifiedExplainer, compute_embeddings_tsne

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
    # NOTE: sklearn breast cancer dataset uses 0=Malignant, 1=Benign
    # predict_proba[:, 1] → P(Benign), so P(Malignant) = 1 - P(Benign)
    if model_config.get('use_tabnet', False):
        X_combined = np.hstack([x_s, z_np])
        prob_benign_cal = float(calibrated_clf.predict_proba(X_combined)[:, 1][0])
        prob_benign_raw = float(clf.predict_proba(X_combined)[:, 1][0])
    else:
        prob_benign_cal = float(calibrated_clf.predict_proba(z_np)[:, 1][0])
        prob_benign_raw = float(clf.predict_proba(z_np)[:, 1][0])
    
    # Convert to P(Malignant) — this is what clinicians care about
    prob_malignant_cal = 1.0 - prob_benign_cal
    prob_malignant_raw = 1.0 - prob_benign_raw
    
    # Predict based on malignant probability
    result = "Malignant" if prob_malignant_cal >= 0.5 else "Benign"
    pred = 1 if result == "Malignant" else 0
    
    # Confidence = probability of the predicted class (always >= 0.5)
    confidence = prob_malignant_cal if result == "Malignant" else prob_benign_cal
    
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
                p_i = 1.0 - float(calibrated_clf.predict_proba(X_i)[:, 1][0])
            else:
                p_i = 1.0 - float(calibrated_clf.predict_proba(z_i_np)[:, 1][0])
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
        "probability": prob_malignant_raw,
        "calibrated_probability": prob_malignant_cal,
        "confidence": confidence,
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

@app.get("/model_evaluation")
def model_evaluation():
    """
    Compute comprehensive model evaluation metrics on test data.
    Returns confusion matrix, ROC curve, PR curve, classification report, and calibration data.
    """
    if not models_loaded:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        roc_curve, auc, precision_recall_curve, average_precision_score,
        accuracy_score, precision_score, recall_score, f1_score
    )
    
    # Load test data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _, _ = prepare_data()
    X_test_np = X_test.values.astype(np.float32)
    
    # Generate embeddings
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test_np)
        z_test = model.encoder(X_test_tensor)
        z_test_np = z_test.cpu().numpy()
    
    # Predictions & probabilities
    if model_config.get('use_tabnet', False):
        X_combined = np.hstack([X_test_np, z_test_np])
        y_pred = calibrated_clf.predict(X_combined)
        y_prob = calibrated_clf.predict_proba(X_combined)[:, 1]
    else:
        y_pred = calibrated_clf.predict(z_test_np)
        y_prob = calibrated_clf.predict_proba(z_test_np)[:, 1]
    
    y_true = y_test.values
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred).tolist()
    
    # 2. Classification Report
    report = classification_report(y_true, y_pred, target_names=['Malignant', 'Benign'], output_dict=True)
    
    # 3. ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc = float(auc(fpr, tpr))
    # Downsample for JSON (max 200 points)
    step = max(1, len(fpr) // 200)
    roc_data = {
        "fpr": fpr[::step].tolist(),
        "tpr": tpr[::step].tolist(),
        "auc": roc_auc
    }
    
    # 4. Precision-Recall Curve
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = float(average_precision_score(y_true, y_prob))
    step_pr = max(1, len(pr_precision) // 200)
    pr_data = {
        "precision": pr_precision[::step_pr].tolist(),
        "recall": pr_recall[::step_pr].tolist(),
        "average_precision": avg_precision
    }
    
    # 5. Calibration data (bin predictions into 10 bins)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    cal_predicted = []
    cal_actual = []
    cal_counts = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:  # Include upper edge in last bin
            mask = mask | (y_prob == bin_edges[i + 1])
        if np.sum(mask) > 0:
            cal_predicted.append(float(np.mean(y_prob[mask])))
            cal_actual.append(float(np.mean(y_true[mask])))
            cal_counts.append(int(np.sum(mask)))
    
    calibration_data = {
        "predicted": cal_predicted,
        "actual": cal_actual,
        "counts": cal_counts
    }
    
    # 6. Overall summary metrics
    summary = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "avg_precision": avg_precision
    }
    
    # 7. Top feature importances (from classifier coefficients if available)
    feature_importance = []
    if hasattr(clf, 'coef_'):
        # Map embedding coefficients back to feature space
        emb_coefs = clf.coef_.flatten()
        if decoder_lin is not None:
            feat_coefs = np.abs(np.dot(decoder_lin, emb_coefs))
        else:
            feat_coefs = np.abs(emb_coefs)
        if len(feat_coefs) == len(feature_names):
            top_idx = np.argsort(feat_coefs)[::-1][:15]
            feature_importance = [
                {"name": feature_names[i], "importance": float(feat_coefs[i])}
                for i in top_idx
            ]
    
    return {
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
        "calibration": calibration_data,
        "summary": summary,
        "feature_importance": feature_importance,
        "model_version": "2.0.0",
        "test_samples": int(len(y_true))
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

