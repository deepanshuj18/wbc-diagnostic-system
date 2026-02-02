# ml/explainers.py - Multiple explainability methods
import numpy as np
import shap
import joblib
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

try:
    from lime import lime_tabular
except ImportError:
    lime_tabular = None

class UnifiedExplainer:
    """
    Unified interface for SHAP, LIME, and Integrated Gradients
    """
    def __init__(self, model, scaler, feature_names, clf, decoder_weight_matrix=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.clf = clf
        self.decoder_weight_matrix = decoder_weight_matrix
        
    def explain_shap(self, X, background=None, nsamples=50):
        """
        SHAP explanation using KernelExplainer on embeddings
        Returns: shap_values, feature_importance
        """
        # Generate embeddings
        with torch.no_grad():
            X_tensor = torch.from_numpy(X.astype(np.float32))
            if hasattr(self.model, 'encoder'):
                _, z, _ = self.model(X_tensor)
                z_np = z.cpu().numpy()
            else:
                z_np = X
        
        # Create background
        if background is None:
            background = shap.kmeans(z_np, k=min(50, len(z_np)))
        
        # Explain classifier predictions
        explainer = shap.KernelExplainer(
            lambda z_vals: self.clf.predict_proba(z_vals)[:,1],
            background
        )
        shap_values = explainer.shap_values(z_np, nsamples=nsamples)
        
        # Map to feature space if decoder available
        if self.decoder_weight_matrix is not None:
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values[0])
            feat_shap = np.dot(self.decoder_weight_matrix, shap_values.T).T
            return feat_shap, shap_values
        else:
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            return shap_values, shap_values
    
    def explain_lime(self, X, num_features=10):
        """
        LIME explanation
        Returns: lime_values, feature_importance
        """
        if lime_tabular is None:
            raise ImportError("LIME not installed. Install with: pip install lime")
        
        # Generate embeddings
        with torch.no_grad():
            X_tensor = torch.from_numpy(X.astype(np.float32))
            if hasattr(self.model, 'encoder'):
                _, z, _ = self.model(X_tensor)
                z_np = z.cpu().numpy()
            else:
                z_np = X
        
        # Create LIME explainer on embeddings
        explainer = lime_tabular.LimeTabularExplainer(
            z_np,
            mode='classification',
            feature_names=[f'emb_{i}' for i in range(z_np.shape[1])],
            discretize_continuous=True
        )
        
        # Explain first sample
        explanation = explainer.explain_instance(
            z_np[0],
            lambda z_vals: self.clf.predict_proba(z_vals)[:, 1],
            num_features=num_features,
            top_labels=1
        )
        
        # Get feature importance
        lime_values = np.zeros(z_np.shape[1])
        for feat_idx, weight in explanation.as_list():
            idx = int(feat_idx.split('_')[1])
            lime_values[idx] = weight
        
        # Map to feature space if decoder available
        if self.decoder_weight_matrix is not None:
            feat_lime = np.dot(self.decoder_weight_matrix, lime_values).flatten()
            return feat_lime, lime_values
        else:
            return lime_values, lime_values
    
    def integrated_gradients(self, X, baseline=None, steps=50):
        """
        Integrated Gradients for feature attribution
        Returns: ig_values, feature_importance
        """
        if not isinstance(self.model, torch.nn.Module):
            raise ValueError("Integrated Gradients requires PyTorch model")
        
        X_tensor = torch.from_numpy(X.astype(np.float32))
        if baseline is None:
            baseline = torch.zeros_like(X_tensor)
        else:
            baseline = torch.from_numpy(baseline.astype(np.float32))
        
        # Set to eval
        self.model.eval()
        
        # Create interpolated inputs
        alphas = torch.linspace(0, 1, steps + 1)
        grad_sum = torch.zeros_like(X_tensor)
        
        for alpha in alphas[1:-1]:
            interp = baseline + alpha * (X_tensor - baseline)
            interp.requires_grad_(True)
            
            # Forward pass
            recon, z, _ = self.model(interp)
            
            # Use classifier prediction as target
            with torch.no_grad():
                z_np = z.cpu().numpy()
                target = self.clf.predict_proba(z_np)[:, 1]
            
            # Backward
            z.backward(torch.ones_like(z) * target[0])
            grad_sum += interp.grad
        
        # Final gradient
        ig_values = grad_sum / steps
        ig_values = ig_values.cpu().numpy()
        
        # Average over batch
        ig_values = np.mean(ig_values, axis=0)
        
        return ig_values, ig_values
    
    def explain(self, X, method='shap', **kwargs):
        """
        Unified explain interface
        Returns: (feature_importance, embedding_importance)
        """
        if method.lower() == 'shap':
            return self.explain_shap(X, **kwargs)
        elif method.lower() == 'lime':
            return self.explain_lime(X, **kwargs)
        elif method.lower() in ['ig', 'integrated_gradients']:
            return self.integrated_gradients(X, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from: shap, lime, ig")

def compute_fairness_metrics(y_true, y_pred, demographics):
    """
    Compute fairness metrics across demographic groups
    Args:
        y_true: true labels
        y_pred: predicted labels
        demographics: dict with keys like 'age', 'gender'
    
    Returns:
        fairness_metrics: dict with group-wise accuracy, precision, recall
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {}
    
    # Overall metrics
    metrics['overall'] = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Age groups
    if 'age' in demographics:
        age_groups = ['<40', '40-60', '>60']
        for group in age_groups:
            if group == '<40':
                mask = demographics['age'] < 40
            elif group == '40-60':
                mask = (demographics['age'] >= 40) & (demographics['age'] <= 60)
            else:
                mask = demographics['age'] > 60
            
            if np.sum(mask) > 0:
                metrics[f'age_{group}'] = {
                    'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                    'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
                    'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
                    'f1': f1_score(y_true[mask], y_pred[mask], zero_division=0),
                    'count': int(np.sum(mask))
                }
    
    # Gender groups
    if 'gender' in demographics:
        genders = np.unique(demographics['gender'])
        for gender in genders:
            mask = demographics['gender'] == gender
            if np.sum(mask) > 0:
                metrics[f'gender_{gender}'] = {
                    'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                    'precision': precision_score(y_true[mask], y_pred[mask], zero_division=0),
                    'recall': recall_score(y_true[mask], y_pred[mask], zero_division=0),
                    'f1': f1_score(y_true[mask], y_pred[mask], zero_division=0),
                    'count': int(np.sum(mask))
                }
    
    # Demographic parity difference
    if 'gender' in demographics:
        genders = list(np.unique(demographics['gender']))
        if len(genders) == 2:
            mask1 = demographics['gender'] == genders[0]
            mask2 = demographics['gender'] == genders[1]
            if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                rate1 = np.mean(y_pred[mask1])
                rate2 = np.mean(y_pred[mask2])
                metrics['demographic_parity_diff'] = abs(rate1 - rate2)
    
    return metrics

def compute_embeddings_tsne(embeddings, labels, perplexity=30, n_components=2):
    """
    Compute t-SNE visualization of embeddings
    """
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        return embeddings_2d.tolist()
    except ImportError:
        # Fallback to PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        return embeddings_2d.tolist()

