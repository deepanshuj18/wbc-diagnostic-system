# ml/shap_utils.py
import shap
import joblib
import numpy as np
import torch

def compute_shap_for_embedding(clf, embed_X, background=None):
    # Use KernelExplainer on the classifier operating on embeddings
    # clf: scikit-learn classifier; embed_X: numpy array (n_samples, embed_dim)
    if background is None:
        background = shap.sample(embed_X, nsamples=min(50, len(embed_X)))
    explainer = shap.KernelExplainer(lambda z: clf.predict_proba(z)[:,1], background)
    shap_values = explainer.shap_values(embed_X, nsamples=50)
    return shap_values  # list or array

def map_embedding_shap_to_features(shap_embed, decoder_weight_matrix):
    # decoder_weight_matrix shape (output_dim, embed_dim)
    # shap_embed shape (n_samples, embed_dim)
    # Map embedding-level SHAP to original features by linear projection:
    # feature_shap = decoder_weight_matrix @ shap_embed.T  -> (output_dim, n_samples)
    feat_shap = np.dot(decoder_weight_matrix, shap_embed.T).T
    return feat_shap  # (n_samples, output_dim)
