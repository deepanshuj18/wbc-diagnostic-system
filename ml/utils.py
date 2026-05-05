# ml/utils.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_wbc():
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=['target'])
    y = data.frame['target']
    feature_names = list(X.columns)
    return X, y, feature_names

def prepare_data(test_size=0.2, val_size=0.1, cal_size=0.1, random_state=42, include_cal=False):
    """
    Split data into train/val/test (default) or train/val/cal/test (if include_cal=True).
    
    When include_cal=True:
      - test_size: fraction of total for test set (default 0.2)
      - val_size: fraction of remaining for validation (default 0.1)
      - cal_size: fraction of remaining for calibration (default 0.1)
      - rest goes to training
    
    Returns:
      If include_cal=False: (train, val, test, scaler, feature_names) — backward compatible
      If include_cal=True:  (train, val, cal, test, scaler, feature_names)
    """
    X, y, feature_names = load_wbc()
    
    # First split: separate test set
    X_train_rest, X_test, y_train_rest, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    
    if include_cal:
        # Second split: separate val+cal from train
        combined_holdout = val_size + cal_size
        X_train, X_val_cal, y_train, y_val_cal = train_test_split(
            X_train_rest, y_train_rest, test_size=combined_holdout,
            stratify=y_train_rest, random_state=random_state)
        
        # Third split: separate val from cal (50/50 of the holdout)
        cal_frac = cal_size / combined_holdout
        X_val, X_cal, y_val, y_cal = train_test_split(
            X_val_cal, y_val_cal, test_size=cal_frac,
            stratify=y_val_cal, random_state=random_state)
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_rest, y_train_rest, test_size=val_size,
            stratify=y_train_rest, random_state=random_state)
    
    # Scale all splits using training data statistics only
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    
    if include_cal:
        X_cal_s = pd.DataFrame(scaler.transform(X_cal), columns=X_cal.columns, index=X_cal.index)
        return (X_train_s, y_train), (X_val_s, y_val), (X_cal_s, y_cal), (X_test_s, y_test), scaler, feature_names
    
    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), scaler, feature_names

