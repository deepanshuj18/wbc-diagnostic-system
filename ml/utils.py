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

def prepare_data(test_size=0.2, val_size=0.1, random_state=42):
    X, y, feature_names = load_wbc()
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=random_state)
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
    X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), scaler, feature_names
