from sklearn.datasets import load_breast_cancer
import numpy as np

# Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=['target'])
y = data.frame['target']
feature_names = list(X.columns)

# Find malignant cases (target = 0) and benign cases (target = 1)
malignant_indices = np.where(y == 0)[0]
benign_indices = np.where(y == 1)[0]

print("# Malignant Samples")
print()
# Get 3 malignant samples
for i, idx in enumerate(malignant_indices[:3], 1):
    values = X.iloc[idx].values.tolist()
    print(f"## Malignant {i} (ID: {idx})")
    print(f"curl -X POST http://localhost:8000/predict -H \"Content-Type: application/json\" -d '{{\"features\": {values}}}'")
    print()

print()
print("# Benign Samples")
print()
# Get 3 benign samples
for i, idx in enumerate(benign_indices[:3], 1):
    values = X.iloc[idx].values.tolist()
    print(f"## Benign {i} (ID: {idx})")
    print(f"curl -X POST http://localhost:8000/predict -H \"Content-Type: application/json\" -d '{{\"features\": {values}}}'")
    print()

