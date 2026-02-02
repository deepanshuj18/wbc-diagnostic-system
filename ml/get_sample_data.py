from sklearn.datasets import load_breast_cancer
import numpy as np
import json

# Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=['target'])
y = data.frame['target']

# Find samples
malignant_indices = np.where(y == 0)[0]
benign_indices = np.where(y == 1)[0]

# Get samples
malignant_samples = [X.iloc[idx].values.tolist() for idx in malignant_indices[:3]]
benign_samples = [X.iloc[idx].values.tolist() for idx in benign_indices[:3]]

# Print as JSON for easy copy
print("MALIGNANT_SAMPLES:")
print(json.dumps(malignant_samples, indent=2))
print("\nBENIGN_SAMPLES:")
print(json.dumps(benign_samples, indent=2))

# Also print as test commands
print("\n\n=== CURL COMMANDS ===")
for i, sample in enumerate(malignant_samples[:2], 1):
    print(f'\n# Malignant {i}')
    print(f'curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{{\\"features\\": {sample}}}"')

for i, sample in enumerate(benign_samples[:2], 1):
    print(f'\n# Benign {i}')
    print(f'curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{{\\"features\\": {sample}}}"')

