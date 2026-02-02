from sklearn.datasets import load_breast_cancer
import numpy as np
import sys

# Load the dataset
data = load_breast_cancer(as_frame=True)
X = data.frame.drop(columns=['target'])
y = data.frame['target']
feature_names = list(X.columns)

# Find malignant cases (target = 0 for malignant)
malignant_indices = np.where(y == 0)[0]
# Find benign cases (target = 1 for benign)
benign_indices = np.where(y == 1)[0]

sys.stdout.write("=" * 100 + "\n")
sys.stdout.write("MALIGNANT SAMPLES (predicted as Malignant, target = 0)\n")
sys.stdout.write("=" * 100 + "\n\n")

# Get 3 malignant samples
for i, idx in enumerate(malignant_indices[:3], 1):
    malignant_sample = X.iloc[idx]
    values = malignant_sample.values.tolist()
    
    sys.stdout.write(f"MALIGNANT SAMPLE {i} (Patient ID: {idx})\n")
    sys.stdout.write("-" * 100 + "\n")
    sys.stdout.write("Patient Info:\n")
    sys.stdout.write("  Name: Malignant Sample " + str(i) + "\n")
    sys.stdout.write("  Age: 45\n")
    sys.stdout.write("  Sex: F\n\n")
    
    sys.stdout.write("30 WBC Features (copy in order):\n")
    for j, (name, value) in enumerate(malignant_sample.items(), 1):
        sys.stdout.write(f"  {j:2d}. {name:40s} : {value:.6f}\n")
    
    sys.stdout.write("\nValues as Array (for copy-paste):\n")
    sys.stdout.write("  " + str(values) + "\n")
    sys.stdout.write("\n" + "=" * 100 + "\n\n")

sys.stdout.write("=" * 100 + "\n")
sys.stdout.write("BENIGN SAMPLES (predicted as Benign, target = 1)\n")
sys.stdout.write("=" * 100 + "\n\n")

# Get 3 benign samples
for i, idx in enumerate(benign_indices[:3], 1):
    benign_sample = X.iloc[idx]
    values = benign_sample.values.tolist()
    
    sys.stdout.write(f"BENIGN SAMPLE {i} (Patient ID: {idx})\n")
    sys.stdout.write("-" * 100 + "\n")
    sys.stdout.write("Patient Info:\n")
    sys.stdout.write("  Name: Benign Sample " + str(i) + "\n")
    sys.stdout.write("  Age: 35\n")
    sys.stdout.write("  Sex: F\n\n")
    
    sys.stdout.write("30 WBC Features (copy in order):\n")
    for j, (name, value) in enumerate(benign_sample.items(), 1):
        sys.stdout.write(f"  {j:2d}. {name:40s} : {value:.6f}\n")
    
    sys.stdout.write("\nValues as Array (for copy-paste):\n")
    sys.stdout.write("  " + str(values) + "\n")
    sys.stdout.write("\n" + "=" * 100 + "\n\n")

sys.stdout.flush()

