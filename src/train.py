# ==========================================================
# train.py
# ==========================================================
"""
Research-Grade QSAR Training Pipeline for BBBP Prediction

This script performs:

1) Load BBBP dataset
2) Convert SMILES → numerical feature vectors (Morgan fingerprints + physicochemical descriptors)
3) Split dataset into:
      - Training set (64% of total data)
      - Validation set (16% of total data, used for hyperparameter tuning)
      - Held-out test set (20% of total data, never touched until final evaluation)
4) Perform 5-fold cross-validation on the training set
5) Train model on training data and evaluate on validation set
6) Retrain final model on full training+validation data (80%) before testing
7) Save:
      - Trained model
      - Held-out test set (X_test, y_test) for evaluation in eval.py

IMPORTANT:
- The test set is created ONLY here and must NEVER be used during training or cross-validation.
- Validation data is used strictly for monitoring performance and hyperparameter tuning.
- This structure avoids data leakage and ensures unbiased evaluation.
"""

# ==============================
# Imports
# ==============================

import os                                  # File path handling
import pandas as pd                        # Reading CSV
import numpy as np                         # Numerical arrays
import joblib                              # Save model
from rdkit import Chem                      # Molecular parsing
from rdkit.Chem import AllChem, Descriptors

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ==============================
# Define Project Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "BBBP.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================
# Load Dataset
# ==============================

df = pd.read_csv(DATA_PATH)

# ==============================
# Convert SMILES to Features
# ==============================

X = []   # Molecular feature vectors
y = []   # Labels (0/1)

def featurize_molecule(smiles):
    """
    Convert a SMILES string to combined features:
    - Morgan fingerprint (2048 bits)
    - Molecular descriptors: MolWt, LogP, TPSA, HBD, HBA, Rotatable Bonds, Ring Count, Heavy Atom Count
    Returns a 2056-length vector
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fp_array = np.zeros((2048,), dtype=int)
    AllChem.DataStructs.ConvertToNumpyArray(fp, fp_array)

    # Physicochemical descriptors
    descriptors = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        mol.GetNumHeavyAtoms()
    ], dtype=float)

    # Concatenate fingerprint + descriptors
    features = np.concatenate([fp_array, descriptors])
    return features

# Loop over dataset
for smiles, label in zip(df["smiles"], df["p_np"]):
    features = featurize_molecule(smiles)
    if features is not None:
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# ==============================
# Step 1 — Train / Test Split
# ==============================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ==============================
# Step 2 — Train / Validation Split
# ==============================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=42
)

# ==============================
# Step 3 — Define Model
# ==============================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# ==============================
# Step 4 — Cross-Validation (Training Only)
# ==============================

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

print("=================================")
print("Cross-Validation ROC-AUC (mean):", cv_scores.mean())
print("Cross-Validation ROC-AUC (std):", cv_scores.std())
print("=================================")

# ==============================
# Step 5 — Train Model on Training Set
# ==============================

model.fit(X_train, y_train)

# ==============================
# Step 6 — Validation Evaluation
# ==============================

val_probs = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_probs)
print("Validation ROC-AUC:", val_auc)

# ==============================
# Step 7 — Retrain on Full Training Data
# ==============================

model.fit(X_train_full, y_train_full)

# ==============================
# Step 8 — Save Model
# ==============================

MODEL_PATH = os.path.join(RESULTS_DIR, "rf_bbbp_model.pkl")
joblib.dump(model, MODEL_PATH)

# ==============================
# Step 9 — Save Held-Out Test Set
# ==============================

TEST_PATH = os.path.join(RESULTS_DIR, "test_set.npz")
np.savez(TEST_PATH, X_test=X_test, y_test=y_test)

print("Training complete.")
print("Model saved to:", MODEL_PATH)
print("Test set saved to:", TEST_PATH)


# Output:
# =================================
# Cross-Validation ROC-AUC (mean): 0.9051324815510657
# Cross-Validation ROC-AUC (std): 0.017106155923554387
# =================================
# Validation ROC-AUC: 0.921974025974026
# Training complete.
