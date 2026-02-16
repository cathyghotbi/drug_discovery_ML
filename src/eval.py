# ==========================================================
# eval.py
# how to run: PS C:\Users\Cathy\PycharmProjects\pythonProject1\drug_discovery_ML> python src/eval_old.py
# ==========================================================

"""
Research-Grade Evaluation Pipeline

This script performs:

1) Loads the trained QSAR / Random Forest model
2) Loads the held-out test set
3) Computes:
      - ROC-AUC
      - Confusion Matrix

IMPORTANT:
- The model was trained on the combination of training + validation data
  (80% of the total dataset).
- The test set (20%) has never been seen by the model.
- Evaluation is strictly performed on this held-out test set to
  measure unbiased generalization performance.
"""

# ==============================
# Imports
# ==============================

import os
import numpy as np
import joblib

from sklearn.metrics import roc_auc_score, confusion_matrix


# ==============================
# Define Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Path to trained model
MODEL_PATH = os.path.join(RESULTS_DIR, "rf_bbbp_model.pkl")

# Path to saved held-out test set
TEST_PATH = os.path.join(RESULTS_DIR, "test_set.npz")


# ==============================
# Load Trained Model
# ==============================

"""
Loads the serialized Random Forest model.

If this fails:
    → train.py was not executed or model file is missing
"""

model = joblib.load(MODEL_PATH)


# ==============================
# Load Held-Out Test Set
# ==============================

"""
The test set was split and saved in train.py:

- 20% of the total dataset
- Never used in training or cross-validation
- Provides unbiased estimate of model performance
"""

data = np.load(TEST_PATH)

X_test = data["X_test"]   # Feature matrix
y_test = data["y_test"]   # True labels


# ==============================
# Step 1 — Predict Probabilities
# ==============================

"""
ROC-AUC requires predicted probabilities rather than class labels.

predict_proba returns:
    - column 0 → probability of class 0 (non-permeable)
    - column 1 → probability of class 1 (BBBP permeable)

We select column 1 to evaluate BBBP permeability ranking.
"""

y_probs = model.predict_proba(X_test)[:, 1]


# ==============================
# Step 2 — Compute ROC-AUC
# ==============================

"""
ROC-AUC measures ranking ability of the classifier:

- 0.5 → random guessing
- 1.0 → perfect separation of classes

This metric is commonly used in QSAR and drug discovery papers.
"""

auc = roc_auc_score(y_test, y_probs)


# ==============================
# Step 3 — Compute Confusion Matrix
# ==============================

"""
Confusion matrix layout:

                Predicted
               0        1
True 0       TN       FP
True 1       FN       TP

Useful for:
- Sensitivity / Recall
- Specificity
- False positive rate
- Precision
"""

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)


# ==============================
# Step 4 — Print Results
# ==============================

print("=================================")
print("Final Test ROC-AUC:", auc)
print("Confusion Matrix:\n", cm)
print("=================================")

"""
Notes:

- The model was trained on the combination of training + validation data.
- The held-out test set ensures an unbiased evaluation.
- Do NOT use training or validation data for final performance reporting.
"""


# Output:
# =================================
# Final Test ROC-AUC: 0.9420572916666666
# Confusion Matrix:
#  [[ 62  34]
#  [  7 305]]
# =================================
