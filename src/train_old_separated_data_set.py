# # ==========================================================
# # train.py
# # how to run: PS C:\Users\Cathy\PycharmProjects\pythonProject1\drug_discovery_ML> python src/train_old.py
# # ==========================================================
# """
# Research-Grade QSAR Training Pipeline for BBBP Prediction

# This script performs:

# 1) Load BBBP dataset
# 2) Convert SMILES → numerical feature vectors
# 3) Split dataset into:
#       - Training set (64% of total data)
#       - Validation set (16% of total data, used for hyperparameter tuning)
#       - Held-out test set (20% of total data, never touched until final evaluation)
# 4) Perform 5-fold cross-validation on the training set
# 5) Train model on training data and evaluate on validation set
# 6) Retrain final model on full training+validation data (80%) before testing
# 7) Save:
#       - Trained model
#       - Held-out test set (X_test, y_test) for evaluation in eval.py

# IMPORTANT:
# - The test set is created ONLY here and must NEVER be used during training or cross-validation.
# - Validation data is used strictly for monitoring performance and hyperparameter tuning.
# - This structure avoids data leakage and ensures unbiased evaluation.
# """

# # ==============================
# # Imports
# # ==============================

# import os                                  # For file path handling
# import pandas as pd                        # For reading CSV data
# import numpy as np                         # For numerical arrays
# import joblib                              # For saving trained model

# from sklearn.model_selection import (
#     train_test_split,                      # For splitting data
#     StratifiedKFold,                       # For stratified cross-validation
#     cross_val_score                        # For running cross-validation
# )
# from sklearn.ensemble import RandomForestClassifier  # ML model

# from featurization import smiles_to_features        # Custom feature generator
# from sklearn.metrics import roc_auc_score

# # ==============================
# # Define Project Paths
# # ==============================

# # Get project root directory
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Construct path to BBBP dataset
# DATA_PATH = os.path.join(BASE_DIR, "data", "BBBP.csv")

# # Directory where models & artifacts are stored
# RESULTS_DIR = os.path.join(BASE_DIR, "results")

# # Create results directory if it doesn't exist
# os.makedirs(RESULTS_DIR, exist_ok=True)


# # ==============================
# # Load Dataset
# # ==============================

# # Read BBBP dataset into pandas DataFrame
# df = pd.read_csv(DATA_PATH)


# # ==============================
# # Convert SMILES to Features
# # ==============================

# X = []   # Will store molecular feature vectors
# y = []   # Will store labels (0 or 1)

# """
# Machine learning models cannot use SMILES strings directly.
# We convert each molecule into a numeric feature vector
# using a custom featurization function.
# """

# # Loop over each molecule in dataset
# for smiles, label in zip(df["smiles"], df["p_np"]):

#     # Convert SMILES string to feature vector
#     features = smiles_to_features(smiles)

#     # Skip molecules if featurization fails
#     if features is not None:
#         X.append(features)
#         y.append(label)

# # Convert lists into NumPy arrays (required by sklearn)
# X = np.array(X)
# y = np.array(y)


# # ==============================
# # Step 1 — Train / Test Split
# # ==============================

# # Split dataset into:
# # 80% training+validation
# # 20% held-out test set (never touched until final evaluation)

# X_train_full, X_test, y_train_full, y_test = train_test_split(
#     X,
#     y,
#     test_size=0.2,        # 20% test set
#     stratify=y,           # Preserve class distribution
#     random_state=42       # Reproducibility
# )


# # ==============================
# # Step 2 — Train / Validation Split
# # ==============================

# # Further split training data:
# # 80% training
# # 20% validation

# X_train, X_val, y_train, y_val = train_test_split(
#     X_train_full,
#     y_train_full,
#     test_size=0.2,
#     stratify=y_train_full,
#     random_state=42
# )

# """
# At this point:

# Total dataset:
#     64% → training
#     16% → validation
#     20% → test

# The test set remains completely untouched.
# """


# # ==============================
# # Step 3 — Define Model
# # ==============================

# model = RandomForestClassifier(
#     n_estimators=200,
#     random_state=42,
#     n_jobs=-1
# )


# # ==============================
# # Step 4 — Cross-Validation (Training Only)
# # ==============================

# """
# Cross-validation is performed ONLY on X_train.

# Why?
# Because validation and test sets must remain unseen
# during model selection and tuning.
# """

# cv = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )

# cv_scores = cross_val_score(
#     model,
#     X_train,
#     y_train,
#     cv=cv,
#     scoring="roc_auc"
# )

# print("=================================")
# print("Cross-Validation ROC-AUC (mean):", cv_scores.mean())
# print("Cross-Validation ROC-AUC (std):", cv_scores.std())
# print("=================================")


# # ==============================
# # Step 5 — Train Model on Training Set
# # ==============================

# model.fit(X_train, y_train)


# # ==============================
# # Step 6 — Validation Evaluation
# # ==============================

# """
# Validation set is used for:
# - Monitoring performance
# - Hyperparameter tuning
# - Detecting overfitting

# This is NOT the final metric.
# """

# val_probs = model.predict_proba(X_val)[:, 1]
# val_auc = roc_auc_score(y_val, val_probs)

# print("Validation ROC-AUC:", val_auc)


# # ==============================
# # Step 7 — Retrain on Full Training Data
# # ==============================

# """
# After tuning decisions are finalized,
# we retrain using BOTH training and validation data
# to maximize learning before final testing.
# """

# model.fit(X_train_full, y_train_full)


# # ==============================
# # Step 8 — Save Model
# # ==============================

# MODEL_PATH = os.path.join(RESULTS_DIR, "rf_bbbp_model.pkl")
# joblib.dump(model, MODEL_PATH)


# # ==============================
# # Step 9 — Save Held-Out Test Set
# # ==============================

# TEST_PATH = os.path.join(RESULTS_DIR, "test_set.npz")
# np.savez(TEST_PATH, X_test=X_test, y_test=y_test)


# print("Training complete.")
# print("Model saved to:", MODEL_PATH)
# print("Test set saved to:", TEST_PATH)

# # Output:
# # Cross-Validation ROC-AUC (mean): 0.905951260798971
# # Cross-Validation ROC-AUC (std): 0.014198717520972449

