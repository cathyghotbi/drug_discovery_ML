# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.metrics import roc_auc_score, confusion_matrix

# #from featurization import smiles_to_morgan
# from featurization import smiles_to_features

# import os

# # how to run: PS C:\Users\Cathy\PycharmProjects\pythonProject1\drug_discovery_ML> python src/eval.py
# """
# In this file, the evaluation loop used in real drug-discovery ML papers is implemented:
# Loading a trained QSAR model
# Recomputing molecular fingerprints
# Measuring model performance
# Producing research-grade metrics
# """

# # Load model
# #model = joblib.load("../results/rf_bbbp_model.pkl")

# # Load dataset
# #df = pd.read_csv("../data/BBBP.csv")

# """
# Finds project root reliably
# Builds safe, absolute paths
# """
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "BBBP.csv")
# RESULTS_DIR = os.path.join(BASE_DIR, "results")
# MODEL_PATH = os.path.join(RESULTS_DIR, "rf_bbbp_model.pkl")

# """
# Loads your trained Random Forest model from disk
# If this fails → model was not trained correctly.
# """
# # Load model
# model = joblib.load(MODEL_PATH)

# """
# Loads the BBBP dataset again

# Evaluation must use the same data format as training
# """
# df = pd.read_csv(DATA_PATH)

# # Generate features
# """
# Converts molecules → fingerprints
# Builds:
# X = feature matrix
# y = true labels
# """

# X = []
# y = []

# for smiles, label in zip(df["smiles"], df["p_np"]):
#     #fp = smiles_to_morgan(smiles)
#     fp = smiles_to_features(smiles)
#     if fp is not None:
#         X.append(fp)
#         y.append(label)

# X = np.array(X)
# y = np.array(y)

# # Predict probabilities
# """
# Predicts probability of BBBP penetration
# Uses probabilities (not class labels) for ROC-AUC
# """
# y_probs = model.predict_proba(X)[:, 1]

# # Evaluate
# """
# Computes ROC-AUC score
# Measures ranking quality of the model
# """
# auc = roc_auc_score(y, y_probs)

# """
# Converts probabilities → class predictions
# Computes confusion matrix
# """
# y_pred = model.predict(X)
# cm = confusion_matrix(y, y_pred)

# #cm = confusion_matrix(y, model.predict(X))

# print("ROC-AUC:", auc)
# print("Confusion Matrix:\n", cm)


# """
# result of running the file using only Morgan fingerprint:

# # how to run: PS r"Cathy\PycharmProjects\pythonProject1\drug_discovery_ML"> python src/eval.py


# ROC-AUC: 0.9920888871045448
# Confusion Matrix:
#  [[ 433   46]
#  [   7 1553]]

# ROC-AUC ≈ 0.99 means the model almost perfectly separates the two classes
#            Predicted 0   Predicted 1
# Actual 0        433           46
# Actual 1          7         1553

# Meaning:

# True negatives: 433
# False positives: 46
# False negatives: 7
# True positives: 1553

# So the model:
# Rarely misses BBB-penetrating molecules (only 7 FN)
# Makes some false positives (46 FP), which is common in drug discovery

# Right now, we evaluated on the same data we trained on.
# That means:
# The model has already “seen” these molecules
# The ROC-AUC is optimistically biased
# A score of 0.99 here does NOT mean real-world performance, it is Training-set performance




# result of running the file using only Morgan fingerprint together with physicochemical descriptors:

# ROC-AUC: 0.9938319683100476
# Confusion Matrix:
#  [[ 437   42]
#  [   9 1551]]

# True negatives: 437
# False positives: 42
# False negatives: 9
# True positives: 1551
# """
