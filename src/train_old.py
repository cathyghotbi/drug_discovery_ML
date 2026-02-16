# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# #from featurization import smiles_to_morgan
# from featurization import smiles_to_features

# # how to run: PS C:\Users\Cathy\PycharmProjects\pythonProject1\drug_discovery_ML> python src/train.py

# # Load dataset
# import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_PATH = os.path.join(BASE_DIR, "data", "BBBP.csv")

# df = pd.read_csv(DATA_PATH)


# # Convert SMILES to fingerprints
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

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )

# # Train model
# model = RandomForestClassifier(n_estimators=200, random_state=42)
# model.fit(X_train, y_train)

# # Save trained model
# RESULTS_DIR = os.path.join(
#     os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
#     "results"
# )

# os.makedirs(RESULTS_DIR, exist_ok=True)

# model_path = os.path.join(RESULTS_DIR, "rf_bbbp_model.pkl")
# joblib.dump(model, model_path)

# """
# Python:

# ✅ Creates files

# ❌ Does not create missing directories

# Every production ML pipeline explicitly creates output folders."""
# # joblib.dump(model, "../results/rf_bbbp_model.pkl")
