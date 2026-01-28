# 🧪 Drug Discovery with Machine Learning

### Predicting Blood–Brain Barrier Permeability (BBBP)

This project implements an end-to-end **machine learning pipeline for drug discovery**, focused on predicting **blood–brain barrier (BBB) permeability** from molecular structure.

The workflow follows standard **QSAR (Quantitative Structure–Activity Relationship)** modeling practices used in real pharmaceutical and cheminformatics research.

---

## 📌 Project Overview

* **Task:** Binary classification (BBB permeable vs non-permeable)
* **Input:** Molecular structures represented as SMILES
* **Features:** Morgan (circular) fingerprints
* **Model:** Random Forest classifier
* **Evaluation metrics:** ROC-AUC, confusion matrix
* **Tools:** Python, RDKit, scikit-learn

---

## 🧬 Dataset

This project uses the **BBBP (Blood–Brain Barrier Penetration) dataset** from **MoleculeNet**, a widely used benchmark collection for molecular machine learning.

### Dataset source

* **MoleculeNet (via DeepChem)**
* Original publication: *Wu et al., MoleculeNet: A Benchmark for Molecular Machine Learning*
* Publicly available dataset

### Direct download link

```
https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
```

### Dataset details

* ~2,000 small molecules
* Binary labels indicating BBB permeability
* Frequently used to benchmark QSAR and molecular ML models

### File location in this project

```
data/BBBP.csv
```

### Columns

* `smiles` — molecular structure in SMILES format
* `p_np` — label

  * `1`: BBB permeable
  * `0`: not BBB permeable

---

## 🗂️ Project Structure

```
drug_discovery_ML/
├── data/
│   └── BBBP.csv
├── notebooks/
│   └── exploration.ipynb
├── results/
│   ├── figures/
│   ├── metrics.txt
│   └── rf_bbbp_model.pkl
├── src/
│   ├── featurization.py
│   ├── train.py
│   └── eval.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

* **Windows**

```bash
venv\Scripts\activate
```

* **macOS / Linux**

```bash
source venv/bin/activate
```

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If RDKit installation fails with pip, install via conda:

```bash
conda install -c conda-forge rdkit
```

---

## 🚀 Usage

All commands should be run **from the project root**.

---

### ▶️ Train the model

```bash
python -m src.train
```

This step:

* Loads the BBBP dataset
* Converts SMILES to Morgan fingerprints
* Trains a Random Forest classifier
* Saves the trained model to:

  ```
  results/rf_bbbp_model.pkl
  ```

---

### 📊 Evaluate the model

```bash
python -m src.eval
```

This step:

* Loads the trained model
* Recomputes molecular fingerprints
* Reports ROC-AUC and confusion matrix

**Example output:**

```
ROC-AUC: 0.99
Confusion Matrix:
[[ 433   46]
 [   7 1553]]
```

> ⚠️ **Note:**
> Current evaluation is performed on the full dataset (training data).
> Proper train/test splitting and cross-validation will be added in future iterations.

---

## 🧠 Methodology

### Molecular Representation

* SMILES parsed using RDKit
* Morgan fingerprints (radius = 2, 2048 bits)

### Machine Learning

* Random Forest classifier
* Chosen as a strong, interpretable baseline commonly used in QSAR modeling

### Evaluation

* ROC-AUC for ranking performance
* Confusion matrix for classification error analysis

---

## 🔬 Current Limitations

* Evaluation on training data only (optimistic performance)
* No external test set
* No cross-validation
* Limited model interpretability analysis

These limitations are intentional at this stage and will be addressed in future updates.

---

## 🛠️ Planned Improvements

* Proper train/test split and held-out evaluation
* Cross-validation
* ROC curve and confusion matrix visualizations
* Model comparison (Logistic Regression, SVM)
* Deep learning models (Graph Neural Networks)
* Chemical interpretation of predictions

---

## 🤝 Collaboration & Acknowledgment

This project was developed **collaboratively** through an interactive learning and development process involving:

* **Cathy** — project owner, implementation, experimentation, and analysis
* **AI-assisted guidance** — support with project design, debugging, best practices, and documentation

All code was written, executed, debugged, and validated by the project owner, with guidance used as a learning and acceleration tool.

---

## 📚 References

* Wu, Z. et al. *MoleculeNet: A Benchmark for Molecular Machine Learning*
* RDKit: Open-source cheminformatics toolkit
* scikit-learn: Machine learning in Python

---

## 👤 Author

**Cathy**
Machine Learning & Drug Discovery

I asked AI to help me start learning python and machine learning in drug discovery

---


screen shot of result:
<img width="320" height="89" alt="{FE51FC70-6EBC-4F60-9F8E-3DC3161542E3}" src="https://github.com/user-attachments/assets/493804c3-d8f8-4fb0-839c-5a6ecebf09ac" />



