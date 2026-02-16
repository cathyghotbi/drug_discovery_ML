# 🧪 Drug Discovery with Machine Learning

### Predicting Blood–Brain Barrier Permeability (BBBP)

This project implements an end-to-end **machine learning pipeline for drug discovery**, focused on predicting **blood–brain barrier (BBB) permeability** from molecular structure.

The workflow follows standard **QSAR (Quantitative Structure–Activity Relationship)** modeling practices used in real pharmaceutical and cheminformatics research.

---

## Background

This project implements a **machine learning model** to predict **Blood–Brain Barrier (BBB) penetration** of small molecules from their SMILES representations.

The **blood–brain barrier (BBB)** is a highly selective biological barrier that protects the brain by restricting which compounds can pass from the bloodstream into the central nervous system (CNS).
Accurately predicting BBB penetration is critical in **drug discovery**, especially for:

* CNS drug development (e.g. Alzheimer’s, Parkinson’s, epilepsy)
* Avoiding unwanted brain side effects for peripheral drugs

The model is trained on the **BBBP dataset** using RDKit-based molecular features and a Random Forest classifier.

---

## What is BBB Penetration?

**BBB penetration (BBBP)** refers to a molecule’s ability to cross the blood–brain barrier.

* **BBB+ (penetrant):** molecule can reach the brain
* **BBB− (non-penetrant):** molecule is blocked by the BBB

The BBB strongly favors molecules that are:

* Small
* Moderately lipophilic
* Low in polarity
* Capable of forming few hydrogen bonds

---

## Molecular Features Used

Each molecule is featurized using a **combination of structural and physicochemical descriptors**.

### 1. Morgan Fingerprints (ECFP)

* Circular fingerprints capturing molecular substructures
* Radius = 2
* 2048-bit binary vector
* Encodes *what chemical fragments are present*

These features allow the model to learn **structure–activity relationships (SAR)**.

---

### 2. Physicochemical Descriptors

These descriptors encode global molecular properties that are strongly correlated with BBB permeability:

| Feature                      | Description                             | Relevance to BBB                                 |
| ---------------------------- | --------------------------------------- | ------------------------------------------------ |
| **Molecular Weight (MolWt)** | Size of the molecule                    | Smaller molecules cross BBB more easily          |
| **LogP**                     | Lipophilicity (octanol/water partition) | Higher LogP favors membrane crossing             |
| **TPSA**                     | Topological Polar Surface Area          | Lower TPSA improves BBB penetration              |
| **HBD**                      | Hydrogen Bond Donors                    | Fewer donors favor BBB crossing                  |
| **HBA**                      | Hydrogen Bond Acceptors                 | Fewer acceptors favor BBB crossing               |
| **Rotatable Bonds**          | Molecular flexibility                   | Too much flexibility reduces permeability        |
| **Ring Count**               | Number of rings                         | Often associated with rigidity and lipophilicity |
| **Heavy Atom Count**         | Non-hydrogen atoms                      | Proxy for molecular size                         |

---

## Final Feature Vector

For each molecule, the final feature vector is:

```
[Morgan Fingerprint (2048 bits) | Physicochemical Descriptors (8 floats)]
```

Total feature dimension:

```
2056 features per molecule
```

---

## Model

* **Algorithm:** Random Forest Classifier
* **Input:** Combined molecular features
* **Output:** Probability of BBB penetration
* **Evaluation metric:** ROC-AUC

---

## Dependencies

* Python
* RDKit
* NumPy
* Pandas
* scikit-learn
* joblib

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

### ▶️ Training the model (train.py)

```bash
python -m src.train
```

This step:

* Loads the BBBP dataset
* Convert SMILES → 2056-length feature vectors (2048 Morgan + 8 descriptors).
* Trains a Random Forest classifier
* Split dataset:
** Training: 64%
** Validation: 16%
** Held-out Test: 20%
* Perform 5-fold cross-validation on training set.
* Train on training set and evaluate on validation set.
* Retrain on full training+validation set.
* Saves the trained model to:

  ```
  results/rf_bbbp_model.pkl
  Trained model → results/rf_bbbp_model.pkl
  Held-out test set → results/test_set.npz (⚠️ The held-out test set is created only here. Never regenerate it in eval.py.)
  ```

---

### 📊 Evaluate the model

```bash
python -m src.eval
```

This step:

* Load the trained Random Forest model (rf_bbbp_model.pkl).
* Load held-out test set (test_set.npz).
* Predict BBB permeability probabilities.
* Compute metrics:
** ROC-AUC — ranking quality
** Confusion matrix — classification errors
* Reports ROC-AUC and confusion matrix

**Recent Example output with separated train ans test data :**
```
Final Test ROC-AUC: 0.94
Confusion Matrix:
[[ 62  34]
 [  7 305]]
```
✅ Evaluation is strictly on held-out test set to avoid data leakage.


**Previous Example output with training and evaluating on the same :**

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


## References:
Morgan fingerprint: to represent chemical molecules as a binary or counter vector of fixed lengthIt encodes the structure of a molecule based on the local environments of the atom
https://blog.dnanexus.com/hs-fs/hubfs/Imported_Blog_Media/Morgan-Algorithm-1024x653.png?width=1024&height=653&name=Morgan-Algorithm-1024x653.png

---

Recent result of trainig with splitted train and test data set:

<img width="287" height="79" alt="{91B10DE9-B3BC-4DBA-85C4-E96FD2178026}" src="https://github.com/user-attachments/assets/d8612f5d-c338-4784-b8d3-08bb02a4ee72" />


<img width="224" height="82" alt="{75597C33-296C-45BE-A14E-EFAC86A34E28}" src="https://github.com/user-attachments/assets/242f488b-6e28-4800-b898-3f7ef0f52102" />



Previous result of trainig with Morgan fingerprint:

<img width="320" height="89" alt="{FE51FC70-6EBC-4F60-9F8E-3DC3161542E3}" src="https://github.com/user-attachments/assets/493804c3-d8f8-4fb0-839c-5a6ecebf09ac" />

result of trainig with Morgan fingerprint together with physicochemical descriptors:

<img width="144" height="58" alt="{1F2376BF-B177-499A-8247-C280D4C2C11E}" src="https://github.com/user-attachments/assets/f78d7906-4c93-41d1-8c90-3fcfde1b372a" />




