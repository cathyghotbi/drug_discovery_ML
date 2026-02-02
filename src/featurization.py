from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np


def compute_descriptors(mol):
    """
    Compute basic physicochemical descriptors for a molecule.
    Returns a numpy array of floats.
    """
    return np.array([
        Descriptors.MolWt(mol),                     # Molecular weight
        Descriptors.MolLogP(mol),                   # LogP
        Descriptors.NumHDonors(mol),                # H-bond donors
        Descriptors.NumHAcceptors(mol),             # H-bond acceptors
        Descriptors.TPSA(mol),                      # Polar surface area
        Descriptors.NumRotatableBonds(mol),         # Flexibility
        Descriptors.RingCount(mol),                 # Ring count
        Descriptors.HeavyAtomCount(mol),            # Heavy atoms
    ], dtype=float)


def smiles_to_features(smiles, radius=2, n_bits=2048):
    """
    Convert a SMILES string into a combined feature vector:
    [Morgan fingerprint | molecular descriptors]
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits
    )
    fp = np.array(fp, dtype=int)

    # Molecular descriptors
    desc = compute_descriptors(mol)

    # Concatenate
    features = np.concatenate([fp, desc])

    return features
