from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    """
    Convert a SMILES string into a Morgan fingerprint.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)



"""
Chem module from RDKit: This module allows us to work with molecules, atoms, and bonds
AllChem from RDKit: contains advanced chemistry functions, including fingerprint generation
NumPy for numerical array handling: Machine learning models require numerical inputs (arrays, matrices)

In function smiles_to_morgan, converts chemical structure → ML-ready numbers , called Molecular featurization
smiles: a string representing a molecule (e.g. "CCO")
radius: how far the fingerprint “looks” around each atom
n_bits: length of the fingerprint vector

mol = Chem.MolFromSmiles(smiles):
Parses the SMILES string into an RDKit Mol object
RDKit can only perform chemistry operations on Mol objects

If RDKit couldn’t understand the molecule, return None to exit the function safely, Prevents pipeline from crashing

fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits):
What this does:
Generates a Morgan fingerprint (also known as ECFP)
Encodes the molecule’s structure as a binary vector

Breaking it down:
mol → molecule to encode
radius → how many bonds away from each atom to consider
nBits → length of the fingerprint (e.g. 2048)

Chemistry intuition:
The fingerprint captures local atomic environments
Similar molecules → similar fingerprints

return np.array(fp):
What this does:
Converts the RDKit fingerprint object into a NumPy array
NumPy arrays are required by scikit-learn models

Result:
Output shape: (2048,)
Values: 0 or 1
"""
