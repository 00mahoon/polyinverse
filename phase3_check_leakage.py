import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
import numpy as np

RDLogger.DisableLog('rdApp.*')

df = pd.read_csv('polymer_data/density_augmented_v2.csv')

def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

print("Converting to canonical SMILES...")
df['canonical'] = df['SMILES'].apply(canonical_smiles)
df = df[df['canonical'].notna()]

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

train_canonical = set(train_data['canonical'])
test_canonical = set(test_data['canonical'])

overlap = train_canonical & test_canonical
print(f"Train samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")
print(f"Overlapping molecules: {len(overlap)}")
print(f"Overlap ratio: {len(overlap)/len(test_canonical)*100:.1f}%")
