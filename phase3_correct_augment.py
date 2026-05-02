import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from sklearn.model_selection import train_test_split

RDLogger.DisableLog('rdApp.*')

def augment_smiles(smiles, n=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    augmented = set()
    augmented.add(smiles)
    for _ in range(n * 3):
        atoms = list(range(mol.GetNumAtoms()))
        np.random.shuffle(atoms)
        new_smiles = Chem.MolToSmiles(mol, rootedAtAtom=int(atoms[0]))
        if new_smiles:
            augmented.add(new_smiles)
        if len(augmented) >= n:
            break
    return list(augmented)

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df_density = df[df['Density'].notna()].reset_index(drop=True)

print(f"Original: {len(df_density)} samples")

# Split BEFORE augmentation
train_orig, test_orig = train_test_split(df_density, test_size=0.2, random_state=42)
print(f"Train original: {len(train_orig)}, Test original: {len(test_orig)}")

# Augment ONLY training data
train_rows = []
for _, row in train_orig.iterrows():
    augmented = augment_smiles(row['SMILES'], n=5)
    for smi in augmented:
        train_rows.append({'SMILES': smi, 'Density': row['Density']})

train_aug = pd.DataFrame(train_rows)
print(f"Train after augmentation: {len(train_aug)}")

train_aug.to_csv('polymer_data/train_augmented.csv', index=False)
test_orig[['SMILES', 'Density']].to_csv('polymer_data/test_original.csv', index=False)
print("Saved: train_augmented.csv, test_original.csv")
print("\nNo data leakage — test set is original molecules only!")
