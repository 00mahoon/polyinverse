import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def augment_smiles(smiles, n=5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
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

print(f"Original density samples: {len(df_density)}")

rows = []
for _, row in df_density.iterrows():
    augmented = augment_smiles(row['SMILES'], n=5)
    for smi in augmented:
        rows.append({'SMILES': smi, 'Density': row['Density']})

df_aug = pd.DataFrame(rows)
print(f"After augmentation: {len(df_aug)}")
df_aug.to_csv('polymer_data/density_augmented_v2.csv', index=False)
print("Saved: polymer_data/density_augmented_v2.csv")
