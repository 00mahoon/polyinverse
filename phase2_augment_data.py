import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# Known polymers with density from literature
extra_data = [
    ("*CC*", 0.92),           # Polyethylene
    ("*CC(C)*", 0.90),        # Polypropylene
    ("*CC(c1ccccc1)*", 1.05), # Polystyrene
    ("*CC(O)*", 1.19),        # Polyvinyl alcohol
    ("*CC(Cl)*", 1.40),       # Polyvinyl chloride
    ("*CC(C(=O)OC)*", 1.19),  # PMMA
    ("*CC(C#N)*", 1.18),      # Polyacrylonitrile
    ("*C(F)(F)*", 2.15),      # PTFE
    ("*CCOC(=O)*", 1.27),     # PET repeat unit
    ("*CCCCCCCCCCCC*", 0.95), # Nylon-12 approx
]

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df_density = df[df['Density'].notna()].copy()

print(f"Original density samples: {len(df_density)}")

extra_rows = pd.DataFrame(extra_data, columns=['SMILES', 'Density'])
df_combined = pd.concat([df_density[['SMILES', 'Density']], extra_rows], ignore_index=True)

print(f"After augmentation: {len(df_combined)}")
df_combined.to_csv('polymer_data/density_augmented.csv', index=False)
print("Saved: polymer_data/density_augmented.csv")
