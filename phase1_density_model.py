import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.RingCount(mol),
        mol.GetNumAtoms(),
        mol.GetNumBonds(),
    ]

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df = df[df['Density'].notna()].reset_index(drop=True)

print(f"Density samples: {len(df)}")

features, targets = [], []
for _, row in df.iterrows():
    feat = smiles_to_features(row['SMILES'])
    if feat:
        features.append(feat)
        targets.append(row['Density'])

X = np.array(features)
y = np.array(targets)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n=== Model Performance ===")
print(f"R² Score : {r2:.3f}")
print(f"MAE      : {mae:.4f} g/ml")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Density')
plt.ylabel('Predicted Density')
plt.title(f'Density Prediction (R²={r2:.3f})')
plt.tight_layout()
plt.savefig('density_prediction.png')
print("\nSaved: density_prediction.png")
