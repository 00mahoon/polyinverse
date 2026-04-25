import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

descriptor_names = [name for name, _ in Descriptors.descList]

def smiles_to_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    features = []
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            features.append(float(val) if val is not None else 0.0)
        except:
            features.append(0.0)
    return features

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df = df[df['Density'].notna()].reset_index(drop=True)

print(f"Density samples: {len(df)}")
print(f"Computing {len(descriptor_names)} descriptors...")

features, targets = [], []
for _, row in df.iterrows():
    feat = smiles_to_features(row['SMILES'])
    if feat:
        features.append(feat)
        targets.append(row['Density'])

X = np.array(features)
y = np.array(targets)

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n=== Model Performance (v2) ===")
print(f"Descriptors used : {len(descriptor_names)}")
print(f"R² Score         : {r2:.3f}")
print(f"MAE              : {mae:.4f} g/ml")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Density')
plt.ylabel('Predicted Density')
plt.title(f'Density Prediction v2 (R²={r2:.3f})')
plt.tight_layout()
plt.savefig('density_prediction_v2.png')
print("Saved: density_prediction_v2.png")
