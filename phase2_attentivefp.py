import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import AttentiveFP
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

def mol_to_graph(smiles, target):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum() / 100.0,
            atom.GetDegree() / 10.0,
            atom.GetFormalCharge() / 5.0,
            float(atom.GetIsAromatic()),
            float(atom.IsInRing()),
            atom.GetTotalNumHs() / 8.0,
            atom.GetMass() / 200.0,
            float(atom.GetHybridization()) / 8.0,
        ]
        atom_features.append(features)
    x = torch.tensor(atom_features, dtype=torch.float)

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
        bond_feat = [
            float(bond.GetBondTypeAsDouble()),
            float(bond.GetIsAromatic()),
            float(bond.IsInRing()),
        ]
        edge_attr += [bond_feat, bond_feat]

    if len(edge_index) == 0:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    y = torch.tensor([target], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

df = pd.read_csv('polymer_data/density_augmented.csv')
print(f"Total samples: {len(df)}")

y_mean = df['Density'].mean()
y_std = df['Density'].std()

graphs = []
for _, row in df.iterrows():
    normalized = (row['Density'] - y_mean) / y_std
    g = mol_to_graph(row['SMILES'], normalized)
    if g:
        graphs.append(g)

print(f"Valid graphs: {len(graphs)}")

train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = AttentiveFP(
    in_channels=8,
    hidden_channels=128,
    out_channels=1,
    edge_dim=3,
    num_layers=4,
    num_timesteps=2,
    dropout=0.2,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
criterion = nn.MSELoss()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining AttentiveFP...")

best_r2 = -999
for epoch in range(300):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = criterion(out.squeeze(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    if (epoch + 1) % 50 == 0:
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y_true.extend(batch.y.numpy())
                y_pred.extend(out.squeeze().numpy())
        y_true = np.array(y_true) * y_std + y_mean
        y_pred = np.array(y_pred) * y_std + y_mean
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        print(f"Epoch {epoch+1:>3}/300 | Loss: {avg_loss:.4f} | R²: {r2:.3f} | MAE: {mae:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), 'best_attentivefp.pt')

model.load_state_dict(torch.load('best_attentivefp.pt'))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y_true.extend(batch.y.numpy())
        y_pred.extend(out.squeeze().numpy())

y_true = np.array(y_true) * y_std + y_mean
y_pred = np.array(y_pred) * y_std + y_mean
r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"\n=== AttentiveFP Best Performance ===")
print(f"R² Score : {r2:.3f}")
print(f"MAE      : {mae:.4f} g/ml")

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='steelblue')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('Actual Density')
plt.ylabel('Predicted Density')
plt.title(f'AttentiveFP Density (R²={r2:.3f})')
plt.tight_layout()
plt.savefig('density_attentivefp.png')
print("Saved: density_attentivefp.png")
