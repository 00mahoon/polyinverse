import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
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
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetTotalNumHs(),
            atom.GetMass(),
            float(atom.GetHybridization()),
        ]
        atom_features.append(features)
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    if len(edge_index) == 0:
        return None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([target], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

class GNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        x = torch.relu(self.bn4(self.conv4(x, edge_index)))
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)
        return self.fc(x).squeeze()

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df = df[df['Density'].notna()].reset_index(drop=True)

print(f"Building graphs from {len(df)} samples...")

graphs = []
for _, row in df.iterrows():
    g = mol_to_graph(row['SMILES'], row['Density'])
    if g:
        graphs.append(g)

print(f"Valid graphs: {len(graphs)}")

train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = GNN(input_dim=8, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
criterion = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining GNN v2...")

best_r2 = -999
for epoch in range(300):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out, batch.y)
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
                out = model(batch)
                y_true.extend(batch.y.numpy())
                y_pred.extend(out.numpy())
        r2 = r2_score(y_true, y_pred)
        print(f"Epoch {epoch+1:>3}/300 | Loss: {avg_loss:.4f} | R²: {r2:.3f}")
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), 'best_gnn_model.pt')

model.load_state_dict(torch.load('best_gnn_model.pt', weights_only=True))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        y_true.extend(batch.y.numpy())
        y_pred.extend(out.numpy())

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"\n=== GNN v2 Best Performance ===")
print(f"R² Score : {r2:.3f}")
print(f"MAE      : {mae:.4f} g/ml")

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='steelblue')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('Actual Density')
plt.ylabel('Predicted Density')
plt.title(f'GNN v2 Density (R²={r2:.3f})')
plt.tight_layout()
plt.savefig('density_gnn_v2.png')
print("Saved: density_gnn_v2.png")
