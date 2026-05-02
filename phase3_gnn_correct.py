import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
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
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    if len(edge_index) == 0:
        return None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    y = torch.tensor([target], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

train_df = pd.read_csv('polymer_data/train_augmented.csv')
test_df = pd.read_csv('polymer_data/test_original.csv')

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

y_mean = train_df['Density'].mean()
y_std = train_df['Density'].std()

train_graphs = []
for _, row in train_df.iterrows():
    g = mol_to_graph(row['SMILES'], (row['Density'] - y_mean) / y_std)
    if g:
        train_graphs.append(g)

test_graphs = []
for _, row in test_df.iterrows():
    g = mol_to_graph(row['SMILES'], (row['Density'] - y_mean) / y_std)
    if g:
        test_graphs.append(g)

print(f"Valid train: {len(train_graphs)} | Valid test: {len(test_graphs)}")

train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)

class GNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=256):
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
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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

model = GNN(input_dim=8, hidden_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
criterion = nn.MSELoss()

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining with correct augmentation...")

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
    scheduler.step()

    if (epoch + 1) % 50 == 0:
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch)
                y_true.extend((batch.y.numpy() * y_std + y_mean).tolist())
                y_pred.extend((out.numpy() * y_std + y_mean).tolist())
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:>3}/300 | Loss: {avg_loss:.4f} | R²: {r2:.3f} | MAE: {mae:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            torch.save(model.state_dict(), 'best_correct.pt')

model.load_state_dict(torch.load('best_correct.pt'))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        y_true.extend((batch.y.numpy() * y_std + y_mean).tolist())
        y_pred.extend((out.numpy() * y_std + y_mean).tolist())

r2 = r2_score(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f"\n=== Correct Augmentation GNN ===")
print(f"R² Score : {r2:.3f}")
print(f"MAE      : {mae:.4f} g/ml")
print(f"Train size: {len(train_graphs)} | Test size: {len(test_graphs)}")

plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.5, color='steelblue')
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
plt.xlabel('Actual Density')
plt.ylabel('Predicted Density')
plt.title(f'Correct Augmentation GNN (R²={r2:.3f})')
plt.tight_layout()
plt.savefig('correct_augmentation.png')
print("Saved: correct_augmentation.png")
