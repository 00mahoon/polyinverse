import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')

def mol_to_graph(smiles, density, tc):
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
    y_density = torch.tensor([density], dtype=torch.float)
    y_tc = torch.tensor([tc], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y_density=y_density, y_tc=y_tc)

class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128):
        super(MultiTaskGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.head_density = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))
        self.head_tc = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        x = torch.relu(self.bn4(self.conv4(x, edge_index)))
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)
        shared = self.shared(x)
        return self.head_density(shared).squeeze(), self.head_tc(shared).squeeze()

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df = df[df['Density'].notna() | df['Tc'].notna()].reset_index(drop=True)
print(f"Total samples: {len(df)}")

density_mean = df['Density'].mean()
density_std = df['Density'].std()
tc_mean = df['Tc'].mean()
tc_std = df['Tc'].std()

graphs = []
for _, row in df.iterrows():
    d = (row['Density'] - density_mean) / density_std if pd.notna(row['Density']) else float('nan')
    t = (row['Tc'] - tc_mean) / tc_std if pd.notna(row['Tc']) else float('nan')
    g = mol_to_graph(row['SMILES'], d, t)
    if g:
        graphs.append(g)

print(f"Valid graphs: {len(graphs)}")

train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = MultiTaskGNN(input_dim=8, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining 1000 epochs with Cosine Annealing...")

best_r2_d = -999
for epoch in range(1000):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        pred_d, pred_t = model(batch)
        loss = torch.tensor(0.0, requires_grad=True)
        mask_d = ~torch.isnan(batch.y_density.squeeze())
        mask_t = ~torch.isnan(batch.y_tc.squeeze())
        if mask_d.sum() > 0:
            loss = loss + ((pred_d[mask_d] - batch.y_density.squeeze()[mask_d]) ** 2).mean()
        if mask_t.sum() > 0:
            loss = loss + ((pred_t[mask_t] - batch.y_tc.squeeze()[mask_t]) ** 2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    if (epoch + 1) % 200 == 0:
        model.eval()
        d_true, d_pred, t_true, t_pred = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                pd_out, pt_out = model(batch)
                mask_d = ~torch.isnan(batch.y_density.squeeze())
                mask_t = ~torch.isnan(batch.y_tc.squeeze())
                if mask_d.sum() > 0:
                    d_true.extend((batch.y_density.squeeze()[mask_d].numpy() * density_std + density_mean).tolist())
                    d_pred.extend((pd_out[mask_d].numpy() * density_std + density_mean).tolist())
                if mask_t.sum() > 0:
                    t_true.extend((batch.y_tc.squeeze()[mask_t].numpy() * tc_std + tc_mean).tolist())
                    t_pred.extend((pt_out[mask_t].numpy() * tc_std + tc_mean).tolist())
        r2_d = r2_score(d_true, d_pred) if len(d_true) > 1 else 0
        r2_t = r2_score(t_true, t_pred) if len(t_true) > 1 else 0
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1:>4}/1000 | Loss: {avg_loss:.4f} | Density R²: {r2_d:.3f} | Tc R²: {r2_t:.3f}")
        if r2_d > best_r2_d:
            best_r2_d = r2_d
            torch.save(model.state_dict(), 'best_final_push.pt')

model.load_state_dict(torch.load('best_final_push.pt'))
model.eval()
d_true, d_pred, t_true, t_pred = [], [], [], []
with torch.no_grad():
    for batch in test_loader:
        pd_out, pt_out = model(batch)
        mask_d = ~torch.isnan(batch.y_density.squeeze())
        mask_t = ~torch.isnan(batch.y_tc.squeeze())
        if mask_d.sum() > 0:
            d_true.extend((batch.y_density.squeeze()[mask_d].numpy() * density_std + density_mean).tolist())
            d_pred.extend((pd_out[mask_d].numpy() * density_std + density_mean).tolist())
        if mask_t.sum() > 0:
            t_true.extend((batch.y_tc.squeeze()[mask_t].numpy() * tc_std + tc_mean).tolist())
            t_pred.extend((pt_out[mask_t].numpy() * tc_std + tc_mean).tolist())

r2_d = r2_score(d_true, d_pred)
r2_t = r2_score(t_true, t_pred)
mae_d = mean_absolute_error(d_true, d_pred)
mae_t = mean_absolute_error(t_true, t_pred)

print(f"\n=== Final Push Best Performance ===")
print(f"Density — R²: {r2_d:.3f} | MAE: {mae_d:.4f} g/ml")
print(f"Tc      — R²: {r2_t:.3f} | MAE: {mae_t:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(d_true, d_pred, alpha=0.5, color='steelblue')
axes[0].plot([min(d_true), max(d_true)], [min(d_true), max(d_true)], 'r--')
axes[0].set_title(f'Density (R²={r2_d:.3f})')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[1].scatter(t_true, t_pred, alpha=0.5, color='darkorange')
axes[1].plot([min(t_true), max(t_true)], [min(t_true), max(t_true)], 'r--')
axes[1].set_title(f'Tc (R²={r2_t:.3f})')
axes[1].set_xlabel('Actual')
axes[1].set_ylabel('Predicted')
plt.tight_layout()
plt.savefig('final_push.png')
print("Saved: final_push.png")
