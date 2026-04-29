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

def mol_to_graph(smiles, targets):
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
    y = torch.tensor(targets, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y)

class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=256, num_tasks=3):
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
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
            for _ in range(num_tasks)
        ])

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        x = torch.relu(self.bn4(self.conv4(x, edge_index)))
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)
        shared = self.shared(x)
        return [head(shared).squeeze() for head in self.heads]

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
df = df[df['Density'].notna() | df['Tc'].notna() | df['Rg'].notna()].reset_index(drop=True)
print(f"Total samples: {len(df)}")

stats = {}
for col in ['Density', 'Tc', 'Rg']:
    stats[col] = {'mean': df[col].mean(), 'std': df[col].std()}

graphs = []
for _, row in df.iterrows():
    targets = []
    for col in ['Density', 'Tc', 'Rg']:
        if pd.notna(row[col]):
            targets.append((row[col] - stats[col]['mean']) / stats[col]['std'])
        else:
            targets.append(float('nan'))
    g = mol_to_graph(row['SMILES'], targets)
    if g:
        graphs.append(g)

print(f"Valid graphs: {len(graphs)}")

train_data, test_data = train_test_split(graphs, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

model = MultiTaskGNN(input_dim=8, hidden_dim=256, num_tasks=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining Multi-task GNN v2 (Density + Tc + Rg)...")

best_r2 = -999
task_names = ['Density', 'Tc', 'Rg']

for epoch in range(500):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        preds = model(batch)
        loss = torch.tensor(0.0, requires_grad=True)
        for i in range(3):
            target = batch.y.view(-1, 3)[:, i]
            mask = ~torch.isnan(target)
            if mask.sum() > 0:
                loss = loss + ((preds[i][mask] - target[mask]) ** 2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)

    if (epoch + 1) % 100 == 0:
        model.eval()
        results = {name: {'true': [], 'pred': []} for name in task_names}
        with torch.no_grad():
            for batch in test_loader:
                preds = model(batch)
                for i, name in enumerate(task_names):
                    target = batch.y.view(-1, 3)[:, i]
                    mask = ~torch.isnan(target)
                    if mask.sum() > 0:
                        true_vals = target[mask].numpy() * stats[name]['std'] + stats[name]['mean']
                        pred_vals = preds[i][mask].numpy() * stats[name]['std'] + stats[name]['mean']
                        results[name]['true'].extend(true_vals.tolist())
                        results[name]['pred'].extend(pred_vals.tolist())

        r2s = []
        print(f"\nEpoch {epoch+1}/500 | Loss: {avg_loss:.4f}")
        for name in task_names:
            if len(results[name]['true']) > 1:
                r2 = r2_score(results[name]['true'], results[name]['pred'])
                mae = mean_absolute_error(results[name]['true'], results[name]['pred'])
                print(f"  {name:<10} R²: {r2:.3f} | MAE: {mae:.4f}")
                r2s.append(r2)

        avg_r2 = np.mean(r2s)
        if avg_r2 > best_r2:
            best_r2 = avg_r2
            torch.save(model.state_dict(), 'best_multitask_v2.pt')

model.load_state_dict(torch.load('best_multitask_v2.pt'))
model.eval()
results = {name: {'true': [], 'pred': []} for name in task_names}
with torch.no_grad():
    for batch in test_loader:
        preds = model(batch)
        for i, name in enumerate(task_names):
            target = batch.y.view(-1, 3)[:, i]
            mask = ~torch.isnan(target)
            if mask.sum() > 0:
                true_vals = target[mask].numpy() * stats[name]['std'] + stats[name]['mean']
                pred_vals = preds[i][mask].numpy() * stats[name]['std'] + stats[name]['mean']
                results[name]['true'].extend(true_vals.tolist())
                results[name]['pred'].extend(pred_vals.tolist())

print(f"\n=== Multi-task GNN v2 Best Performance ===")
for name in task_names:
    r2 = r2_score(results[name]['true'], results[name]['pred'])
    mae = mean_absolute_error(results[name]['true'], results[name]['pred'])
    print(f"{name:<10} R²: {r2:.3f} | MAE: {mae:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = ['steelblue', 'darkorange', 'green']
for i, (name, color) in enumerate(zip(task_names, colors)):
    t = results[name]['true']
    p = results[name]['pred']
    r2 = r2_score(t, p)
    axes[i].scatter(t, p, alpha=0.5, color=color)
    axes[i].plot([min(t), max(t)], [min(t), max(t)], 'r--')
    axes[i].set_title(f'{name} (R²={r2:.3f})')
    axes[i].set_xlabel('Actual')
    axes[i].set_ylabel('Predicted')

plt.tight_layout()
plt.savefig('multitask_gnn_v2.png')
print("Saved: multitask_gnn_v2.png")
