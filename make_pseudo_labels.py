import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from huggingface_hub import hf_hub_download

RDLogger.DisableLog('rdApp.*')

def mol_to_graph(smiles):
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
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)

class MultiTaskGNN(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(hidden_dim)
        self.shared = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(), nn.Dropout(0.1))
        self.head_density = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tg = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        x = torch.relu(self.bn4(self.conv4(x, edge_index)))
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        shared = self.shared(x)
        return self.head_density(shared).squeeze(), self.head_tc(shared).squeeze(), self.head_tg(shared).squeeze()

# 데이터 통계
df_orig = pd.read_csv('polymer_data/combined_dataset.csv')
stats = {}
for col in ['Density', 'Tc', 'Tg']:
    stats[col] = {'mean': df_orig[col].mean(), 'std': df_orig[col].std()}

# 모델 로드
model = MultiTaskGNN()
model_path = hf_hub_download(repo_id='Ethan-Im/polyinverse-model', filename='best_combined.pt', repo_type='model')
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# PI1M에서 SA 좋은 것 3000개 샘플링
df_pi1m = pd.read_csv('polymer_data/PI1M_v2.csv')
df_good = df_pi1m[df_pi1m['SA Score'] <= 4.0].sample(3000, random_state=42)

results = []
print("Pseudo label 생성 중...")
for i, (_, row) in enumerate(df_good.iterrows()):
    if i % 500 == 0:
        print(f"  {i}/3000")
    smiles = row['SMILES']
    graph = mol_to_graph(smiles)
    if graph is None:
        continue
    with torch.no_grad():
        pred_d, pred_tc, pred_tg = model(graph.x, graph.edge_index, graph.batch)
        density = pred_d.item() * stats['Density']['std'] + stats['Density']['mean']
        tc = pred_tc.item() * stats['Tc']['std'] + stats['Tc']['mean']
        tg = pred_tg.item() * stats['Tg']['std'] + stats['Tg']['mean']
    results.append({'SMILES': smiles, 'Density': density, 'Tc': tc, 'Tg': tg})

df_pseudo = pd.DataFrame(results)
print(f"\n생성 완료: {len(df_pseudo)}개")
print(df_pseudo[['Density', 'Tc', 'Tg']].describe())

# 기존 데이터와 병합
df_merged = pd.concat([df_orig, df_pseudo], ignore_index=True)
df_merged.to_csv('polymer_data/combined_dataset_v2.csv', index=False)
print(f"\n최종 데이터셋: {len(df_merged)}개")
print(f"Tg 데이터: {df_merged['Tg'].notna().sum()}개")
