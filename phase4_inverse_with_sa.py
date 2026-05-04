import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDConfig
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import os, sys, random

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

RDLogger.DisableLog('rdApp.*')

def get_sa_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 10.0
    try:
        return sascorer.calculateScore(mol)
    except:
        return 10.0

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
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.head_tc = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

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
density_mean = df['Density'].mean()
density_std = df['Density'].std()

model = MultiTaskGNN(input_dim=8, hidden_dim=128)
model.load_state_dict(torch.load('best_model_final.pt', map_location='cpu'))
model.eval()

def predict_density(smiles):
    graph = mol_to_graph(smiles)
    if graph is None:
        return None
    with torch.no_grad():
        pred_d, _ = model(graph)
        return pred_d.item() * density_std + density_mean

def mutate_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rw_mol = Chem.RWMol(mol)
        atom_idx = random.randint(0, mol.GetNumAtoms()-1)
        atom = rw_mol.GetAtomWithIdx(atom_idx)
        # Only use common safe atoms
        candidates = [6, 7, 8]
        candidates = [c for c in candidates if c != atom.GetAtomicNum()]
        atom.SetAtomicNum(random.choice(candidates))
        new_smiles = Chem.MolToSmiles(rw_mol)
        if Chem.MolFromSmiles(new_smiles):
            return new_smiles
    except:
        pass
    return None

def inverse_design(target_density, tolerance=0.05, n_iterations=300, sa_threshold=4.0):
    print(f"\nTarget Density: {target_density:.3f} g/ml (±{tolerance})")
    print(f"SA Score threshold: < {sa_threshold} (synthesizable only)")

    df_density = df[df['Density'].notna()]
    population = df_density['SMILES'].sample(
        min(50, len(df_density)), random_state=42).tolist()

    best_candidates = []

    for iteration in range(n_iterations):
        scores = []
        for smiles in population:
            pred = predict_density(smiles)
            if pred is None:
                continue
            sa = get_sa_score(smiles)
            density_error = abs(pred - target_density)

            # Combined score: density error + SA Score penalty
            combined = density_error + max(0, sa - sa_threshold) * 0.1
            scores.append((smiles, pred, density_error, sa, combined))

        scores.sort(key=lambda x: x[4])

        # Only accept synthesizable candidates
        for smiles, pred, d_error, sa, combined in scores:
            if d_error <= tolerance and sa <= sa_threshold:
                if smiles not in [c[0] for c in best_candidates]:
                    best_candidates.append((smiles, pred, d_error, sa))

        if (iteration + 1) % 100 == 0:
            print(f"  Iteration {iteration+1} | Best error: {scores[0][2]:.4f} | SA: {scores[0][3]:.2f} | Candidates: {len(best_candidates)}")

        survivors = [s[0] for s in scores[:25]]
        new_population = survivors.copy()
        while len(new_population) < 50:
            parent = random.choice(survivors)
            mutated = mutate_smiles(parent)
            if mutated:
                new_population.append(mutated)
            else:
                new_population.append(random.choice(survivors))
        population = new_population

    return best_candidates[:10]

print("=" * 60)
print("POLYINVERSE v2 — Inverse Design with SA Score Filter")
print("=" * 60)

targets = [
    (0.95, "Light polymer (PP/PE range)"),
    (1.20, "Medium density (PET range)"),
    (1.40, "Heavy polymer (PVC range)"),
]

for target, description in targets:
    print(f"\n{'='*60}")
    print(f"Goal: {description}")
    candidates = inverse_design(target, tolerance=0.08, n_iterations=300, sa_threshold=4.0)

    print(f"\nTop synthesizable candidates for Density ≈ {target} g/ml:")
    print(f"{'SMILES':<45} {'Predicted':>10} {'Error':>7} {'SA Score':>9}")
    print("-" * 75)

    if candidates:
        for smiles, pred, error, sa in candidates[:5]:
            print(f"{smiles[:43]:<45} {pred:>10.4f} {error:>7.4f} {sa:>9.2f}")
    else:
        print("No synthesizable candidates found. Try increasing tolerance.")
