import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io

RDLogger.DisableLog('rdApp.*')

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
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
        return None, None
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch), mol

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
        self.head_tg = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long)
        x = torch.relu(self.bn1(self.conv1(x, edge_index)))
        x = torch.relu(self.bn2(self.conv2(x, edge_index)))
        x = torch.relu(self.bn3(self.conv3(x, edge_index)))
        x = torch.relu(self.bn4(self.conv4(x, edge_index)))
        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=1)
        shared = self.shared(x)
        return self.head_density(shared).squeeze()

# Load model
df = pd.read_csv('polymer_data/combined_dataset.csv')
density_mean = df['Density'].mean()
density_std = df['Density'].std()

model = MultiTaskGNN(input_dim=8, hidden_dim=128)

# Load weights (density head only for explainability)
full_state = torch.load('best_combined.pt', map_location='cpu')
model.load_state_dict(full_state)
model.eval()

# Test SMILES
test_smiles = "*CC(c1ccccc1)*"  # Polystyrene
graph, mol = mol_to_graph(test_smiles)

# GNNExplainer
explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
)

explanation = explainer(
    x=graph.x,
    edge_index=graph.edge_index,
    batch=graph.batch,
)

node_importance = explanation.node_mask.sum(dim=1).detach().numpy()
node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)

print("=== GNNExplainer Results ===")
print(f"SMILES: {test_smiles}")
print(f"Num atoms: {mol.GetNumAtoms()}")
print(f"Node importance scores:")
for i, (atom, score) in enumerate(zip(mol.GetAtoms(), node_importance)):
    print(f"  Atom {i} ({atom.GetSymbol()}): {score:.3f}")

# Visualize
atom_colors = {}
for i, score in enumerate(node_importance):
    r = float(score)
    g = 0.0
    b = float(1.0 - score)
    atom_colors[i] = (r, g, b)

atom_radii = {i: 0.3 + 0.4 * float(s) for i, s in enumerate(node_importance)}

drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)
drawer.drawOptions().addAtomIndices = False
rdMolDraw2D.PrepareMolForDrawing(mol)
drawer.DrawMolecule(mol,
                    highlightAtoms=list(range(mol.GetNumAtoms())),
                    highlightAtomColors=atom_colors,
                    highlightAtomRadii=atom_radii)
drawer.FinishDrawing()
svg = drawer.GetDrawingText()

with open('explanation_test.svg', 'w') as f:
    f.write(svg)
print("\nSaved: explanation_test.svg")
print("Red = High importance, Blue = Low importance")
