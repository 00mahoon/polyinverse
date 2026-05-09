import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.explain import Explainer, GNNExplainer
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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

df = pd.read_csv('polymer_data/combined_dataset.csv')
model = MultiTaskGNN(input_dim=8, hidden_dim=128)
model.load_state_dict(torch.load('best_combined.pt', map_location='cpu'))
model.eval()

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

def explain_polymer(smiles):
    graph, mol = mol_to_graph(smiles)
    if graph is None or mol is None:
        return None, "Invalid SMILES"

    explanation = explainer(
        x=graph.x,
        edge_index=graph.edge_index,
        batch=graph.batch,
    )

    node_importance = explanation.node_mask.sum(dim=1).detach().numpy()
    node_importance = (node_importance - node_importance.min()) / (node_importance.max() - node_importance.min() + 1e-8)

    # Use RdBu colormap — red=high, blue=low
    cmap = plt.cm.RdYlGn
    atom_colors = {}
    atom_radii = {}
    for i, score in enumerate(node_importance):
        rgba = cmap(float(score))
        atom_colors[i] = (rgba[0], rgba[1], rgba[2])
        atom_radii[i] = 0.3 + 0.5 * float(score)

    drawer = rdMolDraw2D.MolDraw2DCairo(600, 500)
    drawer.drawOptions().addAtomIndices = False
    rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(mol.GetNumAtoms())),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
    )
    drawer.FinishDrawing()
    png_data = drawer.GetDrawingText()
    img = Image.open(io.BytesIO(png_data))

    # Explanation text
    feature_names = ['AtomicNum', 'Degree', 'Charge', 'Aromatic', 'InRing', 'NumHs', 'Mass', 'Hybridization']
    top_atoms = sorted(enumerate(node_importance), key=lambda x: x[1], reverse=True)[:3]

    explanation_text = "### Why this prediction?\n\n"
    explanation_text += "**Top influential atoms (green=high, red=low):**\n"
    for idx, score in top_atoms:
        atom = mol.GetAtomWithIdx(idx)
        symbol = atom.GetSymbol()
        aromatic = "aromatic " if atom.GetIsAromatic() else ""
        ring = "in ring " if atom.IsInRing() else ""
        explanation_text += f"- Atom {idx} ({symbol}): {score:.3f} ({aromatic}{ring})\n"

    feat_importance = explanation.node_mask.abs().mean(dim=0).detach().numpy()
    feat_importance = feat_importance / feat_importance.sum()
    top_features = sorted(zip(feature_names, feat_importance), key=lambda x: x[1], reverse=True)[:4]
    explanation_text += "\n**Key features:**\n"
    for feat, imp in top_features:
        bar = "█" * int(imp * 20)
        explanation_text += f"- {feat}: {bar} ({imp:.3f})\n"

    return img, explanation_text

img, text = explain_polymer("*CC(c1ccccc1)*")
print(text)
img.save("explanation_v2.png")
print("Saved: explanation_v2.png")
import subprocess
subprocess.run(["open", "explanation_v2.png"])
