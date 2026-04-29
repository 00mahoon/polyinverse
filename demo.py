import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import gradio as gr
import io
from PIL import Image

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
    return Data(x=x, edge_index=edge_index, batch=torch.zeros(x.size(0), dtype=torch.long))

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

# Load model
public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df = pd.concat([public, private], ignore_index=True)
density_mean = df['Density'].mean()
density_std = df['Density'].std()
tc_mean = df['Tc'].mean()
tc_std = df['Tc'].std()

model = MultiTaskGNN(input_dim=8, hidden_dim=128)
model.load_state_dict(torch.load('best_model_final.pt'))
model.eval()

# Example SMILES
examples = [
    ["*CC(c1ccccc1)*", "Polystyrene"],
    ["*CC(C(=O)OC)*", "PMMA"],
    ["*CC(Cl)*", "PVC"],
    ["*C(F)(F)*", "PTFE"],
    ["*CC*", "Polyethylene"],
]

def predict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "❌ Invalid SMILES", None

    # Predict properties
    graph = mol_to_graph(smiles)
    if graph is None:
        return "❌ Cannot convert to graph", None

    with torch.no_grad():
        pred_d, pred_t = model(graph)
        density = pred_d.item() * density_std + density_mean
        tc = pred_t.item() * tc_std + tc_mean

    # Draw molecule
    img = Draw.MolToImage(mol, size=(300, 300))

    result = f"""## Prediction Results

**SMILES:** `{smiles}`

| Property | Predicted Value |
|----------|----------------|
| Density  | {density:.4f} g/ml |
| Tc       | {tc:.4f} |

---
*Model: Multi-task GNN (R² Density=0.757, Tc=0.512)*
*Data: NeurIPS 2025 Open Polymer Challenge*
"""
    return result, img

with gr.Blocks(title="Polyinverse — Polymer Property Predictor") as demo:
    gr.Markdown("""
# 🧪 Polyinverse
## AI-powered Polymer Property Predictor
Enter a polymer SMILES string to predict **Density** and **Tc** using Graph Neural Network.
    """)

    with gr.Row():
        with gr.Column():
            smiles_input = gr.Textbox(
                label="Polymer SMILES",
                placeholder="e.g. *CC(c1ccccc1)*",
                value="*CC(c1ccccc1)*"
            )
            predict_btn = gr.Button("🔬 Predict", variant="primary")

            gr.Markdown("### Example Polymers")
            for smiles, name in examples:
                gr.Button(f"{name}").click(
                    fn=lambda s=smiles: s,
                    outputs=smiles_input
                )

        with gr.Column():
            output_text = gr.Markdown(label="Results")
            output_img = gr.Image(label="Molecule Structure")

    predict_btn.click(
        fn=predict,
        inputs=smiles_input,
        outputs=[output_text, output_img]
    )

    smiles_input.submit(
        fn=predict,
        inputs=smiles_input,
        outputs=[output_text, output_img]
    )

demo.launch(share=True)
