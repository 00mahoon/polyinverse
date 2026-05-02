import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
import gradio as gr
import random

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
tc_mean = df['Tc'].mean()
tc_std = df['Tc'].std()

from huggingface_hub import hf_hub_download
import os

model = MultiTaskGNN(input_dim=8, hidden_dim=128)
model_path = hf_hub_download(
    repo_id='00mahoon/polyinverse-model',
    filename='best_model_final.pt',
    repo_type='model'
)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
        candidates = [6, 7, 8, 9, 16, 17]
        candidates = [c for c in candidates if c != atom.GetAtomicNum()]
        atom.SetAtomicNum(random.choice(candidates))
        new_smiles = Chem.MolToSmiles(rw_mol)
        if Chem.MolFromSmiles(new_smiles):
            return new_smiles
    except:
        pass
    return None

def predict(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "❌ Invalid SMILES", None
    graph = mol_to_graph(smiles)
    if graph is None:
        return "❌ Cannot convert to graph", None
    with torch.no_grad():
        pred_d, pred_t = model(graph)
        density = pred_d.item() * density_std + density_mean
        tc = pred_t.item() * tc_std + tc_mean
    img = Draw.MolToImage(mol, size=(300, 300))
    result = f"""## Prediction Results
**SMILES:** `{smiles}`

| Property | Predicted Value |
|----------|----------------|
| Density  | {density:.4f} g/ml |
| Tc       | {tc:.4f} |

---
*Model: Multi-task GNN (R² Density=0.757)*
"""
    return result, img

def inverse_design(target_density, tolerance, n_iterations):
    df_density = df[df['Density'].notna()]
    population = df_density['SMILES'].sample(
        min(50, len(df_density)), random_state=42).tolist()

    best_candidates = []
    for iteration in range(int(n_iterations)):
        scores = []
        for smiles in population:
            pred = predict_density(smiles)
            if pred is not None:
                score = abs(pred - target_density)
                scores.append((smiles, pred, score))
        scores.sort(key=lambda x: x[2])
        for smiles, pred, score in scores:
            if score <= tolerance:
                if smiles not in [c[0] for c in best_candidates]:
                    best_candidates.append((smiles, pred, score))
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

    if not best_candidates:
        return "No candidates found. Try increasing tolerance.", None

    best_candidates.sort(key=lambda x: x[2])
    top = best_candidates[:5]

    result = f"## Inverse Design Results\n**Target Density:** {target_density:.3f} g/ml\n\n"
    result += "| SMILES | Predicted | Error |\n|--------|-----------|-------|\n"
    for smiles, pred, error in top:
        result += f"| `{smiles[:40]}` | {pred:.4f} | {error:.4f} |\n"

    best_mol = Chem.MolFromSmiles(top[0][0])
    img = Draw.MolToImage(best_mol, size=(300, 300)) if best_mol else None
    return result, img

with gr.Blocks(title="Polyinverse v2") as demo:
    gr.Markdown("""
# 🧪 Polyinverse
## AI-powered Polymer Property Predictor & Inverse Design
    """)

    with gr.Tabs():
        with gr.Tab("🔬 Forward Prediction"):
            gr.Markdown("### Predict properties from SMILES")
            with gr.Row():
                with gr.Column():
                    smiles_input = gr.Textbox(
                        label="Polymer SMILES",
                        value="*CC(c1ccccc1)*"
                    )
                    predict_btn = gr.Button("Predict", variant="primary")
                with gr.Column():
                    output_text = gr.Markdown()
                    output_img = gr.Image(label="Molecule")
            predict_btn.click(predict, inputs=smiles_input,
                            outputs=[output_text, output_img])
            smiles_input.submit(predict, inputs=smiles_input,
                              outputs=[output_text, output_img])

        with gr.Tab("🎯 Inverse Design"):
            gr.Markdown("### Design molecules with target properties")
            with gr.Row():
                with gr.Column():
                    target_density = gr.Slider(
                        minimum=0.8, maximum=2.0, value=1.2,
                        step=0.05, label="Target Density (g/ml)")
                    tolerance = gr.Slider(
                        minimum=0.01, maximum=0.2, value=0.08,
                        step=0.01, label="Tolerance")
                    n_iter = gr.Slider(
                        minimum=100, maximum=500, value=200,
                        step=100, label="Iterations")
                    design_btn = gr.Button("🎯 Design", variant="primary")
                with gr.Column():
                    design_output = gr.Markdown()
                    design_img = gr.Image(label="Best Candidate")
            design_btn.click(inverse_design,
                           inputs=[target_density, tolerance, n_iter],
                           outputs=[design_output, design_img])

demo.launch(share=True)
