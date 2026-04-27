import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog('rdApp.*')

def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features (atom-level)
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
            atom.GetTotalNumHs(),
        ]
        atom_features.append(features)

    x = torch.tensor(atom_features, dtype=torch.float)

    # Edge index (bond-level)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

# Test
test_smiles = [
    ("Polystyrene", "CC(c1ccccc1)CC(c1ccccc1)"),
    ("Polyethylene", "CCCCCCCC"),
    ("Polylactic acid", "CC(C(=O)O)OC(=O)C(C)O"),
]

print("=== Molecule to Graph Conversion ===\n")
for name, smiles in test_smiles:
    graph = mol_to_graph(smiles)
    if graph:
        print(f"{name}")
        print(f"  Nodes (atoms) : {graph.x.shape[0]}")
        print(f"  Edges (bonds) : {graph.edge_index.shape[1] // 2}")
        print(f"  Node features : {graph.x.shape[1]}")
        print()

print("Done!")
