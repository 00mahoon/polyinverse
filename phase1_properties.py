from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

polymers = {
    "Polystyrene": "CC(c1ccccc1)CC(c1ccccc1)",
    "Polyethylene": "CCCCCCCC",
    "Polylactic acid": "CC(C(=O)O)OC(=O)C(C)O"
}

print("=" * 50)
print(f"{'Name':<20} {'MW':>10} {'LogP':>8} {'TPSA':>8}")
print("=" * 50)

for name, smiles in polymers.items():
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        print(f"{name:<20} {mw:>10.2f} {logp:>8.2f} {tpsa:>8.2f}")

print("=" * 50)
print("Done!")
