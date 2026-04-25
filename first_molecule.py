from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# 폴리스티렌 (Polystyrene) — 가장 흔한 고분자 중 하나
smiles = "CC(c1ccccc1)CC(c1ccccc1)"

mol = Chem.MolFromSmiles(smiles)

if mol:
    print("✅ 분자 로드 성공!")
    print(f"분자식: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print(f"분자량: {Descriptors.MolWt(mol):.2f} g/mol")
    print(f"원자 수: {mol.GetNumAtoms()}")
    
    # 이미지 저장
    img = Draw.MolToFile(mol, 'first_molecule.png', size=(400, 300))
    print("✅ first_molecule.png 저장 완료!")
else:
    print("❌ 분자 로드 실패")
