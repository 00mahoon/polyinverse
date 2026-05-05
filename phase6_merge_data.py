import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

def canonical_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            return Chem.MolToSmiles(mol)
    except:
        pass
    return None

# 기존 데이터
public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')
df_orig = pd.concat([public, private], ignore_index=True)
df_orig = df_orig[['SMILES', 'Tg', 'Density', 'Tc']]
print(f"Original: {len(df_orig)} samples")

# extended_polymer_dataset
df_ext = pd.read_csv('paper_data/extracted/extended_polymer_dataset.csv')
df_ext = df_ext[['SMILES', 'Tg', 'Density', 'Tc']]
print(f"Extended: {len(df_ext)} samples")

# tg_density (Tg가 K 단위 → °C 변환 필요)
df_tg = pd.read_csv('paper_data/extracted/tg_density.csv')
df_tg = df_tg[['SMILES', 'Tg', 'Density']].copy()
# Tg가 K 단위면 °C로 변환
if df_tg['Tg'].mean() > 200:
    print("Tg unit: K → converting to °C")
    df_tg['Tg'] = df_tg['Tg'] - 273.15
df_tg['Tc'] = np.nan
print(f"TG_Density: {len(df_tg)} samples")

# 합치기
df_combined = pd.concat([df_orig, df_ext, df_tg], ignore_index=True)
print(f"\nCombined total: {len(df_combined)}")

# SMILES 정규화 및 중복 제거
print("Canonicalizing SMILES...")
df_combined['canonical'] = df_combined['SMILES'].apply(canonical_smiles)
df_combined = df_combined[df_combined['canonical'].notna()]

# 중복 제거 (같은 SMILES면 물성값 평균)
df_dedup = df_combined.groupby('canonical').agg({
    'Density': 'mean',
    'Tc': 'mean',
    'Tg': 'mean'
}).reset_index()
df_dedup.rename(columns={'canonical': 'SMILES'}, inplace=True)

print(f"After deduplication: {len(df_dedup)}")
print(f"\nFinal counts:")
for col in ['Density', 'Tc', 'Tg']:
    count = df_dedup[col].notna().sum()
    print(f"  {col}: {count} samples")

df_dedup.to_csv('polymer_data/combined_dataset.csv', index=False)
print("\nSaved: polymer_data/combined_dataset.csv")
