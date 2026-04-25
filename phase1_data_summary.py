import pandas as pd

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')

combined = pd.concat([public, private], ignore_index=True)

print("=== Combined Dataset ===")
print(f"Total samples: {len(combined)}")
print(f"\nNon-null counts per property:")
for col in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']:
    count = combined[col].notna().sum()
    pct = count / len(combined) * 100
    print(f"  {col:<10}: {count:>5} samples ({pct:.1f}%)")

print(f"\nProperty statistics:")
print(combined[['Tg', 'FFV', 'Tc', 'Density', 'Rg']].describe().round(3))
