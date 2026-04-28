import os
import pandas as pd
from mp_api.client import MPRester

API_KEY = os.environ.get("MP_API_KEY")

print("Connecting to Materials Project...")

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        fields=["material_id", "formula_pretty", "density"],
        num_chunks=10
    )

print(f"Fetched: {len(docs)} materials")

data = []
for doc in docs:
    data.append({
        "material_id": doc.material_id,
        "formula": doc.formula_pretty,
        "density": doc.density,
    })

df = pd.DataFrame(data)
df = df[df['density'].notna()]
print(f"With density: {len(df)} materials")
print(df.head(5))

df.to_csv('materials_data.csv', index=False)
print("Saved: materials_data.csv")
