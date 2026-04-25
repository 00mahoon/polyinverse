import pandas as pd

public = pd.read_csv('polymer_data/public.csv')
private = pd.read_csv('polymer_data/private.csv')

print("=== Public Dataset ===")
print(f"Shape: {public.shape}")
print(f"Columns: {list(public.columns)}")
print(public.head(3))

print("\n=== Private Dataset ===")
print(f"Shape: {private.shape}")
print(f"Columns: {list(private.columns)}")
print(private.head(3))
