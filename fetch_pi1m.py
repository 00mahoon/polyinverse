from datasets import load_dataset
import pandas as pd

datasets_to_try = [
    "polygnn/pi1m",
    "pi1m/pi1m", 
    "polymerization/pi1m",
]

for name in datasets_to_try:
    try:
        print(f"시도 중: {name}")
        ds = load_dataset(name, split="train")
        df = ds.to_pandas()
        print(f"성공! 컬럼: {df.columns.tolist()}")
        print(f"행 수: {len(df)}")
        break
    except Exception as e:
        print(f"실패: {e}")
