import requests
import pandas as pd
import time
import os

API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "여기에_API_키_직접입력_가능")

HEADERS = {"x-api-key": API_KEY}

queries = [
    "polymer glass transition temperature Tg",
    "polyethylene glass transition temperature",
    "polystyrene Tg measurement",
    "polymer thermal properties Tg DSC",
    "polyimide glass transition",
    "epoxy resin glass transition temperature",
]

def search_papers(query, limit=50):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,abstract,year"
    }
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code == 200:
        return r.json().get("data", [])
    return []

def extract_tg_from_abstract(abstract):
    import re
    if not abstract:
        return []
    patterns = [
        r'T[g]\s*[=:of]*\s*[-]?\d+\.?\d*\s*°?C',
        r'glass transition temperature[s]?\s*[of]*\s*[-]?\d+\.?\d*\s*°?C',
        r'Tg\s*=\s*[-]?\d+\.?\d*',
    ]
    results = []
    for pattern in patterns:
        matches = re.findall(pattern, abstract, re.IGNORECASE)
        for match in matches:
            nums = re.findall(r'[-]?\d+\.?\d*', match)
            if nums:
                val = float(nums[-1])
                if -150 < val < 600:
                    results.append(val)
    return results

all_tg = []
seen_titles = set()

for query in queries:
    print(f"Searching: {query}")
    papers = search_papers(query)
    for paper in papers:
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        if title in seen_titles:
            continue
        seen_titles.add(title)
        tg_values = extract_tg_from_abstract(abstract)
        for tg in tg_values:
            all_tg.append({"title": title, "Tg": tg, "source": "SemanticScholar"})
    time.sleep(1)

df_new = pd.DataFrame(all_tg)
print(f"\n추출된 Tg 데이터: {len(df_new)}개")
print(df_new.head(10))
df_new.to_csv("polymer_data/tg_from_papers.csv", index=False)
print("저장 완료: polymer_data/tg_from_papers.csv")
