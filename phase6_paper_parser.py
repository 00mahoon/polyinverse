import requests
import json
import time
import pandas as pd
from pathlib import Path
import os

# Semantic Scholar API로 논문 검색
def search_polymer_papers(query, limit=10):
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,openAccessPdf,abstract"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json().get('data', [])
    return []

# 테스트
queries = [
    "polymer density glass transition temperature SMILES",
    "polyimide thermal properties synthesis",
    "polyester crystallization temperature machine learning",
]

print("=== Searching polymer papers ===\n")
all_papers = []
for query in queries:
    papers = search_polymer_papers(query, limit=5)
    for p in papers:
        has_pdf = p.get('openAccessPdf') is not None
        all_papers.append({
            'title': p['title'],
            'year': p.get('year'),
            'has_pdf': has_pdf,
            'pdf_url': p.get('openAccessPdf', {}).get('url') if has_pdf else None,
            'abstract': p.get('abstract', '')[:200]
        })
        print(f"{'✅' if has_pdf else '❌'} [{p.get('year')}] {p['title'][:60]}")
    time.sleep(1)

df = pd.DataFrame(all_papers)
pdf_available = df[df['has_pdf'] == True]
print(f"\nTotal papers found: {len(df)}")
print(f"With open access PDF: {len(pdf_available)}")
pdf_available.to_csv('paper_data/papers_list.csv', index=False)
print("Saved: paper_data/papers_list.csv")
