import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json
import time
import fitz  # pymupdf
from pathlib import Path
import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def search_arxiv(query, max_results=20):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "max_results": max_results,
        "sortBy": "relevance"
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip()
        arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
        papers.append({"title": title, "arxiv_id": arxiv_id, "pdf_url": pdf_url})
    return papers

def download_pdf(pdf_url, save_path):
    response = requests.get(pdf_url, timeout=30)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    return False

def extract_text_from_pdf(pdf_path, max_pages=5):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            if i >= max_pages:
                break
            text += page.get_text()
        return text[:5000]
    except:
        return ""

def extract_polymer_data_with_llm(text, title):
    if not OPENAI_API_KEY:
        return None
    
    prompt = f"""You are a chemistry expert. Extract polymer data from this paper text.

Paper title: {title}

Text: {text}

Extract ANY polymer SMILES strings and their measured properties (density, Tg, Tc) from tables or text.
Return ONLY valid JSON in this exact format:
{{
  "polymers": [
    {{
      "smiles": "SMILES string with * for repeat units",
      "density": null or float (g/ml),
      "tg": null or float (celsius),
      "tc": null or float (normalized 0-1 or celsius)
    }}
  ]
}}

If no polymer data found, return {{"polymers": []}}
Only include entries where you found at least one property value."""

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}",
                 "Content-Type": "application/json"},
        json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 1000
        }
    )
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        try:
            clean = content.strip().replace("```json", "").replace("```", "")
            return json.loads(clean)
        except:
            return None
    return None

Path("paper_data/pdfs").mkdir(parents=True, exist_ok=True)

queries = [
    "polymer density glass transition temperature measurement",
    "polyimide thermal properties experimental",
    "polyester density crystallization temperature",
    "polymer chain structure physical properties table",
]

print("=== Searching arXiv papers ===")
all_papers = []
for query in queries:
    papers = search_arxiv(query, max_results=10)
    all_papers.extend(papers)
    print(f"Query '{query[:40]}': {len(papers)} papers")
    time.sleep(1)

all_papers = list({p['arxiv_id']: p for p in all_papers}.values())
print(f"\nTotal unique papers: {len(all_papers)}")

extracted_data = []
for i, paper in enumerate(all_papers[:10]):
    print(f"\n[{i+1}/{min(10, len(all_papers))}] {paper['title'][:50]}")
    
    pdf_path = f"paper_data/pdfs/{paper['arxiv_id']}.pdf"
    if not os.path.exists(pdf_path):
        success = download_pdf(paper['pdf_url'], pdf_path)
        if not success:
            print("  ❌ PDF download failed")
            continue
        time.sleep(2)
    
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print("  ❌ Text extraction failed")
        continue
    
    print(f"  ✅ Text extracted ({len(text)} chars)")
    
    if OPENAI_API_KEY:
        data = extract_polymer_data_with_llm(text, paper['title'])
        if data and data.get('polymers'):
            for polymer in data['polymers']:
                polymer['source'] = paper['arxiv_id']
                polymer['title'] = paper['title']
                extracted_data.append(polymer)
            print(f"  ✅ Found {len(data['polymers'])} polymer entries")
        else:
            print("  ℹ️  No polymer data found")
    else:
        print("  ⚠️  No OpenAI API key - showing text preview:")
        print(f"  {text[:200]}")

if extracted_data:
    df_new = pd.DataFrame(extracted_data)
    df_new.to_csv('paper_data/extracted_polymers.csv', index=False)
    print(f"\n✅ Extracted {len(extracted_data)} polymer entries")
    print(df_new.head())
else:
    print("\n⚠️  No data extracted yet")
    print("Set OPENAI_API_KEY environment variable to enable LLM extraction")
    print("PDF files downloaded to paper_data/pdfs/ for manual inspection")
