import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}

def fetch_polyinfo_density():
    url = "https://polymer.nims.go.jp/PolyInfo/search"
    
    params = {
        'property': 'density',
        'format': 'json',
        'limit': 1000
    }
    
    print("Fetching from PolyInfo...")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"Status: {response.status_code}")
        print(response.text[:500])
    except Exception as e:
        print(f"Error: {e}")

fetch_polyinfo_density()
