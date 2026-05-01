import requests
import pandas as pd

print("Fetching from Polymer Genome...")

url = "https://khazana.gatech.edu/api/v2/properties"
headers = {'Content-Type': 'application/json'}

try:
    response = requests.get(url, headers=headers, timeout=10)
    print(f"Status: {response.status_code}")
    print(response.text[:500])
except Exception as e:
    print(f"Error: {e}")
