import requests
import pandas as pd

url = "https://figshare.com/ndownloader/files/18109017"

print("Downloading PI1M dataset...")
response = requests.get(url, stream=True, timeout=60)

if response.status_code == 200:
    with open('pi1m.csv', 'wb') as f:
        total = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
            if total % (1024*1024) == 0:
                print(f"Downloaded: {total // (1024*1024)} MB")
    print("Done!")
else:
    print(f"Error: {response.status_code}")
