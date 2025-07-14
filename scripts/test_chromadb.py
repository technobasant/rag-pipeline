import requests

url = "http://localhost:8000/api/v2/heartbeat"
try:
    resp = requests.get(url)
    print(f"{url} -> {resp.status_code} {resp.text}")
except Exception as e:
    print(f"{url} -> error: {e}") 