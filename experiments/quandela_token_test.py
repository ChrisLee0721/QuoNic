"""Test all possible auth header formats for Quandela Cloud."""
import requests, json

TOKEN = "YOUR_TOKEN_HERE"

jwt_only = TOKEN[3:]  # without _T_ prefix
print(f"Token length: {len(TOKEN)}")
print(f"JWT length: {len(jwt_only)}")
print(f"JWT parts: {len(jwt_only.split('.'))}")

BASE = "https://api.cloud.quandela.com"

tests = [
    ("Bearer full",    {"Authorization": f"Bearer {TOKEN}"}),
    ("Bearer JWT",     {"Authorization": f"Bearer {jwt_only}"}),
    ("Token header",   {"Authorization": TOKEN}),
    ("X-Api-Key",      {"X-Api-Key": TOKEN}),
    ("X-Api-Key JWT",  {"X-Api-Key": jwt_only}),
]

for name, hdr in tests:
    try:
        r = requests.get(f"{BASE}/api/platforms", headers=hdr, timeout=8)
        try:
            body = r.json()
        except:
            body = r.text[:100]
        print(f"[{r.status_code}] {name:20s} -> {body}")
        if r.status_code == 200:
            print("   *** SUCCESS ***")
            break
    except Exception as e:
        print(f"[ERR] {name} -> {e}")
