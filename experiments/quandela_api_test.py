"""Reproduce the 400 response we got earlier (token was accepted)."""

import requests
import json

TOKEN = "YOUR_TOKEN_HERE"

BASE = "https://api.cloud.quandela.com"
headers = {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}

# This exact payload got 400 (not 401) earlier
test1 = {
    "platform": "sim:slos",
    "payload": {
        "circuit": {"name": "test", "m": 2},
        "input_state": "|1,1>",
        "min_detected_photons": 1
    }
}

print("=== Test 1: Original 400 payload (no job_name) ===")
r = requests.post(f"{BASE}/api/job", headers=headers, json=test1, timeout=10)
print(f"[{r.status_code}] {r.text[:300]}")

# Now add job_name
test2 = dict(test1)
test2["job_name"] = "test_job"
print("\n=== Test 2: With job_name ===")
r = requests.post(f"{BASE}/api/job", headers=headers, json=test2, timeout=10)
print(f"[{r.status_code}] {r.text[:300]}")

# Try with platform_name instead of platform
test3 = {
    "job_name": "test_job",
    "platform_name": "sim:slos",
    "payload": {
        "circuit": {"name": "test", "m": 2},
        "input_state": "|1,1>",
        "min_detected_photons": 1
    }
}
print("\n=== Test 3: With platform_name ===")
r = requests.post(f"{BASE}/api/job", headers=headers, json=test3, timeout=10)
print(f"[{r.status_code}] {r.text[:300]}")
