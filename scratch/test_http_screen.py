import requests
import json

url = "http://localhost:8080/screen"
payload = {
    "application_id": "00000000-0000-0000-0000-000000000001",
    "job_id": "00000000-0000-0000-0000-000000000002",
    "jd_text": "Role: Senior Python Developer\nMust have: FastAPI, Docker, testing experience",
    "api_key": "LDUEOZmAVS5gLbFSC3AsLi06FUnVq1CkHVkZuNvwI1g="
}

files = {
    "resume": ("sample_resume.txt", open("tests/fixtures/sample_resume.txt", "rb"), "text/plain")
}

print("Sending POST request to /screen...")
try:
    response = requests.post(url, data=payload, files=files, timeout=60)
    print("Response Status:", response.status_code)
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print("Request failed:", e)
