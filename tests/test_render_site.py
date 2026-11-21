import requests
import re
import sys

RENDER_URL = "https://ai-mental-health-system-cxkq.onrender.com"
s = requests.Session()
s.headers.update({
    "Referer": f"{RENDER_URL}/login",
    "Origin": RENDER_URL
})

print(f"Testing {RENDER_URL}...")
r1 = s.get(f"{RENDER_URL}/login", timeout=20)
print(f"GET /login: {r1.status_code}")

csrf_token = ""
for pattern in [
    r'name=["\']csrf_token["\'][^>]*?value=["\']([^"\']+)["\']',
    r'value=["\']([^"\']+)["\'][^>]*?name=["\']csrf_token["\']',
    r'<meta name="csrf-token" content="([^"]+)"'
]:
    m = re.search(pattern, r1.text)
    if m:
        csrf_token = m.group(1)
        break

print(f"CSRF token found: {bool(csrf_token)}")

r2 = s.post(
    f"{RENDER_URL}/login",
    data={"email": "demo@example.com", "password": "password123", "csrf_token": csrf_token},
    allow_redirects=True,
    timeout=20
)
print(f"POST /login: {r2.status_code}, URL: {r2.url}")

headers = {"Content-Type": "application/json", "X-CSRFToken": csrf_token}
payload = {"message": "What is artificial intelligence?", "history": []}
r3 = s.post(f"{RENDER_URL}/api/chat", json=payload, headers=headers, timeout=45)
print(f"POST /api/chat: {r3.status_code}")
print("Response JSON:")
try:
    print(r3.json().get("response"))
except Exception as e:
    print(r3.text[:500])
