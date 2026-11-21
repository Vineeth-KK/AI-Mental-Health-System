"""tests/compare_environments.py
Runs the 10 questions across:
1. Localhost: http://127.0.0.1:5000
2. Cloudflare Tunnel: https://estimate-immunology-possess-cabinets.trycloudflare.com
3. Render Production: https://ai-mental-health-system-cxkq.onrender.com
"""

import requests
import json
import time
import re

TARGETS = {
    "Localhost": "http://127.0.0.1:5000",
    "Cloudflare": "https://estimate-immunology-possess-cabinets.trycloudflare.com"
}

QUESTIONS = [
    "What is quantum computing?",
    "Can you give me a simple real-world analogy for that?",
    "What are the three primary laws of robotics proposed by Isaac Asimov?",
    "I'm feeling stuck on a difficult calculus problem. How do you suggest I approach it?",
    "Can you suggest a quick and healthy dinner recipe under 20 minutes?",
    "I have been feeling really overwhelmed with work and exams lately.",
    "What is the single smallest step I should take tonight to decompress?",
    "Write a clean Python function to reverse words in a sentence.",
    "Can you explain how that function handles punctuation or extra spaces?",
    "Tell me a lighthearted joke about programmers."
]

def test_target(name, base_url):
    print(f"\n==========================================")
    print(f"Testing {name}: {base_url}")
    print(f"==========================================")
    s = requests.Session()
    s.headers.update({
        "Referer": f"{base_url}/login",
        "Origin": base_url
    })

    # Login
    r = s.get(f"{base_url}/login", timeout=15)
    csrf_match = re.search(r'name=["\']csrf_token["\'][^>]*?value=["\']([^"\']+)["\']', r.text)
    if not csrf_match:
        csrf_match = re.search(r'value=["\']([^"\']+)["\'][^>]*?name=["\']csrf_token["\']', r.text)
    csrf_token = csrf_match.group(1) if csrf_match else ""
    s.headers["X-CSRFToken"] = csrf_token

    r_login = s.post(
        f"{base_url}/login",
        data={"email": "demo@example.com", "password": "password123", "csrf_token": csrf_token},
        allow_redirects=True,
        timeout=15
    )
    if r_login.status_code != 200:
        print(f"❌ Login failed: status {r_login.status_code}")
        return []

    print(f"✅ Logged in successfully as demo@example.com")

    history = []
    results = []
    for idx, prompt in enumerate(QUESTIONS, 1):
        t0 = time.time()
        payload = {"message": prompt, "history": history[-6:]}
        res = s.post(f"{base_url}/api/chat", json=payload, timeout=45)
        dt = round(time.time() - t0, 2)
        if res.status_code == 200:
            ans = res.json().get("response", "").strip()
            preview = ans.replace("\n", " ")[:110]
            print(f"[{idx}/10] ({dt}s, {len(ans)} chars): {preview}...")
            results.append({"id": idx, "prompt": prompt, "latency": dt, "response": ans})
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": ans})
        else:
            print(f"[{idx}/10] ❌ HTTP {res.status_code}")

    return results

if __name__ == "__main__":
    out = {}
    for name, url in TARGETS.items():
        out[name] = test_target(name, url)

    with open("tests/local_and_cf_comparison.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSaved comparison to tests/local_and_cf_comparison.json")
