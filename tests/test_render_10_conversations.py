"""tests/test_render_10_conversations.py
Comprehensive 10-Question and Multi-Turn Follow-Up Verification Script
for Deployed Production Website: https://ai-mental-health-system-cxkq.onrender.com
"""

import sys
import json
import time
import requests
import re

RENDER_URL = "https://ai-mental-health-system-cxkq.onrender.com"

QUESTIONS = [
    {
        "id": 1,
        "type": "New Topic",
        "prompt": "What is quantum computing?"
    },
    {
        "id": 2,
        "type": "Follow-up",
        "prompt": "Can you give me a simple real-world analogy for that?"
    },
    {
        "id": 3,
        "type": "New Topic",
        "prompt": "What are the three primary laws of robotics proposed by Isaac Asimov?"
    },
    {
        "id": 4,
        "type": "Academic Guidance",
        "prompt": "I'm feeling stuck on a difficult calculus problem. How do you suggest I approach it?"
    },
    {
        "id": 5,
        "type": "Everyday Planning",
        "prompt": "Can you suggest a quick and healthy dinner recipe under 20 minutes?"
    },
    {
        "id": 6,
        "type": "Emotional Wellbeing",
        "prompt": "I have been feeling really overwhelmed with work and exams lately."
    },
    {
        "id": 7,
        "type": "Follow-up / Reflection",
        "prompt": "What is the single smallest step I should take tonight to decompress?"
    },
    {
        "id": 8,
        "type": "Coding / Python",
        "prompt": "Write a clean Python function to reverse words in a sentence."
    },
    {
        "id": 9,
        "type": "Follow-up / Code Refinement",
        "prompt": "Can you explain how that function handles punctuation or extra spaces?"
    },
    {
        "id": 10,
        "type": "General / Humor",
        "prompt": "Tell me a lighthearted joke about programmers."
    }
]

def main():
    print("=" * 70)
    print("Testing Production Website: " + RENDER_URL)
    print("=" * 70)

    session = requests.Session()
    session.headers.update({
        "Referer": f"{RENDER_URL}/login",
        "Origin": RENDER_URL
    })

    # Step 1: Login
    print("\n[Step 1] Authenticating as demo@example.com on Render...")
    r1 = session.get(f"{RENDER_URL}/login", timeout=20)
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

    session.headers["X-CSRFToken"] = csrf_token
    login_res = session.post(
        f"{RENDER_URL}/login",
        data={"email": "demo@example.com", "password": "password123", "csrf_token": csrf_token},
        allow_redirects=True,
        timeout=20
    )
    if login_res.status_code != 200 or "dashboard" not in login_res.url:
        print(f"Login failed: status {login_res.status_code}, url {login_res.url}")
        sys.exit(1)
    print("Authentication successful!")

    # Step 2: Run 10 conversational turns
    history = []
    results = []
    canned_phrases = [
        "There are several angles we can explore regarding",
        "{rag_snippet_str}",
        "We can unpack the core concepts and practical details behind it",
        "We can look at how to apply it step-by-step"
    ]

    for item in QUESTIONS:
        q_id = item["id"]
        q_type = item["type"]
        prompt = item["prompt"]

        print(f"\n--- Turn #{q_id} [{q_type}] ---")
        print(f"User: {prompt}")

        start_time = time.time()
        payload = {
            "message": prompt,
            "history": history[-6:]
        }
        res = session.post(f"{RENDER_URL}/api/chat", json=payload, timeout=50)
        latency = round(time.time() - start_time, 2)

        if res.status_code != 200:
            print(f"❌ Error HTTP {res.status_code}: {res.text[:200]}")
            continue

        data = res.json()
        reply = data.get("response", "").strip()

        # Check for repetitive canned templates
        has_canned = any(p in reply for p in canned_phrases)
        is_repetitive = bool(has_canned)

        preview = reply.replace("\n", " ")[:140]
        status = "⚠️ CANNED" if is_repetitive else "✅ DYNAMIC"
        print(f"{status} ({latency}s) [{len(reply)} chars]: {preview}...")

        results.append({
            "id": q_id,
            "type": q_type,
            "prompt": prompt,
            "latency": latency,
            "is_canned": is_repetitive,
            "response": reply
        })

        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})

    out_file = "tests/render_10_conversations_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved all results to {out_file}")

if __name__ == "__main__":
    main()
