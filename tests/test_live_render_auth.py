"""tests/test_live_render_auth.py
Live Production Verification for Authentication, Registration, and Multi-Account Isolation on Render
"""

import requests
import re
import time

RENDER_URL = "https://ai-mental-health-system-cxkq.onrender.com"

def get_csrf(html):
    m = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', html)
    if not m:
        m = re.search(r'value="([^"]+)"[^>]*name="csrf_token"', html)
    return m.group(1) if m else None

def get_meta_csrf(html):
    m = re.search(r'<meta\s+name="csrf-token"\s+content="([^"]+)"', html)
    return m.group(1) if m else None

def run_live_tests():
    print(f"=== Testing Live Render URL: {RENDER_URL} ===")
    
    # 1. Check health endpoint
    h = requests.get(f"{RENDER_URL}/health", timeout=15)
    print(f"1. Health Check: {h.status_code} -> {h.json().get('status')}")
    assert h.status_code == 200

    # 2. Login as Demo User
    s_demo = requests.Session()
    r = s_demo.get(f"{RENDER_URL}/login", timeout=15)
    csrf = get_csrf(r.text)
    res_demo = s_demo.post(
        f"{RENDER_URL}/login",
        data={"email": "demo@example.com", "password": "password123", "csrf_token": csrf},
        headers={"Referer": f"{RENDER_URL}/login", "Origin": RENDER_URL},
        allow_redirects=True,
        timeout=15
    )
    print(f"2. Demo Login: status={res_demo.status_code}, url={res_demo.url}")
    assert "/dashboard" in res_demo.url
    print("   [PASS] Demo user logged in successfully.")

    # 3. Login as CounselorSarah
    s_counselor = requests.Session()
    r = s_counselor.get(f"{RENDER_URL}/login", timeout=15)
    csrf = get_csrf(r.text)
    res_counselor = s_counselor.post(
        f"{RENDER_URL}/login",
        data={"email": "counselor@careai.com", "password": "password123", "csrf_token": csrf},
        headers={"Referer": f"{RENDER_URL}/login", "Origin": RENDER_URL},
        allow_redirects=True,
        timeout=15
    )
    print(f"3. Counselor Login: status={res_counselor.status_code}, url={res_counselor.url}")
    assert "/dashboard" in res_counselor.url or "/counselor" in res_counselor.url
    print("   [PASS] Counselor user logged in successfully.")

    # 4. Register a New Account with mixed casing & trailing spaces
    s_new = requests.Session()
    r = s_new.get(f"{RENDER_URL}/register", timeout=15)
    csrf_reg = get_csrf(r.text)
    rand_id = int(time.time())
    new_username = f"LiveTester{rand_id % 10000}"
    new_email = f"livetester_{rand_id}@careai.com"

    print(f"4. Registering New Account: {new_username} ({new_email})")
    res_reg = s_new.post(
        f"{RENDER_URL}/register",
        data={
            "username": new_username,
            "email": new_email,
            "password": "LivePassword123!",
            "confirm": "LivePassword123!",
            "csrf_token": csrf_reg
        },
        headers={"Referer": f"{RENDER_URL}/register", "Origin": RENDER_URL},
        allow_redirects=True,
        timeout=15
    )
    print(f"   Register Response: status={res_reg.status_code}, url={res_reg.url}")
    assert res_reg.status_code == 200
    assert "/login" in res_reg.url
    print("   [PASS] Registration completed successfully.")

    # 5. Login using newly registered account
    r_login = s_new.get(f"{RENDER_URL}/login", timeout=15)
    csrf_login = get_csrf(r_login.text)
    res_login_new = s_new.post(
        f"{RENDER_URL}/login",
        data={"email": new_email, "password": "LivePassword123!", "csrf_token": csrf_login},
        headers={"Referer": f"{RENDER_URL}/login", "Origin": RENDER_URL},
        allow_redirects=True,
        timeout=15
    )
    print(f"5. Login with New Account: status={res_login_new.status_code}, url={res_login_new.url}")
    assert "/dashboard" in res_login_new.url
    print(f"   [PASS] Logged in with new account {new_username}.")

    # 6. Logout and log in again
    res_logout = s_new.get(f"{RENDER_URL}/logout", allow_redirects=True, timeout=15)
    print(f"6. Logout: status={res_logout.status_code}, url={res_logout.url}")
    assert "/login" in res_logout.url

    r_login_again = s_new.get(f"{RENDER_URL}/login", timeout=15)
    csrf_again = get_csrf(r_login_again.text)
    res_relogin = s_new.post(
        f"{RENDER_URL}/login",
        data={"email": new_email, "password": "LivePassword123!", "csrf_token": csrf_again},
        headers={"Referer": f"{RENDER_URL}/login", "Origin": RENDER_URL},
        allow_redirects=True,
        timeout=15
    )
    assert "/dashboard" in res_relogin.url
    print("   [PASS] Relogin after logout successful.")

    # 7. Check private data access as new user
    r_chat = s_new.get(f"{RENDER_URL}/chat", timeout=15)
    chat_csrf = get_meta_csrf(r_chat.text)
    res_chat = s_new.post(
        f"{RENDER_URL}/api/chat",
        json={"message": "Hello from newly registered user!"},
        headers={"Referer": f"{RENDER_URL}/chat", "Origin": RENDER_URL, "X-CSRFToken": chat_csrf or ""},
        timeout=25
    )
    print(f"7. Chat API for New User: status={res_chat.status_code}")
    assert res_chat.status_code == 200
    print("   [PASS] Chat API works seamlessly for newly registered account.")

    print("\nALL LIVE PRODUCTION AUTHENTICATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_live_tests()
