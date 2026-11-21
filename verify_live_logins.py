import requests
import re

session = requests.Session()

def test_login(email, password, base_url="http://127.0.0.1:5000"):
    s = requests.Session()
    r1 = s.get(f"{base_url}/login")
    match = re.search(r'name="csrf_token" type="hidden" value="([^"]+)"', r1.text) or re.search(r'value="([^"]+)" id="csrf_token"', r1.text) or re.search(r'id="csrf_token" name="csrf_token" type="hidden" value="([^"]+)"', r1.text)
    csrf = match.group(1) if match else None
    
    r2 = s.post(f"{base_url}/login", data={
        "email": email,
        "password": password,
        "csrf_token": csrf,
        "submit": "Login"
    }, allow_redirects=True)
    
    success = ("Logged in successfully" in r2.text) or ("dashboard" in r2.url)
    print(f"[{base_url}] Login '{email}': status={r2.status_code}, redirect_url={r2.url}, success={success}")
    return success

print("Testing accounts locally:")
test_login("demo@example.com", "password123")
test_login("counselor@careai.com", "password123")
test_login("admin@careai.com", "password123")

print("\nTesting accounts via public tunnel:")
test_login("demo@example.com", "password123", "https://likelihood-savannah-paintings-workplace.trycloudflare.com")
test_login("counselor@careai.com", "password123", "https://likelihood-savannah-paintings-workplace.trycloudflare.com")

