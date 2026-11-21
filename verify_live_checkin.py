import requests
import re

session = requests.Session()
r1 = session.get("http://127.0.0.1:5000/login")
match = (
    re.search(r'name="csrf_token" type="hidden" value="([^"]+)"', r1.text) or
    re.search(r'value="([^"]+)" id="csrf_token"', r1.text) or
    re.search(r'id="csrf_token" name="csrf_token" type="hidden" value="([^"]+)"', r1.text)
)
csrf = match.group(1) if match else None

login_res = session.post("http://127.0.0.1:5000/login", data={
    "email": "demo@example.com",
    "password": "password123",
    "csrf_token": csrf,
    "submit": "Login"
})
print("Login status:", login_res.status_code)

# Post a check-in
r_checkin = session.post(
    "http://127.0.0.1:5000/api/checkin",
    json={
        "mood": 6,
        "anxiety": 5,
        "stress": 4,
        "sleep_quality": 7,
        "isolation_score": 3
    },
    headers={"X-CSRFToken": csrf}
)
print("Checkin response status:", r_checkin.status_code)
data = r_checkin.json()

print(f"Distress Score: {data.get('distress_score')} (Level: {data.get('distress_level')})")
print(f"Trend Trajectory: {data.get('trend_analysis', {}).get('trend')} (Slope: {data.get('trend_analysis', {}).get('slope')})")
print(f"Exercises Returned: {len(data.get('recommendations', []))}")
for i, ex in enumerate(data.get("recommendations", [])):
    print(f"  [{i+1}] {ex.get('title')} ({ex.get('category')}) - {ex.get('steps')[:50]}...")

print("\nMulti-Agent Evaluation Trace:")
eval_data = data.get("multi_agent_evaluation", {})
print(f"Total: {eval_data.get('successful_agents')}/{eval_data.get('total_agents')} Verified")
for t in eval_data.get("trace", []):
    print(f"  {t.get('agent')}: [{t.get('status').upper()}] {t.get('contribution')}")
