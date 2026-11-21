"""
Comprehensive Functional Evaluation for CareAI Conversational Intelligence.
Runs all 10 test scenarios specified in Phase 7 through the multi-agent orchestrator.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from extensions import db
from models.user import User
from models.chat import ChatMessage
from agents.orchestrator import orchestrator

app = create_app()

def run_tests():
    with app.app_context():
        # Ensure a test user exists
        user = User.query.filter_by(email="demo@careai.com").first() or User.query.first()

        user_id = user.id
        user_name = user.username

        results = []

        print("=" * 70)
        print("CARE AI CONVERSATIONAL INTELLIGENCE: 10-SCENARIO FUNCTIONAL TEST SUITE")
        print("=" * 70)

        # Scenarios Definition
        scenarios = [
            {
                "id": 1,
                "name": "General question",
                "input": "What is stress?",
                "history": []
            },
            {
                "id": 2,
                "name": "Personal concern",
                "input": "I feel lonely even when I am around people.",
                "history": []
            },
            {
                "id": 3,
                "name": "Academic concern",
                "input": "I am stressed about my upcoming examinations.",
                "history": []
            },
            {
                "id": 4,
                "name": "Follow-up question (context dependent)",
                "input": "What can I do about it?",
                "history": [
                    {"role": "user", "content": "I am stressed about my upcoming examinations."},
                    {"role": "assistant", "content": "Exam pressure is heavy. Dividing your subjects into manageable chunks can help prevent overwhelm."}
                ]
            },
            {
                "id": 5,
                "name": "Practical request",
                "input": "Give me a simple daily routine to reduce stress.",
                "history": []
            },
            {
                "id": 6,
                "name": "Informational question",
                "input": "What is the difference between anxiety and stress?",
                "history": []
            },
            {
                "id": 7,
                "name": "Casual conversation",
                "input": "What can I do when I feel bored?",
                "history": []
            },
            {
                "id": 8,
                "name": "Sleep-related question",
                "input": "Why am I unable to sleep properly?",
                "history": []
            },
            {
                "id": 9,
                "name": "Context test (2-turn multi-turn thread)",
                "input": "I am also unable to sleep properly tonight.",
                "history": [
                    {"role": "user", "content": "I feel stressed because of my final exams tomorrow."},
                    {"role": "assistant", "content": "Final exams can trigger intense physical and mental strain. Your performance improves when your nervous system is calm."}
                ]
            },
            {
                "id": 10,
                "name": "Unrelated topic shift",
                "input": "Can you explain how mindful walking works?",
                "history": [
                    {"role": "user", "content": "Why am I unable to sleep properly?"},
                    {"role": "assistant", "content": "Sleep problems often stem from bedtime hyperarousal."}
                ]
            }
        ]

        for sc in scenarios:
            print(f"\n[{sc['id']}/10] Testing: {sc['name']}")
            print(f"      Input: \"{sc['input']}\"")
            if sc['history']:
                print(f"      Prior turns: {len(sc['history'])} message(s)")

            t0 = time.time()
            # Format message array
            messages = list(sc["history"]) + [{"role": "user", "content": sc["input"]}]

            agent_result = orchestrator.handle_chat(
                user_id=user_id,
                user_name=user_name,
                messages=messages
            )
            elapsed = round(time.time() - t0, 2)
            response = agent_result.get("response", "").strip()

            print(f"      Duration: {elapsed}s")
            print(f"      Response preview (first 180 chars):")
            print(f"      \"{response[:180]}...\"")

            # Basic quality checks
            is_relevant = len(response) > 50 and not response.startswith("I am having difficulty")
            results.append({
                "id": sc["id"],
                "name": sc["name"],
                "input": sc["input"],
                "response": response,
                "elapsed": elapsed,
                "is_relevant": is_relevant
            })

        # Save test report
        output_file = PROJECT_ROOT / "tests" / "conversational_ai_test_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 70)
        print(f"TEST RUN COMPLETE. Results saved to: {output_file}")
        print("=" * 70)

if __name__ == "__main__":
    run_tests()
