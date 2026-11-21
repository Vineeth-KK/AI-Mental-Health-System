# verify_care_ai_scenarios.py
"""Comprehensive functional test script for Care AI conversation intelligence.
Executes all 10 scenario benchmarks specified in the prompt.
"""

from app import create_app
from agents.conversation import generate_llm_response
from agents.orchestrator import orchestrator

app = create_app()

def run_tests():
    print("=" * 80)
    print("CARE AI CONVERSATION INTELLIGENCE: 10 SCENARIO EVALUATION")
    print("=" * 80)

    results = {}

    # Scenario 1: General question
    q1 = "What is stress?"
    r1 = generate_llm_response([{"role": "user", "content": q1}], user_name="Vineeth")
    print(f"\n[SCENARIO 1] General Question: '{q1}'")
    print(f"Response ({len(r1)} chars):\n{r1}\n{'-'*60}")
    results["scenario_1"] = ("stress" in r1.lower() and len(r1) > 100)

    # Scenario 2: Personal concern
    q2 = "I feel lonely even when I am around people."
    r2 = generate_llm_response([{"role": "user", "content": q2}], user_name="Vineeth")
    print(f"\n[SCENARIO 2] Personal Concern: '{q2}'")
    print(f"Response ({len(r2)} chars):\n{r2}\n{'-'*60}")
    results["scenario_2"] = ("lonel" in r2.lower() and len(r2) > 100)

    # Scenario 3: Academic concern
    q3 = "I am stressed about my upcoming examinations."
    r3 = generate_llm_response([{"role": "user", "content": q3}], user_name="Vineeth")
    print(f"\n[SCENARIO 3] Academic Concern: '{q3}'")
    print(f"Response ({len(r3)} chars):\n{r3}\n{'-'*60}")
    results["scenario_3"] = ("exam" in r3.lower() and len(r3) > 100)

    # Scenario 4: Follow-up question
    q4 = "What can I do about it?"
    history_s4 = [
        {"role": "user", "content": q3},
        {"role": "assistant", "content": r3},
        {"role": "user", "content": q4}
    ]
    r4 = generate_llm_response(history_s4, user_name="Vineeth")
    print(f"\n[SCENARIO 4] Follow-up Question: '{q4}' (Following Exam Stress)")
    print(f"Response ({len(r4)} chars):\n{r4}\n{'-'*60}")
    results["scenario_4"] = (("exam" in r4.lower() or "study" in r4.lower() or "syllabus" in r4.lower()) and len(r4) > 100)

    # Scenario 5: Practical request
    q5 = "Give me a simple daily routine to reduce stress."
    r5 = generate_llm_response([{"role": "user", "content": q5}], user_name="Vineeth")
    print(f"\n[SCENARIO 5] Practical Request: '{q5}'")
    print(f"Response ({len(r5)} chars):\n{r5}\n{'-'*60}")
    results["scenario_5"] = ("routine" in r5.lower() and "morning" in r5.lower() and len(r5) > 100)

    # Scenario 6: Informational question
    q6 = "What is the difference between anxiety and stress?"
    r6 = generate_llm_response([{"role": "user", "content": q6}], user_name="Vineeth")
    print(f"\n[SCENARIO 6] Informational Question: '{q6}'")
    print(f"Response ({len(r6)} chars):\n{r6}\n{'-'*60}")
    results["scenario_6"] = ("stress" in r6.lower() and "anxiety" in r6.lower() and "difference" in r6.lower() or "distinction" in r6.lower())

    # Scenario 7: Casual conversation
    q7 = "What can I do when I feel bored?"
    r7 = generate_llm_response([{"role": "user", "content": q7}], user_name="Vineeth")
    print(f"\n[SCENARIO 7] Casual Conversation: '{q7}'")
    print(f"Response ({len(r7)} chars):\n{r7}\n{'-'*60}")
    results["scenario_7"] = ("bored" in r7.lower() and len(r7) > 100)

    # Scenario 8: Sleep-related question
    q8 = "Why am I unable to sleep properly?"
    r8 = generate_llm_response([{"role": "user", "content": q8}], user_name="Vineeth")
    print(f"\n[SCENARIO 8] Sleep-related Question: '{q8}'")
    print(f"Response ({len(r8)} chars):\n{r8}\n{'-'*60}")
    results["scenario_8"] = ("sleep" in r8.lower() and len(r8) > 100)

    # Scenario 9: Multi-turn Context Test (Exam Stress -> Sleep -> What should I do tonight?)
    t1_msg = "I feel stressed because of my exams."
    t1_resp = generate_llm_response([{"role": "user", "content": t1_msg}], user_name="Vineeth")
    t2_msg = "I am also unable to sleep properly."
    t2_history = [
        {"role": "user", "content": t1_msg},
        {"role": "assistant", "content": t1_resp},
        {"role": "user", "content": t2_msg}
    ]
    t2_resp = generate_llm_response(t2_history, user_name="Vineeth")
    t3_msg = "What should I do tonight?"
    t3_history = t2_history + [
        {"role": "assistant", "content": t2_resp},
        {"role": "user", "content": t3_msg}
    ]
    t3_resp = generate_llm_response(t3_history, user_name="Vineeth")
    print(f"\n[SCENARIO 9] Multi-turn Context Synthesis:")
    print(f"  Turn 1: '{t1_msg}'")
    print(f"  Turn 2: '{t2_msg}'")
    print(f"  Turn 3: '{t3_msg}'")
    print(f"Turn 3 Synthesized Response ({len(t3_resp)} chars):\n{t3_resp}\n{'-'*60}")
    results["scenario_9"] = (("tonight" in t3_resp.lower() or "evening" in t3_resp.lower()) and 
                             ("exam" in t3_resp.lower() or "study" in t3_resp.lower()) and 
                             ("sleep" in t3_resp.lower() or "rest" in t3_resp.lower()))

    # Scenario 10: Unrelated Question Pivot
    # Immediately after talking about exams, user asks a completely unrelated casual question
    q10 = "What can I do when I feel bored?"
    history_s10 = t3_history + [
        {"role": "assistant", "content": t3_resp},
        {"role": "user", "content": q10}
    ]
    r10 = generate_llm_response(history_s10, user_name="Vineeth")
    print(f"\n[SCENARIO 10] Unrelated Question Pivot: '{q10}' (Pivoting away from exam/sleep)")
    print(f"Response ({len(r10)} chars):\n{r10}\n{'-'*60}")
    # Must answer about boredom, NOT repeat exam study steps
    results["scenario_10"] = ("bored" in r10.lower() and len(r10) > 100)

    # Verification of non-repetitiveness across all 10 responses
    responses = [r1, r2, r3, r4, r5, r6, r7, r8, t3_resp, r10]
    unique_responses = len(set(responses))
    print("\n" + "=" * 80)
    print("DIVERSITY & REPETITION CHECK:")
    print(f"Total Scenarios Tested: {len(responses)}")
    print(f"Unique Distinct Responses: {unique_responses} / {len(responses)}")
    for s_name, passed in results.items():
        status_str = "PASS" if passed else "FAIL"
        print(f"  - {s_name}: {status_str}")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()
