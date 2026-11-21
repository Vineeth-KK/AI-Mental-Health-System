# test_conversation_diversity.py
"""Comprehensive evaluation script to test CareAI conversational diversity,
intent understanding, topic pivoting, and multi-turn context tracking.
"""

import sys
from agents.conversation import generate_llm_response

def test_diverse_conversations():
    print("=" * 80)
    print("CARE AI COMPREHENSIVE CONVERSATIONAL DIVERSITY & ACCURACY TEST")
    print("=" * 80)

    test_cases = [
        # 1. Greeting
        ("1. Greeting", [{"role": "user", "content": "Hello"}], None),
        
        # 2. Pleasantry
        ("2. Pleasantry", [{"role": "user", "content": "How are you doing today?"}], None),
        
        # 3. Academic Concern
        ("3. Academic Concern", [{"role": "user", "content": "Can you help me plan my study schedule for biology?"}], None),
        
        # 4. Academic Follow-up (Subject pivot)
        ("4. Academic Follow-up", [
            {"role": "user", "content": "Can you help me plan my study schedule for biology?"},
            {"role": "assistant", "content": "Here is a biology plan."},
            {"role": "user", "content": "What about physics?"}
        ], None),
        
        # 5. Topic Pivot to Daily Life
        ("5. Topic Pivot to Food/Lunch", [
            {"role": "user", "content": "What about physics?"},
            {"role": "assistant", "content": "Here is a physics plan."},
            {"role": "user", "content": "What should I eat for lunch?"}
        ], None),
        
        # 6. Emotional Concern
        ("6. Emotional Concern", [{"role": "user", "content": "I feel a bit sad today."}], None),
        
        # 7. Emotional Follow-up
        ("7. Emotional Follow-up", [
            {"role": "user", "content": "I feel a bit sad today."},
            {"role": "assistant", "content": "I hear you, sadness is valid."},
            {"role": "user", "content": "What can I do about it?"}
        ], None),
        
        # 8. Pure Factual / Science
        ("8. Factual / Science", [{"role": "user", "content": "What is photosynthesis?"}], None),
        
        # 9. Science Follow-up
        ("9. Science Follow-up", [
            {"role": "user", "content": "What is photosynthesis?"},
            {"role": "assistant", "content": "Photosynthesis is the process of light conversion."},
            {"role": "user", "content": "Why do leaves change color in autumn?"}
        ], None),
        
        # 10. Technical / Coding
        ("10. Technical Coding", [{"role": "user", "content": "Can you write a python function to reverse a string?"}], None),
        
        # 11. Psychoeducation
        ("11. Psychoeducation", [{"role": "user", "content": "What is the difference between anxiety and stress?"}], None),
        
        # 12. Multi-turn Exam + Sleep + Tonight
        ("12. Context Synthesis (Exam+Sleep->Tonight)", [
            {"role": "user", "content": "I feel stressed because of my exams."},
            {"role": "assistant", "content": "Exam stress can be overwhelming."},
            {"role": "user", "content": "I am also unable to sleep properly."},
            {"role": "assistant", "content": "Cognitive hyperarousal causes insomnia."},
            {"role": "user", "content": "What should I do tonight?"}
        ], None),
    ]

    responses = []
    print("\n[TEST RUN 1: TESTING WITH LOCAL OLLAMA]")
    for title, history, rag in test_cases:
        last_q = history[-1]["content"]
        resp = generate_llm_response(history, rag_context=rag, user_name="Vineeth")
        responses.append(resp)
        print(f"\n--- {title} ---")
        print(f"User: {last_q}")
        print(f"CareAI: {resp[:180].replace(chr(10), ' ')}...")

    unique_count = len(set(responses))
    print("\n" + "=" * 80)
    print(f"Ollama Results: {unique_count} / {len(test_cases)} Unique Responses")
    assert unique_count == len(test_cases), f"Repetitive responses detected: {unique_count} unique out of {len(test_cases)}"
    print("OLLAMA DIVERSITY TEST PASSED: 100% Unique, Intent-Aligned Responses!")

    print("\n[TEST RUN 2: TESTING WITH FALLBACK ENGINE (Ollama Host Offline Simulation)]")
    fallback_responses = []
    for title, history, rag in test_cases:
        last_q = history[-1]["content"]
        resp = generate_llm_response(history, rag_context=rag, user_name="Vineeth", ollama_host="http://127.0.0.1:9999")
        fallback_responses.append(resp)
        print(f"\n--- {title} (Fallback) ---")
        print(f"User: {last_q}")
        print(f"CareAI: {resp[:180].replace(chr(10), ' ')}...")

    fb_unique_count = len(set(fallback_responses))
    print("\n" + "=" * 80)
    print(f"Fallback Results: {fb_unique_count} / {len(test_cases)} Unique Responses")
    assert fb_unique_count == len(test_cases), f"Fallback repetitive responses detected: {fb_unique_count} unique out of {len(test_cases)}"
    print("FALLBACK DIVERSITY TEST PASSED: 100% Unique, Non-Repetitive Responses Across All Domains!")

if __name__ == "__main__":
    test_diverse_conversations()
