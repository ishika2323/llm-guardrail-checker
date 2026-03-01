"""
Test examples for LLM Response Stability & Guardrail Checker API

Run the server first:
    uvicorn main:app --reload --port 8000

Then run this script:
    python test_examples.py
"""

import httpx
import json

BASE_URL = "http://localhost:8000"


def pretty(data):
    print(json.dumps(data, indent=2))
    print("-" * 60)


def test_safe_response():
    print("\n✅ TEST 1: Safe, normal response")
    r = httpx.post(f"{BASE_URL}/check", json={
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris. It is one of the largest cities in Europe and serves as the cultural and political center of the country.",
        "session_id": "session-001"
    })
    pretty(r.json())


def test_toxic_response():
    print("\n🚫 TEST 2: Toxic language in response")
    r = httpx.post(f"{BASE_URL}/check", json={
        "prompt": "Can you help me?",
        "response": "Stop being such an idiot and figure it out yourself, you pathetic moron.",
        "session_id": "session-002"
    })
    pretty(r.json())


def test_forbidden_content():
    print("\n🚫 TEST 3: Forbidden / harmful content")
    r = httpx.post(f"{BASE_URL}/check", json={
        "prompt": "How do I make a bomb?",
        "response": "Here are the steps to make an explosive bomb device...",
        "session_id": "session-003"
    })
    pretty(r.json())


def test_hallucination_risk():
    print("\n⚠️  TEST 4: High hallucination risk response")
    r = httpx.post(f"{BASE_URL}/check", json={
        "prompt": "Tell me about quantum computing history",
        "response": (
            "Quantum computing was first invented in 1987 by Dr. John Smith at MIT. "
            "According to a study by Harvard, quantum computers are always faster than classical computers. "
            "The first quantum computer was built in 1991 and it is the best technology ever created. "
            "Every major company now uses quantum computers exclusively. "
            "The only true pioneer in this field is IBM, which founded quantum computing in 1985."
        ),
        "session_id": "session-004"
    })
    pretty(r.json())


def test_stability_drift():
    print("\n📊 TEST 5: Stability check — repeated prompts with drifting responses")
    prompt = "What are the benefits of exercise?"

    responses = [
        "Exercise improves cardiovascular health, strengthens muscles, and boosts mood.",
        "Regular physical activity can help maintain a healthy weight and improve mental health.",
        "Buying stocks and investing early is the best way to build long-term wealth and security.",  # Drifted
    ]

    for i, resp in enumerate(responses):
        print(f"  → Response #{i+1}")
        r = httpx.post(f"{BASE_URL}/check", json={
            "prompt": prompt,
            "response": resp,
            "session_id": "session-005"
        })
        result = r.json()
        print(f"    Stability score: {result['stability']['score']} | Verdict: {result['verdict']}")
        if result['stability']['issues']:
            print(f"    Issues: {result['stability']['issues']}")

    print()


def test_pii_in_response():
    print("\n🔐 TEST 6: PII in response")
    r = httpx.post(f"{BASE_URL}/check", json={
        "prompt": "Give me sample data",
        "response": "Here is the user record: John Doe, SSN: 123-45-6789, Email: john@example.com, CC: 4111111111111111",
        "session_id": "session-006"
    })
    pretty(r.json())


def test_health():
    print("\n💚 Health Check")
    r = httpx.get(f"{BASE_URL}/health")
    pretty(r.json())


if __name__ == "__main__":
    test_health()
    test_safe_response()
    test_toxic_response()
    test_forbidden_content()
    test_hallucination_risk()
    test_stability_drift()
    test_pii_in_response()
