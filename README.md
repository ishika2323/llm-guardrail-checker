# LLM Response Stability & Guardrail Checker API

A FastAPI-based service that evaluates LLM-generated responses across four critical dimensions:

| Check | What It Detects |
|---|---|
| **Guardrail** | Harmful content, PII, forbidden topics |
| **Toxicity** | Abusive language, aggression signals |
| **Hallucination Risk** | Absolute claims, unverifiable citations, missing uncertainty markers |
| **Stability** | Response drift across repeated prompts in the same session |

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Open docs: http://localhost:8000/docs

---

## 📡 API Endpoints

### `POST /check`
Evaluate an LLM response.

**Request Body:**
```json
{
  "prompt": "What is the capital of France?",
  "response": "The capital of France is Paris.",
  "session_id": "user-session-abc",
  "context": "Optional system prompt or conversation history"
}
```

**Response:**
```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "prompt_hash": "abc123...",
  "guardrail": { "passed": true, "score": 1.0, "issues": [], "details": {} },
  "toxicity":  { "passed": true, "score": 1.0, "issues": [], "details": {} },
  "hallucination_risk": { "passed": true, "score": 0.85, "issues": [], "details": {} },
  "stability": { "passed": true, "score": 0.9, "issues": [], "details": {} },
  "overall_score": 0.94,
  "verdict": "PASS",
  "summary": "Overall score: 0.94. Verdict: PASS. No issues detected."
}
```

**Verdict Logic:**
- `FAIL` → guardrail or toxicity check failed
- `WARN` → overall_score < 0.65
- `PASS` → all checks passed with score ≥ 0.65

---

### `GET /history/{session_id}`
Retrieve stored response history for a session (used for stability tracking).

### `DELETE /history/{session_id}`
Clear response history for a session.

### `GET /health`
Health check.

---

## 🧪 Run Tests

```bash
# Start the server first, then:
python test_examples.py
```

---

## 🏗️ Architecture

```
main.py
├── check_guardrails()     → Regex-based forbidden content + PII detection
├── check_toxicity()       → Toxic word list + caps/exclamation heuristics
├── check_hallucination_risk() → Absolute claims, citation patterns, missing hedges
├── check_stability()      → Word overlap across repeated prompts per session
└── POST /check            → Aggregates all checks → score + verdict
```

---

## 🔧 Extending

- **Swap regex with ML models**: Replace `check_toxicity()` with a call to a model like `unitary/toxic-bert`
- **Add LLM-as-judge**: Call Claude/GPT to evaluate factual accuracy
- **Persist history**: Replace in-memory dict with Redis for production use
- **Add authentication**: Use FastAPI's `Depends` with API key middleware
- **Webhook alerts**: POST to Slack/PagerDuty when verdict is FAIL
