from __future__ import annotations
from fastapi.testclient import TestClient
from api.main import app
from app.exceptions import OllamaUnavailableError, StockDataUnavailableError, VectorStoreUnavailableError

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "ollama_model" in body


def test_companies_lists_twenty():
    resp = client.get("/companies")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["companies"]) == 20
    assert set(body["fiscal_years"]) == {2023, 2024, 2025}


def test_query_blank_message_is_rejected():
    resp = client.post("/query", json={"message": "   "})
    assert resp.status_code == 422


def test_query_rule_based_happy_path(monkeypatch):
    def fake_ask_agent(message, conversation_company=None):
        return "Nvidia's main risks are X, Y, Z.", "nvidia"

    monkeypatch.setattr("app.agent.ask_agent", fake_ask_agent)

    resp = client.post("/query", json={"message": "What are Nvidia's risks?", "mode": "rule_based"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active_company"] == "nvidia"
    assert body["active_company_display"] == "Nvidia"
    assert body["mode"] == "rule_based"
    assert body["tools_invoked"] == []
    assert body["latency_ms"] >= 0


def test_query_agentic_happy_path(monkeypatch):
    def fake_run_agentic_query(message, conversation_company=None):
        return {
            "answer": "Nvidia's risks include X.",
            "active_company": "nvidia",
            "tools_invoked": [{"tool": "search_filing", "args": {"company": "nvidia", "question": message}}],
            "rounds": 1,
        }

    monkeypatch.setattr("app.agentic_router.run_agentic_query", fake_run_agentic_query)

    resp = client.post("/query", json={"message": "What are Nvidia's risks?", "mode": "agentic"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "agentic"
    assert len(body["tools_invoked"]) == 1
    assert body["tools_invoked"][0]["tool"] == "search_filing"


def test_query_maps_ollama_unavailable_to_503(monkeypatch):
    def fake_ask_agent(message, conversation_company=None):
        raise OllamaUnavailableError("connection refused")

    monkeypatch.setattr("app.agent.ask_agent", fake_ask_agent)

    resp = client.post("/query", json={"message": "What are Nvidia's risks?"})
    assert resp.status_code == 503
    assert "Ollama" in resp.json()["error"]


def test_query_maps_vector_store_unavailable_to_503(monkeypatch):
    def fake_ask_agent(message, conversation_company=None):
        raise VectorStoreUnavailableError("chroma_db missing")

    monkeypatch.setattr("app.agent.ask_agent", fake_ask_agent)

    resp = client.post("/query", json={"message": "What are Nvidia's risks?"})
    assert resp.status_code == 503
    assert "Vector store" in resp.json()["error"]


def test_query_maps_stock_data_unavailable_to_502(monkeypatch):
    def fake_ask_agent(message, conversation_company=None):
        raise StockDataUnavailableError("yfinance timeout")

    monkeypatch.setattr("app.agent.ask_agent", fake_ask_agent)

    resp = client.post("/query", json={"message": "How is NVDA stock doing?"})
    assert resp.status_code == 502
    assert "Stock data" in resp.json()["error"]
