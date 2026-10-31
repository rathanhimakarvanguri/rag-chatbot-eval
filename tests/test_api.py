from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ask_returns_cited_answer():
    r = client.post("/ask", json={"question": "What is the daily meal allowance during travel?"})
    body = r.json()
    assert r.status_code == 200
    assert "1500" in body["answer"].replace(",", "")
    assert body["citations"][0]["chunk_id"] == "expenses#hotels-and-meals"


def test_ask_validates_input():
    assert client.post("/ask", json={"question": ""}).status_code == 422
