from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_empty_query_returns_400():
    response = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": ""}]}
    )
    assert response.status_code == 400


def test_whitespace_only_query_returns_400():
    response = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "   "}]}
    )
    assert response.status_code == 400


def test_no_user_messages_returns_400():
    response = client.post(
        "/api/chat", json={"messages": [{"role": "system", "content": "hi"}]}
    )
    assert response.status_code == 400


def test_health_endpoint_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama" in data
    assert "pgvector" in data


def test_search_empty_query_returns_400():
    response = client.post("/api/search", json={"query": ""})
    assert response.status_code == 400


def test_search_whitespace_query_returns_400():
    response = client.post("/api/search", json={"query": "   "})
    assert response.status_code == 400


def _mock_chain():
    async def _fake_stream(_input):
        yield {"context": []}
        yield {"answer": "Mocked answer."}

    chain = MagicMock()
    chain.astream = _fake_stream
    return chain


@patch("app.api.endpoints.chat.get_rag_chain", return_value=_mock_chain())
def test_long_query_handled(_mock):
    long_query = "What is the law? " * 200
    response = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": long_query}]}
    )
    assert response.status_code == 200


@patch("app.api.endpoints.chat.get_rag_chain", return_value=_mock_chain())
def test_special_characters_handled(_mock):
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "What is § 302 BNS?"}]},
    )
    assert response.status_code == 200


@patch("app.api.endpoints.chat.get_rag_chain", return_value=_mock_chain())
def test_long_query_truncated(_mock):
    long_query = "Section 103 BNS " * 300
    response = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": long_query}]}
    )
    assert response.status_code == 200
    _mock.assert_called_once_with(k=3)


@patch("app.api.endpoints.chat.get_rag_chain")
def test_greeting_bypasses_rag(_mock):
    response = client.post(
        "/api/chat", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200
    assert "legal research assistant for Indian law" in response.text
    _mock.assert_not_called()


@patch("app.api.endpoints.chat.get_rag_chain")
def test_out_of_scope_query_bypasses_rag(_mock):
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "tell me a joke"}]},
    )
    assert response.status_code == 200
    assert "limited to Indian legal research" in response.text
    _mock.assert_not_called()


@patch("app.api.endpoints.chat.get_rag_chain", return_value=_mock_chain())
def test_legal_query_still_uses_rag(_mock):
    response = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": "What is Section 103 BNS?"}]},
    )
    assert response.status_code == 200
    assert "Mocked answer." in response.text
    _mock.assert_called_once_with(k=5)
