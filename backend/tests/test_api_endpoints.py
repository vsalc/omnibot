"""API-level tests for the FastAPI endpoints (mirrored in ``conftest.test_app``).

These exercise request validation, response shape, and the wiring between each
route and ``RAGSystem`` (mocked). They do not touch ChromaDB, the embedding
model, or Anthropic. See ``conftest.py`` for why the real ``app.py`` isn't
imported directly.
"""

import pytest


# --------------------------------------------------------------------------- #
# POST /api/query
# --------------------------------------------------------------------------- #
def test_query_returns_answer_and_sources(client, mock_rag_system):
    resp = client.post("/api/query", json={"query": "What is MCP?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "MCP lets clients call tools over a server."
    assert body["sources"] == [
        {"text": "MCP Course - Lesson 1", "link": "https://example.com/mcp/1"}
    ]
    assert body["session_id"] == "session-1"
    mock_rag_system.query.assert_called_once_with("What is MCP?", "session-1")


def test_query_creates_session_when_missing(client, mock_rag_system):
    client.post("/api/query", json={"query": "hi"})

    # No session_id provided → one is created and used for the query.
    mock_rag_system.session_manager.create_session.assert_called_once()
    assert mock_rag_system.query.call_args.args[1] == "session-1"


def test_query_reuses_provided_session(client, mock_rag_system):
    resp = client.post(
        "/api/query", json={"query": "hi", "session_id": "existing-session"}
    )

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "existing-session"
    mock_rag_system.session_manager.create_session.assert_not_called()
    mock_rag_system.query.assert_called_once_with("hi", "existing-session")


def test_query_requires_query_field(client):
    resp = client.post("/api/query", json={"session_id": "s1"})
    assert resp.status_code == 422  # pydantic validation error


def test_query_propagates_rag_failure_as_500(client, mock_rag_system):
    mock_rag_system.query.side_effect = RuntimeError("vector store down")

    resp = client.post("/api/query", json={"query": "boom"})

    assert resp.status_code == 500
    assert "vector store down" in resp.json()["detail"]


def test_query_allows_empty_sources(client, mock_rag_system):
    mock_rag_system.query.return_value = ("No relevant content found.", [])

    resp = client.post("/api/query", json={"query": "unrelated"})

    assert resp.status_code == 200
    assert resp.json()["sources"] == []


# --------------------------------------------------------------------------- #
# GET /api/courses
# --------------------------------------------------------------------------- #
def test_courses_returns_stats(client, mock_rag_system):
    resp = client.get("/api/courses")

    assert resp.status_code == 200
    assert resp.json() == {
        "total_courses": 2,
        "course_titles": ["MCP Course", "Advanced RAG"],
    }
    mock_rag_system.get_course_analytics.assert_called_once()


def test_courses_propagates_failure_as_500(client, mock_rag_system):
    mock_rag_system.get_course_analytics.side_effect = RuntimeError("no db")

    resp = client.get("/api/courses")

    assert resp.status_code == 500
    assert "no db" in resp.json()["detail"]


# --------------------------------------------------------------------------- #
# POST /api/session/clear
# --------------------------------------------------------------------------- #
def test_clear_session_calls_manager(client, mock_rag_system):
    resp = client.post("/api/session/clear", json={"session_id": "s1"})

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
    mock_rag_system.session_manager.clear_session.assert_called_once_with("s1")


def test_clear_session_requires_session_id(client):
    resp = client.post("/api/session/clear", json={})
    assert resp.status_code == 422


# --------------------------------------------------------------------------- #
# GET /  (frontend root)
# --------------------------------------------------------------------------- #
def test_root_is_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
