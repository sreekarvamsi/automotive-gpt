"""Tests for src/api/main.py — FastAPI endpoints."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

# We need to mock heavy dependencies before importing the app
# so they don't try to connect to real services at import time.
_patches = [
    patch("src.api.main.init_db"),
    patch("src.api.main.ConversationStore"),
    patch("src.api.main.HybridRetriever"),
    patch("src.api.main.AutomotiveGenerator"),
    patch("src.api.main._redis"),
]


@pytest.fixture(autouse=True)
def mock_deps():
    """Start all patches before each test, stop after."""
    started = [p.start() for p in _patches]
    yield started
    for p in _patches:
        p.stop()


@pytest.fixture
def client():
    """Create a fresh TestClient, triggering the startup event."""
    from src.api.main import app
    with TestClient(app) as c:
        yield c


# ── Health endpoint ───────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "redis" in data
        assert "db" in data


# ── Chat endpoint ─────────────────────────────────────────────────────
class TestChatEndpoint:
    def test_chat_requires_message(self, client):
        resp = client.post("/api/v1/chat", json={"session_id": "s1"})
        assert resp.status_code == 422  # validation error — message is required

    def test_chat_returns_answer(self, client):
        from src.api import main as api_main

        # Wire up mocks
        mock_store = MagicMock()
        mock_store.create.return_value = "conv-123"
        mock_store.get_history_for_prompt.return_value = []
        api_main._store = mock_store

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []  # empty context
        api_main._retriever = mock_retriever

        mock_gen_result = MagicMock()
        mock_gen_result.answer = "The torque spec is 25 N·m."
        mock_gen_result.confidence = 0.85
        mock_gen_result.sources = [{"source_id": 1, "source_file": "test.pdf", "page": 10, "section_type": "paragraph", "score": 0.9}]
        mock_gen_result.latency_ms = 1500
        mock_gen_result.usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_gen_result
        api_main._generator = mock_generator

        # Mock Redis cache miss
        api_main._cache_get = lambda key: None
        api_main._cache_set = lambda key, val: None

        resp = client.post("/api/v1/chat", json={
            "session_id": "test-session",
            "message": "What is the torque spec?",
            "filters": {"make": "Honda"},
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "The torque spec is 25 N·m."
        assert data["conversation_id"] == "conv-123"
        assert data["confidence"] == 0.85
        assert len(data["sources"]) == 1

    def test_chat_resumes_existing_conversation(self, client):
        from src.api import main as api_main

        mock_store = MagicMock()
        mock_store.get_history_for_prompt.return_value = [
            {"role": "user", "content": "Previous Q"},
            {"role": "assistant", "content": "Previous A"},
        ]
        api_main._store = mock_store

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = []
        api_main._retriever = mock_retriever

        mock_gen_result = MagicMock()
        mock_gen_result.answer = "Follow-up answer."
        mock_gen_result.confidence = 0.7
        mock_gen_result.sources = []
        mock_gen_result.latency_ms = 800
        mock_gen_result.usage = {}

        mock_generator = MagicMock()
        mock_generator.generate.return_value = mock_gen_result
        api_main._generator = mock_generator

        api_main._cache_get = lambda key: None
        api_main._cache_set = lambda key, val: None

        resp = client.post("/api/v1/chat", json={
            "session_id": "s1",
            "conversation_id": "existing-conv-id",
            "message": "Follow-up question",
        })

        assert resp.status_code == 200
        # Generator should have received the history
        call_kwargs = mock_generator.generate.call_args[1]
        assert len(call_kwargs["conversation_history"]) == 2


# ── Conversation endpoints ────────────────────────────────────────────
class TestConversationEndpoints:
    def test_get_conversation_not_found(self, client):
        from src.api import main as api_main
        api_main._store = MagicMock()
        api_main._store.get_conversation.return_value = None

        resp = client.get("/api/v1/conversations/nonexistent-id")
        assert resp.status_code == 404

    def test_get_conversation_success(self, client):
        from src.api import main as api_main
        api_main._store = MagicMock()
        api_main._store.get_conversation.return_value = {
            "id": "conv-1",
            "session_id": "sess-1",
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:01:00+00:00",
            "messages": [{"role": "user", "content": "Hi"}],
            "filters": {"make": "Honda"},
        }

        resp = client.get("/api/v1/conversations/conv-1")
        assert resp.status_code == 200
        assert resp.json()["id"] == "conv-1"

    def test_delete_conversation_not_found(self, client):
        from src.api import main as api_main
        api_main._store = MagicMock()
        api_main._store.delete.return_value = False

        resp = client.delete("/api/v1/conversations/bad-id")
        assert resp.status_code == 404

    def test_delete_conversation_success(self, client):
        from src.api import main as api_main
        api_main._store = MagicMock()
        api_main._store.delete.return_value = True

        resp = client.delete("/api/v1/conversations/conv-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"


# ── Vehicles endpoint ─────────────────────────────────────────────────
class TestVehiclesEndpoint:
    def test_vehicles_returns_list(self, client):
        resp = client.get("/api/v1/vehicles")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
