"""HTTP contract tests for the FastAPI layer.

``TestClient`` is used without its context manager so the application lifespan
does not rebuild the index: each test installs an already-initialized pipeline
instead.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from src.api import server


@pytest_asyncio.fixture
async def client(monkeypatch, make_pipeline):
    """A test client whose pipeline is scripted per test."""

    async def _factory(decisions=None, answers=None) -> TestClient:
        pipeline = await make_pipeline(decisions=decisions, answers=answers)
        monkeypatch.setattr(server, "pipeline", pipeline)
        test_client = TestClient(server.app)
        test_client.pipeline = pipeline  # type: ignore[attr-defined]
        return test_client

    return _factory


@pytest.mark.asyncio
async def test_health_reports_ready_components(client) -> None:
    test_client = await client()

    response = test_client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["components"]["retriever"] is True
    assert body["components"]["workflow"] is True
    assert body["components"]["knowledge_base"] is True


@pytest.mark.asyncio
async def test_health_reports_503_when_the_pipeline_is_down(client) -> None:
    test_client = await client()
    test_client.pipeline.ready = False

    response = test_client.get("/health")

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_chat_returns_the_documented_response_shape(client) -> None:
    test_client = await client(
        decisions=[{"route": "answer"}],
        answers=["Standard delivery takes three to five business days."],
    )

    response = test_client.post(
        "/chat",
        json={
            "session_id": "demo-session-1",
            "message": "How long does standard shipping take?",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "success": True,
        "session_id": "demo-session-1",
        "response": "Standard delivery takes three to five business days.",
        "sources": ["shipping.md"],
        "ticket_id": None,
    }


@pytest.mark.asyncio
async def test_chat_rejects_a_blank_message(client) -> None:
    test_client = await client()

    response = test_client.post(
        "/chat",
        json={"session_id": "demo-session-1", "message": ""},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_returns_502_when_the_model_is_unavailable(client) -> None:
    test_client = await client(decisions=[RuntimeError("connection refused")])

    response = test_client.post(
        "/chat",
        json={"session_id": "demo-session-1", "message": "How long is shipping?"},
    )

    assert response.status_code == 502
    assert response.json()["detail"]


@pytest.mark.asyncio
async def test_chat_returns_503_when_the_pipeline_is_not_ready(client) -> None:
    test_client = await client()
    test_client.pipeline.ready = False

    response = test_client.post(
        "/chat",
        json={"session_id": "demo-session-1", "message": "How long is shipping?"},
    )

    assert response.status_code == 503


@pytest.mark.asyncio
async def test_a_ticket_created_in_chat_is_retrievable_by_id(client) -> None:
    test_client = await client(
        decisions=[
            {
                "route": "ticket",
                "customer_name": "Asha Rao",
                "customer_email": "asha@example.com",
                "issue_description": "Charged twice for a single order",
            }
        ],
    )

    chat = test_client.post(
        "/chat",
        json={
            "session_id": "demo-session-2",
            "message": (
                "I am Asha Rao, asha@example.com, my payment was charged twice. "
                "Please open a ticket."
            ),
        },
    )

    ticket_id = chat.json()["ticket_id"]
    assert ticket_id

    response = test_client.get(f"/tickets/{ticket_id}")

    assert response.status_code == 200
    ticket = response.json()
    assert ticket["ticket_id"] == ticket_id
    assert ticket["category"] == "payment"
    assert ticket["customer_name"] == "Asha Rao"
    assert ticket["customer_email"] == "asha@example.com"
    assert ticket["status"] == "open"
    assert ticket["created_at"]
    assert ticket["summary"]


@pytest.mark.asyncio
async def test_unknown_ticket_id_returns_404(client) -> None:
    test_client = await client()

    response = test_client.get("/tickets/CST-2026-9999")

    assert response.status_code == 404
