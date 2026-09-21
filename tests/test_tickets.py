"""The mock ticket repository and the validated tool bound to it."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import TicketCreate
from src.tools.ticket_tool import TicketRepository, create_ticket_tool


def sample(**overrides) -> TicketCreate:
    values = {
        "customer_name": "Test User",
        "customer_email": "test@example.com",
        "issue_description": "Payment was charged twice",
        "category": "payment",
        "summary": "Possible duplicate payment charge",
    }
    values.update(overrides)
    return TicketCreate(**values)


def test_repository_prevents_duplicate_ticket_per_session() -> None:
    # The repository is also an integration contract for the agent tool:
    # one session must produce one ticket.
    repository = TicketRepository()
    request = sample()

    first = repository.create("session-1", request)
    second = repository.create("session-1", request)

    assert first.ticket_id == second.ticket_id
    assert len(list(repository.all())) == 1


def test_different_sessions_receive_unique_ticket_ids() -> None:
    repository = TicketRepository()

    first = repository.create("session-1", sample())
    second = repository.create("session-2", sample(customer_name="Other User"))

    assert first.ticket_id != second.ticket_id
    assert len(list(repository.all())) == 2


def test_a_created_ticket_preserves_every_supplied_detail() -> None:
    repository = TicketRepository()

    ticket = repository.create("session-1", sample())

    assert ticket.customer_name == "Test User"
    assert ticket.customer_email == "test@example.com"
    assert ticket.issue_description == "Payment was charged twice"
    assert ticket.category == "payment"
    assert ticket.summary == "Possible duplicate payment charge"

    # Repository-owned fields the model must never supply.
    assert ticket.ticket_id
    assert ticket.status == "open"
    assert ticket.created_at is not None


def test_a_created_ticket_is_retrievable_by_id() -> None:
    repository = TicketRepository()

    ticket = repository.create("session-1", sample())

    assert repository.get(ticket.ticket_id) == ticket


def test_an_unknown_ticket_id_returns_none() -> None:
    assert TicketRepository().get("CST-2026-9999") is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"customer_email": "not-an-email"},
        {"category": "billing"},
        {"customer_name": ""},
        {"issue_description": "x"},
        {"summary": "y" * 161},
    ],
)
def test_invalid_ticket_details_are_rejected_before_any_side_effect(overrides) -> None:
    with pytest.raises(ValidationError):
        sample(**overrides)


def test_the_tool_validates_before_writing_to_the_repository() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-1")

    with pytest.raises(Exception):
        tool.invoke(
            {
                "customer_name": "Test User",
                "customer_email": "not-an-email",
                "issue_description": "Payment was charged twice",
                "category": "payment",
                "summary": "Possible duplicate payment charge",
            }
        )

    assert list(repository.all()) == []


def test_the_tool_returns_the_repository_issued_id() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-1")

    ticket_id = tool.invoke(
        {
            "customer_name": "Test User",
            "customer_email": "test@example.com",
            "issue_description": "Payment was charged twice",
            "category": "payment",
            "summary": "Possible duplicate payment charge",
        }
    )

    assert repository.get(ticket_id) is not None


def test_the_tool_is_bound_to_one_session() -> None:
    repository = TicketRepository()
    tool = create_ticket_tool(repository, "session-1")

    payload = {
        "customer_name": "Test User",
        "customer_email": "test@example.com",
        "issue_description": "Payment was charged twice",
        "category": "payment",
        "summary": "Possible duplicate payment charge",
    }

    first = tool.invoke(payload)
    second = tool.invoke(dict(payload))

    assert first == second
    assert len(list(repository.all())) == 1
