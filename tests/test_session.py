"""Conversation state: what is still missing, and what stays isolated."""

from __future__ import annotations

import pytest

from src.sessions.store import ConversationState, SessionStore


def test_session_reports_only_missing_ticket_fields() -> None:
    # This supplied test establishes the expected collection order.
    state = ConversationState(customer_name="Asha", customer_email="asha@example.com")

    assert state.missing_ticket_fields() == ["issue_description", "category"]


def test_a_new_session_is_missing_every_ticket_field() -> None:
    assert ConversationState().missing_ticket_fields() == [
        "customer_name",
        "customer_email",
        "issue_description",
        "category",
    ]


def test_a_complete_session_is_missing_nothing() -> None:
    state = ConversationState(
        customer_name="Asha",
        customer_email="asha@example.com",
        issue_description="Charged twice for a single order",
        category="payment",
    )

    assert state.missing_ticket_fields() == []


def test_blank_values_still_count_as_missing() -> None:
    state = ConversationState(customer_name="", customer_email="   ".strip())

    assert "customer_name" in state.missing_ticket_fields()
    assert "customer_email" in state.missing_ticket_fields()


def test_fields_collected_over_several_turns_are_not_requested_again() -> None:
    state = ConversationState()

    state.customer_name = "Asha Rao"
    assert state.missing_ticket_fields()[0] == "customer_email"

    state.customer_email = "asha@example.com"
    assert state.missing_ticket_fields()[0] == "issue_description"

    state.issue_description = "Charged twice for a single order"
    assert state.missing_ticket_fields() == ["category"]

    state.category = "payment"
    assert state.missing_ticket_fields() == []


def test_the_store_returns_the_same_state_for_one_session_id() -> None:
    store = SessionStore()

    first = store.get_or_create("session-1")
    first.customer_name = "Asha Rao"

    assert store.get_or_create("session-1") is first
    assert store.get_or_create("session-1").customer_name == "Asha Rao"


def test_one_session_never_sees_another_customer_values() -> None:
    store = SessionStore()

    store.get_or_create("session-1").customer_email = "asha@example.com"
    other = store.get_or_create("session-2")

    assert other.customer_email is None
    assert other.ticket_id is None
    assert other.history == []


def test_a_blank_session_id_is_rejected() -> None:
    store = SessionStore()

    with pytest.raises(ValueError):
        store.get_or_create("   ")


def test_history_keeps_role_and_content_in_order() -> None:
    state = ConversationState()

    state.history.append({"role": "user", "content": "hello"})
    state.history.append({"role": "assistant", "content": "hi"})

    assert [item["role"] for item in state.history] == ["user", "assistant"]
    assert state.history[-1]["content"] == "hi"
