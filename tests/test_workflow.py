"""End-to-end agent behavior through the LangGraph workflow.

Retrieval is real; only the language model is scripted. Each test therefore
covers the same path a customer turn takes in production apart from the model
call itself.
"""

from __future__ import annotations

import pytest

from src.llm.workflow import NO_KNOWLEDGE_RESPONSE, Decision
from src.utils.errors import AgentProcessingError


ANSWER = {"route": "answer"}


def ticket_turn(**fields) -> dict:
    return {"route": "ticket", **fields}


@pytest.mark.asyncio
async def test_policy_question_is_answered_from_the_knowledge_base(make_pipeline) -> None:
    pipeline = await make_pipeline(
        decisions=[ANSWER],
        answers=["Standard delivery usually takes three to five business days."],
    )

    result = await pipeline.process(
        "session-policy",
        "How long does standard shipping take?",
    )

    # The correct document is cited, and only that document.
    assert result.sources == ["shipping.md"]
    assert result.ticket_id is None
    assert result.success is True

    # The retrieved policy text really reached the model, so the answer is
    # grounded rather than recalled from model weights.
    prompt = pipeline.model.answer_prompts[0]
    assert "three to five business days" in prompt
    assert "[Source: shipping.md]" in prompt


@pytest.mark.asyncio
async def test_each_topic_cites_its_own_document(make_pipeline) -> None:
    pipeline = await make_pipeline(
        decisions=[ANSWER, ANSWER, ANSWER],
        answers=["a", "b", "c"],
    )

    expected = {
        "How do I reset my password?": "accounts.md",
        "Can I return an item I bought three weeks ago?": "returns.md",
        "I was charged twice for one order": "payments.md",
    }

    for index, (question, source) in enumerate(expected.items()):
        result = await pipeline.process(f"session-topic-{index}", question)
        assert result.sources == [source], question


@pytest.mark.asyncio
async def test_out_of_scope_question_is_refused_without_fabrication(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline(decisions=[ANSWER])

    result = await pipeline.process(
        "session-unknown",
        "What is the capital of France?",
    )

    # Nothing is cited, because nothing relevant was retrieved.
    assert result.sources == []
    assert result.ticket_id is None
    assert result.response == NO_KNOWLEDGE_RESPONSE

    # The model was never asked to answer, so it had no opportunity to reply
    # from its own weights. Prompt instructions alone do not stop a small
    # instruction-tuned model from doing exactly that.
    assert pipeline.model.answer_prompts == []


@pytest.mark.asyncio
async def test_several_off_topic_questions_are_all_refused(make_pipeline) -> None:
    questions = [
        "What is the capital of France?",
        "Write me a poem about dragons",
        "Who won the 2014 world cup?",
    ]
    pipeline = await make_pipeline(decisions=[ANSWER] * len(questions))

    for index, question in enumerate(questions):
        result = await pipeline.process(f"session-off-{index}", question)
        assert result.response == NO_KNOWLEDGE_RESPONSE, question
        assert result.sources == [], question


@pytest.mark.asyncio
async def test_multi_turn_conversation_creates_one_validated_ticket(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline(
        decisions=[
            ticket_turn(issue_description="Charged twice for a single order"),
            ticket_turn(customer_name="Asha Rao"),
            ticket_turn(customer_email="asha@example.com"),
        ],
    )

    first = await pipeline.process(
        "session-ticket",
        "My payment was charged twice and nobody replied. Please open a ticket.",
    )
    assert first.ticket_id is None
    assert "your name" in first.response

    second = await pipeline.process("session-ticket", "Asha Rao")
    assert second.ticket_id is None
    assert "email" in second.response.lower()

    third = await pipeline.process("session-ticket", "asha@example.com")

    assert third.ticket_id is not None
    assert third.ticket_id in third.response

    ticket = pipeline.tickets.get(third.ticket_id)
    assert ticket is not None
    assert ticket.customer_name == "Asha Rao"
    assert ticket.customer_email == "asha@example.com"
    assert ticket.category == "payment"
    assert ticket.status == "open"
    assert ticket.created_at is not None
    assert len(list(pipeline.tickets.all())) == 1


@pytest.mark.asyncio
async def test_missing_fields_produce_follow_up_questions_not_a_ticket(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline(decisions=[ticket_turn()])

    result = await pipeline.process("session-missing", "I want to raise a ticket")

    assert result.ticket_id is None
    assert result.response.rstrip().endswith("?")
    assert list(pipeline.tickets.all()) == []

    # Nothing was invented on the customer's behalf.
    session = pipeline.sessions.get_or_create("session-missing")
    assert session.customer_name is None
    assert session.customer_email is None


@pytest.mark.asyncio
async def test_a_category_the_customer_never_mentioned_is_not_accepted(
    make_pipeline,
) -> None:
    # The model guesses a category from an issue that names none.
    pipeline = await make_pipeline(
        decisions=[
            ticket_turn(
                issue_description="The screen keeps flickering",
                category="technical",
            )
        ],
    )

    await pipeline.process("session-guess", "My screen keeps flickering, please help")

    session = pipeline.sessions.get_or_create("session-guess")
    assert session.category is None
    assert "category" in session.missing_ticket_fields()


@pytest.mark.asyncio
async def test_customer_supplied_category_is_accepted(make_pipeline) -> None:
    pipeline = await make_pipeline(decisions=[ticket_turn(), ticket_turn()])

    await pipeline.process("session-cat", "I want to raise a ticket")
    await pipeline.process("session-cat", "technical")

    session = pipeline.sessions.get_or_create("session-cat")
    assert session.category == "technical"


@pytest.mark.asyncio
async def test_invalid_email_is_rejected_and_asked_for_again(make_pipeline) -> None:
    pipeline = await make_pipeline(
        decisions=[
            ticket_turn(
                customer_name="Asha Rao",
                customer_email="asha@@example",
                issue_description="Charged twice for a single order",
            ),
            ticket_turn(customer_email="asha@example.com"),
        ],
    )

    first = await pipeline.process(
        "session-email",
        "I am Asha Rao, asha@@example, my payment was charged twice",
    )

    assert first.ticket_id is None
    assert "email" in first.response.lower()

    session = pipeline.sessions.get_or_create("session-email")
    assert session.customer_email is None

    # The valid address supplied next completes the ticket.
    second = await pipeline.process("session-email", "asha@example.com")
    assert second.ticket_id is not None


@pytest.mark.asyncio
async def test_retrying_a_completed_request_returns_the_same_ticket(
    make_pipeline,
) -> None:
    complete = ticket_turn(
        customer_name="Asha Rao",
        customer_email="asha@example.com",
        issue_description="Charged twice for a single order",
    )

    pipeline = await make_pipeline(decisions=[complete, dict(complete)])

    message = (
        "I am Asha Rao, asha@example.com, my payment was charged twice. "
        "Please open a ticket."
    )

    first = await pipeline.process("session-dup", message)
    second = await pipeline.process("session-dup", message)

    assert first.ticket_id is not None
    assert second.ticket_id == first.ticket_id
    assert "already" in second.response.lower()
    assert len(list(pipeline.tickets.all())) == 1


@pytest.mark.asyncio
async def test_separate_sessions_do_not_share_state_or_tickets(make_pipeline) -> None:
    pipeline = await make_pipeline(
        decisions=[
            ticket_turn(
                customer_name="Asha Rao",
                customer_email="asha@example.com",
                issue_description="Charged twice for a single order",
            ),
            ticket_turn(),
        ],
    )

    complete = (
        "I am Asha Rao, asha@example.com, my payment was charged twice. "
        "Please open a ticket."
    )

    first = await pipeline.process("session-a", complete)
    second = await pipeline.process("session-b", "I want to raise a ticket")

    assert first.ticket_id is not None
    assert second.ticket_id is None

    other = pipeline.sessions.get_or_create("session-b")
    assert other.customer_name is None
    assert other.customer_email is None


@pytest.mark.asyncio
async def test_model_failure_while_routing_is_reported_not_guessed(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline(decisions=[RuntimeError("connection refused")])

    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-down", "How long does shipping take?")


@pytest.mark.asyncio
async def test_model_failure_while_answering_is_reported_not_guessed(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline(
        decisions=[ANSWER],
        answers=[RuntimeError("connection refused")],
    )

    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-down-2", "How long does shipping take?")


@pytest.mark.asyncio
async def test_an_empty_model_response_is_never_returned_to_the_customer(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline(decisions=[ANSWER], answers=["   "])

    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-empty", "How long does shipping take?")


@pytest.mark.asyncio
async def test_a_failed_ticket_call_leaves_no_ticket_id_in_the_session(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline(
        decisions=[
            ticket_turn(
                customer_name="Asha Rao",
                customer_email="asha@example.com",
                issue_description="Charged twice for a single order",
            )
        ],
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("ticket backend unavailable")

    pipeline.tickets.create = explode  # type: ignore[method-assign]

    with pytest.raises(AgentProcessingError):
        await pipeline.process(
            "session-toolfail",
            "I am Asha Rao, asha@example.com, my payment was charged twice.",
        )

    session = pipeline.sessions.get_or_create("session-toolfail")
    assert session.ticket_id is None


@pytest.mark.asyncio
async def test_an_invalid_route_is_rejected(make_pipeline) -> None:
    pipeline = await make_pipeline(
        decisions=[Decision.model_construct(route="escalate")]
    )

    with pytest.raises(AgentProcessingError):
        await pipeline.process("session-route", "hello")


@pytest.mark.asyncio
async def test_history_records_both_roles_in_order(make_pipeline) -> None:
    pipeline = await make_pipeline(decisions=[ANSWER], answers=["Three to five days."])

    await pipeline.process("session-history", "How long does standard shipping take?")

    history = pipeline.sessions.get_or_create("session-history").history
    assert [item["role"] for item in history] == ["user", "assistant"]
    assert history[0]["content"] == "How long does standard shipping take?"
    assert history[1]["content"] == "Three to five days."
