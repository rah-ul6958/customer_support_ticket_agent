from __future__ import annotations

from collections.abc import Iterable

from langchain_core.tools import StructuredTool

from src.models import Ticket, TicketCreate


class TicketRepository:
    """Working in-memory ticket store with session-level idempotency."""

    def __init__(self) -> None:
        self._tickets: dict[str, Ticket] = {}
        self._session_ticket: dict[str, str] = {}

    def create(self, session_id: str, data: TicketCreate) -> Ticket:
        existing_id = self._session_ticket.get(session_id)
        if existing_id:
            return self._tickets[existing_id]

        ticket_id = f"CST-2026-{len(self._tickets) + 1:04d}"
        ticket = Ticket(ticket_id=ticket_id, **data.model_dump())
        self._tickets[ticket_id] = ticket
        self._session_ticket[session_id] = ticket_id
        return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        return self._tickets.get(ticket_id)

    def all(self) -> Iterable[Ticket]:
        return self._tickets.values()


def create_ticket_tool(repository: TicketRepository, session_id: str) -> StructuredTool:
    """Bind the repository to one session as a validated LangChain tool."""

    def create_ticket(
        customer_name: str,
        customer_email: str,
        issue_description: str,
        category: str,
        summary: str,
    ) -> str:
        # Pydantic validation occurs before the side effect. The model must not
        # generate the repository-issued ID or call this with incomplete values.
        request = TicketCreate(
            customer_name=customer_name,
            customer_email=customer_email,
            issue_description=issue_description,
            category=category,  # type: ignore[arg-type]
            summary=summary,
        )
        return repository.create(session_id, request).ticket_id

    return StructuredTool.from_function(
        func=create_ticket,
        name="create_support_ticket",
        description="Create one support ticket after all required customer fields are validated.",
    )
