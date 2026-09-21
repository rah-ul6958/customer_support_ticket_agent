from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationState:
    customer_name: str | None = None
    customer_email: str | None = None
    issue_description: str | None = None
    category: str | None = None
    ticket_id: str | None = None
    history: list[dict[str, str]] = field(default_factory=list)

    def missing_ticket_fields(self) -> list[str]:
        values = {
            "customer_name": self.customer_name,
            "customer_email": self.customer_email,
            "issue_description": self.issue_description,
            "category": self.category,
        }
        return [name for name, value in values.items() if not value]


class SessionStore:
    """Supplied in-memory isolation boundary for conversation state."""

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str) -> ConversationState:
        if not session_id.strip():
            raise ValueError("session_id must not be blank")
        return self._sessions.setdefault(session_id, ConversationState())
