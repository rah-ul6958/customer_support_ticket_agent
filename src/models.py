from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, EmailStr, Field


TicketCategory = Literal["order", "payment", "account", "technical", "other"]

# MODEL CONTRACT NOTES
#
# These models define the public boundary between Streamlit, FastAPI, the agent,
# and the mock ticket repository. Extend them only when the assignment requires
# additional data; avoid passing unvalidated dictionaries between components.
#
# Candidate expectations:
# - Keep ``session_id`` client-supplied and stable across a conversation.
# - Trim or reject blank text rather than forwarding it to retrieval/the model.
# - Validate email through Pydantic before calling the ticket repository.
# - Keep category values constrained to the supplied list.
# - Do not accept ``ticket_id``, status, or creation time from the model/user.
# - Preserve backward compatibility with the documented minimum JSON examples.


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    message: str = Field(min_length=1)


class TicketCreate(BaseModel):
    customer_name: str = Field(min_length=1)
    customer_email: EmailStr
    issue_description: str = Field(min_length=5)
    category: TicketCategory
    summary: str = Field(min_length=5, max_length=160)


class Ticket(TicketCreate):
    ticket_id: str
    status: Literal["open"] = "open"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatResponse(BaseModel):
    # ``sources`` contains filenames used to ground this exact response.
    # ``ticket_id`` remains null until a real ticket has been created.
    success: bool = True
    session_id: str
    response: str
    sources: list[str] = Field(default_factory=list)
    ticket_id: str | None = None
