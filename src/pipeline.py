from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm.client import build_chat_model
from src.llm.workflow import build_support_workflow
from src.models import ChatResponse
from src.rag.retriever import KnowledgeRetriever
from src.sessions.store import SessionStore
from src.tools.ticket_tool import TicketRepository
from src.utils.errors import AgentProcessingError, ComponentNotReadyError


class SupportPipeline:
    """Top-level binding for model, RAG, workflow, sessions, and ticket tool."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.model = build_chat_model(settings)
        self.retriever = KnowledgeRetriever(settings, documents_dir)
        self.sessions = SessionStore()
        self.tickets = TicketRepository()
        self.documents_dir = documents_dir
        self.workflow = None
        self.ready = False

    @property
    def retriever_ready(self) -> bool:
        """Whether the vector index is built and answering searches."""
        return getattr(self.retriever, "_store", None) is not None

    async def initialize(self) -> None:
        await self.retriever.initialize()

        self.workflow = build_support_workflow(
            self.model,
            self.retriever,
            self.sessions,
            self.tickets,
        )

        self.ready = True

    async def process(
        self,
        session_id: str,
        message: str,
    ) -> ChatResponse:

        if not self.ready or self.workflow is None:
            raise ComponentNotReadyError(
                "Support pipeline is not ready"
            )

        session_id = session_id.strip()
        message = message.strip()

        if not session_id:
            raise ValueError("session_id must not be blank")

        if not message:
            raise ValueError("message must not be blank")

        session = self.sessions.get_or_create(session_id)

        initial = {
            "session_id": session_id,
            "customer_message": message,
            "messages": [
                {
                    "role": item["role"],
                    "content": item["content"],
                }
                for item in session.history[-6:]
            ],
            "retrieved_chunks": [],
            "sources": [],
            # Left unset on purpose: only the node that actually creates or
            # re-confirms a ticket may put an identifier into the response.
            "ticket_id": None,
        }

        try:
            result = await self.workflow.ainvoke(initial)

            response = str(
                result.get("response_text", "")
            ).strip()

            if not response:
                raise AgentProcessingError(
                    "The agent returned an empty response."
                )

            sources = []
            seen = set()

            for source in result.get("sources", []) or []:
                source = Path(str(source)).name

                if source and source not in seen:
                    sources.append(source)
                    seen.add(source)

            ticket_id = result.get("ticket_id")

            if (
                ticket_id
                and self.tickets.get(str(ticket_id)) is None
            ):
                ticket_id = None

            session.history.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

            session.history.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            return ChatResponse(
                session_id=session_id,
                response=response,
                sources=sources,
                ticket_id=(
                    str(ticket_id)
                    if ticket_id
                    else None
                ),
            )

        except AgentProcessingError:
            raise

        except Exception as exc:
            raise AgentProcessingError(
                "Unable to process the support request."
            ) from exc