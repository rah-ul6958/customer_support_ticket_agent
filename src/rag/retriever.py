from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval with stable document IDs and safe outputs."""

    #: Minimum cosine relevance for a chunk to count as on-topic at all. Below
    #: this the workflow receives no context and must say the knowledge base
    #: does not cover the question rather than improvise an answer.
    RELEVANCE_THRESHOLD = 0.25

    #: A chunk is also dropped when it scores far below the best hit for the
    #: same query. This keeps source attribution precise: a shipping question
    #: should cite shipping.md alone, not every loosely related policy.
    RELATIVE_MARGIN = 0.6

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None
        self.relevance_threshold = self.RELEVANCE_THRESHOLD
        self.relative_margin = self.RELATIVE_MARGIN

    async def initialize(self) -> None:
        """Build or reuse the persistent index for the supplied Markdown docs."""
        documents = split_support_documents(
            load_support_documents(self.documents_dir)
        )

        embeddings = build_embeddings(self.settings)

        db_path = Path(self.settings.vector_db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        store = Chroma(
            collection_name=self.settings.rag_collection,
            persist_directory=str(db_path),
            embedding_function=embeddings,
            # Cosine distance keeps relevance scores comparable across queries,
            # which is what the thresholds above assume.
            collection_metadata={"hnsw:space": "cosine"},
        )

        # Content-derived IDs make initialization idempotent: restarting the API
        # re-indexes nothing, and an edited document replaces only its own chunk.
        ids = [
            hashlib.sha256(
                f"{doc.metadata.get('source', '')}\n{doc.page_content}".encode("utf-8")
            ).hexdigest()
            for doc in documents
        ]

        existing = set(store.get(ids=ids).get("ids", [])) if ids else set()

        new_docs = [
            doc
            for doc, doc_id in zip(documents, ids)
            if doc_id not in existing
        ]

        new_ids = [
            doc_id
            for doc_id in ids
            if doc_id not in existing
        ]

        if new_docs:
            store.add_documents(new_docs, ids=new_ids)

        self._store = store

    async def search(
        self,
        query: str,
        limit: int | None = None,
    ) -> list[dict[str, str]]:
        """Return relevant chunks as ``{"content", "source"}`` dictionaries.

        An empty list is a meaningful answer: it tells the workflow that the
        knowledge base has nothing on topic.
        """
        if not query or not query.strip():
            return []

        if self._store is None:
            raise ComponentNotReadyError(
                "Knowledge retriever is not ready"
            )

        k = limit if limit is not None else self.settings.rag_top_k

        if k <= 0:
            return []

        # Cosine distance in [0, 2]; converting here rather than using
        # ``similarity_search_with_relevance_scores`` keeps the metric explicit
        # and avoids LangChain's warning for scores outside [0, 1].
        scored = [
            (document, 1.0 - distance)
            for document, distance in self._store.similarity_search_with_score(
                query.strip(),
                k=k,
            )
        ]

        if not scored:
            return []

        top_score = max(score for _, score in scored)
        cutoff = max(
            self.relevance_threshold,
            top_score * self.relative_margin,
        )

        output: list[dict[str, str]] = []
        seen = set()

        for document, score in scored:
            if score < cutoff:
                continue

            # Keep only the bare filename so absolute developer paths never
            # reach the API response.
            source = Path(
                str(document.metadata.get("source", ""))
            ).name

            content = document.page_content.strip()

            if not source or not content:
                continue

            key = (source, content)

            if key in seen:
                continue

            seen.add(key)

            output.append(
                {
                    "content": content,
                    "source": source,
                }
            )

        return output[:k]
