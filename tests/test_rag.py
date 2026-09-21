"""Document loading and retrieval behavior.

These tests use the real embedding model against a temporary Chroma directory,
because the point of most of them is whether the relevance rule actually
separates on-topic from off-topic questions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


def test_loader_preserves_source_names(knowledge_dir) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


def test_loader_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_support_documents(tmp_path / "does-not-exist")


def test_loader_rejects_a_directory_without_documents(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_support_documents(tmp_path)


@pytest.mark.asyncio
async def test_search_before_initialization_is_refused(settings, knowledge_dir) -> None:
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    with pytest.raises(ComponentNotReadyError):
        await retriever.search("How long does shipping take?")


@pytest.mark.asyncio
async def test_relevant_questions_retrieve_their_own_document(
    settings, knowledge_dir
) -> None:
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    expected = {
        "How long does standard shipping take?": "shipping.md",
        "When does tracking become available?": "shipping.md",
        "What is the return window?": "returns.md",
        "How do I reset my forgotten password?": "accounts.md",
        "I was charged twice for one purchase": "payments.md",
    }

    for question, source in expected.items():
        hits = await retriever.search(question)
        assert hits, f"expected a hit for {question!r}"
        assert [hit["source"] for hit in hits] == [source], question
        assert hits[0]["content"]


@pytest.mark.asyncio
async def test_off_topic_questions_retrieve_nothing(settings, knowledge_dir) -> None:
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    for question in (
        "What is the capital of France?",
        "Write me a poem about dragons",
        "Who won the 2014 world cup?",
        "asha@example.com",
    ):
        assert await retriever.search(question) == [], question


@pytest.mark.asyncio
async def test_blank_queries_retrieve_nothing(settings, knowledge_dir) -> None:
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    assert await retriever.search("") == []
    assert await retriever.search("   ") == []


@pytest.mark.asyncio
async def test_a_non_positive_limit_retrieves_nothing(settings, knowledge_dir) -> None:
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    assert await retriever.search("How long does shipping take?", limit=0) == []


@pytest.mark.asyncio
async def test_initialization_is_idempotent(settings, knowledge_dir) -> None:
    retriever = KnowledgeRetriever(settings, knowledge_dir)

    await retriever.initialize()
    first = len(retriever._store.get()["ids"])

    # A restart must reuse the persisted index instead of duplicating chunks.
    await retriever.initialize()
    second = len(retriever._store.get()["ids"])

    assert first > 0
    assert first == second


@pytest.mark.asyncio
async def test_results_expose_only_bare_filenames(settings, knowledge_dir) -> None:
    retriever = KnowledgeRetriever(settings, knowledge_dir)
    await retriever.initialize()

    hits = await retriever.search("How long does standard shipping take?")

    for hit in hits:
        assert set(hit) == {"content", "source"}
        assert "/" not in hit["source"]
        assert "\\" not in hit["source"]
