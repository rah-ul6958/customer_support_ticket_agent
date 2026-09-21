# RAG Implementation Guide

The starter already provides the component boundaries and supported integrations. Complete the marked retrieval logic without moving agent decisions into the RAG layer.

## Supplied flow

```text
knowledge_base/*.md
  -> load_support_documents()
  -> split_support_documents()
  -> HuggingFaceEmbeddings
  -> persistent Chroma collection
  -> KnowledgeRetriever.search()
  -> LangGraph workflow
```

`document_loader.py` provides validated Markdown loading, filename metadata, and a default text splitter. `embeddings.py` binds the configured open-source sentence-transformer model. `retriever.py` owns Chroma initialization and result normalization.

## Candidate implementation

Complete `KnowledgeRetriever.initialize()` and `KnowledgeRetriever.search()`.

- Use `settings.vector_db_path` for persistence.
- Use `settings.rag_collection` as the collection name.
- Avoid duplicating chunks every time the API restarts. Stable IDs derived from source and chunk content are one acceptable approach.
- Preserve only safe source filenames in metadata; do not expose absolute paths.
- Default to `settings.rag_top_k`, while allowing an explicit method limit.
- Define a relevance threshold or another defensible unknown-answer rule.
- Return plain dictionaries containing `content` and `source` so orchestration is independent of Chroma-specific objects.

Do not generate final responses inside the retriever. The LangGraph workflow combines retrieved evidence with the system prompt and decides between answering and ticket collection.

## Verification

Test document loading, metadata preservation, initialization idempotency, relevant retrieval, irrelevant retrieval, and search-before-initialization. Tests should use temporary persistence directories and may substitute deterministic fake embeddings to remain fast.
