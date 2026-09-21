from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings


@lru_cache(maxsize=4)
def _load_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    """Load a sentence-transformer once per process.

    Loading the model takes several seconds, so the instance is reused across
    retriever rebuilds and across the test suite.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        # L2-normalized vectors make Chroma's cosine distance -- and therefore
        # the retriever's relevance threshold -- comparable across queries.
        encode_kwargs={"normalize_embeddings": True},
    )


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Create the configured local/open-source embedding adapter."""
    return _load_embeddings(settings.embedding_model)
