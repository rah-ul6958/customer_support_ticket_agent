from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_support_documents(documents_dir: Path) -> list[Document]:
    """Load supplied Markdown policies and retain safe source metadata."""
    if not documents_dir.is_dir():
        raise FileNotFoundError(f"Knowledge directory not found: {documents_dir}")

    documents = [
        Document(page_content=path.read_text(encoding="utf-8"), metadata={"source": path.name})
        for path in sorted(documents_dir.glob("*.md"))
    ]
    if not documents:
        raise ValueError("No Markdown knowledge documents were found")
    return documents


def split_support_documents(documents: list[Document]) -> list[Document]:
    """Provide a consistent splitter while leaving tuning to the candidate."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    return splitter.split_documents(documents)
