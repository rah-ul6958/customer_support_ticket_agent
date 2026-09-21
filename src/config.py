from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Environment-controlled model and retrieval settings.

    Add new provider or embedding options here when required. Keep safe defaults
    suitable for local development and document every new variable in both
    ``.env.example`` and the project README.
    """
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    vector_db_path: str
    embedding_model: str
    rag_collection: str
    rag_top_k: int
    api_host: str
    api_port: int
    streamlit_host: str
    streamlit_port: int
    # Voice is an optional add-on: when its libraries or services are missing the
    # text agent must keep working unchanged, so nothing here is required.
    voice_enabled: bool
    stt_model: str
    stt_device: str
    stt_compute_type: str
    tts_voice: str


def load_settings() -> Settings:
    """Load the small, provider-neutral configuration used by the starter."""
    load_dotenv()
    # Configuration rules:
    # - Never commit a real API key.
    # - Avoid absolute developer-machine paths.
    # - Keep model identity configurable so reviewers can use another compatible
    #   open-source endpoint without editing source code.
    # - Validate required configuration during startup, not on the first request.
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "not-required"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", ".data/vector_db"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        rag_collection=os.getenv("RAG_COLLECTION", "customer-support"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        streamlit_host=os.getenv("STREAMLIT_HOST", "127.0.0.1"),
        streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
        voice_enabled=os.getenv("VOICE_ENABLED", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        stt_model=os.getenv("STT_MODEL", "tiny"),
        stt_device=os.getenv("STT_DEVICE", "cpu"),
        stt_compute_type=os.getenv("STT_COMPUTE_TYPE", "int8"),
        tts_voice=os.getenv("TTS_VOICE", "en-US-AriaNeural"),
    )
