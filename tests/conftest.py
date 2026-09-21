"""Shared fixtures.

The suite runs without a language-model endpoint. ``FakeChatModel`` stands in
for the open-source model so routing, grounding, ticket collection, and failure
handling are all deterministic. Retrieval is exercised against the real
embedding model and a temporary Chroma directory, which is what makes the source
attribution assertions meaningful.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from src.config import Settings
from src.llm.workflow import Decision
from src.pipeline import SupportPipeline


@pytest.fixture
def knowledge_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "knowledge_base"


class FakeResponse:
    """Minimal stand-in for a LangChain ``AIMessage``."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StructuredView:
    def __init__(self, model: "FakeChatModel") -> None:
        self._model = model

    async def ainvoke(self, messages: Any) -> Decision:
        return self._model._next_decision(messages)


class FakeChatModel:
    """Scripted chat model.

    ``decisions`` and ``answers`` are consumed one per turn. Either may be an
    exception instance, which the model raises instead of returning -- that is
    how the suite simulates an unavailable model endpoint.
    """

    def __init__(
        self,
        decisions: list[Any] | None = None,
        answers: list[Any] | None = None,
    ) -> None:
        self.decisions = list(decisions or [])
        self.answers = list(answers or [])
        # Prompts are recorded so tests can assert what the model was actually
        # told -- in particular whether retrieved context reached it.
        self.decision_prompts: list[str] = []
        self.answer_prompts: list[str] = []

    @staticmethod
    def _render(messages: Any) -> str:
        return "\n".join(str(getattr(item, "content", item)) for item in messages)

    def _next_decision(self, messages: Any) -> Decision:
        self.decision_prompts.append(self._render(messages))

        if not self.decisions:
            raise AssertionError("FakeChatModel ran out of scripted decisions")

        item = self.decisions.pop(0)

        if isinstance(item, BaseException):
            raise item
        if isinstance(item, Decision):
            return item
        return Decision.model_validate(item)

    def with_structured_output(self, _schema: Any) -> _StructuredView:
        return _StructuredView(self)

    async def ainvoke(self, messages: Any) -> FakeResponse:
        self.answer_prompts.append(self._render(messages))

        if not self.answers:
            raise AssertionError("FakeChatModel ran out of scripted answers")

        item = self.answers.pop(0)

        if isinstance(item, BaseException):
            raise item
        return FakeResponse(str(item))


def build_settings(vector_db_path: Path) -> Settings:
    """Settings pointed at a throwaway vector directory."""
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(vector_db_path),
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
        voice_enabled=True,
        stt_model="tiny",
        stt_device="cpu",
        stt_compute_type="int8",
        tts_voice="en-US-AriaNeural",
    )


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return build_settings(tmp_path / "vector_db")


@pytest_asyncio.fixture
async def make_pipeline(settings: Settings, knowledge_dir: Path):
    """Build an initialized pipeline whose model is scripted by the test."""

    async def _factory(
        decisions: list[Any] | None = None,
        answers: list[Any] | None = None,
    ) -> SupportPipeline:
        pipeline = SupportPipeline(settings, knowledge_dir)
        pipeline.model = FakeChatModel(decisions=decisions, answers=answers)
        await pipeline.initialize()
        return pipeline

    return _factory


# ---------------------------------------------------------------------------
# Voice fakes
#
# The voice suite never loads Whisper or calls the Edge TTS service. These
# adapters implement the supplied contracts so transcription, synthesis, and
# every failure path stay deterministic and offline.
# ---------------------------------------------------------------------------

from src.voice.contracts import STTService, TTSService  # noqa: E402
from src.voice.pipeline import VoicePipeline  # noqa: E402
from src.voice.service import VoiceSupport  # noqa: E402


class ConfigurableFakeSTT(STTService):
    """Fake speech-to-text. ``error`` makes every call fail."""

    def __init__(
        self,
        transcript: str = "My payment was charged twice",
        error: BaseException | None = None,
    ) -> None:
        self.transcript = transcript
        self.error = error
        self.calls: list[tuple[bytes, str]] = []

    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        self.calls.append((audio_bytes, media_type))

        if self.error is not None:
            raise self.error

        return self.transcript


class ConfigurableFakeTTS(TTSService):
    """Fake text-to-speech. ``error`` makes every call fail."""

    def __init__(
        self,
        audio: bytes = b"fake-audio",
        media_type: str = "audio/mpeg",
        error: BaseException | None = None,
    ) -> None:
        self.audio = audio
        self.media_type = media_type
        self.error = error
        self.calls: list[str] = []

    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        self.calls.append(text)

        if self.error is not None:
            raise self.error

        # Audio is derived from the text so tests can prove that the audio
        # returned belongs to the message that asked for it.
        if self.audio is None:
            return b"", self.media_type

        return self.audio + text.encode("utf-8"), self.media_type


def build_voice_support(
    settings: Settings,
    stt: STTService | None = None,
    tts: TTSService | None = None,
) -> VoiceSupport:
    """A ready :class:`VoiceSupport` backed by fake adapters."""
    support = VoiceSupport(settings)
    support.pipeline = VoicePipeline(
        stt or ConfigurableFakeSTT(),
        tts or ConfigurableFakeTTS(),
    )
    support.unavailable_reason = None
    return support
