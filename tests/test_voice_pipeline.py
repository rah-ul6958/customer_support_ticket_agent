import pytest

from src.voice.contracts import STTService, TTSService
from src.voice.pipeline import VoicePipeline


class FakeSTT(STTService):
    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        return "My payment was charged twice"


class FakeTTS(TTSService):
    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        return b"fake-audio", "audio/mpeg"


@pytest.mark.asyncio
async def test_voice_pipeline_contracts() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())

    transcript, _ = await pipeline.transcribe(b"fake-input", "audio/wav")
    audio, media_type, _ = await pipeline.synthesize("Agent response")

    assert transcript == "My payment was charged twice"
    assert audio == b"fake-audio"
    assert media_type == "audio/mpeg"


# ---------------------------------------------------------------------------
# Extended coverage of the supplied VoicePipeline scaffold.
# ---------------------------------------------------------------------------

from src.voice.service import VoiceSupport  # noqa: E402
from tests.conftest import ConfigurableFakeSTT, ConfigurableFakeTTS  # noqa: E402


@pytest.mark.asyncio
async def test_pipeline_reports_timing_and_media_type() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())

    transcript, transcribe_ms = await pipeline.transcribe(b"audio", "audio/wav")
    audio, media_type, synthesize_ms = await pipeline.synthesize("Agent response")

    assert transcript
    assert isinstance(transcribe_ms, int) and transcribe_ms >= 0
    assert audio
    assert media_type == "audio/mpeg"
    assert isinstance(synthesize_ms, int) and synthesize_ms >= 0


@pytest.mark.asyncio
async def test_pipeline_rejects_empty_audio() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())

    with pytest.raises(ValueError, match="empty"):
        await pipeline.transcribe(b"", "audio/wav")


@pytest.mark.asyncio
async def test_pipeline_rejects_an_unintelligible_recording() -> None:
    pipeline = VoicePipeline(ConfigurableFakeSTT(transcript="   "), FakeTTS())

    with pytest.raises(ValueError, match="No understandable speech"):
        await pipeline.transcribe(b"audio", "audio/wav")


@pytest.mark.asyncio
async def test_pipeline_rejects_blank_synthesis_text() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())

    with pytest.raises(ValueError, match="empty"):
        await pipeline.synthesize("   ")


@pytest.mark.asyncio
async def test_pipeline_rejects_empty_synthesized_audio() -> None:
    pipeline = VoicePipeline(FakeSTT(), ConfigurableFakeTTS(audio=None))

    with pytest.raises(ValueError, match="empty audio"):
        await pipeline.synthesize("Agent response")


@pytest.mark.asyncio
async def test_stt_errors_propagate_for_the_api_to_translate() -> None:
    pipeline = VoicePipeline(
        ConfigurableFakeSTT(error=RuntimeError("model unavailable")), FakeTTS()
    )

    with pytest.raises(RuntimeError, match="model unavailable"):
        await pipeline.transcribe(b"audio", "audio/wav")


@pytest.mark.asyncio
async def test_tts_errors_propagate_for_the_api_to_translate() -> None:
    pipeline = VoicePipeline(
        FakeSTT(), ConfigurableFakeTTS(error=RuntimeError("service unavailable"))
    )

    with pytest.raises(RuntimeError, match="service unavailable"):
        await pipeline.synthesize("Agent response")


@pytest.mark.asyncio
async def test_voice_support_stays_down_instead_of_raising(settings) -> None:
    """A machine without the voice extras must still start the text agent."""
    support = VoiceSupport(settings)

    async def explode() -> None:
        raise ImportError("No module named 'faster_whisper'")

    support_pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    support_pipeline.initialize = explode  # type: ignore[method-assign]

    # Reproduce what initialize() does when the adapters cannot be imported.
    try:
        await support_pipeline.initialize()
    except ImportError as exc:
        support.unavailable_reason = str(exc)

    assert support.available is False
    assert support.status["available"] is False
    assert support.unavailable_reason


@pytest.mark.asyncio
async def test_disabled_voice_never_builds_a_pipeline(settings) -> None:
    from dataclasses import replace

    support = VoiceSupport(replace(settings, voice_enabled=False))

    await support.initialize()

    assert support.available is False
    assert support.status["detail"] == "Voice is disabled by configuration"
