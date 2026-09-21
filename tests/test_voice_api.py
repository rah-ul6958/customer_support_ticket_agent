"""HTTP contract for the voice endpoints, and proof that typed chat is intact.

No real speech model or speech service is involved: the fake adapters from
``conftest`` stand in for both, so these tests run offline and deterministically.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from src.api import server
from src.voice.service import VoiceSupport
from src.voice.stt_whisper import UnsupportedAudioError
from tests.conftest import ConfigurableFakeSTT, ConfigurableFakeTTS, build_voice_support


WAV = ("recording.wav", b"fake-audio-bytes", "audio/wav")


@pytest_asyncio.fixture
async def voice_client(monkeypatch, make_pipeline, settings):
    """Test client with a scripted agent pipeline and fake voice adapters."""

    async def _factory(
        stt=None,
        tts=None,
        voice_support: VoiceSupport | None = None,
        decisions=None,
        answers=None,
    ) -> TestClient:
        pipeline = await make_pipeline(decisions=decisions, answers=answers)
        support = (
            voice_support
            if voice_support is not None
            else build_voice_support(settings, stt, tts)
        )

        monkeypatch.setattr(server, "pipeline", pipeline)
        monkeypatch.setattr(server, "voice", support)

        client = TestClient(server.app)
        client.pipeline = pipeline  # type: ignore[attr-defined]
        client.voice = support  # type: ignore[attr-defined]
        return client

    return _factory


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_transcription_returns_the_documented_payload(voice_client) -> None:
    client = await voice_client(stt=ConfigurableFakeSTT("My payment was charged twice"))

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["transcript"] == "My payment was charged twice"
    assert isinstance(body["processing_time_ms"], int)
    assert body["processing_time_ms"] >= 0


@pytest.mark.asyncio
async def test_transcription_passes_the_audio_and_media_type_through(
    voice_client,
) -> None:
    stt = ConfigurableFakeSTT()
    client = await voice_client(stt=stt)

    client.post("/voice/transcribe", files={"audio": WAV})

    assert stt.calls == [(b"fake-audio-bytes", "audio/wav")]


@pytest.mark.asyncio
async def test_empty_audio_is_rejected_with_a_clear_json_error(voice_client) -> None:
    client = await voice_client()

    response = client.post(
        "/voice/transcribe",
        files={"audio": ("recording.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert "empty" in body["error"].lower()


@pytest.mark.asyncio
async def test_unintelligible_audio_is_rejected_not_invented(voice_client) -> None:
    # The model heard nothing it could transcribe.
    client = await voice_client(stt=ConfigurableFakeSTT(transcript="   "))

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 400
    body = response.json()
    assert body["success"] is False
    assert body["error"] == "No understandable speech was detected"


@pytest.mark.asyncio
async def test_unsupported_audio_format_is_reported_as_415(voice_client) -> None:
    client = await voice_client(
        stt=ConfigurableFakeSTT(error=UnsupportedAudioError("Unsupported audio format: text/plain"))
    )

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 415
    assert response.json()["success"] is False


@pytest.mark.asyncio
async def test_stt_failure_returns_502_without_a_transcript(voice_client) -> None:
    client = await voice_client(
        stt=ConfigurableFakeSTT(error=RuntimeError("whisper exploded"))
    )

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 502
    body = response.json()
    assert body["success"] is False
    assert "transcript" not in body
    # The upstream error text is not leaked to the customer.
    assert "exploded" not in body["error"]


@pytest.mark.asyncio
async def test_transcription_is_503_when_voice_is_unavailable(
    voice_client, settings
) -> None:
    unavailable = VoiceSupport(settings)
    unavailable.unavailable_reason = "Voice extras are not installed."
    client = await voice_client(voice_support=unavailable)

    response = client.post("/voice/transcribe", files={"audio": WAV})

    assert response.status_code == 503
    assert response.json()["success"] is False


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_synthesis_returns_audio_with_the_correct_media_type(
    voice_client,
) -> None:
    client = await voice_client()

    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Ticket CST-2026-0001 has been created."},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert response.content


@pytest.mark.asyncio
async def test_synthesis_sends_the_exact_displayed_text(voice_client) -> None:
    tts = ConfigurableFakeTTS()
    client = await voice_client(tts=tts)
    displayed = "Standard delivery usually takes three to five business days."

    client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": displayed},
    )

    assert tts.calls == [displayed]


@pytest.mark.asyncio
async def test_audio_is_associated_with_the_requesting_message(voice_client) -> None:
    client = await voice_client()

    first = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "First agent response."},
    )
    second = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-2", "text": "Second agent response."},
    )

    # Each response is tagged with the message that asked for it...
    assert first.headers["X-Message-Id"] == "msg-1"
    assert second.headers["X-Message-Id"] == "msg-2"

    # ...and the audio really differs, so the UI cannot attach one message's
    # playback under another message.
    assert first.content != second.content
    assert b"First agent response." in first.content
    assert b"Second agent response." in second.content


@pytest.mark.asyncio
async def test_synthesis_reports_processing_time(voice_client) -> None:
    client = await voice_client()

    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Agent response."},
    )

    assert int(response.headers["X-Processing-Time-Ms"]) >= 0


@pytest.mark.asyncio
async def test_tts_failure_returns_502_and_leaves_the_text_reply_alone(
    voice_client,
) -> None:
    client = await voice_client(
        tts=ConfigurableFakeTTS(error=RuntimeError("edge-tts exploded"))
    )

    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Agent response."},
    )

    assert response.status_code == 502
    body = response.json()
    assert body["success"] is False
    assert "exploded" not in body["error"]

    # The agent and its tickets are untouched by a speech failure.
    assert client.pipeline.ready is True


@pytest.mark.asyncio
async def test_empty_synthesis_audio_is_reported_not_served(voice_client) -> None:
    client = await voice_client(tts=ConfigurableFakeTTS(audio=None))

    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Agent response."},
    )

    assert response.status_code == 400
    assert response.json()["success"] is False


@pytest.mark.asyncio
async def test_synthesis_rejects_blank_text(voice_client) -> None:
    client = await voice_client()

    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": ""},
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_synthesis_is_503_when_voice_is_unavailable(
    voice_client, settings
) -> None:
    unavailable = VoiceSupport(settings)
    unavailable.unavailable_reason = "Voice is disabled by configuration"
    client = await voice_client(voice_support=unavailable)

    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Agent response."},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "Voice is disabled by configuration"


# ---------------------------------------------------------------------------
# The original agent pipeline is unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_typed_chat_is_unchanged_by_the_voice_integration(voice_client) -> None:
    client = await voice_client(
        decisions=[{"route": "answer"}],
        answers=["Standard delivery takes three to five business days."],
    )

    response = client.post(
        "/chat",
        json={
            "session_id": "demo-session-1",
            "message": "How long does standard shipping take?",
        },
    )

    assert response.status_code == 200
    # Byte-for-byte the documented contract, with no voice fields added.
    assert response.json() == {
        "success": True,
        "session_id": "demo-session-1",
        "response": "Standard delivery takes three to five business days.",
        "sources": ["shipping.md"],
        "ticket_id": None,
    }


@pytest.mark.asyncio
async def test_a_transcript_reaches_the_agent_through_the_normal_chat_route(
    voice_client,
) -> None:
    """Dictated input must run the same session, RAG, and ticket workflow."""
    client = await voice_client(
        stt=ConfigurableFakeSTT("How long does standard shipping take?"),
        decisions=[{"route": "answer"}],
        answers=["Standard delivery takes three to five business days."],
    )

    transcription = client.post("/voice/transcribe", files={"audio": WAV})
    transcript = transcription.json()["transcript"]

    # The UI sends the confirmed transcript through the ordinary endpoint.
    chat = client.post(
        "/chat",
        json={"session_id": "voice-session", "message": transcript},
    )

    assert chat.status_code == 200
    body = chat.json()
    assert body["sources"] == ["shipping.md"]
    assert body["session_id"] == "voice-session"

    # The turn was recorded in the ordinary session history.
    history = client.pipeline.sessions.get_or_create("voice-session").history
    assert history[0]["content"] == "How long does standard shipping take?"


@pytest.mark.asyncio
async def test_voice_failure_does_not_affect_the_chat_endpoint(voice_client) -> None:
    client = await voice_client(
        stt=ConfigurableFakeSTT(error=RuntimeError("whisper exploded")),
        tts=ConfigurableFakeTTS(error=RuntimeError("edge-tts exploded")),
        decisions=[{"route": "answer"}],
        answers=["Standard delivery takes three to five business days."],
    )

    assert client.post("/voice/transcribe", files={"audio": WAV}).status_code == 502
    assert (
        client.post(
            "/voice/synthesize", json={"message_id": "m", "text": "hi"}
        ).status_code
        == 502
    )

    chat = client.post(
        "/chat",
        json={"session_id": "s", "message": "How long does standard shipping take?"},
    )

    assert chat.status_code == 200
    assert chat.json()["sources"] == ["shipping.md"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health_reports_voice_readiness(voice_client) -> None:
    client = await voice_client()

    body = client.get("/health").json()

    assert body["status"] == "ready"
    assert body["voice"]["available"] is True


@pytest.mark.asyncio
async def test_health_is_still_ready_when_voice_is_unavailable(
    voice_client, settings
) -> None:
    unavailable = VoiceSupport(settings)
    unavailable.unavailable_reason = "Voice extras are not installed."
    client = await voice_client(voice_support=unavailable)

    response = client.get("/health")

    # Voice is an optional add-on; its absence must not make the service
    # look unhealthy.
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["voice"]["available"] is False
    assert body["voice"]["detail"] == "Voice extras are not installed."
