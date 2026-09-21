from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.models import SynthesisRequest, TranscriptionResponse, VoiceErrorResponse
from src.voice.service import VoiceSupport
from src.voice.stt_whisper import UnsupportedAudioError


settings = load_settings()
pipeline = SupportPipeline(settings, Path(__file__).resolve().parents[2] / "knowledge_base")
voice = VoiceSupport(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup owns expensive shared initialization. Request handlers reuse the
    # resulting components, while shutdown makes readiness false immediately.
    await pipeline.initialize()
    # Voice is optional and initializes itself defensively, so a missing model
    # or offline TTS service degrades voice alone and never the text agent.
    await voice.initialize()
    yield
    await voice.cleanup()
    pipeline.ready = False


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)


VOICE_ERRORS = {
    400: {"model": VoiceErrorResponse},
    415: {"model": VoiceErrorResponse},
    502: {"model": VoiceErrorResponse},
    503: {"model": VoiceErrorResponse},
}


def voice_error(status_code: int, message: str) -> JSONResponse:
    """Render a voice failure in the documented ``{success, error}`` shape."""
    return JSONResponse(
        status_code=status_code,
        content=VoiceErrorResponse(error=message).model_dump(),
    )


@app.get("/health")
async def health() -> dict[str, object]:
    """Report component readiness.

    The retriever and workflow are checked directly. The language model is only
    reported as configured, never as reachable: confirming that would mean
    calling the upstream endpoint on every health probe. A model outage instead
    surfaces as a 502 from ``/chat``.

    Voice readiness is reported but never gates the response: the text agent is
    healthy whether or not speech is available.
    """
    components = {
        "retriever": pipeline.retriever_ready,
        "workflow": pipeline.workflow is not None,
        "knowledge_base": pipeline.documents_dir.is_dir(),
    }

    if not pipeline.ready or not all(components.values()):
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Support pipeline is not ready",
                "components": components,
                "voice": voice.status,
            },
        )

    return {
        "status": "ready",
        "components": components,
        "model": settings.llm_model,
        "voice": voice.status,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    #
    # Voice never enters this path. Speech is transcribed to text first and
    # arrives here as an ordinary message, so session state, RAG, and the
    # ticket workflow behave identically for typed and dictated input.
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    # Keep this endpoint read-only. It returns the exact repository record
    # rather than asking the model to reconstruct ticket details.
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@app.post(
    "/voice/transcribe",
    response_model=TranscriptionResponse,
    responses=VOICE_ERRORS,
)
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe uploaded audio. The caller decides what to do with the text.

    This endpoint deliberately does not call the agent: the transcript goes back
    to the UI so the customer can correct it and submit it explicitly through
    ``POST /chat``.
    """
    if voice.pipeline is None:
        return voice_error(
            503,
            voice.unavailable_reason or "Voice support is not available",
        )

    audio_bytes = await audio.read()

    try:
        transcript, elapsed_ms = await voice.pipeline.transcribe(
            audio_bytes,
            audio.content_type or "",
        )
    except UnsupportedAudioError as exc:
        return voice_error(415, str(exc))
    except ValueError as exc:
        # Empty upload, or nothing intelligible in the recording.
        return voice_error(400, str(exc))
    except ComponentNotReadyError as exc:
        return voice_error(503, str(exc))
    except Exception:  # noqa: BLE001 - upstream STT failure
        return voice_error(502, "Speech recognition failed. Please try again.")

    return TranscriptionResponse(
        transcript=transcript,
        processing_time_ms=elapsed_ms,
    )


@app.post(
    "/voice/synthesize",
    responses={200: {"content": {"audio/mpeg": {}}}, **VOICE_ERRORS},
)
async def synthesize(request: SynthesisRequest):
    """Render one agent response as audio.

    ``message_id`` is echoed back in a header so the UI can confirm the audio it
    received belongs to the message whose speaker icon was clicked.
    """
    if voice.pipeline is None:
        return voice_error(
            503,
            voice.unavailable_reason or "Voice support is not available",
        )

    try:
        audio_bytes, media_type, elapsed_ms = await voice.pipeline.synthesize(
            request.text
        )
    except ValueError as exc:
        return voice_error(400, str(exc))
    except Exception:  # noqa: BLE001 - upstream TTS failure
        return voice_error(502, "Speech synthesis failed. The text reply is unaffected.")

    return Response(
        content=audio_bytes,
        media_type=media_type,
        headers={
            "X-Message-Id": request.message_id,
            "X-Processing-Time-Ms": str(elapsed_ms),
        },
    )
