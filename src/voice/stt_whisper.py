from __future__ import annotations

import asyncio
import io

from src.utils.errors import ComponentNotReadyError
from src.voice.contracts import STTService


#: Container formats browsers and desktop recorders actually produce. PyAV (a
#: faster-whisper dependency) decodes all of them, so no system ffmpeg is needed.
SUPPORTED_MEDIA_TYPES = frozenset(
    {
        "audio/wav",
        "audio/wave",
        "audio/x-wav",
        "audio/webm",
        "audio/ogg",
        "audio/mpeg",
        "audio/mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/flac",
        "video/webm",  # what some browsers label a MediaRecorder audio track
    }
)


class UnsupportedAudioError(ValueError):
    """Raised for audio this adapter cannot decode.

    A ``ValueError`` on purpose: a recording the decoder rejects is bad input,
    not a server fault, so the API answers 415 rather than 502.
    """


def _is_decode_error(exc: BaseException) -> bool:
    """Whether an exception came from PyAV failing to decode the container.

    Identified by module rather than by importing ``av``, so this module still
    imports cleanly when the voice extras are absent.
    """
    return type(exc).__module__.split(".")[0] == "av"


class WhisperSTTService(STTService):
    """Speech-to-text using faster-whisper, running fully on this machine.

    The model is loaded once at startup. Both loading and transcription are
    blocking CPU work, so each runs in a worker thread to keep the event loop
    free for other requests.
    """

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    async def initialize(self) -> None:
        # Imported here so the text agent still starts on a machine where the
        # voice extras were never installed.
        from faster_whisper import WhisperModel

        def _load():
            return WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

        self._model = await asyncio.to_thread(_load)

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        if self._model is None:
            raise ComponentNotReadyError("Speech-to-text model is not ready")

        normalized = (media_type or "").split(";")[0].strip().lower()

        if normalized and normalized not in SUPPORTED_MEDIA_TYPES:
            raise UnsupportedAudioError(
                f"Unsupported audio format: {media_type}"
            )

        def _run() -> str:
            segments, _info = self._model.transcribe(
                io.BytesIO(audio_bytes),
                beam_size=1,
                # Drops silence and background noise before decoding, which is
                # what turns an empty or unintelligible recording into an empty
                # transcript instead of hallucinated words.
                vad_filter=True,
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

        try:
            return await asyncio.to_thread(_run)
        except Exception as exc:
            if _is_decode_error(exc):
                raise UnsupportedAudioError(
                    "The recording could not be decoded. Please record again."
                ) from exc
            raise

    async def cleanup(self) -> None:
        self._model = None
