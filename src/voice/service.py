from __future__ import annotations

import logging

from src.config import Settings
from src.voice.pipeline import VoicePipeline
from src.voice.stt_whisper import WhisperSTTService
from src.voice.tts_edge import EdgeTTSService


logger = logging.getLogger(__name__)


class VoiceSupport:
    """Optional voice layer wrapped around the supplied :class:`VoicePipeline`.

    Voice is strictly additive: it sits beside the text agent and never inside
    it. If the adapters cannot start -- the extras are not installed, the model
    will not load, voice is switched off -- initialization records why and the
    text agent carries on untouched. ``/chat`` never consults this class.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.pipeline: VoicePipeline | None = None
        self.unavailable_reason: str | None = (
            None if settings.voice_enabled else "Voice is disabled by configuration"
        )

    @property
    def available(self) -> bool:
        return self.pipeline is not None

    @property
    def status(self) -> dict[str, object]:
        """Readiness detail for ``/health``."""
        return {
            "available": self.available,
            "stt_model": self.settings.stt_model,
            "tts_voice": self.settings.tts_voice,
            "detail": self.unavailable_reason,
        }

    async def initialize(self) -> None:
        """Bring voice up if possible. Never raises: voice is optional."""
        if not self.settings.voice_enabled:
            logger.info("Voice support disabled by configuration")
            return

        pipeline = VoicePipeline(
            WhisperSTTService(
                model_size=self.settings.stt_model,
                device=self.settings.stt_device,
                compute_type=self.settings.stt_compute_type,
            ),
            EdgeTTSService(voice=self.settings.tts_voice),
        )

        try:
            await pipeline.initialize()
        except ImportError as exc:
            self.unavailable_reason = (
                "Voice extras are not installed. Run "
                "'pip install -r requirements.txt' to enable voice."
            )
            logger.warning("Voice support unavailable: %s", exc)
            return
        except Exception as exc:  # noqa: BLE001 - voice must never block startup
            self.unavailable_reason = f"Voice adapters failed to start: {exc}"
            logger.warning("Voice support unavailable: %s", exc)
            return

        self.pipeline = pipeline
        self.unavailable_reason = None
        logger.info(
            "Voice support ready (stt=%s, tts=%s)",
            self.settings.stt_model,
            self.settings.tts_voice,
        )

    async def cleanup(self) -> None:
        if self.pipeline is None:
            return

        try:
            await self.pipeline.cleanup()
        except Exception as exc:  # noqa: BLE001 - shutdown must not fail
            logger.warning("Voice cleanup failed: %s", exc)
        finally:
            self.pipeline = None
