from __future__ import annotations

from time import perf_counter

from .contracts import STTService, TTSService


class VoicePipeline:
    """Shared orchestration scaffold; candidates supply concrete adapters."""

    def __init__(self, stt: STTService, tts: TTSService) -> None:
        self.stt = stt
        self.tts = tts

    async def initialize(self) -> None:
        await self.stt.initialize()
        await self.tts.initialize()

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> tuple[str, int]:
        if not audio_bytes:
            raise ValueError("Audio input is empty")
        started = perf_counter()
        transcript = (await self.stt.transcribe(audio_bytes, media_type)).strip()
        if not transcript:
            raise ValueError("No understandable speech was detected")
        return transcript, round((perf_counter() - started) * 1000)

    async def synthesize(self, text: str) -> tuple[bytes, str, int]:
        if not text.strip():
            raise ValueError("Text input is empty")
        started = perf_counter()
        audio, media_type = await self.tts.synthesize(text)
        if not audio:
            raise ValueError("TTS returned empty audio")
        return audio, media_type, round((perf_counter() - started) * 1000)

    async def cleanup(self) -> None:
        await self.stt.cleanup()
        await self.tts.cleanup()
