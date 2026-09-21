from __future__ import annotations

from src.voice.contracts import TTSService


class EdgeTTSService(TTSService):
    """Text-to-speech using Microsoft Edge's neural voices via ``edge-tts``.

    The service needs no API key and no local model; it does need outbound
    network access, so every failure path here has to stay recoverable. The
    caller keeps the text response on screen when synthesis fails.
    """

    MEDIA_TYPE = "audio/mpeg"

    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        self.voice = voice
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def initialize(self) -> None:
        # Imported here so the text agent still starts on a machine where the
        # voice extras were never installed.
        import edge_tts  # noqa: F401

        self._ready = True

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)

        chunks = bytearray()

        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.extend(chunk["data"])

        return bytes(chunks), self.MEDIA_TYPE

    async def cleanup(self) -> None:
        self._ready = False
