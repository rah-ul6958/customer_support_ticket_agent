from __future__ import annotations

from abc import ABC, abstractmethod


class STTService(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the selected STT client or local model."""

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        """Return only the recognized text."""

    async def cleanup(self) -> None:
        """Release optional STT resources."""


class TTSService(ABC):
    @abstractmethod
    async def initialize(self) -> None:
        """Prepare the selected TTS client or local model."""

    @abstractmethod
    async def synthesize(self, text: str) -> tuple[bytes, str]:
        """Return audio bytes and their media type."""

    async def cleanup(self) -> None:
        """Release optional TTS resources."""
