from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    success: bool = True
    transcript: str = Field(min_length=1)
    processing_time_ms: int = Field(ge=0)


class SynthesisRequest(BaseModel):
    message_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4000)


class VoiceErrorResponse(BaseModel):
    """Clear JSON error shape shared by both voice endpoints."""

    success: bool = False
    error: str = Field(min_length=1)
