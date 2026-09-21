"""Pipeline-level guards that run before any model or retrieval call."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline import SupportPipeline
from src.utils.errors import ComponentNotReadyError


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(settings, tmp_path: Path) -> None:
    pipeline = SupportPipeline(settings, tmp_path)

    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_an_uninitialized_pipeline_reports_its_components_as_not_ready(
    settings, tmp_path: Path
) -> None:
    pipeline = SupportPipeline(settings, tmp_path)

    assert pipeline.ready is False
    assert pipeline.retriever_ready is False
    assert pipeline.workflow is None


@pytest.mark.asyncio
async def test_an_initialized_pipeline_reports_its_components_as_ready(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline()

    assert pipeline.ready is True
    assert pipeline.retriever_ready is True
    assert pipeline.workflow is not None


@pytest.mark.asyncio
async def test_blank_input_is_rejected_before_the_model_is_called(
    make_pipeline,
) -> None:
    pipeline = await make_pipeline()

    with pytest.raises(ValueError):
        await pipeline.process("   ", "hello")

    with pytest.raises(ValueError):
        await pipeline.process("session-1", "   ")

    # No scripted responses were consumed, so nothing reached the model.
    assert pipeline.model.decision_prompts == []
    assert pipeline.model.answer_prompts == []
