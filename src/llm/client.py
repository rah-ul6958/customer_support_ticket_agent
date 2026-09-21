from langchain_openai import ChatOpenAI

from src.config import Settings


def build_chat_model(settings: Settings) -> ChatOpenAI:
    """Create the configured LangChain chat-model adapter.

    The upstream endpoint must expose an OpenAI-compatible chat-completions API.
    This binding is supplied so candidates focus on agent behavior rather than
    rediscovering how the selected model connects to LangChain.
    """
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.2,
        max_retries=1,
        timeout=30,
    )
