"""Construct LangChain chat models for Ollama or Google Gemini."""

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from config import Settings, get_settings

Provider = Literal["ollama", "gemini"]


def get_chat_model(
    provider: str,
    model_name: str,
    *,
    temperature: float = 0.7,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Return a `BaseChatModel` for the given provider.

    Gemini API key is only required when ``provider`` is ``"gemini"``; missing
    ``GOOGLE_API_KEY`` at settings load time does not raise until this path runs.
    """
    cfg = settings or get_settings()
    p = provider.strip().lower()
    if p == "ollama":
        return ChatOllama(
            model=model_name,
            base_url=cfg.ollama_base_url,
            temperature=temperature,
        )
    if p == "gemini":
        key = (cfg.google_api_key or "").strip()
        if not key:
            raise ValueError(
                "GOOGLE_API_KEY is required when using the Gemini provider. "
                "Set it in your environment or `.env` file."
            )
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=key,
            temperature=temperature,
        )
    raise ValueError(
        f"Unknown LLM provider {provider!r}. Expected 'ollama' or 'gemini'."
    )
