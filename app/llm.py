from __future__ import annotations
import logging
from app.config import GROQ_MODEL, LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger("sovereign_fa.llm")

_SUPPORTED_PROVIDERS = {"ollama", "groq"}


def get_llm(temperature: float = 0):
    """Return a chat model instance for the configured LLM_PROVIDER.

    Raises ValueError on an unrecognized provider (fails fast at import
    time rather than surfacing as a confusing downstream error).
    """
    if LLM_PROVIDER not in _SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Expected one of {sorted(_SUPPORTED_PROVIDERS)}."
        )

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        logger.info("llm_provider_selected: groq model=%s", GROQ_MODEL)
        return ChatGroq(model=GROQ_MODEL, temperature=temperature)

    from langchain_ollama import ChatOllama

    logger.info("llm_provider_selected: ollama model=%s base_url=%s", OLLAMA_MODEL, OLLAMA_BASE_URL)
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=temperature)
