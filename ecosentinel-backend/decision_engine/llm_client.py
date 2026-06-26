"""
decision_engine/llm_client.py
-------------------------------
Provider-agnostic LLM client built on LiteLLM.

All provider configuration comes from config.settings.LLM_CONFIG,
which is populated from environment variables. To switch providers,
change environment variables only:

    # Local Ollama (default)
    LLM_PROVIDER=ollama
    LLM_MODEL=llama3.1:8b
    LLM_API_BASE=http://localhost:11434

    # OpenAI
    LLM_PROVIDER=openai
    LLM_MODEL=gpt-4o-mini
    LLM_API_KEY=sk-...

    # Azure OpenAI
    LLM_PROVIDER=azure
    LLM_MODEL=<deployment-name>
    LLM_API_BASE=https://<resource>.openai.azure.com
    LLM_API_KEY=<key>
    AZURE_API_VERSION=2024-02-15-preview

    # Anthropic
    LLM_PROVIDER=anthropic
    LLM_MODEL=claude-sonnet-4-6
    LLM_API_KEY=sk-ant-...

No code in this module ever needs to change when switching providers.
"""

import sys
import os
import json
import logging
from time import perf_counter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import litellm
from litellm import completion
from litellm.exceptions import APIConnectionError, Timeout, APIError

from config.settings import LLM_CONFIG

logger = logging.getLogger(__name__)

# Suppress LiteLLM's verbose default logging — we handle our own
litellm.suppress_debug_info = True


class LLMClientError(Exception):
    """Raised when the LLM call fails after all retries."""
    pass


def _build_model_string() -> str:
    """
    Builds the LiteLLM model string from LLM_CONFIG.

    LiteLLM convention: "{provider}/{model}"
    Examples:
        ollama/llama3.1:8b
        openai/gpt-4o-mini
        azure/<deployment-name>
        anthropic/claude-sonnet-4-6
    """
    provider = LLM_CONFIG["provider"]
    model    = LLM_CONFIG["model"]
    return f"{provider}/{model}"


def _build_completion_kwargs(messages: list[dict]) -> dict:
    """
    Builds the kwargs dict for litellm.completion() based on
    the configured provider. Only includes parameters relevant
    to that provider — e.g. api_base is only meaningful for
    self-hosted providers like Ollama or Azure.
    """
    kwargs = {
        "model":       _build_model_string(),
        "messages":    messages,
        "temperature": LLM_CONFIG["temperature"],
        "max_tokens":  LLM_CONFIG["max_tokens"],
        "timeout":     LLM_CONFIG["timeout"],
        "num_retries": LLM_CONFIG["max_retries"],
    }

    # api_base is used by Ollama, Azure, and any self-hosted/proxy setup
    if LLM_CONFIG.get("api_base"):
        kwargs["api_base"] = LLM_CONFIG["api_base"]

    # api_key is required for cloud providers, not used by Ollama
    if LLM_CONFIG.get("api_key"):
        kwargs["api_key"] = LLM_CONFIG["api_key"]

    # Azure-specific
    if LLM_CONFIG.get("api_version"):
        kwargs["api_version"] = LLM_CONFIG["api_version"]

    # Request JSON output where the provider supports it.
    # LiteLLM passes this through as response_format for
    # OpenAI/Azure/compatible providers. Ollama (via LiteLLM)
    # also supports "format": "json" — handled by passing
    # response_format and letting LiteLLM translate per-provider.
    # If unsupported, the provider simply ignores it and we
    # rely on prompt instructions + parsing fallback.
    try:
        kwargs["response_format"] = {"type": "json_object"}
    except Exception:
        pass

    return kwargs


def call_llm(messages: list[dict]) -> str:
    """
    Sends a chat completion request to the configured LLM provider
    and returns the raw text response.

    Parameters
    ----------
    messages : list of {"role": ..., "content": ...} dicts,
               standard OpenAI-style chat format. Works across
               all LiteLLM-supported providers.

    Returns
    -------
    The raw text content of the LLM's response.

    Raises
    ------
    LLMClientError on connection failure, timeout, or API error
    after retries are exhausted.
    """
    kwargs = _build_completion_kwargs(messages)
    model_str = kwargs["model"]
    started = perf_counter()

    logger.info(
        f"Calling LLM: {model_str} with {len(messages)} message(s), max_tokens={kwargs.get('max_tokens')}, timeout={kwargs.get('timeout')}, retries={kwargs.get('num_retries')}."
    )
    logger.debug(
        f"LLM request content lengths: {[len(m.get('content', '')) for m in messages]}."
    )

    try:
        response = completion(**kwargs)
    except (APIConnectionError, Timeout) as e:
        raise LLMClientError(
            f"LLM connection failed for model '{model_str}': {e}. "
            f"If using Ollama, ensure it is running at "
            f"{LLM_CONFIG.get('api_base')} and the model "
            f"'{LLM_CONFIG['model']}' is pulled (ollama pull {LLM_CONFIG['model']})."
        ) from e
    except APIError as e:
        raise LLMClientError(f"LLM API error for model '{model_str}': {e}") from e
    except Exception as e:
        # response_format may be unsupported by some Ollama models —
        # retry once without it before giving up.
        if "response_format" in kwargs:
            logger.warning(
                f"LLM call failed with response_format set ({e}); "
                f"retrying without it."
            )
            kwargs.pop("response_format")
            try:
                response = completion(**kwargs)
            except Exception as e2:
                raise LLMClientError(
                    f"LLM call failed for model '{model_str}': {e2}"
                ) from e2
        else:
            raise LLMClientError(
                f"LLM call failed for model '{model_str}': {e}"
            ) from e

    content = response.choices[0].message.content
    if content is None:
        raise LLMClientError(f"LLM '{model_str}' returned empty response.")

    logger.info(
        f"LLM call completed for {model_str} in {(perf_counter() - started) * 1000:.1f} ms; response_length={len(content)}."
    )

    return content


def get_provider_info() -> dict:
    """Returns current provider/model config for logging and provenance."""
    return {
        "provider": LLM_CONFIG["provider"],
        "model":    LLM_CONFIG["model"],
    }