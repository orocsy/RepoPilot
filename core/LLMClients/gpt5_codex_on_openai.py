"""
GPT-5-Codex on OpenAI LLM client implementation.
"""
from __future__ import annotations

import openai

from .openai_chat_base import OpenAIChatClient


class GPT5CodexOnOpenAIClient(OpenAIChatClient):
    """Adapter for GPT-5-Codex models hosted directly on OpenAI."""

    PROVIDER = "GPT-5-Codex on OpenAI"
    DEFAULT_MODEL = "gpt-5-codex"
    MAX_TOKENS = 16384
    MAX_TOOLS = 128
    TOKEN_LIMIT_KWARG = "max_completion_tokens"

    FIELDS = [
        {
            "key": "model_id",
            "label": "Model ID",
            "placeholder": "gpt-5-codex",
            "default": "gpt-5-codex",
            "required": True,
            "secret": False,
        },
        {
            "key": "api_key",
            "label": "OpenAI API Key",
            "placeholder": "<your-openai-api-key>",
            "default": "",
            "required": True,
            "secret": True,
        },
    ]

    def __init__(self, api_key: str, model_id: str = DEFAULT_MODEL):
        self._api_key = api_key
        super().__init__(
            client=openai.OpenAI(api_key=self._api_key),
            model_id=model_id,
        )
