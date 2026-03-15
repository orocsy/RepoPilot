"""
Claude on Anthropic LLM client implementation.
"""
from __future__ import annotations

import anthropic

from .anthropic_messages_base import AnthropicMessagesClient


class ClaudeOnAnthropicClient(AnthropicMessagesClient):
    """Adapter for Claude models hosted directly on Anthropic."""

    PROVIDER = "Claude on Anthropic"
    DEFAULT_MODEL = "claude-sonnet-4-5"
    MAX_TOKENS = 4096
    MAX_TOOLS = 256

    FIELDS = [
        {
            "key": "model_id",
            "label": "Model ID",
            "placeholder": "claude-sonnet-4-5",
            "default": "claude-sonnet-4-5",
            "required": True,
            "secret": False,
        },
        {
            "key": "api_key",
            "label": "Anthropic API Key",
            "placeholder": "<your-anthropic-api-key>",
            "default": "",
            "required": True,
            "secret": True,
        },
    ]

    def __init__(self, api_key: str, model_id: str = DEFAULT_MODEL):
        self._api_key = api_key
        super().__init__(
            client=anthropic.Anthropic(api_key=self._api_key),
            model_id=model_id,
        )
