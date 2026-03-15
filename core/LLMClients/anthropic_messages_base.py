"""
Shared helpers for Anthropic Messages API providers.
"""
from __future__ import annotations

from typing import Any

from .base import LLMClient, LLMResponse, ToolCall


class AnthropicMessagesClient(LLMClient):
    """Base adapter for providers that expose Anthropic's Messages API."""

    MAX_TOOLS = 256

    def __init__(self, client: Any, model_id: str):
        self._client = client
        self._model_id = model_id

    def provider_name(self) -> str:
        return self.PROVIDER

    def model_id(self) -> str:
        return self._model_id

    def send_message(self, message: str,
                     history: list[dict] | None = None) -> str:
        messages = list(history) if history else []
        messages.append({"role": "user", "content": message})
        return self.send_messages(messages)

    def send_messages(self, messages: list[dict]) -> str:
        response = self._messages_create(messages)
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def is_available(self) -> bool:
        try:
            self._messages_create(
                [{"role": "user", "content": "ping"}],
                max_tokens=16,
            )
            return True
        except Exception:
            return False

    def _messages_create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        system: str = "",
        max_tokens: int | None = None,
    ):
        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "max_tokens": max_tokens or self.MAX_TOKENS,
            "messages": messages,
        }
        if tools is not None:
            kwargs["tools"] = tools
        if system:
            kwargs["system"] = system
        return self._call_with_retry(self._client.messages.create, **kwargs)

    def send_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        response = self._messages_create(messages, tools=tools, system=system)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, input=block.input)
                )

        return LLMResponse(
            text="\n".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=(
                "tool_use" if response.stop_reason == "tool_use" else "end_turn"
            ),
            assistant_message={"role": "assistant", "content": response.content},
        )

    def make_tool_results(self, results: list[dict]) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": result["tool_use_id"],
                        "content": str(result["output"]),
                    }
                    for result in results
                ],
            }
        ]
