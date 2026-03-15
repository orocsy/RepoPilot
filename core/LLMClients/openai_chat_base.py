"""
Shared helpers for OpenAI-compatible chat-completions providers.
"""
from __future__ import annotations

import json
from typing import Any

from .base import LLMClient, LLMResponse, ToolCall


class OpenAIChatClient(LLMClient):
    """Base adapter for providers that expose the OpenAI chat API."""

    MAX_TOOLS = 128
    TOKEN_LIMIT_KWARG = "max_completion_tokens"

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
        response = self._chat_create(messages)
        choice = response.choices[0]
        return choice.message.content or ""

    def is_available(self) -> bool:
        try:
            self._chat_create(
                [{"role": "user", "content": "ping"}],
                max_tokens=16,
            )
            return True
        except Exception:
            return False

    def _chat_create(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
    ):
        kwargs = {
            "model": self._model_id,
            "messages": messages,
            self.TOKEN_LIMIT_KWARG: max_tokens or self.MAX_TOKENS,
        }
        if tools is not None:
            kwargs["tools"] = self._convert_tools(tools)
        return self._call_with_retry(self._client.chat.completions.create, **kwargs)

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "input_schema",
                        {"type": "object", "properties": {}},
                    ),
                },
            }
            for tool in tools
        ]

    def send_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
    ) -> LLMResponse:
        msgs = list(messages)
        if system:
            msgs.insert(0, {"role": "system", "content": system})

        response = self._chat_create(msgs, tools=tools)
        choice = response.choices[0]

        tool_calls: list[ToolCall] = []
        assistant_msg: dict = {
            "role": "assistant",
            "content": choice.message.content,
        }

        if choice.message.tool_calls:
            raw_tool_calls = []
            for tool_call in choice.message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments or "{}"),
                    )
                )
                raw_tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            assistant_msg["tool_calls"] = raw_tool_calls

        return LLMResponse(
            text=choice.message.content or "",
            tool_calls=tool_calls,
            stop_reason=(
                "tool_use" if choice.finish_reason == "tool_calls" else "end_turn"
            ),
            assistant_message=assistant_msg,
        )

    def make_tool_results(self, results: list[dict]) -> list[dict]:
        return [
            {
                "role": "tool",
                "tool_call_id": result["tool_use_id"],
                "content": str(result["output"]),
            }
            for result in results
        ]
