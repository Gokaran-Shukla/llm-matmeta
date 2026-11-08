"""Minimal, provider-agnostic LLM client.

Any server that speaks the OpenAI-compatible ``/chat/completions`` API works, e.g.
OpenAI, a local Ollama server (``http://localhost:11434/v1``), vLLM or LM Studio.
Configuration is read from environment variables:

* ``MATMETA_BASE_URL`` (default ``https://api.openai.com/v1``)
* ``MATMETA_API_KEY``  (not needed for local servers)
* ``MATMETA_MODEL``    (default ``gpt-4o-mini``)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Protocol

import requests


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict[str, Any]: ...


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("MATMETA_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("MATMETA_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("MATMETA_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.timeout = timeout

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=self.timeout)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        return parse_json_reply(content)


class MockClient:
    """Offline client for tests and demos: returns pre-defined replies (or calls a function)."""

    def __init__(self, reply: dict[str, Any] | list[dict[str, Any]] | Callable[[str, str], dict[str, Any]]):
        self._reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append((system, user))
        if callable(self._reply):
            return self._reply(system, user)
        if isinstance(self._reply, list):
            return self._reply[min(len(self.calls), len(self._reply)) - 1]
        return self._reply


def parse_json_reply(content: str) -> dict[str, Any]:
    """Parse a JSON object from a model reply, tolerating ```json fences or leading text."""
    content = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", content, re.S)
    if fence:
        content = fence.group(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(content[start : end + 1])
