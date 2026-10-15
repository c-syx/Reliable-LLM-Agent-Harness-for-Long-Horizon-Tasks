"""Minimal OpenRouter chat client with tool calling, throttling and retries."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from pathlib import Path

import requests


def load_dotenv(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a .env file into the environment. Existing variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class ToolCall:
    id: str
    name: str
    raw_arguments: str  # exactly what the model sent

    @property
    def arguments(self) -> dict | None:
        """Parsed arguments, or None if the model sent invalid JSON."""
        try:
            v = json.loads(self.raw_arguments) if self.raw_arguments else {}
        except json.JSONDecodeError:
            return None
        return v if isinstance(v, dict) else None


@dataclass
class LLMReply:
    content: str
    tool_calls: list[ToolCall]
    usage: dict = field(default_factory=dict)
    latency: float = 0.0


class OpenRouterLLM:
    URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model: str, api_key: str | None = None, temperature: float = 0.7,
                 min_interval: float = 3.2, max_retries: int = 6, timeout: int = 120):
        # min_interval of 3.2s keeps us under the free tier limit of 20 requests a minute.
        self.model = model
        load_dotenv()
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("No API key found. Put OPENROUTER_API_KEY=your_key in a file called .env, "
                               "or set the OPENROUTER_API_KEY environment variable.")
        self.temperature = temperature
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_call = 0.0

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if tools:  # a plain question with no tools must not send an empty tools list
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err = "unknown"
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            self._last_call = t0
            try:
                r = requests.post(self.URL, headers=headers, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last_err = f"network error {e}"
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {r.status_code}"
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text[:300]}")
            data = r.json()
            if "choices" not in data or not data["choices"]:
                # OpenRouter sometimes returns 200 with an error body from the upstream provider.
                last_err = f"bad body {str(data)[:200]}"
                time.sleep(min(60, 3 * 2 ** attempt))
                continue
            msg = data["choices"][0]["message"]
            calls = []
            for i, tc in enumerate(msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                calls.append(ToolCall(tc.get("id") or f"call_{i}", fn.get("name", ""), args or ""))
            return LLMReply(msg.get("content") or "", calls, data.get("usage", {}), time.time() - t0)
        raise RuntimeError(f"OpenRouter retries exhausted ({last_err})")


def assistant_message(reply: LLMReply) -> dict:
    """The assistant turn exactly as it should be appended to the history."""
    msg: dict = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        msg["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.raw_arguments}}
            for c in reply.tool_calls
        ]
    return msg