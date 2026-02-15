from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroqResponse:
  raw: dict[str, Any]
  parsed: Optional[dict[str, Any]]


class GroqClient:
  def __init__(self, api_key: str, model: str) -> None:
    self._api_key = api_key
    self._model = model

  def chat_json(self, system: str, user: str, timeout_s: float = 30.0) -> GroqResponse:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
    payload = {
      "model": self._model,
      "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
      "temperature": 0,
      "response_format": {"type": "json_object"},
    }

    try:
      with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        raw = r.json()
    except Exception as e:
      logger.exception("Groq request failed: %s", e)
      return GroqResponse(raw={"error": str(e)}, parsed=None)

    try:
      content = raw["choices"][0]["message"]["content"]
      parsed = json.loads(content)
      return GroqResponse(raw=raw, parsed=parsed)
    except Exception as e:
      logger.exception("Groq response parse failed: %s", e)
      return GroqResponse(raw=raw, parsed=None)
