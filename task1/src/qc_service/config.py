from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
  groq_api_key: str | None = None
  groq_model: str = "llama-3.1-8b-instant"
  use_llm: bool = False
  log_level: str = "INFO"


def load_settings() -> Settings:
  groq_api_key = os.getenv("GROQ_API_KEY") or None
  groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
  use_llm = os.getenv("USE_LLM", "0").strip() in {"1", "true", "TRUE", "yes", "YES"}
  log_level = os.getenv("LOG_LEVEL", "INFO")
  return Settings(groq_api_key=groq_api_key, groq_model=groq_model, use_llm=use_llm, log_level=log_level)
