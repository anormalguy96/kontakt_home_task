from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import requests

from .prompt_loader import load_prompt_bundle
from .preprocess import mask_pii_in_text, normalize_text
from .models import CallTranscript, CriterionResult


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


@dataclass
class StubLLMClient:
    """Deterministic client for tests and offline runs."""

    # Map (criterion_code) -> response dict
    canned: Dict[str, Dict[str, Any]]

    def complete(self, system: str, user: str) -> str:
        # Expect criterion code in the user prompt as "CRITERION: KR2.x"
        m = None
        for code in self.canned.keys():
            if code in user:
                m = code
                break
        key: str = m if m is not None else "DEFAULT"
        resp = self.canned.get(key, {"score": 2, "reasoning": "Stub reasoning", "evidence": "", "probability": "LOW"})
        return json.dumps(resp, ensure_ascii=False)


class GroqClient:
    """Minimal Groq chat completions client (optional)."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


class GeminiClient:
    """Minimal Gemini REST client (optional)."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        # Gemini API format may evolve; keep this as an optional adapter.
        # We keep it minimal and best-effort.
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": system + "\n\n" + user}]}
            ],
            "generationConfig": {"temperature": 0.0},
        }
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Extract first candidate text (best-effort)
        return data["candidates"][0]["content"]["parts"][0]["text"]


def build_client() -> Optional[LLMClient]:
    provider = os.getenv("KONTAKT_LLM_PROVIDER", "none").strip().lower()
    model = os.getenv("KONTAKT_LLM_MODEL", "").strip()

    if provider in {"none", "", "off"}:
        return None

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            return None
        return GroqClient(key, model or "llama-3.1-8b-instant")

    if provider == "gemini":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            return None
        return GeminiClient(key, model or "gemini-1.5-flash")

    if provider == "stub":
        # Useful for local experimentation; caller should pass real stub via injection.
        return StubLLMClient(canned={})

    return None


def llm_score_criterion(call: CallTranscript, criterion_code: str, client: LLMClient) -> CriterionResult:
    bundle = load_prompt_bundle()
    system = bundle["system"]
    user_tmpl = bundle["user_template"]

    # Mask PII before sending to an LLM (privacy + compliance)
    transcript_text = "\n".join(
        f"{s.speaker}: {mask_pii_in_text(s.text)}" for s in call.segments if s.text is not None
    ).strip()

    user = user_tmpl.format(criterion=criterion_code, transcript=transcript_text)

    raw = client.complete(system=system, user=user)

    # Parse as JSON (strict). If parsing fails, treat as low-confidence fallback.
    try:
        obj = json.loads(raw)
    except Exception:
        return CriterionResult(
            score=0,
            reasoning="LLM cavabı JSON formatında deyildi; fallback tətbiq edildi.",
            probability="LOW",
            evidence=None,
        )

    score = int(obj.get("score", 0))
    reasoning = str(obj.get("reasoning", "")).strip() or "LLM reasoning boşdur."
    evidence = str(obj.get("evidence", "")).strip()
    prob = str(obj.get("probability", "LOW")).strip().upper()
    if prob not in {"HIGH", "MEDIUM", "LOW"}:
        prob = "LOW"

    # Anti-hallucination guard: evidence must be present verbatim (normalized substring check)
    if evidence:
        if normalize_text(evidence) not in normalize_text(transcript_text):
            return CriterionResult(
                score=0,
                reasoning="LLM evidence transkriptdə tapılmadı (anti-hallucination). Rule-based nəticə saxlanıldı.",
                probability="LOW",
                evidence=None,
            )

    return CriterionResult(score=score, reasoning=reasoning, probability=prob, evidence=evidence or None)
