from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import requests

from .models import CallTranscript, CriterionResult
from .preprocess import mask_pii_in_text, normalize_text
from .prompt_loader import load_prompt_bundle

LOG = logging.getLogger(__name__)

_ALLOWED_PROB = {"HIGH", "MEDIUM", "LOW"}


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


@dataclass(frozen=True)
class StubLLMClient:
    """
    Deterministic client for tests / offline runs.

    `canned` maps criterion code (e.g. "KR2.1") -> JSON-like dict response.
    """
    canned: Dict[str, Dict[str, Any]]

    def complete(self, system: str, user: str) -> str:
        picked: Optional[str] = None
        for code in self.canned.keys():
            if code in user:
                picked = code
                break

        key = picked if picked is not None else "DEFAULT"
        resp = self.canned.get(
            key,
            {"score": 2, "reasoning": "Stub reasoning", "evidence": "", "probability": "LOW"},
        )
        return json.dumps(resp, ensure_ascii=False)


class GroqClient:
    """
    Minimal Groq OpenAI-compatible chat completions client (optional).

    Endpoint:
      https://api.groq.com/openai/v1/chat/completions
    """
    def __init__(self, api_key: str, model: str, timeout_s: float) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout_s = timeout_s
        self._session = requests.Session()

    def complete(self, system: str, user: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
        }

        r = self._session.post(url, headers=headers, json=payload, timeout=self._timeout_s)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_client() -> Optional[LLMClient]:
    """
    Builds an optional LLM client from environment variables.

    - KONTAKT_LLM_PROVIDER: none|groq|stub
    - KONTAKT_LLM_MODEL: optional (default: llama-3.1-8b-instant)
    - GROQ_API_KEY: required if provider=groq
    - KONTAKT_LLM_TIMEOUT_SECONDS: optional (default: 30)
    """
    provider = os.getenv("KONTAKT_LLM_PROVIDER", "none").strip().lower()
    model = os.getenv("KONTAKT_LLM_MODEL", "").strip()
    timeout_s = _env_float("KONTAKT_LLM_TIMEOUT_SECONDS", 30.0)

    if provider in {"none", "", "off", "false", "0"}:
        return None

    if provider == "groq":
        key = os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            LOG.warning("KONTAKT_LLM_PROVIDER=groq but GROQ_API_KEY is missing -> LLM disabled")
            return None
        return GroqClient(
            api_key=key,
            model=(model or "llama-3.1-8b-instant"),
            timeout_s=timeout_s,
        )

    if provider == "stub":
        return StubLLMClient(canned={})

    LOG.warning("Unknown KONTAKT_LLM_PROVIDER=%s -> LLM disabled", provider)
    return None


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if not s.startswith("```"):
        return s

    # İlk sətir ``` və ya ```json ola bilər
    lines = s.splitlines()
    if len(lines) <= 1:
        return s.strip("`").strip()
    body_lines = lines[1:]

    # Son sətir ``` ilə bitirsə silinsin
    if body_lines and body_lines[-1].strip().startswith("```"):
        body_lines = body_lines[:-1]

    body = "\n".join(body_lines).strip()

    # json sözü ilə başlayırsa silinsin
    if body.lower().startswith("json"):
        body = body[4:].strip()

    return body


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort JSON object extraction.
    Handles:
      - pure JSON
      - code-fenced ```json ... ```
      - extra text around JSON (find first '{' and last '}')
    """
    s = (raw or "").strip()
    if not s:
        return None

    s = _strip_code_fence(s)

    # Try direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # Try substring parse
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            obj2 = json.loads(candidate)
            return obj2 if isinstance(obj2, dict) else None
        except Exception:
            return None

    return None


def llm_score_criterion(call: CallTranscript, criterion_code: str, client: LLMClient) -> CriterionResult:
    """
    LLM-ə əsasən bir KR-i qiymətləndirir.

    Qaydalar:
      - Transkript göndərilməzdən əvvəl PII (Fərdi məlumatları) maskalanır
      - JSON parsing (best-effort extraction)
      - Ehtimalın normalizasiyası
      - Anti-hallucination (yəni sübut transkriptdə sözü-sözünə olmalıdır)
      - Defense-in-depth (yəni sübut və əsaslandırma maskalanır)
    """
    bundle = load_prompt_bundle()
    system = bundle["system"]
    user_tmpl = bundle["user_template"]

    transcript_text = "\n".join(
        f"{s.speaker}: {mask_pii_in_text(s.text)}"
        for s in call.segments
        if s.text is not None
    ).strip()

    user = user_tmpl.format(criterion=criterion_code, transcript=transcript_text)

    try:
        raw = client.complete(system=system, user=user)
    except requests.RequestException:
        LOG.exception("LLM request failed -> fallback")
        return CriterionResult(
            score=0,
            reasoning="LLM servisinə sorğu uğursuz oldu; fallback tətbiq edildi.",
            probability="LOW",
            evidence=None,
        )

    except Exception:
        LOG.exception("LLM call failed -> fallback")
        return CriterionResult(
            score=0,
            reasoning="LLM çağırışı uğursuz oldu; fallback tətbiq edildi.",
            probability="LOW",
            evidence=None,
        )

    obj = _extract_json_object(raw)
    if obj is None:
        return CriterionResult(
            score=0,
            reasoning="LLM cavabı JSON formatında parse olunmadı; fallback tətbiq edildi.",
            probability="LOW",
            evidence=None,
        )

    try:
        score = int(obj.get("score", 0))
    except Exception:
        score = 0

    reasoning = str(obj.get("reasoning", "")).strip() or "LLM reasoning boşdur."
    evidence = str(obj.get("evidence", "")).strip()
    prob = str(obj.get("probability", "LOW")).strip().upper()
    if prob not in _ALLOWED_PROB:
        prob = "LOW"

    reasoning = mask_pii_in_text(reasoning)
    if evidence:
        evidence = mask_pii_in_text(evidence)

    # Anti-hallucination guard: evidence must exist verbatim (normalized substring check)
    if evidence:
        if normalize_text(evidence) not in normalize_text(transcript_text):
            return CriterionResult(
                score=0,
                reasoning="LLM evidence transkriptdə tapılmadı (anti-hallucination); fallback tətbiq edildi.",
                probability="LOW",
                evidence=None,
            )

    return CriterionResult(
        score=score,
        reasoning=reasoning,
        probability=prob,
        evidence=evidence or None,
    )