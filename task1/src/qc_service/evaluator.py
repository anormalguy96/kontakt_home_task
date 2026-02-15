from __future__ import annotations

import json
import logging
from typing import Any

from .config import Settings
from .models import EvaluationResult, MetricResult, Transcript
from .preprocess import transcript_duration_s
from .rules.kr2 import score_all_kr2
from .llm.groq_client import GroqClient
from .llm.prompts import load_prompt_yaml
from .pii import redact_pii

logger = logging.getLogger(__name__)


def _validate_llm_output(llm: dict[str, Any], transcript: Transcript) -> dict[str, MetricResult] | None:
  # Hallucination guard: evidence_snippet must be present verbatim in transcript text. fileciteturn9file0L235-L242
  full_text = " ".join(f"{s.speaker}: {s.text}" for s in transcript.segments)
  results: dict[str, MetricResult] = {}
  for key in ["KR2.1", "KR2.2", "KR2.3", "KR2.4", "KR2.5"]:
    item = llm.get(key)
    if not isinstance(item, dict):
      return None
    try:
      score = int(item.get("score"))
      reasoning = str(item.get("reasoning", "")).strip()
      prob = str(item.get("probability", "LOW")).strip().upper()
      evidence = str(item.get("evidence_snippet", "")).strip()
    except Exception:
      return None

    if score < 0 or score > 3:
      return None
    if prob not in {"HIGH", "LOW"}:
      prob = "LOW"

    if evidence and evidence not in full_text:
      logger.warning("LLM evidence not found in transcript; dropping LLM output for %s", key)
      return None

    results[key] = MetricResult(score=score, reasoning=reasoning or "LLM qiymətləndirməsi", probability=prob, evidence_snippet=evidence or "")
  return results


def evaluate_transcript(transcript: Transcript, settings: Settings) -> EvaluationResult:
  # Short audio/transcript guard from task requirements. fileciteturn9file0L107-L109
  dur = transcript_duration_s(transcript.segments)
  if dur < 0.1:
    empty = {
      k: MetricResult(
        score=0,
        reasoning="Transkript çox qısadır (<0.1s), qiymətləndirmə mümkün deyil.",
        probability="LOW",
        evidence_snippet="",
      )
      for k in ["KR2.1", "KR2.2", "KR2.3", "KR2.4", "KR2.5"]
    }
    return EvaluationResult(call_id=transcript.call_id, results=empty, meta={"duration_s": dur, "llm_used": False})

  rule_results = score_all_kr2(transcript.segments)

  llm_used = False
  final_results = rule_results

  if settings.use_llm and settings.groq_api_key:
    try:
      prompt = load_prompt_yaml("prompts/kr2_scoring.yaml")
      # redact PII before sending to LLM. fileciteturn9file0L246-L248
      redacted = {
        "call_id": transcript.call_id,
        "segments": [
          {"speaker": s.speaker, "text": redact_pii(s.text), "start": s.start, "end": s.end}
          for s in transcript.segments
        ],
      }
      user = prompt["user"].replace("{{transcript_json}}", json.dumps(redacted, ensure_ascii=False))
      client = GroqClient(api_key=settings.groq_api_key, model=settings.groq_model)
      resp = client.chat_json(system=prompt["system"], user=user)

      if resp.parsed:
        validated = _validate_llm_output(resp.parsed, transcript)
        if validated:
          final_results = validated
          llm_used = True
    except Exception:
      logger.exception("LLM path failed; falling back to rule-based")

  return EvaluationResult(
    call_id=transcript.call_id,
    results=final_results,
    meta={"duration_s": dur, "llm_used": llm_used, "model": settings.groq_model if llm_used else None},
  )
