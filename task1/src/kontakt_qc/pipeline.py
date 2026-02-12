from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from .hybrid import apply_llm_overrides
from .llm import build_client
from .rules import evaluate_rule_based
from .models import CallTranscript, CriterionResult, parse_transcript

LOG = logging.getLogger("kontakt_qc")


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def evaluate_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Main entrypoint.

    Modes (env):
      - KONTAKT_QC_MODE=rule   (default)
      - KONTAKT_QC_MODE=hybrid (LLM only for LOW-confidence criteria)
      - KONTAKT_QC_MODE=llm    (attempt LLM for all criteria; falls back if no provider)
    """
    try:
        call: CallTranscript = parse_transcript(payload)
    except Exception as e:
        LOG.exception("Failed to parse transcript: %s", e)
        call = CallTranscript(call_id="UNKNOWN_CALL", segments=tuple())

    # Chaos guard: if transcript effectively empty or too short, skip heavy work.
    # Specification: if audio < 0.1 sec -> model shouldn't run.
    if call.total_duration < 0.1 or len(call.segments) == 0:
        empty: Dict[str, CriterionResult] = {
            "KR2.1": CriterionResult(0, "Transkript çox qısadır və ya boşdur; qiymətləndirmə aparılmadı.", "LOW"),
            "KR2.2": CriterionResult(0, "Transkript çox qısadır və ya boşdur; qiymətləndirmə aparılmadı.", "LOW"),
            "KR2.3": CriterionResult(0, "Transkript çox qısadır və ya boşdur; qiymətləndirmə aparılmadı.", "LOW"),
            "KR2.4": CriterionResult(0, "Transkript çox qısadır və ya boşdur; qiymətləndirmə aparılmadı.", "LOW"),
            "KR2.5": CriterionResult(0, "Transkript çox qısadır və ya boşdur; qiymətləndirmə aparılmadı.", "LOW"),
        }
        return {call.call_id: {k: v.to_dict() for k, v in empty.items()}}

    base = evaluate_rule_based(call)

    mode = os.getenv("KONTAKT_QC_MODE", "rule").strip().lower()
    client = build_client()

    if mode == "hybrid":
        # LLM only for LOW confidence KR(s), with anti-hallucination guardrails.
        base = apply_llm_overrides(call, base, client, only_if_probability=("LOW",))
    elif mode == "llm":
        # Try to score everything with LLM. If provider missing, returns rule-based.
        base = apply_llm_overrides(call, base, client, only_if_probability=("LOW", "MEDIUM", "HIGH"))

    return {call.call_id: {k: v.to_dict() for k, v in base.items()}}


def evaluate_call_json(input_json: str) -> str:
    payload = json.loads(input_json)
    out = evaluate_call(payload)
    return json.dumps(out, ensure_ascii=False, indent=2)

        