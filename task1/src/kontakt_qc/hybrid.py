from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

from .llm import LLMClient, llm_score_criterion
from .models import CallTranscript, CriterionResult


def apply_llm_overrides(
    call: CallTranscript,
    base: Dict[str, CriterionResult],
    client: Optional[LLMClient],
    only_if_probability: Sequence[str] = ("LOW",),
) -> Dict[str, CriterionResult]:
    """Optionally override some criteria with LLM outputs.

    Guardrails:
    - If client is None -> no-op
    - Only run on criteria whose base probability is in only_if_probability
    - LLM must return evidence that exists in transcript; otherwise no override
    """
    if client is None:
        return base

    out = dict(base)
    for code, res in base.items():
        # Trigger LLM if probability is LOW or if evidence is missing (even if rules were optimistic)
        should_override = res.probability.upper() in set(only_if_probability) or res.evidence is None
        if not should_override:
            continue
        llm_res = llm_score_criterion(call, code, client)
        # If LLM failed evidence validation it returns score=0 with fallback reasoning;
        # we do NOT override in that case.
        if "anti-hallucination" in llm_res.reasoning.lower():
            continue
        if "fallback" in llm_res.reasoning.lower() and llm_res.score == 0:
            continue
        out[code] = llm_res
    return out
