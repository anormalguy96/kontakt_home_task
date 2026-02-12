from __future__ import annotations

from kontakt_qc.hybrid import apply_llm_overrides
from kontakt_qc.llm import StubLLMClient
from kontakt_qc.models import CallTranscript, CriterionResult, Segment


def test_llm_override_rejected_when_evidence_missing():
    call = CallTranscript(
        call_id="H1",
        segments=(Segment("Operator", "Salam, necə kömək edim?", 0.0, 2.0),),
    )
    base = {"KR2.1": CriterionResult(1, "Base", "LOW")}
    stub = StubLLMClient(
        canned={
            "KR2.1": {"score": 3, "reasoning": "LLM", "evidence": "Sizə 50% endirim edəcəyik", "probability": "HIGH"}
        }
    )
    out = apply_llm_overrides(call, base, stub, only_if_probability=("LOW",))
    assert out["KR2.1"].score == 1


def test_llm_override_accepts_when_evidence_present():
    call = CallTranscript(
        call_id="H2",
        segments=(Segment("Operator", "Sizə 50% endirim edəcəyik.", 0.0, 2.0),),
    )
    base = {"KR2.1": CriterionResult(1, "Base", "LOW")}
    stub = StubLLMClient(
        canned={
            "KR2.1": {"score": 3, "reasoning": "LLM", "evidence": "Sizə 50% endirim edəcəyik.", "probability": "HIGH"}
        }
    )
    out = apply_llm_overrides(call, base, stub, only_if_probability=("LOW",))
    assert out["KR2.1"].score == 3
