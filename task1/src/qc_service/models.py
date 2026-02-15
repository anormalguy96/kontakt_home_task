from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Segment(BaseModel):
  speaker: str = Field(..., description="Who is speaking (e.g., Operator, Customer)")
  text: str = Field(default="", description="Utterance text")
  start: float = Field(default=0.0, description="Seconds (ingestion accepts start or start_time)")
  end: float = Field(default=0.0, description="Seconds (ingestion accepts end or end_time)")

  @property
  def duration(self) -> float:
    return max(0.0, float(self.end) - float(self.start))


class Transcript(BaseModel):
  call_id: str = Field(..., description="Call identifier")
  segments: list[Segment] = Field(default_factory=list, description="Ordered list of transcript segments")


Probability = Literal["HIGH", "LOW"]


class MetricResult(BaseModel):
  score: int = Field(..., ge=0, le=3, description="Discrete score for the metric")
  reasoning: str = Field(..., description="Short explanation why this score was chosen")
  probability: Probability = Field(..., description="Confidence level for the score")
  evidence_snippet: Optional[str] = Field(default=None, description="Best-effort supporting snippet from transcript")


MetricKey = Literal["KR2.1", "KR2.2", "KR2.3", "KR2.4", "KR2.5"]


class EvaluationResult(BaseModel):
  call_id: str = Field(..., description="Echoed call identifier")
  results: dict[MetricKey, MetricResult] = Field(..., description="Per-metric scoring results")
  meta: dict[str, Any] = Field(default_factory=dict, description="Optional extra debug metadata")