
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Segment:
    speaker: str
    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def format_evidence(self) -> str:
        return f"[{self.start:.1f}-{self.end:.1f}] {self.speaker}: {self.text}"


@dataclass(frozen=True)
class CallTranscript:
    call_id: str
    segments: Tuple[Segment, ...]

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        start = min(s.start for s in self.segments)
        end = max(s.end for s in self.segments)
        return max(0.0, float(end) - float(start))


@dataclass(frozen=True)
class CriterionResult:
    score: int
    reasoning: str
    probability: str
    evidence: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "score": int(self.score),
            "reasoning": self.reasoning,
            "probability": self.probability,
        }
        if self.evidence:
            d["evidence_snippet"] = self.evidence
        return d


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def parse_transcript(payload: Dict[str, Any]) -> CallTranscript:
    call_id = str(payload.get("call_id", "")).strip() or "UNKNOWN_CALL"

    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        raw_segments = []

    segments: List[Segment] = []
    for i, seg in enumerate(raw_segments):
        if not isinstance(seg, dict):
            continue
        speaker = str(seg.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        text = str(seg.get("text", "")).strip()
        # Support both (start,end) and (start_time,end_time)
        start = safe_float(seg.get("start", seg.get("start_time", 0.0)), 0.0)
        end = safe_float(seg.get("end", seg.get("end_time", start)), start)
        if end < start:
            start, end = end, start
        # Keep even empty text segments for robustness; downstream will ignore appropriately.
        segments.append(Segment(speaker=speaker, text=text, start=start, end=end))

    return CallTranscript(call_id=call_id, segments=tuple(segments))
