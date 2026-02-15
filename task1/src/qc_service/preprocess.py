from __future__ import annotations

import logging
from typing import Iterable

from .models import Segment, Transcript

logger = logging.getLogger(__name__)


def normalize_transcript(payload: dict) -> Transcript:
  """
  Normalize input JSON payload into an internal Transcript model.

  Accepts both (start, end) and (start_time, end_time) formats.
  call_id is a required string
  segments is a required list of objects
  each segment has:
      speaker: str (optional, default "")
      text: str (optional, default "")
      start/start_time: number-like (optional, default 0.0)
      end/end_time: number-like (optional, default 0.0)
  """
  call_id = payload.get("call_id")
  if not call_id or not isinstance(call_id, str):
    raise ValueError("Missing or invalid call_id")

  raw_segments = payload.get("segments")
  if not isinstance(raw_segments, list):
    raise ValueError("Missing or invalid segments list")

  segments: list[Segment] = []

  for i, seg in enumerate(raw_segments):
    if not isinstance(seg, dict):
      logger.warning("Segment %s is not an object; skipped", i)
      continue

    speaker = str(seg.get("speaker", "") or "").strip()
    text = str(seg.get("text", "") or "")

    start_raw = seg.get("start", seg.get("start_time", 0.0))
    end_raw = seg.get("end", seg.get("end_time", 0.0))

    try:
      start_f = float(start_raw)
      end_f = float(end_raw)
    except Exception:
      logger.warning(
        "Segment %s has non-numeric times (start=%r end=%r); forcing 0.0",
        i,
        start_raw,
        end_raw,
      )
      start_f, end_f = 0.0, 0.0

    # Fix swapped timestamps (happens in messy real transcripts)
    if end_f < start_f:
      logger.warning(
        "Segment %s has end < start (start=%s end=%s); swapping",
        i,
        start_f,
        end_f,
      )
      start_f, end_f = end_f, start_f

    # Keep empty segments but warn; real logs can contain blanks
    if not text.strip() or text.strip() in {"...", "..", "."}:
      logger.warning("Empty/broken text segment at index %s (speaker=%s)", i, speaker)

    segments.append(Segment(speaker=speaker, text=text, start=start_f, end=end_f))

  # Stable order for downstream logic
  segments.sort(key=lambda s: (s.start, s.end))
  return Transcript(call_id=call_id, segments=segments)


def transcript_duration_s(segments: Iterable[Segment]) -> float:
  starts = [s.start for s in segments]
  ends = [s.end for s in segments]
  if not starts or not ends:
    return 0.0
  return max(ends) - min(starts)