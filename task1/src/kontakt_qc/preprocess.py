
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from .models import Segment

# --- PII patterns (simple, fast, interpretable) ---
_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
# CVV/CVC: very common ways agents ask
_CVV_RE = re.compile(r"\b(?:cvv|cvc)\b", re.IGNORECASE)
# Azerbaijan FIN (often 7 chars, alnum), but we keep it loose.
_FIN_RE = re.compile(r"\bFIN\b", re.IGNORECASE)

# Bracketed silence patterns in dataset: "[140 saniyə süküt]"
_SILENCE_RE = re.compile(r"\[(\d+)\s*saniyə(?:\s+[^\]]+)?\s*s[üu]k[üu]t[^\]]*\]", re.IGNORECASE)


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def segment_text(seg: Segment) -> str:
    return normalize_text(seg.text)


def contains_card_number(text: str) -> bool:
    return bool(_CARD_RE.search(text or ""))


def contains_cvv(text: str) -> bool:
    return bool(_CVV_RE.search(text or ""))


def contains_fin(text: str) -> bool:
    return bool(_FIN_RE.search(text or ""))


def extract_silence_seconds(text: str) -> Optional[int]:
    m = _SILENCE_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def mask_pii(text: str) -> str:
    # Mask card-like digit runs
    def _mask_digits(m: re.Match[str]) -> str:
        raw = m.group(0)
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 13:
            return raw
        # keep last 4 for traceability
        return "**** **** **** " + digits[-4:]

    out = _CARD_RE.sub(_mask_digits, text or "")
    # Mask potential CVV mentions without removing context
    out = re.sub(r"\b(\d{3})\b", "***", out) if contains_cvv(text or "") else out
    return out


def pick_first_operator_segment(segments: Iterable[Segment], predicate) -> Optional[Segment]:
    for seg in segments:
        if seg.speaker.lower().startswith("oper") and predicate(seg):
            return seg
    return None


def pick_any_segment(segments: Iterable[Segment], predicate) -> Optional[Segment]:
    for seg in segments:
        if predicate(seg):
            return seg
    return None


def format_evidence(seg: Segment) -> str:
    # Evidence is masked to avoid leaking PII in logs/outputs.
    safe_text = mask_pii(seg.text)
    return f"[{seg.start:.1f}-{seg.end:.1f}] {seg.speaker}: {safe_text}"


# Public alias used by LLM layer

def mask_pii_in_text(text: str) -> str:
    return mask_pii(text)
