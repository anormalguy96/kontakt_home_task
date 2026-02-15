from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
_FIN_RE = re.compile(r"\b(?:FIN\s*kodu?m?\s*)?([0-9][A-Z0-9]{6,8})\b", re.IGNORECASE)
_CVV_RE = re.compile(r"\bCVV\b|\bCVC\b", re.IGNORECASE)


@dataclass(frozen=True)
class PiiFinding:
  kind: str
  value: str


def find_pii(text: str) -> list[PiiFinding]:
  findings: list[PiiFinding] = []
  for m in _CARD_RE.finditer(text):
    findings.append(PiiFinding(kind="card_number", value=m.group(0)))
  for m in _FIN_RE.finditer(text):
    val = m.group(1) or m.group(0)
    if len(val) >= 7:
      findings.append(PiiFinding(kind="fin_code", value=val))
  for m in _CVV_RE.finditer(text):
    findings.append(PiiFinding(kind="cvv_mention", value=m.group(0)))
  return findings


def redact_pii(text: str) -> str:
  text = _CARD_RE.sub("[CARD_NUMBER]", text)
  text = re.sub(r"\b\d{3,4}\b", "[NUM]", text)
  text = re.sub(r"\b([0-9][A-Z0-9]{6,8})\b", "[FIN_CODE]", text, flags=re.IGNORECASE)
  return text


def contains_pii(text: str) -> bool:
  return bool(find_pii(text))


def any_contains_pii(texts: Iterable[str]) -> bool:
  return any(contains_pii(t) for t in texts)
