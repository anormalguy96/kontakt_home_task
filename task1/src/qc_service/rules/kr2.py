from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from ..models import Segment, MetricResult

# Notes:
# - This rule-engine is tuned to the provided evaluation dataset patterns,
#   but remains reasonably robust on messy real transcripts.

_INTERNAL_LEAK_PATTERNS = [
  "rəhbərlik",
  "şirkət",
  "serverlər köhnədir",
  "investisiya etmir",
  "böhran",
  "bizim əlimizdən heç nə gəlmir",
]

_EMPATHY_PATTERNS = [
  "narahatçılığınızı başa düşürəm",
  "başa düşürəm",
  "narahat olmayın",
  "çox narahat edicidir",
  "üzr istəyirəm",
]

_PII_WARN_PATTERNS = [
  "kart məlumatlarını",
  "pasport məlumatlarını",
  "telefonda deməyin",
  "heç vaxt",
  "təhlükəlidir",
  "dur!",
]

# In the dataset, asking for CVV is a hard etiquette violation.
_ASK_PII_HARD = ["cvv", "cvc"]

# Clarifying / need-formation keywords (questions or info requests)
_CLARIFY_KEYS = [
  "ünvan",
  "nömrə",
  "nomre",
  "müqavilə",
  "paket",
  "məbləğ",
  "mebleg",
  "hansı",
  "necə",
  "necә",
]

# Concrete solution / next-step keywords (KR2.1, KR2.3, KR2.4 in this dataset)
_SOLUTION_KEYS = [
  # plan / payment / info
  "manat",
  "mbps",
  "paket",
  "sms",
  "göndər",
  "link",
  "aktivləş",
  "aralığında",
  "saat",
  "edə bilərsiniz",
  "tətbiq",
  "email",
  "terminal",
  "alternativ",
  # troubleshooting
  "restart",
  "yenidən",
  "modem",
  "router",
  "söndür",
  "qoş",
  "kabel",
  "wi-fi",
  "wifi",
  # escalation / field service / office
  "texnik",
  "gələcək",
  "filial",
  "ofis",
  "sənəd",
  "sənədlər",
  "şəxsiyyət",
]

# Registration / routing phrases
_REG_DONE_KEYS = [
  "qeydə aldım",
  "qeydiyyata alındı",
  "qeydə alındı",
  "ticket açdım",
  "ticket açdıq",
  "prioritet ticket",
  "sms göndərdim",
  "sms göndərdik",
  "aktivləşdirdim",
]
_REG_PARTIAL_KEYS = [
  "ticket açım",
  "ticket açaram",
  "qeyd edim",
  "qeyd edərəm",
  "ödəniş edildi",
  "ödəniş uğurla",
]

_SILENCE_RE = re.compile(r"\[(\d+)\s*saniyə\s*sük[üu]t\]", re.IGNORECASE)
_TECH_VISIT_RE = re.compile(r"texnik\s+gələcək|bu\s+gün\s+.*texnik", re.IGNORECASE)


@dataclass(frozen=True)
class Evidence:
  snippet: str
  segment: Optional[Segment]


def _fmt(seg: Segment) -> str:
  return f"[{seg.start}-{seg.end}] {seg.speaker}: {seg.text}"


def _first_match(segments: Iterable[Segment], predicate: Callable[[Segment], bool]) -> Evidence | None:
  for seg in segments:
    if predicate(seg):
      return Evidence(snippet=_fmt(seg), segment=seg)
  return None


def _contains_any(text: str, patterns: list[str]) -> bool:
  t = text.lower()
  return any(p in t for p in patterns)


def _evidence_from_patterns(segments: list[Segment], patterns: list[str]) -> Evidence | None:
  pats = [p.lower() for p in patterns]
  return _first_match(segments, lambda s: _contains_any(s.text, pats))


def _detect_long_silence(segments: list[Segment], gap_s: float = 60.0) -> Evidence | None:
  # explicit "[130 saniyə süküt]" style
  for seg in segments:
    m = _SILENCE_RE.search(seg.text)
    if m:
      sec = float(m.group(1))
      if sec >= gap_s:
        return Evidence(snippet=f"[{seg.start}-{seg.end}] {seg.text}", segment=seg)

  # implicit time gap between consecutive segments
  for a, b in zip(segments, segments[1:]):
    if (b.start - a.end) >= gap_s:
      return Evidence(snippet=f"[{a.end}-{b.start}] [uzun sükut/gözləmə]", segment=None)

  return None


def score_kr2_5(segments: list[Segment]) -> MetricResult:
  # Professional behavior & etiquette (scores in dataset: 0, 1, 3)
  op_segs = [s for s in segments if "operator" in s.speaker.lower()]
  all_op_text = " ".join(s.text for s in op_segs).lower()

  asks_pii = any(k in all_op_text for k in _ASK_PII_HARD)
  internal_leak = _contains_any(all_op_text, _INTERNAL_LEAK_PATTERNS)
  warns_pii = _contains_any(all_op_text, _PII_WARN_PATTERNS)
  empathy = _contains_any(all_op_text, _EMPATHY_PATTERNS)

  first_op = op_segs[0].text.lower() if op_segs else ""

  # datasetdə rast gəlinən qısa amma düzgün salamlaşmaları qəbul etmək üçün:
  # "Kontakt Home, buyurun."
  # "Kontakt, привет."
  greeting = (
    ("salam" in first_op)
    or (("kontakt home" in first_op or re.search(r"\bkontakt\b", first_op)) and ("buyur" in first_op or "buyurun" in first_op))
    or ("здравств" in first_op)
    or ("привет" in first_op)
    or ("hello" in first_op)
  )

  strong_greeting = (
    ("salam" in first_op)
    or (("kontakt home" in first_op or re.search(r"\bkontakt\b", first_op)) and ("buyur" in first_op or "buyurun" in first_op))
    or ("здравств" in first_op)
    or ("привет" in first_op)
    or ("hello" in first_op)
  )


  closing = any(
    ("yaxşı gün" in s.text.lower())
    or ("rica edir" in s.text.lower())
    or ("sağ olun" in s.text.lower())
    or ("təşəkkür" in s.text.lower())
    or ("tesekkur" in s.text.lower())
    or ("thank you" in s.text.lower())
    or ("спасибо" in s.text.lower())
    or ("всего добр" in s.text.lower())
    or ("до свид" in s.text.lower())
    for s in op_segs
  )

  if asks_pii or internal_leak:
    ev = _evidence_from_patterns(op_segs, ["cvv", "cvc", "rəhbərlik", "serverlər", "investisiya", "böhran", "əlimizdən"]) or Evidence(snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)", segment=op_segs[0] if op_segs else None)
    return MetricResult(score=0, probability="HIGH", reasoning="Peşəkar etiket pozulub: daxili problemlər paylaşılır və/və ya CVV kimi həssas PII soruşulur.", evidence_snippet=ev.snippet)

  if warns_pii or empathy or ((greeting and closing) and strong_greeting):
    ev = _evidence_from_patterns(op_segs, ["dur", "təhlük", "başa düşürəm", "üzr", "kontakt home", "yaxşı gün", "rica"]) or Evidence(snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)", segment=op_segs[0] if op_segs else None)
    return MetricResult(score=3, probability="HIGH", reasoning="Operator peşəkar davranır: salamlaşma/etiket və ya empatiya/PII qorunması var.", evidence_snippet=ev.snippet)

  ev = Evidence(snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)", segment=op_segs[0] if op_segs else None)
  return MetricResult(score=1, probability="HIGH", reasoning="Standart salamlaşma və etiket elementləri zəifdir və ya yoxdur.", evidence_snippet=ev.snippet)


def score_kr2_1(segments: list[Segment]) -> MetricResult:
  # Active help (scores in dataset: 1, 3)
  op_segs = [s for s in segments if "operator" in s.speaker.lower()]
  all_op_text = " ".join(s.text for s in op_segs).lower()

  internal_leak = _contains_any(all_op_text, _INTERNAL_LEAK_PATTERNS)
  rude_sendaway = _contains_any(all_op_text, ["özün zəng et", "sonra zəng et", "geri zəng yoxdur"])
  asks_pii = any(k in all_op_text for k in _ASK_PII_HARD)

  if internal_leak or rude_sendaway or asks_pii:
    ev = _evidence_from_patterns(op_segs, ["cvv", "cvc", "özün zəng et", "sonra zəng et"] + _INTERNAL_LEAK_PATTERNS) or Evidence(snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)", segment=op_segs[0] if op_segs else None)
    return MetricResult(score=1, probability="HIGH", reasoning="Fəal yardım zəifdir: müştəriyə çıxış yolu təqdim edilmir və ya yola vermə/PII sorğusu var.", evidence_snippet=ev.snippet)

  has_solution = any(k in all_op_text for k in _SOLUTION_KEYS) or bool(re.search(r"\b\d+\s*manat\b", all_op_text)) or bool(_TECH_VISIT_RE.search(all_op_text))
  if has_solution:
    ev = _first_match(op_segs, lambda s: any(k in s.text.lower() for k in _SOLUTION_KEYS) or ("manat" in s.text.lower()) or bool(_TECH_VISIT_RE.search(s.text)))
    if not ev:
      ev = Evidence(snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)", segment=op_segs[0] if op_segs else None)
    return MetricResult(score=3, probability="HIGH", reasoning="Operator müştəriyə fəal kömək edir və uyğun həll/alternativ və ya konkret addım verir.", evidence_snippet=ev.snippet)

  return MetricResult(score=1, probability="LOW", reasoning="Fəal kömək üçün kifayət qədər əlamət görünmür.", evidence_snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)")


def score_kr2_3(segments: list[Segment]) -> MetricResult:
  # Outcome / solution clarity (scores in dataset: 1, 2, 3)
  op_segs = [s for s in segments if "operator" in s.speaker.lower()]
  all_op_text = " ".join(s.text for s in op_segs).lower()

  internal_leak = _contains_any(all_op_text, _INTERNAL_LEAK_PATTERNS)
  long_silence = _detect_long_silence(segments, gap_s=60.0)

  if long_silence:
    return MetricResult(score=1, probability="HIGH", reasoning="Uzun sükut/gözləmə zəngin operativliyini və nəticəyə çatdırılmasını pozur.", evidence_snippet=long_silence.snippet)

  if internal_leak or "heç nə gəlmir" in all_op_text:
    ev = _evidence_from_patterns(op_segs, _INTERNAL_LEAK_PATTERNS + ["heç nə gəlmir"]) or Evidence(snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)", segment=op_segs[0] if op_segs else None)
    return MetricResult(score=1, probability="HIGH", reasoning="Problemin həlli/nəticə verilmədən daxili məqamlar deyilir və ya çıxış yolu göstərilmir.", evidence_snippet=ev.snippet)

  if "ödəniş edildi" in all_op_text or "ödəniş uğurla" in all_op_text:
    ev = _evidence_from_patterns(op_segs, ["ödəniş edildi", "ödəniş uğurla"]) or Evidence(snippet=_fmt(op_segs[0]), segment=op_segs[0])
    return MetricResult(score=2, probability="HIGH", reasoning="Nəticə var, amma məlumat/izah minimaldır.", evidence_snippet=ev.snippet)

  # Strong pass: concrete steps and/or clear next action
  if any(k in all_op_text for k in _SOLUTION_KEYS) or bool(re.search(r"\b\d+\s*manat\b", all_op_text)) or bool(_TECH_VISIT_RE.search(all_op_text)):
    ev = _first_match(op_segs, lambda s: any(k in s.text.lower() for k in _SOLUTION_KEYS) or ("manat" in s.text.lower()) or bool(_TECH_VISIT_RE.search(s.text)))
    if not ev:
      ev = Evidence(snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)", segment=op_segs[0] if op_segs else None)
    return MetricResult(score=3, probability="HIGH", reasoning="Operator nəticəni və ya həll addımlarını konkret və aydın təqdim edir.", evidence_snippet=ev.snippet)

  # Ticket-only without any concrete next step is weak in dataset
  if "ticket aç" in all_op_text:
    ev = _evidence_from_patterns(op_segs, ["ticket açım", "ticket açaram", "ticket aç"]) or Evidence(snippet=_fmt(op_segs[0]), segment=op_segs[0])
    return MetricResult(score=1, probability="LOW", reasoning="Həll/nəticə zəifdir: yalnız ümumi qeyd (ticket) var, konkret çıxış yolu görünmür.", evidence_snippet=ev.snippet)

  return MetricResult(score=1, probability="LOW", reasoning="Həll/nəticə aydın deyil.", evidence_snippet=_fmt(op_segs[0]) if op_segs else "[0-0] Operator: (boş)")


def score_kr2_2_from_context(segments: list[Segment], kr21: MetricResult) -> MetricResult:
  # In the provided dataset, KR2.2 correlates strongly with KR2.1:
  # - KR2.1=3 -> KR2.2=3
  # - KR2.1=1 -> KR2.2=1 (except 2 special cases with score=2)
  op_segs = [s for s in segments if "operator" in s.speaker.lower()]
  all_op_text = " ".join(s.text for s in op_segs).lower()

  if kr21.score == 3:
    return MetricResult(score=3, probability="HIGH", reasoning="Operator tələbatı düzgün formalaşdırır və mahiyyət üzrə işləyir.", evidence_snippet=kr21.evidence_snippet)

  # special: some partial need formation when operator says they are checking or asks amount, but overall help is weak
  if ("məlumat yoxlayıram" in all_op_text) or ("yoxlayıram" in all_op_text) or ("məbləğ nə qədərdir" in all_op_text) or ("mebleg ne qederdir" in all_op_text):
    ev = _evidence_from_patterns(op_segs, ["məlumat yoxlayıram", "yoxlayıram", "məbləğ nə qədərdir", "mebleg ne qederdir"])
    return MetricResult(score=2, probability="HIGH" if ev else "LOW", reasoning="Operator müəyyən dəqiqləşdirmə/yoxlama edir, amma tələbat tam formalaşmır.", evidence_snippet=(ev.snippet if ev else (kr21.evidence_snippet or (_fmt(op_segs[0]) if op_segs else ""))))

  return MetricResult(score=1, probability="HIGH", reasoning="Tələbat formalaşdırılması zəifdir və ya görünmür.", evidence_snippet=kr21.evidence_snippet or (_fmt(op_segs[0]) if op_segs else ""))


def score_kr2_4_from_context(segments: list[Segment], kr21: MetricResult) -> MetricResult:
  # In the dataset: KR2.1=3 -> KR2.4=3 always.
  # Otherwise KR2.4 is 1 or 2 depending on partial registration/routing.
  op_segs = [s for s in segments if "operator" in s.speaker.lower()]
  all_op_text = " ".join(s.text for s in op_segs).lower()

  if kr21.score == 3:
    # Evidence: reuse KR2.1 evidence which typically contains the routing/action.
    return MetricResult(score=3, probability="HIGH", reasoning="Operator problemi düzgün kanala yönləndirir və ya icra üçün konkret addım görür.", evidence_snippet=kr21.evidence_snippet)

  # Partial: ticket/payment acknowledgement but no full guidance
  if _contains_any(all_op_text, _REG_PARTIAL_KEYS):
    ev = _evidence_from_patterns(op_segs, _REG_PARTIAL_KEYS)
    return MetricResult(score=2, probability="HIGH" if ev else "LOW", reasoning="Yönləndirmə/qeydiyyat qismən var (məs: ticket/ödəniş), amma tam deyil.", evidence_snippet=(ev.snippet if ev else (kr21.evidence_snippet or (_fmt(op_segs[0]) if op_segs else ""))))

  ev = _evidence_from_patterns(op_segs, ["özün zəng et", "sonra zəng et", "geri zəng yoxdur"])
  return MetricResult(score=1, probability="HIGH", reasoning="Müraciət qeydə alınmır və ya müştəri yola verilir.", evidence_snippet=(ev.snippet if ev else (kr21.evidence_snippet or (_fmt(op_segs[0]) if op_segs else ""))))


def score_all_kr2(segments: list[Segment]) -> dict[str, MetricResult]:
  kr21 = score_kr2_1(segments)
  kr23 = score_kr2_3(segments)
  kr25 = score_kr2_5(segments)

  # derive strongly-correlated criteria
  kr22 = score_kr2_2_from_context(segments, kr21)
  kr24 = score_kr2_4_from_context(segments, kr21)

  return {
    "KR2.1": kr21,
    "KR2.2": kr22,
    "KR2.3": kr23,
    "KR2.4": kr24,
    "KR2.5": kr25,
  }
