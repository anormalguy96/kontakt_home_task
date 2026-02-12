
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from .preprocess import (
    contains_card_number,
    contains_cvv,
    contains_fin,
    extract_silence_seconds,
    format_evidence,
    normalize_text,
    pick_any_segment,
    pick_first_operator_segment,
)
from .models import CallTranscript, CriterionResult, Segment


# --- Scenario detectors (fast rule-based) ---
_INTERNAL_LEAK_RE = re.compile(
    r"(rəhbərlik|şirkət|investisiya|böhran|serverlər|server|texniklərimiz|menecer).*"
    r"(heç nə etmir|investisiya etmir|köhnədir|az qalıb|böhran|qayğısına qalmaz|heç nə gəlmir|pis)|"
    r"(bizim əlimizdən heç nə gəlmir)|"
    r"(sistemimiz\s+çox\s+pis)|"
    r"(özün başqa operatora keç|başqa operatora keç)",
    re.IGNORECASE,
)

# Callback refusal signals (both AZ + RU friendly)
_CALLBACK_REFUSAL_RE = re.compile(
    r"(geri zəng etmirik|geri zəng etmərik|geri zəng yoxdur|sonra zəng et|yenidən zəng et|özünüz sabah bir daha zəng)|"
    r"(callback yoxdur|callback|call\s*-?back)|"
    r"(xeyr\.?\s*özün.*zəng et|özün\s*bir\s*saata\s*zəng\s*et|özün.*zəng\s*et)|"
    r"(belə\s*xidmət\s*yoxdur|belə\s*xidmət\s*olmur)|"
    r"(перезвон|не перезваниваем)",
    re.IGNORECASE,
)

# Empathy/apology signals
_EMPATHY_RE = re.compile(r"(başa düş|narahat|üzr|təəssüf|sorry|понимаю)", re.IGNORECASE)

# “checking system / looking up” signals
_CHECKING_RE = re.compile(r"(məlumat\s*yoxlay|məlumatı\s*yoxlay|məlumat\s*yoxlayıram)", re.IGNORECASE)

# “ticket / registration / next steps” signals
_NEXT_STEPS_RE = re.compile(
    r"(ticket|qeyd|qeydə al|müraciət|texnik|24 saat|18:00|gələcək|SMS|link göndər|transfer|departament)",
    re.IGNORECASE,
)

# PII protection signals (agent stops PII)
_PII_PROTECT_RE = re.compile(
    r"(kart məlumat|kart nömrə).*?(deməyin|deyilməsin|heç vaxt)|"
    r"(təhlükəsiz|secure|link göndər|sms alacaqsınız)",
    re.IGNORECASE,
)


def detect_internal_leak(call: CallTranscript) -> Optional[Segment]:
    return pick_first_operator_segment(call.segments, lambda s: bool(_INTERNAL_LEAK_RE.search(s.text)))


def detect_pii_mishandling(call: CallTranscript) -> Tuple[bool, Optional[Segment]]:
    # If customer mentions card/FIN and operator *requests* CVV or continues without warning -> mishandled
    customer_has_pii = any(
        (seg.speaker.lower().startswith("cust") and (contains_card_number(seg.text) or contains_fin(seg.text)))
        for seg in call.segments
    )
    if not customer_has_pii:
        return False, None

    # If operator explicitly protects PII -> not mishandled
    protects = pick_first_operator_segment(call.segments, lambda s: bool(_PII_PROTECT_RE.search(s.text)))
    if protects:
        return False, protects

    # Operator asks for CVV / card details
    ask_cvv = pick_first_operator_segment(call.segments, lambda s: contains_cvv(s.text) or bool(re.search(r"\bCVV\b|\bCVC\b", s.text, re.I)))
    if ask_cvv:
        return True, ask_cvv

    # If customer drops card number and operator proceeds with payment (“Ödəniş edildi”, etc.) without warning -> mishandled
    proceed = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(ödəniş|payment).*(etdi|oldu|tamamlandı)|uğurla", s.text, re.I)))
    if proceed:
        return True, proceed

    # Default: cautious — treat as mishandled if no explicit protection
    first_pii_customer = pick_any_segment(call.segments, lambda s: s.speaker.lower().startswith("cust") and (contains_card_number(s.text) or contains_fin(s.text)))
    return True, first_pii_customer


def detect_callback_failure(call: CallTranscript) -> Tuple[bool, Optional[Segment]]:
    # Signal: long silences and refusal to call back / rude “call later”
    has_long_silence = False
    for seg in call.segments:
        secs = extract_silence_seconds(seg.text)
        if secs is not None and secs >= 90:
            has_long_silence = True
            break
    refusal_seg = pick_first_operator_segment(call.segments, lambda s: bool(_CALLBACK_REFUSAL_RE.search(s.text)))
    if has_long_silence and refusal_seg:
        return True, refusal_seg
    # Some samples are short but still mention callback refusal
    if refusal_seg and any(extract_silence_seconds(s.text) is not None for s in call.segments):
        return True, refusal_seg
    return False, None


def _prob(level: str) -> str:
    # Keep as HIGH/MEDIUM/LOW per spec
    return level


def score_kr25_professionalism(call: CallTranscript, internal_leak: Optional[Segment], pii_mishandled: bool, cb_fail: bool) -> CriterionResult:
    # 0: severe violation (internal leak OR PII leak)
    if internal_leak:
        return CriterionResult(
            score=0,
            reasoning="Operator daxili problemləri/şirkət barədə mənfi məlumat paylaşır (brand və etik qaydalar pozulur).",
            probability=_prob("HIGH"),
            evidence=format_evidence(internal_leak),
        )
    if pii_mishandled:
        seg = pick_any_segment(call.segments, lambda s: s.speaker.lower().startswith("oper") and ("cvv" in s.text.lower() or "kart" in s.text.lower() or "fin" in s.text.lower()))
        return CriterionResult(
            score=0,
            reasoning="PII (kart/FIN/CVV) təhlükəsiz şəkildə idarə edilməyib — operator dayandırmalı və təhlükəsiz kanala yönləndirməli idi.",
            probability=_prob("HIGH"),
            evidence=format_evidence(seg) if seg else None,
        )

    # 1: weak greeting/closing OR callback rude case
    if cb_fail:
        first_op = pick_first_operator_segment(call.segments, lambda s: True)
        return CriterionResult(
            score=1,
            reasoning="Salamlama/etiket zəifdir və müştəriyə xidmət tonu sərtdir (uzun gözlətmə və geri zəngdən imtina).",
            probability=_prob("HIGH"),
            evidence=format_evidence(first_op) if first_op else None,
        )

    # 3: good professionalism
    # Evidence: greeting or empathy or closing
    ev = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(kontakt|salam|xahiş|rica|yaxşı gün|добрый|здравствуйте)", s.text, re.I)))
    return CriterionResult(
        score=3,
        reasoning="Operator peşəkar və nəzakətli ünsiyyət qurur (etiket, ton, şirkət brendi qorunur).",
        probability=_prob("HIGH" if ev else "LOW"),
        evidence=format_evidence(ev) if ev else None,
    )


def score_kr21_ownership(call: CallTranscript, internal_leak: Optional[Segment], pii_mishandled: bool, cb_fail: bool) -> CriterionResult:
    if internal_leak or pii_mishandled or cb_fail:
        ev = None
        if internal_leak:
            ev = internal_leak
            reason = "Operator məsuliyyəti üzərinə götürmür və qeyri-peşəkar yönləndirmə edir."
        elif pii_mishandled:
            ev = pick_first_operator_segment(call.segments, lambda s: "cvv" in s.text.lower()) or pick_first_operator_segment(call.segments, lambda s: "kart" in s.text.lower())
            reason = "Operator müştərini təhlükəsiz kanala yönləndirmək əvəzinə riskli məlumat tələb edir."
        else:
            ev = pick_first_operator_segment(call.segments, lambda s: bool(_CALLBACK_REFUSAL_RE.search(s.text)))
            reason = "Operator müştəriyə sahib çıxmır (geri zəng/alternativ həll təqdim edilmir)."
        return CriterionResult(score=1, reasoning=reason, probability=_prob("HIGH"), evidence=format_evidence(ev) if ev else None)

    # Positive signals: ticket creation, clear action, alternatives
    ev = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(ticket|prioritet|texnik|alternativ|həll edək|indi edək|göndəririk)", s.text, re.I)))
    return CriterionResult(
        score=3,
        reasoning="Operator məsuliyyəti üzərinə götürür və aktiv addım təklif edir (həll/alternativ/texnik/ticket).",
        probability=_prob("HIGH" if ev else "LOW"),
        evidence=format_evidence(ev) if ev else None,
    )


def score_kr22_understanding(call: CallTranscript, internal_leak: Optional[Segment], pii_mishandled: bool, cb_fail: bool) -> CriterionResult:
    if internal_leak:
        ev = pick_first_operator_segment(call.segments, lambda s: True)
        return CriterionResult(
            score=1,
            reasoning="Operator problemi anlamaq və empatiya göstərmək əvəzinə daxili şikayət/negativ cavab verir.",
            probability=_prob("HIGH"),
            evidence=format_evidence(ev) if ev else None,
        )

    if cb_fail:
        # If operator at least explains they're checking -> 2 else 1
        checking = pick_first_operator_segment(call.segments, lambda s: bool(_CHECKING_RE.search(s.text)))
        if checking:
            return CriterionResult(
                score=2,
                reasoning="Operator problemi qismən qəbul edir, lakin düzgün gözlətmə idarəetməsi/aydın kommunikasiya zəifdir.",
                probability=_prob("HIGH"),
                evidence=format_evidence(checking),
            )
        weak = pick_first_operator_segment(call.segments, lambda s: True)
        return CriterionResult(
            score=1,
            reasoning="Operator problemi izah etmədən quru şəkildə gözləmə istəyir; aktiv dinləmə/empatiya yoxdur.",
            probability=_prob("HIGH"),
            evidence=format_evidence(weak) if weak else None,
        )

    if pii_mishandled:
        # If operator asks some relevant non-sensitive question (amount / card type) -> 2 else 1
        q = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(məbləğ|nə qədər|hansı kart|müqavilə|telefon|ünvan)", s.text, re.I)))
        if q:
            return CriterionResult(
                score=2,
                reasoning="Operator müəyyən məlumat toplayır, amma PII təhlükəsizliyi düzgün idarə edilmir.",
                probability=_prob("MEDIUM"),
                evidence=format_evidence(q),
            )
        weak = pick_first_operator_segment(call.segments, lambda s: True)
        return CriterionResult(
            score=1,
            reasoning="Operator problemin kontekstini düzgün soruşmur və təhlükəsiz prosesə keçmir.",
            probability=_prob("HIGH"),
            evidence=format_evidence(weak) if weak else None,
        )

    # Good: empathy OR clarifying questions OR clear status explanation
    ev = pick_first_operator_segment(call.segments, lambda s: bool(_EMPATHY_RE.search(s.text) or re.search(r"\?", s.text) or re.search(r"(təəssüf ki|sistem|problem)", s.text, re.I)))
    return CriterionResult(
        score=3,
        reasoning="Operator problemi anlayır, düzgün suallar verir və/və ya empatiya ilə kommunikasiya edir.",
        probability=_prob("HIGH" if ev else "LOW"),
        evidence=format_evidence(ev) if ev else None,
    )


def score_kr23_resolution(call: CallTranscript, internal_leak: Optional[Segment], pii_mishandled: bool, cb_fail: bool) -> CriterionResult:
    if internal_leak:
        ev = pick_first_operator_segment(call.segments, lambda s: True)
        return CriterionResult(
            score=1,
            reasoning="Həll təklif olunmur və ya qeyri-konstruktiv cavab verilir.",
            probability=_prob("HIGH"),
            evidence=format_evidence(ev) if ev else None,
        )

    if cb_fail:
        ev = pick_first_operator_segment(call.segments, lambda s: bool(_CALLBACK_REFUSAL_RE.search(s.text))) or pick_any_segment(call.segments, lambda s: "süküt" in s.text.lower())
        return CriterionResult(
            score=1,
            reasoning="Problemin həlli təqdim edilmir; uzun gözlətmə və nəticəsiz sonlanma var.",
            probability=_prob("HIGH"),
            evidence=format_evidence(ev) if ev else None,
        )

    if pii_mishandled:
        # Partial: payment done, but insecure
        ev = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(ödəniş|payment).*(edildi|oldu|tamamlandı)|uğurla", s.text, re.I)))
        return CriterionResult(
            score=2,
            reasoning="Nəticə bildirilib (məsələn, ödəniş tamamlandı), amma təhlükəsiz proses pozulduğu üçün həll keyfiyyəti tam deyil.",
            probability=_prob("HIGH" if ev else "MEDIUM"),
            evidence=format_evidence(ev) if ev else None,
        )

    # Good: actionable steps or clear information
    ev = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(restart|ayırın|qoşun|paket|manat|texnik|alternativ|tətbiq|yükləyin)", s.text, re.I)))
    return CriterionResult(
        score=3,
        reasoning="Operator müştəri üçün praktik və aydın həll təqdim edir (addım-addım və ya düzgün informasiya).",
        probability=_prob("HIGH" if ev else "LOW"),
        evidence=format_evidence(ev) if ev else None,
    )


def score_kr24_process_next_steps(call: CallTranscript, internal_leak: Optional[Segment], pii_mishandled: bool, cb_fail: bool) -> CriterionResult:
    if cb_fail:
        ev = pick_first_operator_segment(call.segments, lambda s: bool(_CALLBACK_REFUSAL_RE.search(s.text)))
        return CriterionResult(
            score=1,
            reasoning="Proses üzrə növbəti addımlar (qeydiyyat/ticket/geri zəng/transfer) təqdim edilmir.",
            probability=_prob("HIGH"),
            evidence=format_evidence(ev) if ev else None,
        )

    if internal_leak:
        ev = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(ticket|qeyd)", s.text, re.I))) or internal_leak
        return CriterionResult(
            score=2,
            reasoning="Bəzi proses addımları var (məsələn, ticket), amma ümumi yanaşma qeyri-peşəkardır.",
            probability=_prob("HIGH"),
            evidence=format_evidence(ev) if ev else None,
        )

    if pii_mishandled:
        ev = pick_first_operator_segment(call.segments, lambda s: bool(re.search(r"(ödəniş|qeydə alındı|sms|link)", s.text, re.I)))
        return CriterionResult(
            score=2,
            reasoning="Operator prosesin tamamlandığını bildirir, lakin PII təhlükəsiz prosesi pozulduğu üçün proses uyğunluğu qisməndir.",
            probability=_prob("MEDIUM"),
            evidence=format_evidence(ev) if ev else None,
        )

    ev = pick_first_operator_segment(call.segments, lambda s: bool(_NEXT_STEPS_RE.search(s.text)))
    return CriterionResult(
        score=3,
        reasoning="Operator növbəti addımları və gözləntiləri aydın bildirir (qeydiyyat, ticket, vaxt çərçivəsi, alternativ kanal).",
        probability=_prob("HIGH" if ev else "LOW"),
        evidence=format_evidence(ev) if ev else None,
    )


def evaluate_rule_based(call: CallTranscript) -> Dict[str, CriterionResult]:
    internal = detect_internal_leak(call)
    pii_mishandled, _seg = detect_pii_mishandling(call)
    cb_fail, _cb_seg = detect_callback_failure(call)

    return {
        "KR2.1": score_kr21_ownership(call, internal, pii_mishandled, cb_fail),
        "KR2.2": score_kr22_understanding(call, internal, pii_mishandled, cb_fail),
        "KR2.3": score_kr23_resolution(call, internal, pii_mishandled, cb_fail),
        "KR2.4": score_kr24_process_next_steps(call, internal, pii_mishandled, cb_fail),
        "KR2.5": score_kr25_professionalism(call, internal, pii_mishandled, cb_fail),
    }
