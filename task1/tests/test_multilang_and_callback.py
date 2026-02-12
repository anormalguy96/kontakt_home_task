from __future__ import annotations

from kontakt_qc.types import CallTranscript, Segment
from kontakt_qc.rules import detect_callback_failure


def test_callback_failure_russian_keyword():
    call = CallTranscript(
        call_id="RU1",
        segments=(
            Segment("Customer", "[140 saniyə süküt]", 0.0, 140.0),
            Segment("Operator", "Мы не перезваниваем, завтра снова позвоните.", 140.0, 145.0),
        ),
    )
    fail, seg = detect_callback_failure(call)
    assert fail is True
    assert seg is not None
