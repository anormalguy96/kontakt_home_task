from __future__ import annotations

from kontakt_qc.pipeline import evaluate_call


def test_evidence_masks_card_digits():
    payload = {
        "call_id": "PII",
        "segments": [
            {"speaker": "Customer", "text": "Kart nömrəm 4111 1111 1111 1111", "start_time": 0.0, "end_time": 2.0},
            {"speaker": "Operator", "text": "CVV-ni də deyin", "start_time": 2.0, "end_time": 4.0},
        ],
    }
    out = evaluate_call(payload)["PII"]
    ev = out["KR2.5"].get("evidence_snippet", "")
    assert "4111 1111 1111 1111" not in ev
