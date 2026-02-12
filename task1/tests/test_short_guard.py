from __future__ import annotations

from kontakt_qc.pipeline import evaluate_call


def test_audio_too_short_guard():
    payload = {
        "call_id": "SHORT",
        "segments": [
            {"speaker": "Operator", "text": "Salam", "start_time": 0.0, "end_time": 0.05},
            {"speaker": "Customer", "text": "Alo", "start_time": 0.05, "end_time": 0.09},
        ],
    }
    out = evaluate_call(payload)["SHORT"]
    assert all(v["score"] == 0 for v in out.values())
