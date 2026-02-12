from __future__ import annotations

from kontakt_qc.pipeline import evaluate_call


def test_handles_non_list_segments():
    payload = {"call_id": "X", "segments": "not-a-list"}
    out = evaluate_call(payload)
    assert "X" in out
    assert set(out["X"].keys()) == {"KR2.1", "KR2.2", "KR2.3", "KR2.4", "KR2.5"}


def test_handles_bad_segment_items():
    payload = {
        "call_id": "Y",
        "segments": [
            None,
            123,
            "oops",
            {"speaker": "Operator", "text": "Salam", "start_time": 0, "end_time": 0.05},
        ],
    }
    out = evaluate_call(payload)
    assert "Y" in out
