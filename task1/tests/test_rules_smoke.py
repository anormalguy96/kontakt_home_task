
from __future__ import annotations

import json
from pathlib import Path

from kontakt_qc.pipeline import evaluate_call


def test_smoke_eval_dataset_subset():
    data = json.loads(Path("docs/Task_1_Eval_dataset.json").read_text(encoding="utf-8"))
    # A tiny subset for fast CI
    subset_ids = {"001_ideal_sales", "003_internal_leak_critical", "005_pii_leak_ignored", "105_callback_fail_long_hold_1"}
    subset = [x for x in data if x["dataset_id"] in subset_ids]
    assert len(subset) == 4

    for item in subset:
        call_id = item["input"]["call_id"]
        out = evaluate_call(item["input"])[call_id]
        for kr, exp in item["expected_output"].items():
            assert out[kr]["score"] == exp["score"]
