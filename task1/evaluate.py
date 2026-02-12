from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

# Ensure src is in path for easier direct execution
sys.path.append(str(Path(__file__).resolve().parent / "src"))

from kontakt_qc.pipeline import evaluate_call


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate scoring accuracy on provided dataset.")
    p.add_argument("--mode", choices=["rule", "hybrid", "llm"], default="rule", help="Scoring mode.")
    args = p.parse_args(argv)

    os.environ["KONTAKT_QC_MODE"] = args.mode

    dataset_path = Path(__file__).parent / "docs" / "Task_1_Eval_dataset.json"
    if not dataset_path.exists():
        dataset_path = Path("docs/Task_1_Eval_dataset.json")
    
    data = json.loads(dataset_path.read_text(encoding="utf-8"))

    total: int = 0
    correct: int = 0
    per_kr_total: Dict[str, int] = {}
    per_kr_correct: Dict[str, int] = {}

    for item in data:
        payload = item["input"]
        expected = item["expected_output"]

        predicted_wrapped = evaluate_call(payload)  # {call_id: {KR: {...}}}
        call_id = payload.get("call_id", "UNKNOWN_CALL")
        predicted = predicted_wrapped.get(call_id, {})

        for kr, exp in expected.items():
            total += 1
            per_kr_total[kr] = per_kr_total.get(kr, 0) + 1
            pred_score = int(predicted.get(kr, {}).get("score", -999))
            if pred_score == int(exp["score"]):
                correct += 1
                per_kr_correct[kr] = per_kr_correct.get(kr, 0) + 1

    print(f"Mode: {args.mode}")
    # Explicitly casting to float to avoid linter confusion with division operator
    overall_acc: float = float(correct) / float(total) if total > 0 else 0.0
    print(f"Overall accuracy: {correct}/{total} = {overall_acc:.3f}")
    for kr in sorted(per_kr_total.keys()):
        c: int = per_kr_correct.get(kr, 0)
        t: int = per_kr_total[kr]
        kr_acc: float = float(c) / float(t) if t > 0 else 0.0
        print(f"{kr}: {c}/{t} = {kr_acc:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
