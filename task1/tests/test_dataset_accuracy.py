import json
from pathlib import Path

from qc_service.preprocess import normalize_transcript
from qc_service.config import load_settings
from qc_service.evaluator import evaluate_transcript


def _dataset_path() -> Path:
  root = Path(__file__).resolve().parents[1]
  return root / "data" / "Task_1_Eval_dataset.json"


def test_rule_based_accuracy_reasonable():
  ds = json.loads(_dataset_path().read_text(encoding="utf-8"))
  settings = load_settings()
  correct = 0
  total = 0

  for item in ds:
    transcript = normalize_transcript(item["input"])
    out = evaluate_transcript(transcript, settings).results
    exp = item["expected_output"]
    for k in ["KR2.1","KR2.2","KR2.3","KR2.4","KR2.5"]:
      total += 1
      if out[k].score == exp[k]["score"]:
        correct += 1

  acc = correct / max(1,total)
  assert acc >= 0.95
