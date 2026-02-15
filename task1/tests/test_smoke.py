import json
from pathlib import Path

from qc_service.preprocess import normalize_transcript
from qc_service.config import load_settings
from qc_service.evaluator import evaluate_transcript


def _dataset_path() -> Path:
  root = Path(__file__).resolve().parents[1]
  return root / "data" / "Task_1_Eval_dataset.json"


def test_smoke_single_example():
  ds = json.loads(_dataset_path().read_text(encoding="utf-8"))
  first = ds[0]
  transcript = normalize_transcript(first["input"])
  settings = load_settings()
  result = evaluate_transcript(transcript, settings)
  assert result.call_id == first["input"]["call_id"]
  assert set(result.results.keys()) == {"KR2.1","KR2.2","KR2.3","KR2.4","KR2.5"}
  for m in result.results.values():
    assert 0 <= m.score <= 3
