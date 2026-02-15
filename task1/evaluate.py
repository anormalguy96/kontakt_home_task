from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# bu hissənin nə işə yaradığını belə izah etmək olar:
# əgər biz bu proqramı pip install -e . etmədən birbaşa işə salırıqsa, o zaman proqram "src" qovluğunu tapıb içindəki modulları (config, preprocess, evaluator) import edə bilmir.
# ROOT = Path(__file__).resolve().parent — bu hissə proqramın yerləşdiyi qovluğu tapır.
# sys.path.insert(0, str(ROOT / "src")) — bu hissə tapdığımız "src" qovluğunu Python-un import yollarına əlavə edir ki, proqram "src" içindəki faylları (məsələn, qc_service.config) rahatlıqla import edə bilsin.
# əgər proqramı pip install -e . ilə quraşdırsaq, bu hissəyə ehtiyac qalmır, çünki quraşdırma zamanı Python import yolları avtomatik düzəldilir.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from qc_service.config import load_settings
from qc_service.preprocess import normalize_transcript
from qc_service.evaluator import evaluate_transcript


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default="data/Task_1_Eval_dataset.json")
  parser.add_argument("--use-llm", action="store_true", help="USE_LLM=1 ilə eynidir, amma yalnız bu run üçün")

  # debugging üçün nəzərdə tutulan arqumentlər
  # production-ready versiyada bu arqumentlər silinə də bilər
  parser.add_argument("--debug", action="store_true", help="Uyğunsuz (mismatch) nümunələri çap et")
  parser.add_argument("--debug-kr", default="", help="Məs: KR2.5 — yalnız seçilmiş KR üçün mismatch çap et")
  parser.add_argument("--max-mismatches", type=int, default=50, help="Maksimum mismatch sayı (debug üçün)")
  args = parser.parse_args()

  load_dotenv()
  if args.use_llm:
    os.environ["USE_LLM"] = "1"

  settings = load_settings()
  ds = json.loads((ROOT / args.dataset).read_text(encoding="utf-8"))

  metrics = ["KR2.1", "KR2.2", "KR2.3", "KR2.4", "KR2.5"]
  per_metric = {k: {"total": 0, "correct": 0} for k in metrics}

  total = 0
  correct = 0
  mismatches_printed = 0

  for item in ds:
    call_id = item.get("input", {}).get("call_id") or item.get("call_id") or "UNKNOWN_CALL_ID"
    transcript = normalize_transcript(item["input"])
    res = evaluate_transcript(transcript, settings).results
    exp = item["expected_output"]

    for k in metrics:
      per_metric[k]["total"] += 1
      total += 1

      pred_score = res[k].score
      exp_score = exp[k]["score"]

      if pred_score == exp_score:
        per_metric[k]["correct"] += 1
        correct += 1
        continue

      # mismatching case ləri üçün
      if args.debug:
        if args.debug_kr and k != args.debug_kr:
          continue
        if mismatches_printed >= args.max_mismatches:
          continue

        prob = getattr(res[k], "probability", None)
        ev = getattr(res[k], "evidence_snippet", None)
        reason = getattr(res[k], "reasoning", None)

        print("\n--- MISMATCH ---")
        print("call_id:", call_id)
        print("metric :", k)
        print("expected:", exp_score, "| predicted:", pred_score)
        if prob is not None:
          print("probability:", prob)
        if ev:
          print("evidence:", ev)
        if reason:
          print("reasoning:", reason)

        mismatches_printed += 1

  print("\nOverall accuracy:", round(correct / max(1, total), 4), f"({correct}/{total})")
  for k, v in per_metric.items():
    acc = v["correct"] / max(1, v["total"])
    print(f"{k}: {acc:.4f} ({v['correct']}/{v['total']})")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())