from __future__ import annotations

import argparse, json, logging, os, sys
from pathlib import Path
from typing import Dict, Any

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from kontakt_qc.pipeline import evaluate_call  # noqa: E402


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    logger = logging.getLogger("evaluate")

    p = argparse.ArgumentParser(description="Evaluate scoring accuracy on provided dataset.")
    p.add_argument("--mode", choices=["rule", "hybrid", "llm"], default="rule", help="Scoring mode.")
    p.add_argument(
        "--dataset",
        default=str(Path(__file__).parent / "docs" / "Task_1_Eval_dataset.json"),
        help="Path to evaluation dataset JSON.",
    )
    p.add_argument(
        "--out",
        default="",
        help="Optional path to write results JSON (in addition to stdout).",
    )
    args = p.parse_args(argv)

    os.environ["KONTAKT_QC_MODE"] = args.mode

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        fallback = Path("docs/Task_1_Eval_dataset.json")
        if fallback.exists():
            dataset_path = fallback

    if not dataset_path.exists():
        logger.error("Dataset file not found: %s", dataset_path)
        return 2

    try:
        data = json.loads(dataset_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to read/parse dataset JSON: %s", dataset_path)
        return 2

    total: int = 0
    correct: int = 0
    per_kr_total: Dict[str, int] = {}
    per_kr_correct: Dict[str, int] = {}

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Skipping non-object dataset item at index=%d", idx)
            continue

        payload = item.get("input")
        expected = item.get("expected_output")

        if not isinstance(payload, dict) or not isinstance(expected, dict):
            logger.warning("Skipping invalid dataset item at index=%d (missing input/expected_output)", idx)
            continue

        predicted_wrapped = evaluate_call(payload)
        call_id = payload.get("call_id", "UNKNOWN_CALL")
        predicted = predicted_wrapped.get(call_id, {})

        for kr, exp in expected.items():
            if not isinstance(exp, dict):
                logger.warning("Skipping expected_output[%s] at index=%d (not an object)", kr, idx)
                continue

            total += 1
            per_kr_total[kr] = per_kr_total.get(kr, 0) + 1

            pred_score_raw = predicted.get(kr, {}).get("score", -999)
            exp_score_raw = exp.get("score", -999)

            try:
                pred_score = int(pred_score_raw)
                exp_score = int(exp_score_raw)
            except Exception:
                logger.warning(
                    "Non-integer score at index=%d kr=%s pred=%r exp=%r",
                    idx,
                    kr,
                    pred_score_raw,
                    exp_score_raw,
                )
                continue

            if pred_score == exp_score:
                correct += 1
                per_kr_correct[kr] = per_kr_correct.get(kr, 0) + 1

    overall_acc: float = float(correct) / float(total) if total > 0 else 0.0

    results: Dict[str, Any] = {
        "mode": args.mode,
        "dataset": str(dataset_path),
        "overall": {"correct": correct, "total": total, "accuracy": round(overall_acc, 6)},
        "per_kr": {},
    }

    for kr in sorted(per_kr_total.keys()):
        c = per_kr_correct.get(kr, 0)
        t = per_kr_total[kr]
        kr_acc: float = float(c) / float(t) if t > 0 else 0.0
        results["per_kr"][kr] = {"correct": c, "total": t, "accuracy": round(kr_acc, 6)}

    sys.stdout.write(json.dumps(results, ensure_ascii=False) + "\n")

    if args.out:
        try:
            Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            logger.info("Wrote results to %s", args.out)
        except Exception:
            logger.exception("Failed to write results to %s", args.out)
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
