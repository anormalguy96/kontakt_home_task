import argparse
from pathlib import Path

from optimum.exporters.onnx import main_export
from transformers import AutoTokenizer


def export_one(model_dir: str, out_dir: str, task: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Fast + compatible ONNX export
    main_export(
        model_name_or_path=model_dir,
        output=out,
        task=task,
        device="cpu",
        framework="pt",
        library_name="transformers",
        monolith=True,
        do_validation=False,  # keep it fast
    )

    # Save tokenizer next to ONNX (used by benchmark/inference)
    AutoTokenizer.from_pretrained(model_dir, use_fast=True).save_pretrained(str(out))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier_pt", default="models/classifier/pytorch")
    ap.add_argument("--ner_pt", default="models/ner/pytorch")
    ap.add_argument("--classifier_out", default="models/classifier/onnx")
    ap.add_argument("--ner_out", default="models/ner/onnx")
    args = ap.parse_args()

    export_one(args.classifier_pt, args.classifier_out, "text-classification")
    export_one(args.ner_pt, args.ner_out, "token-classification")
    print("✅ ONNX export finished.")


if __name__ == "__main__":
    main()
