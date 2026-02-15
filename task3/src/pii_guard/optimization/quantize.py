import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def q_one(in_path: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(in_path),
        model_output=str(out_path),
        weight_type=QuantType.QInt8,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier_onnx", default="models/classifier/onnx/model.onnx")
    ap.add_argument("--ner_onnx", default="models/ner/onnx/model.onnx")
    ap.add_argument("--classifier_out", default="models/classifier/onnx/model.int8.onnx")
    ap.add_argument("--ner_out", default="models/ner/onnx/model.int8.onnx")
    args = ap.parse_args()

    q_one(Path(args.classifier_onnx), Path(args.classifier_out))
    q_one(Path(args.ner_onnx), Path(args.ner_out))

    print("Quantized:")
    print(" -", args.classifier_out)
    print(" -", args.ner_out)


if __name__ == "__main__":
    main()