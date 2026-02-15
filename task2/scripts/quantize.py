from __future__ import annotations

import argparse
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType


def sizeof_mb(path: Path) -> float:
  return round(path.stat().st_size / (1024 * 1024), 2)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--onnx_path", required=True, help="Path to ONNX model")
  p.add_argument("--output_path", required=True, help="Path for quantized output")
  args = p.parse_args()

  onnx_path = Path(args.onnx_path)
  output_path = Path(args.output_path)

  if not onnx_path.exists():
    raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

  print(f"[quantize] Input: {onnx_path} ({sizeof_mb(onnx_path)} MB)")
  print(f"[quantize] Quantizing to INT8...")

  try:
    quantize_dynamic(
      model_input=str(onnx_path),
      model_output=str(output_path),
      weight_type=QuantType.QInt8,
    )
    print(f"[quantize] Output: {output_path} ({sizeof_mb(output_path)} MB)")
    print("[quantize] Done.")
  except Exception as e:
    print(f"[quantize] Warning: Quantization failed: {e}")
    print("[quantize] The float32 model is still available for use.")


if __name__ == "__main__":
  main()
