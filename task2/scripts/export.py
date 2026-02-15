from __future__ import annotations

import argparse
from pathlib import Path
import inspect

from transformers import Wav2Vec2Processor
from onnxruntime.quantization import QuantType, quantize_dynamic

from optimum.onnxruntime import ORTModelForCTC


def sizeof_mb(path: Path) -> float:
  return round(path.stat().st_size / (1024 * 1024), 2)


def ensure_model_onnx_name(out_dir: Path) -> Path:
  """
  Optimum exports an ONNX file into the output dir, but the name can vary.
  Normalize it to model.onnx for the rest of the pipeline.
  """
  onnx_path = out_dir / "model.onnx"
  if onnx_path.exists():
    return onnx_path

  candidates = sorted(out_dir.glob("*.onnx"))
  if not candidates:
    # Sometimes Optimum places it inside a subfolder; search recursively
    candidates = sorted(out_dir.rglob("*.onnx"))
  if not candidates:
    raise FileNotFoundError(f"No .onnx file produced in: {out_dir}")

  # Move/rename first candidate to model.onnx at root
  candidates[0].replace(onnx_path)
  return onnx_path


def export_with_optimum(model_id_or_path: str, out_dir: Path) -> None:
  """
  Export model to ONNX using Optimum, supporting multiple Optimum versions.
  """
  sig = inspect.signature(ORTModelForCTC.from_pretrained)
  kwargs = {"export": True}

  # Older Optimum used from_transformers=True
  if "from_transformers" in sig.parameters:
    kwargs["from_transformers"] = True

  ort_model = ORTModelForCTC.from_pretrained(model_id_or_path, **kwargs)
  ort_model.save_pretrained(out_dir.as_posix())


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument(
    "--checkpoint_dir",
    required=True,
    help="HF repo id (e.g. cahya/wav2vec2-base-turkish) OR local folder path",
  )
  p.add_argument("--onnx_dir", required=True, help="Output folder for ONNX artifacts")
  args = p.parse_args()

  ckpt_id = args.checkpoint_dir
  out_dir = Path(args.onnx_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  print(f"[export] checkpoint_dir={ckpt_id}")
  print(f"[export] onnx_dir={out_dir}")

  # 1) Export to ONNX
  print("[export] exporting to ONNX via Optimum...")
  export_with_optimum(ckpt_id, out_dir)

  # 2) Save processor next to ONNX (API loads from this folder)
  processor = Wav2Vec2Processor.from_pretrained(ckpt_id)
  processor.save_pretrained(out_dir.as_posix())

  # 3) Normalize ONNX name
  onnx_path = ensure_model_onnx_name(out_dir)
  print(f"[export] ONNX saved: {onnx_path} ({sizeof_mb(onnx_path)} MB)")

  # 4) Dynamic INT8 quantization
  int8_path = out_dir / "model_int8.onnx"
  
  quantize_dynamic(
    model_input=onnx_path.as_posix(),
    model_output=int8_path.as_posix(),
    weight_type=QuantType.QInt8,
    op_types_to_quantize=["MatMul", "Gemm"]
  )
  print(f"[export] INT8 saved: {int8_path} ({sizeof_mb(int8_path)} MB)")
  print("[export] Done.")


if __name__ == "__main__":
  main()