from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import librosa
import onnxruntime as ort
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def load_audio(path: str) -> np.ndarray:
  audio, _ = librosa.load(path, sr=16_000, mono=True)
  return audio.astype(np.float32)


def dir_size_mb(path: Path) -> float:
  total = 0
  for p in path.rglob("*"):
    if p.is_file():
      total += p.stat().st_size
  return round(total / (1024 * 1024), 2)


def file_size_mb(path: Path) -> float:
  return round(path.stat().st_size / (1024 * 1024), 2)


def run_pytorch(model_dir: str, audio: np.ndarray, runs: int) -> float:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  processor = Wav2Vec2Processor.from_pretrained(model_dir)
  model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(device)
  model.eval()

  inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
  input_values = inputs.input_values.to(device)
  attn = getattr(inputs, "attention_mask", None)
  if attn is not None:
    attn = attn.to(device)

  # warmup
  with torch.no_grad():
    _ = model(input_values, attention_mask=attn)

  times = []
  for _ in range(runs):
    start = time.perf_counter()
    with torch.no_grad():
      out = model(input_values, attention_mask=attn)
      _ = out.logits
    times.append(time.perf_counter() - start)

  return float(np.mean(times))


def run_onnx(model_dir: str, onnx_path: str, audio: np.ndarray, runs: int) -> float:
  processor = Wav2Vec2Processor.from_pretrained(model_dir)
  inputs = processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
  input_values = inputs.input_values.numpy()
  attention_mask = getattr(inputs, "attention_mask", None)
  attention_mask_np = attention_mask.numpy() if attention_mask is not None else None

  sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

  # warmup
  ort_inputs = {"input_values": input_values}
  if attention_mask_np is not None and any(i.name == "attention_mask" for i in sess.get_inputs()):
    ort_inputs["attention_mask"] = attention_mask_np
  _ = sess.run(None, ort_inputs)

  times = []
  for _ in range(runs):
    start = time.perf_counter()
    _ = sess.run(None, ort_inputs)
    times.append(time.perf_counter() - start)

  return float(np.mean(times))


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--checkpoint_dir", required=True)
  p.add_argument("--onnx_path", required=True)
  p.add_argument("--onnx_int8_path", required=True)
  p.add_argument("--audio_path", required=True)
  p.add_argument("--runs", type=int, default=20)
  p.add_argument("--out", default="artifacts/benchmark_report.md")
  args = p.parse_args()

  ckpt = Path(args.checkpoint_dir)
  onnx = Path(args.onnx_path)
  int8 = Path(args.onnx_int8_path)
  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)

  audio = load_audio(args.audio_path)

  pt_size = dir_size_mb(ckpt)
  onnx_size = file_size_mb(onnx)
  int8_size = file_size_mb(int8)

  pt_time = run_pytorch(ckpt.as_posix(), audio, args.runs)
  onnx_time = run_onnx(ckpt.as_posix(), onnx.as_posix(), audio, args.runs)
  int8_time = run_onnx(ckpt.as_posix(), int8.as_posix(), audio, args.runs)

  md = []
  md.append("# Benchmark Report: PyTorch vs ONNX\n")
  md.append(f"- Audio: `{args.audio_path}`\n")
  md.append(f"- Runs: `{args.runs}` (average latency)\n\n")
  md.append("## Model ölçüsü (MB)\n")
  md.append(f"- PyTorch checkpoint folder: **{pt_size} MB**\n")
  md.append(f"- ONNX (float): **{onnx_size} MB**\n")
  md.append(f"- ONNX (int8): **{int8_size} MB**\n\n")
  md.append("## Inference time (seconds, average)\n")
  md.append(f"- PyTorch: **{pt_time:.4f} s**\n")
  md.append(f"- ONNX (float): **{onnx_time:.4f} s**\n")
  md.append(f"- ONNX (int8): **{int8_time:.4f} s**\n")

  out.write_text("".join(md), encoding="utf-8")
  print(f"[benchmark] wrote: {out.as_posix()}")


if __name__ == "__main__":
  main()
