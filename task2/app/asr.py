from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
import librosa
import onnxruntime as ort
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

Backend = Literal["pytorch", "onnx", "onnx_int8"]


@dataclass
class TranscribeResult:
  text: str
  inference_time: float


def _load_audio_to_16k_mono(path: str) -> np.ndarray:
  # librosa can read wav/mp3 (mp3 typically needs ffmpeg in the system)
  audio, sr = librosa.load(path, sr=16_000, mono=True)
  # ensure float32
  if audio.dtype != np.float32:
    audio = audio.astype(np.float32)
  return audio


class ASRService:
  """
  Minimal ASR service with interchangeable backends:
  - PyTorch (Wav2Vec2ForCTC)
  - ONNXRuntime (float)
  - ONNXRuntime (int8 quantized)
  """

  def __init__(
    self,
    model_dir: str,
    backend: Backend = "onnx_int8",
    device: Optional[str] = None,
  ) -> None:
    self.model_dir = model_dir
    self.backend: Backend = backend
    self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Processor is needed for feature extraction & decoding.
    # We keep processor in the same checkpoint folder OR in onnx folder (export script copies config).
    self.processor = Wav2Vec2Processor.from_pretrained(model_dir)

    self._pt_model: Optional[Wav2Vec2ForCTC] = None
    self._ort_session: Optional[ort.InferenceSession] = None

    if backend == "pytorch":
      self._pt_model = Wav2Vec2ForCTC.from_pretrained(model_dir).to(self.device)
      self._pt_model.eval()
    else:
      onnx_name = "model_int8.onnx" if backend == "onnx_int8" else "model.onnx"
      onnx_path = os.path.join(model_dir, onnx_name)
      if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

      # Providers
      providers = ["CPUExecutionProvider"]
      if ort.get_device().lower() == "gpu":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

      self._ort_session = ort.InferenceSession(onnx_path, providers=providers)
      self._ort_input_names = {i.name for i in self._ort_session.get_inputs()}

  def transcribe_file(self, audio_path: str) -> TranscribeResult:
    audio = _load_audio_to_16k_mono(audio_path)

    # Feature extraction
    inputs = self.processor(audio, sampling_rate=16_000, return_tensors="pt", padding=True)
    input_values = inputs.input_values  # [1, T]
    attention_mask = getattr(inputs, "attention_mask", None)

    start = time.perf_counter()
    if self.backend == "pytorch":
      assert self._pt_model is not None
      with torch.no_grad():
        input_values_dev = input_values.to(self.device)
        attn_dev = attention_mask.to(self.device) if attention_mask is not None else None
        out = self._pt_model(input_values_dev, attention_mask=attn_dev)
        logits = out.logits.detach().cpu().numpy()
    else:
      assert self._ort_session is not None
      ort_inputs = {"input_values": np.ascontiguousarray(input_values.numpy())}

      # Only send attention_mask if the model actually expects it
      if attention_mask is not None and "attention_mask" in self._ort_input_names:
        ort_inputs["attention_mask"] = np.ascontiguousarray(attention_mask.numpy())

      ort_outs = self._ort_session.run(None, ort_inputs)
      # Usually first output is logits
      logits = ort_outs[0]
    elapsed = time.perf_counter() - start

    pred_ids = np.argmax(logits, axis=-1)
    text = self.processor.batch_decode(pred_ids)[0]

    return TranscribeResult(text=text, inference_time=float(elapsed))