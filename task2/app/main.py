from __future__ import annotations

import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

from app.asr import ASRService

app = FastAPI(title="Turkish ASR API", version="1.0")

@app.get("/", include_in_schema=False)
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

_asr: ASRService | None = None


def get_asr() -> ASRService:
  global _asr
  if _asr is not None:
    return _asr

  backend = os.getenv("ASR_BACKEND", "onnx_int8").strip().lower()
  model_dir = os.getenv("ASR_MODEL_DIR", "models/onnx").strip()

  # fallback: if onnx not present, allow pytorch base model directory
  if not os.path.exists(model_dir):
    model_dir = os.getenv("ASR_FALLBACK_MODEL_DIR", "models/checkpoint").strip()

  _asr = ASRService(model_dir=model_dir, backend=backend)  # type: ignore[arg-type]
  return _asr


@app.get("/health")
def health():
  return {"status": "ok"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
  if not file.filename:
    raise HTTPException(status_code=400, detail="Missing filename")

  # accept wav/mp3
  ext = os.path.splitext(file.filename)[1].lower()
  if ext not in [".wav", ".mp3", ".m4a", ".flac", ".ogg"]:
    raise HTTPException(status_code=400, detail="Unsupported file type. Use WAV/MP3 (or similar).")

  asr = get_asr()

  with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
    data = await file.read()
    if not data:
      raise HTTPException(status_code=400, detail="Empty file")
    tmp.write(data)
    tmp_path = tmp.name

  try:
    result = asr.transcribe_file(tmp_path)
    return JSONResponse({"text": result.text, "inference_time": round(result.inference_time, 4)})
  finally:
    try:
      os.remove(tmp_path)
    except Exception:
      pass
