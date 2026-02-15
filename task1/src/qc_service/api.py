from __future__ import annotations

from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import load_settings
from .evaluator import evaluate_transcript
from .logging_setup import setup_logging
from .preprocess import normalize_transcript

load_dotenv()
settings = load_settings()
setup_logging(settings.log_level)

app = FastAPI(title="Kontakt Home Task 1 - QC Prototype", version="1.0.0")


@app.get("/health")
def health() -> dict:
  return {"ok": True}


class EvaluateResponse(BaseModel):
  dataset_id: Optional[str] = None
  call_id: str
  results: Dict[str, Any]


def _unwrap_payload(payload: dict) -> tuple[dict, Optional[str]]:
  """
  Hər iki formatı qəbul edir:
  1) düz: { "call_id": "...", "segments": [...] }
  2) dataset: { "dataset_id": "...", "input": { "call_id": "...", "segments": [...] }, ... }
  """
  dataset_id = payload.get("dataset_id")
  inner = payload.get("input")
  if isinstance(inner, dict):
    return inner, dataset_id
  return payload, dataset_id


@app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(payload: dict) -> EvaluateResponse:
  payload_inner, dataset_id = _unwrap_payload(payload)

  try:
    transcript = normalize_transcript(payload_inner)
    result = evaluate_transcript(transcript, settings)
    out = result.model_dump()

    # dataset_id verilibsə echo et
    if dataset_id is not None:
      out["dataset_id"] = dataset_id

    return EvaluateResponse(**out)

  except ValueError as e:
    # "bad input" tipli səhvlər üçün 400 error daha uyğundur
    raise HTTPException(status_code=400, detail=str(e)) from e
  except KeyError as e:
    raise HTTPException(status_code=400, detail=f"Missing field: {e}") from e
  except Exception as e:
    # gözlənilməyən runtime error üçün 500 error versin
    raise HTTPException(status_code=500, detail="Internal server error") from e