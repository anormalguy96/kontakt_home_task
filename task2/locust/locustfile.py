from locust import HttpUser, task, between
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "samples" / "sample.wav"

class ASRUser(HttpUser):
  wait_time = between(0.5, 1.5)

  @task
  def transcribe(self):
    if not SAMPLE.exists():
      # show in Locust exceptions if sample missing
      raise FileNotFoundError(f"Missing sample file: {SAMPLE}")

    with SAMPLE.open("rb") as f:
      files = {"file": (SAMPLE.name, f, "audio/wav")}
      self.client.post("/transcribe", files=files, timeout=60)