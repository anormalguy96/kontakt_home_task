from __future__ import annotations

import pathlib
import yaml


def load_prompt_yaml(path: str) -> dict:
  p = pathlib.Path(path)
  data = yaml.safe_load(p.read_text(encoding="utf-8"))
  if not isinstance(data, dict):
    raise ValueError("Prompt yaml must be a dict with system/user keys")
  if "system" not in data or "user" not in data:
    raise ValueError("Prompt yaml must contain 'system' and 'user'")
  return data
