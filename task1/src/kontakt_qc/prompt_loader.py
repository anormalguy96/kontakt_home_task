from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml

def load_prompt_bundle() -> Dict[str, str]:
    """Loads prompts from YAML so prompt management is not embedded in code."""
    path = Path(__file__).resolve().parents[2] / "prompts" / "kr_scoring_system_prompt.yaml"
    if not path.exists():
        # Fallback if running from a different context or if directory is misassigned
        alt_path = Path("prompts/kr_scoring_system_prompt.yaml")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Prompt bundle not found at {path} or {alt_path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    system = str(data.get("system", "")).strip()
    user_template = str(data.get("user_template", "")).strip()
    if not user_template:
        # Safe default
        user_template = (
            "CRITERION: {criterion}\n"
            "Return ONLY JSON with keys: score (0-3), reasoning, evidence, probability (HIGH|MEDIUM|LOW).\n"
            "TRANSCRIPT:\n{transcript}\n"
        )
    return {"system": system, "user_template": user_template}
