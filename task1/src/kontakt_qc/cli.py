from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import configure_logging, evaluate_call


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kontakt Home call quality scoring prototype (Task 1).")
    p.add_argument("--input", "-i", required=True, help="Path to input transcript JSON file.")
    p.add_argument("--output", "-o", required=False, help="Path to write output JSON. Default: stdout.")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG/INFO/WARN/ERROR).")
    p.add_argument("--mode", default=None, choices=["rule", "hybrid", "llm"], help="Override KONTAKT_QC_MODE.")
    args = p.parse_args(argv)

    configure_logging(args.log_level)

    if args.mode:
        os.environ["KONTAKT_QC_MODE"] = args.mode

    in_path = Path(args.input)
    payload = json.loads(in_path.read_text(encoding="utf-8"))

    result = evaluate_call(payload)
    out_text = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(out_text, encoding="utf-8")
    else:
        sys.stdout.write(out_text + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
