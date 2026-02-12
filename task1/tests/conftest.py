import sys
from pathlib import Path

# Add src to sys.path so that tests can import the package without pip install -e .
# This matches the behavior of evaluate.py and makes the repo more robust for quick cloning.
src_path = str(Path(__file__).resolve().parents[1] / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
