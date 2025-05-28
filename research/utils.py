from pathlib import Path

ROOT = Path(__file__).parent.parent

class PathStorage:
    raw_root = ROOT / "data/raw"
    processed_root = ROOT / "data/processed"
