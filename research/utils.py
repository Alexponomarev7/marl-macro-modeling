from pathlib import Path

ROOT = Path(__file__).parent.parent

class PathStorage:
    def __init__(self, data_folder: str = "data"):
        self.raw_root = ROOT / data_folder / "raw"
        self.processed_root = ROOT / data_folder / "processed"
        self.dynare_configs_root = ROOT / "dynare/docker/dynare_models"
