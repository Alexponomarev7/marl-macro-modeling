from pathlib import Path

class PathStorage:
    def __init__(self, data_folder: str = "data"):
        self.root = Path(__file__).parent.parent
        self.raw_root = self.root / data_folder / "raw"
        self.processed_root = self.root / data_folder / "interim"
        self.dynare_configs_root = self.root / "dynare/docker/dynare_models"
