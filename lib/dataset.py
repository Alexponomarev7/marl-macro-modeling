import json
import torch
import pandas as pd
import numpy as np

from pathlib import Path

class Dataset:
    def __init__(self, data_path: Path):
        metadata_path = data_path / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        data = pd.read_parquet(self.metadata[idx]["output_dir"])
        
        states = []
        actions = []
        for _, row in data.iterrows():
            states.append(row["state"]["Capital"])
            actions.append(row["info"]["consumption"])
        return torch.from_numpy(np.concatenate(states, dtype=np.float64)).reshape(-1, 1), torch.from_numpy(np.array(actions, dtype=np.float64)).reshape(-1, 1), torch.LongTensor([0])
