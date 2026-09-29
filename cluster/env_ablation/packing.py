"""One env's episodes packed into a few memory-mapped arrays, so that training reads them from the page
cache (shared by every loader worker and run) instead of opening one parquet per sample on NFS.

Layout of a packed directory: <field>.npy for each of FIELDS (all episodes' rows concatenated, padded to
the env's widest episode), offsets.npy [E + 1], widths.npy [E, 3] (state, action, endogenous) and
episodes.json (source parquet, env group and info of each episode).
"""
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pyarrow.parquet as pq
from loguru import logger

from lib.dataset import EconomicsDataset

FIELDS = ("state", "action", "executed", "endogenous", "reward")
WIDTH_OF = {"state": 0, "action": 1, "executed": 1, "endogenous": 2}
INFO_KEYS = ("model_params", "state_description", "action_description", "endogenous_description")


def pack_episodes(files: list[Path], out_dir: Path) -> int:
    """Packs the non-empty episode parquets among `files` into out_dir (replaced atomically); returns
    the number of episodes packed."""
    episodes, arrays = [], {k: [] for k in FIELDS}
    for f in sorted(files):
        try:
            if pq.ParquetFile(f).metadata.num_rows == 0:
                continue
            data, info = EconomicsDataset.read_episode(f)
        except Exception as e:  # a bad draw must not block the others
            logger.warning(f"skipping unreadable episode {f}: {e}")
            continue
        episodes.append({
            "file": f.name, "path": str(f), "env_group": info.get("env_group") or f.name.rsplit("_config_", 1)[0],
            "info": {k: info.get(k) for k in INFO_KEYS},
        })
        for k in FIELDS:
            arrays[k].append(data[k])

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    lengths = [len(a) for a in arrays["state"]]
    np.save(tmp / "offsets.npy", np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    widths = np.array([[s.shape[1], a.shape[1], e.shape[1]] for s, a, e in
                       zip(arrays["state"], arrays["action"], arrays["endogenous"])], dtype=np.int64).reshape(-1, 3)
    np.save(tmp / "widths.npy", widths)
    for k in FIELDS:
        width = int(widths[:, WIDTH_OF[k]].max(initial=0)) if k in WIDTH_OF else 1
        packed = np.zeros((sum(lengths), width), dtype=np.float32)
        for start, a in zip(np.cumsum([0] + lengths[:-1]), arrays[k]):
            packed[start:start + len(a), :a.shape[1]] = a
        np.save(tmp / f"{k}.npy", packed)
    (tmp / "episodes.json").write_text(json.dumps(episodes))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.replace(tmp, out_dir)
    return len(episodes)


def index_entries(packed_dir: Path) -> list[dict]:
    """EconomicsDataset metadata entries of a packed directory."""
    episodes = json.loads((packed_dir / "episodes.json").read_text())
    return [
        {"env_name": e["file"], "env_group": e["env_group"], "output_dir": e["path"],
         "packed": [str(packed_dir), i]}
        for i, e in enumerate(episodes)
    ]


class _Store:
    def __init__(self, packed_dir: str):
        d = Path(packed_dir)
        self.offsets = np.load(d / "offsets.npy")
        self.widths = np.load(d / "widths.npy")
        self.arrays = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in FIELDS}
        self.info = [e["info"] for e in json.loads((d / "episodes.json").read_text())]

    def episode(self, i: int) -> tuple[dict[str, np.ndarray], dict]:
        a, b = self.offsets[i], self.offsets[i + 1]
        width = lambda k: int(self.widths[i, WIDTH_OF[k]]) if k in WIDTH_OF else 1
        return {k: np.array(self.arrays[k][a:b, :width(k)]) for k in FIELDS}, dict(self.info[i])


class PackedEconomicsDataset(EconomicsDataset):
    """EconomicsDataset whose metadata entries (see index_entries) point into packed directories."""

    _stores: dict[str, _Store] = {}  # per process: opened lazily, after the loader workers fork

    def _load_episode(self, idx: int):
        packed_dir, i = self.metadata[idx]["packed"]
        if packed_dir not in self._stores:
            self._stores[packed_dir] = _Store(packed_dir)
        return self._stores[packed_dir].episode(i)
