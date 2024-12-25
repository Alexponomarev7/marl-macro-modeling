#!/usr/bin/env python3
import wandb
import hydra
import hashlib
import time
from omegaconf import DictConfig, OmegaConf
from typing import Any
from loguru import logger
from pathlib import Path

import torch

from lib.generate_dataset import run_generation_batch
from lib.dataset import Dataset
from lib.models.transformer import AlgorithmDistillationTransformer

from torch.utils.data import DataLoader
from torch import optim

def get_run_id():
    return hashlib.md5(str(time.time()).encode()).hexdigest()

def create_dataset_node(dataset_cfg: dict[str, Any]):
    logger.info("stage 1: data generation")
    workdir = Path(dataset_cfg['workdir'])
    workdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"WorkDir: {workdir}")

    run_generation_batch(dataset_cfg['envs'], workdir)

def create_train_node(train_cfg: dict[str, Any], track: bool):
    dataloader = DataLoader(
        Dataset(data_path=Path(train_cfg["data_root"])),
        batch_size=1, # used for testing
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )

    model = AlgorithmDistillationTransformer(
        state_dim=1, action_dim=1, num_tasks=1
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    device = torch.device(train_cfg["device"])

    model = model.to(train_cfg["device"], dtype=torch.float64)
    model.train()

    criterion = torch.nn.MSELoss()
    for epoch in range(train_cfg["epochs"]):
        total_loss = 0.0
        for states, actions, task_ids in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            task_ids = task_ids.to(device)

            optimizer.zero_grad()
            predicted_actions = model(states, task_ids)
            
            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * states.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        logger.info(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
        if track:
            wandb.log({"loss": avg_loss, "epoch": epoch})


def set_global_seed(seed: int):
    import numpy as np
    import torch
    logger.info(f"setting global seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

@hydra.main(config_name='pipeline.yaml', config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.metadata.run_id = get_run_id()
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    metadata = cfg['metadata']
    track = metadata["track"]
    if track:
        wandb.init(project=metadata["project"], config=metadata)
        wandb.run.name = metadata["run_id"]
        wandb.run.save()

    set_global_seed(metadata['seed'])
    # post process the config
    logger.info(f"WorkDir: {metadata["workdir"]}")
    
    # todo(aponomarev): make this like true pipeline?
    create_dataset_node(cfg['dataset'])
    create_train_node(cfg['train'], track)

    if track:
        wandb.finish()

if __name__ == '__main__':
    main()