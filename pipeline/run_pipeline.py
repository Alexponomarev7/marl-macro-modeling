#!/usr/bin/env python3
from typing import (
    Any,
    Optional,
)
from pathlib import Path

import hydra
import torch
import clearml
import numpy as np
from loguru import logger
from omegaconf import (
    DictConfig,
    OmegaConf,
)

import lightning as L
from lib.generate_dataset import run_generation_batch, run_generation_batch_dynare
from lib.dataset import Dataset

from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from lib.dataset import EconomicsDataset
from lib.my_utils import (
    set_global_seed,
    get_run_id,
)
from lib.envs.environment_base import AbstractEconomicEnv


def create_test_envs(dataset_cfg: dict[str, Any]) -> list[tuple[str, AbstractEconomicEnv]]:
    """Create test environments for validation."""
    test_envs = []
    for env_config in dataset_cfg['test']['envs']:
        env_name = env_config['env_name']
        env_cfg = dataset_cfg['envs'][env_name]

        module_path, class_name = env_cfg['env_class'].rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        env_class = getattr(module, class_name)

        env_params = {
            k: hydra.utils.instantiate(v) if isinstance(v, dict) and '_target_' in v else v
            for k, v in env_cfg['params'].items()
        }

        env = env_class(**env_params)
        test_envs.append((env_name, env))

    return test_envs


class DataModule(L.LightningDataModule):
    """PyTorch Lightning data module for handling datasets."""

    def __init__(self, data_root: Path, state_max_dim: int, batch_size: int = 32):
        """
        Initialize DataModule.

        Args:
            data_root: Root directory for datasets
            batch_size: Batch size for dataloaders
        """
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.state_max_dim = state_max_dim

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = EconomicsDataset(self.data_root / "train", self.state_max_dim)
            self.val_dataset = EconomicsDataset(self.data_root / "val", self.state_max_dim)
        if stage == "test":
            self.test_dataset = EconomicsDataset(self.data_root / "test", self.state_max_dim)

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )


class EconomicPolicyModel(L.LightningModule):
    """PyTorch Lightning module for training economic policies."""

    def __init__(
            self,
            model_cfg: dict[str, Any],
            optimizer_cfg: dict[str, Any],
            criterion_cfg: dict[str, Any],
            test_envs: list[tuple[str, AbstractEconomicEnv]],
            state_max_dim: int,
            val_episodes: int = 10
    ):
        """
        Initialize the policy model.

        Args:
            model_cfg: Model architecture configuration
            optimizer_cfg: Optimizer configuration
            criterion_cfg: Loss function configuration
            test_envs: List of test environments for validation
            val_episodes: Number of episodes for environment validation
        """
        super().__init__()
        self.save_hyperparameters(ignore=['test_envs'])

        self.model = hydra.utils.instantiate(model_cfg)
        self.criterion = hydra.utils.instantiate(criterion_cfg)
        self.optimizer_cfg = optimizer_cfg
        self.test_envs = test_envs
        self.val_episodes = val_episodes
        self.state_max_dim = state_max_dim
    def forward(self, states, task_ids):
        """Forward pass of the model."""
        return self.model(states, task_ids)

    def configure_optimizers(self):
        """Configure optimizer for training."""
        return hydra.utils.instantiate(
            self.optimizer_cfg,
            params=self.parameters()
        )

    def training_step(self, batch, batch_idx):
        """Training step logic."""
        states, actions, task_ids = batch
        predicted_actions = self(states, task_ids)
        loss = self.criterion(predicted_actions, actions)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step logic."""
        states, actions, task_ids = batch
        predicted_actions = self(states, task_ids)
        loss = self.criterion(predicted_actions, actions)

        self.log('val_loss', loss, on_epoch=True)
        return loss

    def on_fit_end(self):
        """Perform environment-based validation at the end of training."""
        if not self.test_envs:
            return

        for env_name, env in self.test_envs:
            avg_reward = self._validate_with_env(env)
            self.log(f'env_reward/{env_name}', avg_reward)

    def _validate_with_env(self, env: AbstractEconomicEnv) -> float:
        """
        Validate model performance in an environment.

        Args:
            env: Economic environment instance

        Returns:
            float: Average reward across episodes
        """
        total_rewards = []

        for _ in range(self.val_episodes):
            episode_reward = 0
            state_dict, _ = env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                # TODO(aponomarev): fix this, should pass the history of states
                state = torch.FloatTensor([list([value[0] for value in state_dict.values()])]).unsqueeze(0).to(self.device)
                state = torch.nn.functional.pad(state, (0, self.state_max_dim - state.shape[2], 0, 0, 0, 0), "constant", 0)
                task_id = torch.tensor([[0]]).to(self.device)

                with torch.no_grad():
                    action = self(state, task_id)
                action = action.cpu().numpy().squeeze()

                state_dict, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                break # TODO(aponomarev): fix this becase while loop is infinite

            total_rewards.append(episode_reward)

        return float(np.mean(total_rewards))

class DatasetGenerator:
    """Handles the creation and organization of datasets."""

    def __init__(self, dataset_cfg: dict[str, Any]):
        """
        Initialize DatasetCreator with configuration.
        Args:
            dataset_cfg: Configuration dictionary for dataset creation
        """
        self.cfg = dataset_cfg
        self.workdir = Path(dataset_cfg['workdir'])
        self.enabled = dataset_cfg['enabled']

    # todo: rm
    def create(self):
        """Generate datasets for all stages (train, val, test)."""
        if not self.enabled:
            logger.info("dataset generation is disabled, skipping")
            return

        logger.info("stage 1: data generation")
        self.workdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"WorkDir: {self.workdir}")

        for stage in ['train', 'val', 'test']:
            logger.info(f"generating stage: {stage}")
            stage_dir = self.workdir / stage
            stage_dir.mkdir(parents=True, exist_ok=True)

            stage_cfg = self.cfg[stage]
            if stage_cfg['type'] == 'envs':
                run_generation_batch(stage_cfg, self.cfg['envs'], stage_dir)
            elif stage_cfg['type'] == 'dynare':
                run_generation_batch_dynare(Path(stage_cfg['dynare_output_path']), stage_dir)
            else:
                raise ValueError(f"Unknown dataset type: {stage_cfg['type']}")


@hydra.main(config_name='pipeline.yaml', config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main entry point of the training pipeline."""
    # Initialize configuration
    cfg.metadata.run_id = get_run_id()
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    metadata = cfg['metadata']

    # Set up tracking if enabled
    task = None
    if metadata["track"]:
        task = clearml.Task.init(
            project_name=metadata["project"],
            task_name=metadata["run_id"],
            task_type=clearml.Task.TaskTypes.training
        )
        task.set_parameters_as_dict(cfg)

    # Set random seed
    set_global_seed(metadata['seed'])
    dataset_generator = DatasetGenerator(cfg['dataset'])
    dataset_generator.create()

    # Create test environments
    test_envs = create_test_envs(cfg['dataset'])


    # Initialize model and data module
    model = EconomicPolicyModel(
        model_cfg=cfg['train']['model'],
        optimizer_cfg=cfg['train']['optimizer'],
        criterion_cfg=cfg['train']['loss'],
        test_envs=test_envs,
        val_episodes=cfg['train'].get('val_episodes', 10),
        state_max_dim=cfg['train']['max_state_dim'],
    )

    data_module = DataModule(
        data_root=Path(cfg['train']['data_root']),
        state_max_dim=cfg['train']['max_state_dim'],
        batch_size=cfg['train'].get('batch_size', 32)
    )

    # Set up training
    trainer = L.Trainer(
        max_epochs=cfg['train']['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[
            ModelCheckpoint(
                dirpath='checkpoints',
                filename='{epoch}-{val_loss:.2f}',
                save_top_k=3,
                monitor='val_loss'
            )
        ],
        val_check_interval=cfg['train']['val_freq']
    )

    # Train model
    trainer.fit(model, data_module)

    if task:
        task.close()


if __name__ == '__main__':
    main()
