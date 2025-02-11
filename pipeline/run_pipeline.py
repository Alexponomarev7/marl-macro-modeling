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
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import ModelCheckpoint

from lib.my_utils import (
    set_global_seed,
    get_run_id,
)
from lib.dataset import EconomicsDataset
from lib.envs.environment_base import AbstractEconomicEnv
from lib.generate_dataset import (
    run_generation_batch,
    run_generation_batch_dynare,
)


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

    def __init__(self, data_root: Path, state_max_dim: int, action_max_dim: int, batch_size: int = 32):
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
        self.action_max_dim = action_max_dim

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = EconomicsDataset(self.data_root / "train", self.state_max_dim, self.action_max_dim)
            self.val_dataset = EconomicsDataset(self.data_root / "val", self.state_max_dim, self.action_max_dim)
        if stage == "test":
            self.test_dataset = EconomicsDataset(self.data_root / "test", self.state_max_dim, self.action_max_dim)

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
            scheduler_cfg: dict[str, Any],
            criterion_cfg: dict[str, Any],
            test_envs: list[tuple[str, AbstractEconomicEnv]],
            state_max_dim: int,
            action_max_dim: int,
            val_episodes: int = 10,
            val_steps: int = 1000,
    ):
        """
        Initialize the policy model.

        Args:
            model_cfg: Model architecture configuration
            optimizer_cfg: Optimizer configuration
            criterion_cfg: Loss function configuration
            test_envs: List of test environments for validation
            val_episodes: Number of episodes for environment validation
            val_steps: Number of steps within validation episode
        """
        super().__init__()
        self.save_hyperparameters(ignore=['test_envs'])

        self.model = hydra.utils.instantiate(model_cfg)
        self.criterion = hydra.utils.instantiate(criterion_cfg)
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.test_envs = test_envs
        self.val_episodes = val_episodes
        self.val_steps = val_steps
        self.state_max_dim = state_max_dim
        self.action_max_dim = action_max_dim

    def forward(self, states, states_info, actions, actions_info, rewards, task_ids, padding_mask=None):
        """Forward pass matching the transformer's interface"""
        return self.model(
            states=states,
            states_info=states_info,
            actions=actions,
            actions_info=actions_info,
            rewards=rewards,
            task_ids=task_ids, 
            # padding_mask=padding_mask
        )

    def configure_optimizers(self):
        """Configure optimizer for training."""
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=self.parameters()
        )
        scheduler = hydra.utils.instantiate(self.scheduler_cfg, optimizer=optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Updated training step to handle the new batch format"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['reward']
        task_ids = batch['task_id']
        states_info = batch['states_info']
        actions_info = batch['actions_info']

        # weird bug with nan values
        states = torch.clamp(torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0), min=-1000.0, max=1000.0)
        actions = torch.clamp(torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0), min=-1000.0, max=1000.0)
        rewards = torch.clamp(torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0), min=-1000.0, max=1000.0)

        predicted_actions = self(
            states=states,
            states_info=states_info,
            actions=actions,
            actions_info=actions_info,
            rewards=rewards,
            task_ids=task_ids,
        )

        # predicted_actions shape should be [batch_size, seq_length - 1, action_dim]
        # actions shape: [batch_size, seq_length, action_dim]
        loss = self.criterion(predicted_actions, actions[:, 1:, :])
        assert not torch.isnan(loss)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Updated validation step to match training step"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['reward']
        task_ids = batch['task_id']
        states_info = batch['states_info']
        actions_info = batch['actions_info']

        # weird bug with nan values
        states = torch.clamp(torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0), min=-1000.0, max=1000.0)
        actions = torch.clamp(torch.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0), min=-1000.0, max=1000.0)
        rewards = torch.clamp(torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0), min=-1000.0, max=1000.0)

        predicted_actions = self(
            states=states,
            actions=actions,
            rewards=rewards,
            task_ids=task_ids,
            states_info=states_info,
            actions_info=actions_info,
        )

        loss = self.criterion(predicted_actions, actions[:, 1:, :])
        self.log('val_loss', loss, on_epoch=True)
        return loss

    # def on_fit_end(self):
    #     """Perform environment-based validation at the end of training."""
    #     if not self.test_envs:
    #         return

    #     for env_name, env in self.test_envs:
    #         avg_reward = self._validate_with_env(env)
    #         self.log(f'env_reward/{env_name}', avg_reward)

    def _validate_with_env(self, env: AbstractEconomicEnv) -> float:
        """Environment dynamic validation"""
        total_rewards = []

        for _ in range(self.val_episodes):
            episode_reward = 0
            state_dict, _ = env.reset()
            terminated = truncated = False

            # Keep track of sequence history
            states_history = []
            actions_history = []
            rewards_history = []

            # Get initial state
            current_state = torch.FloatTensor([
                list([value[0] for value in state_dict.values()])
            ]).to(self.device)

            for _ in range(self.val_steps):
                # Prepare input for model
                if len(states_history) > self.model.max_seq_len:
                    states_history = states_history[-self.model.max_seq_len:]
                    actions_history = actions_history[-self.model.max_seq_len:]
                    rewards_history = rewards_history[-self.model.max_seq_len:]

                # Create states tensor
                if states_history:  # If we have previous states
                    states = torch.cat(states_history, dim=0).unsqueeze(0)
                    states = torch.cat([states, current_state.unsqueeze(0)], dim=1)
                else:  # First step
                    states = current_state.unsqueeze(0)  # [1, 1, state_dim]

                # Pad states if necessary
                states = torch.nn.functional.pad(
                    states,
                    (0, self.state_max_dim - states.shape[2]),
                    "constant",
                    0
                )

                # Create actions and rewards sequences
                if actions_history:
                    actions = torch.stack(actions_history, dim=0).unsqueeze(0)
                    actions = torch.nn.functional.pad(
                        actions,
                        (0, self.action_max_dim - actions.shape[-1]),
                        "constant",
                        0
                    )
                    rewards = torch.tensor(rewards_history).unsqueeze(0).unsqueeze(-1)
                else:
                    actions = torch.zeros(1, 0, self.action_max_dim).to(self.device)
                    rewards = torch.zeros(1, 0, 1).to(self.device)

                task_ids = torch.tensor([0]).to(self.device)
                attention_mask = torch.ones(1, states.shape[1], dtype=torch.bool).to(self.device)

                # Ensure all tensors are float32
                states = states.to(torch.float32)
                actions = actions.to(torch.float32)
                rewards = rewards.to(torch.float32)

                with torch.no_grad():
                    predicted_actions = self(
                        states=states,
                        actions=actions,
                        rewards=rewards,
                        task_ids=task_ids,
                        # padding_mask=~attention_mask
                    )
                    action = predicted_actions[0, -1].cpu().numpy()

                # Update histories
                states_history.append(current_state)

                # Take action in environment
                state_dict, reward, terminated, truncated, _ = env.step(action)

                # Prepare for next step
                current_state = torch.FloatTensor([
                    list([value[0] for value in state_dict.values()])
                ]).to(self.device)

                actions_history.append(torch.from_numpy(action).to(self.device))
                rewards_history.append(reward)
                episode_reward += reward

                if terminated or truncated:
                    break

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
        scheduler_cfg=cfg['train']['scheduler'],
        criterion_cfg=cfg['train']['loss'],
        test_envs=test_envs,
        val_episodes=cfg['train'].get('val_episodes', 10),
        state_max_dim=cfg['train']['max_state_dim'],
        action_max_dim=cfg['train']['max_action_dim'],
    )

    data_module = DataModule(
        data_root=Path(cfg['train']['data_root']),
        state_max_dim=cfg['train']['max_state_dim'],
        action_max_dim=cfg['train']['max_action_dim'],
        batch_size=cfg['train'].get('batch_size', 32),
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
