#!/usr/bin/env python3
from typing import (
    Any,
    Optional,
    cast,
)
import math
from pathlib import Path

import hydra
import torch
import clearml
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
from lib.dataset import EconomicsDataset, Tokenizer
from lib.models.transformer import symlog
from lib.envs.environment_base import AbstractEconomicEnv
from lib.generate_dataset import (
    DatasetGenerator,
    run_generation_batch,
    run_generation_batch_dynare,
)
from research.utils import PathStorage


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

    def __init__(
        self, data_root: Path, state_max_dim: int, action_max_dim: int,
        endogenous_max_dim: int, model_params_max_dim: int, max_seq_len: int,
        batch_size: int = 32
    ):
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
        self.endogenous_max_dim = endogenous_max_dim
        self.model_params_max_dim = model_params_max_dim
        self.max_seq_len = max_seq_len

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            self.train_dataset = EconomicsDataset(
                self.data_root / "train",
                self.state_max_dim,
                self.action_max_dim,
                self.endogenous_max_dim,
                self.model_params_max_dim,
                self.max_seq_len,
                random_window=True,
            )
            self.val_dataset = EconomicsDataset(
                self.data_root / "val",
                self.state_max_dim,
                self.action_max_dim,
                self.endogenous_max_dim,
                self.model_params_max_dim,
                self.max_seq_len,
                random_window=False,
            )
        if stage == "test":
            self.test_dataset = EconomicsDataset(
                self.data_root / "test",
                self.state_max_dim,
                self.action_max_dim,
                self.endogenous_max_dim,
                self.model_params_max_dim,
                self.max_seq_len,
                random_window=False,
            )

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
        state_max_dim: int,
        action_max_dim: int,
        endogenous_max_dim: int,
        test_envs: list[tuple[str, AbstractEconomicEnv]] = [],
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
        # elementwise, for masking
        self.criterion = hydra.utils.instantiate(criterion_cfg, reduction="none")
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.test_envs = test_envs
        self.val_episodes = val_episodes
        self.val_steps = val_steps
        self.state_max_dim = state_max_dim
        self.action_max_dim = action_max_dim
        self.endogenous_max_dim = endogenous_max_dim

    def forward(self, states, states_info, actions, actions_info, rewards, task_ids, model_params):
        """Forward pass matching the transformer's interface"""
        return self.model(
            states=states,
            states_info=states_info,
            actions=actions,
            actions_info=actions_info,
            rewards=rewards,
            task_ids=task_ids,
            model_params=model_params,
        )

    def configure_optimizers(self):
        """Optimizer from config, with a per-step linear warmup then cosine decay to eta_min."""
        optimizer = hydra.utils.instantiate(
            self.optimizer_cfg,
            params=self.parameters()
        )
        warmup = int(self.scheduler_cfg["warmup_steps"])
        total = max(int(self.trainer.estimated_stepping_batches), warmup + 1)
        floor = float(self.scheduler_cfg["eta_min"]) / optimizer.defaults["lr"]

        def lr_factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = min((step - warmup) / (total - warmup), 1.0)
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    @staticmethod
    def _sanitize(x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), min=-1000.0, max=1000.0)

    def _masked_loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per_elem = self.criterion(pred, target)
        mask = mask.to(per_elem.dtype)
        return (per_elem * mask).sum() / mask.sum().clamp(min=1.0)

    def _shared_step(self, batch, stage: str):
        # step t sees (s_t, a_{t-1}, r_{t-1}) and is scored on a_t
        predicted_actions, pinn_preds = self(
            states=self._sanitize(batch['states']),
            states_info=batch['states_info'],
            actions=self._sanitize(batch['prev_actions']),
            actions_info=batch['actions_info'],
            rewards=self._sanitize(batch['prev_reward']),
            task_ids=batch['task_id'],
            model_params=batch['model_params'],
        )

        valid_steps = batch['attention_mask'].unsqueeze(-1)  # [bs, seq, 1]
        action_mask = valid_steps & (batch['actions_info'] != 0).unsqueeze(1)
        # NMSE: errors in units of each episode's action scale
        scale = batch['action_scale']  # [bs, seq, action_dim]
        loss = self._masked_loss(predicted_actions / scale, self._sanitize(batch['actions']) / scale, action_mask)
        self.log(f'{stage}_action_loss', loss, on_step=(stage == 'train'), on_epoch=True)

        if pinn_preds is not None and (batch['endogenous_info'] != 0).any():
            endo_mask = valid_steps & (batch['endogenous_info'] != 0).unsqueeze(1)
            pinn_loss = self._masked_loss(pinn_preds, symlog(self._sanitize(batch['endogenous'])), endo_mask)
            self.log(f'{stage}_pinn_loss', pinn_loss, on_step=(stage == 'train'), on_epoch=True)
            loss = loss + pinn_loss

        assert not torch.isnan(loss)
        self.log(f'{stage}_loss', loss, on_step=(stage == 'train'), on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')


@hydra.main(config_name='pipeline.yaml', config_path="configs", version_base=None)
def main(hydra_cfg: DictConfig) -> None:
    """Main entry point of the training pipeline."""
    # Initialize configuration
    if 'run_id' not in hydra_cfg.metadata or not hydra_cfg.metadata.run_id:
        hydra_cfg.metadata.run_id = get_run_id()

    logger.info(f"run id: {hydra_cfg.metadata.run_id}")

    cfg = cast(dict, OmegaConf.to_container(hydra_cfg, resolve=True, throw_on_missing=True))
    metadata = cfg['metadata']

    task = None
    if metadata["track"]:
        task = clearml.Task.init(
            project_name=metadata["project"],
            task_name=metadata["comment"],
            tags=[metadata["run_id"]],
            task_type=clearml.Task.TaskTypes.training
        )
        task.set_parameters_as_dict(cfg)

    set_global_seed(metadata['seed'])
    dataset_generator = DatasetGenerator(cfg['dataset'])
    dataset_generator.create()

    # Automatically compute num_tasks from Tokenizer's ENV_MAPPING
    tokenizer = Tokenizer()
    num_tasks = tokenizer.num_tasks
    cfg['train']['model']['num_tasks'] = num_tasks

    test_envs = create_test_envs(cfg['dataset'])
    model = EconomicPolicyModel(
        model_cfg=cfg['train']['model'],
        optimizer_cfg=cfg['train']['optimizer'],
        scheduler_cfg=cfg['train']['scheduler'],
        criterion_cfg=cfg['train']['loss'],
        test_envs=test_envs,
        val_episodes=cfg['train'].get('val_episodes', 10),
        state_max_dim=cfg['train']['max_state_dim'],
        action_max_dim=cfg['train']['max_action_dim'],
        endogenous_max_dim=cfg['train']['max_endogenous_dim'],
    )

    data_module = DataModule(
        data_root=Path(cfg['train']['data_root']),
        state_max_dim=cfg['train']['max_state_dim'],
        action_max_dim=cfg['train']['max_action_dim'],
        endogenous_max_dim=cfg['train']['max_endogenous_dim'],
        model_params_max_dim=cfg['train']['max_model_params_dim'],
        max_seq_len=cfg['train']['max_seq_len'],
        batch_size=cfg['train'].get('batch_size', 32),
    )

    checkpoint_dir = Path('checkpoints') / metadata['run_id']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = L.Trainer(
        max_epochs=cfg['train']['epochs'],
        gradient_clip_val=cfg['train']['gradient_clip_val'],
        accelerator=cfg['train'].get('device', 'auto'),
        devices=1,
        callbacks=[
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename='model-{epoch:03d}',
                save_top_k=3,
                monitor='val_loss',
                save_last=True
            )
        ],
        check_val_every_n_epoch=cfg['train']['val_freq'],
        # val_check_interval=cfg['train']['val_freq'],
        # logger=L.pytorch.loggers.ClearMLLogger(task=task) if task else True
    )
    trainer.fit(model, data_module)

    if task:
        task.close()


if __name__ == '__main__':
    main()
