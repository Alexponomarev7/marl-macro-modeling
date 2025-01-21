#!/usr/bin/env python3
import typing
from typing import Any
from pathlib import Path

import hydra
import torch
import clearml
import numpy as np
from loguru import logger
from torch.utils.data import DataLoader
from omegaconf import (
    DictConfig,
    OmegaConf,
)

from lib.dataset import Dataset
from lib.my_utils import set_global_seed
from lib.generate_dataset import run_generation_batch, get_run_id
from lib.envs.environment_base import AbstractEconomicEnv


def create_dataset_node(dataset_cfg: dict[str, Any]):
    logger.info("stage 1: data generation")
    workdir = Path(dataset_cfg['workdir'])
    workdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"WorkDir: {workdir}")
    for stage in ['train', 'val', 'test']:
        logger.info(f"generating stage: {stage}")
        stage_dir = workdir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        run_generation_batch(dataset_cfg[stage], dataset_cfg['envs'], stage_dir)


def create_model(model_cfg: dict[str, Any]) -> torch.nn.Module:
    return hydra.utils.instantiate(model_cfg)


def create_optimizer(optimizer_cfg: dict[str, Any], model: torch.nn.Module) -> torch.optim.Optimizer:
    return hydra.utils.instantiate(optimizer_cfg, params=model.parameters())


def create_criterion(criterion_cfg: dict[str, Any]) -> torch.nn.Module:
    return hydra.utils.instantiate(criterion_cfg)


def create_dataloader(data_root: Path, stage: str) -> DataLoader:
    return DataLoader(
        Dataset(data_path=data_root / "train"),
        batch_size=1,  # used for testing
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
    )


def validate_model(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        task: clearml.Task | None,
        epoch: int,
):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for states, actions, task_ids in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            task_ids = task_ids.to(device)

            predicted_actions = model(states, task_ids)
            loss = criterion(predicted_actions, actions)
            total_loss += loss.item() * states.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    if task:
        task.get_logger().report_scalar("loss", "val_loss", value=avg_loss, iteration=epoch)


def validate_model_with_env(
        model: torch.nn.Module,
        device: torch.device,
        task: clearml.Task | None,
        env: AbstractEconomicEnv,
        num_episodes: int = 10,
) -> float:
    """
    Validate model by analysing its performance in Environment (reward-based)

    Args:
        model: Neural network model to validate
        device: Device to run computations on
        task: ClearML task for logging
        env: Economic environment instance
        num_episodes: Number of episodes to evaluate

    Returns:
        float: Average cumulative reward across episodes
    """
    model.eval()
    total_rewards = []

    with torch.no_grad():
        for episode in range(num_episodes):
            episode_reward = 0
            state_dict, _ = env.reset()
            terminated = truncated = False

            while not (terminated or truncated):
                # Convert observation to tensor
                state = torch.FloatTensor([list(observation.values())]).to(device, dtype=torch.float64)
                task_id = torch.tensor([0]).to(device)

                # Get model's action prediction
                action = model(state, task_id)
                action = action.cpu().numpy().squeeze()

                # Take step in environment
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)

    avg_reward = sum(total_rewards) / num_episodes
    std_reward = np.std(total_rewards)

    logger.info(f"Validation Environment Performance:")
    logger.info(f"Average Reward: {avg_reward:.4f} ± {std_reward:.4f}")
    logger.info(f"Min Reward: {min(total_rewards):.4f}")
    logger.info(f"Max Reward: {max(total_rewards):.4f}")

    if task:
        task.get_logger().report_scalar(
            "environment_performance",
            "avg_reward",
            value=avg_reward,
            iteration=0
        )
        task.get_logger().report_scalar(
            "environment_performance",
            "std_reward",
            value=std_reward,
            iteration=0
        )

    return avg_reward


def create_train_node(train_cfg: dict[str, Any], dataset_cfg: dict[str, Any], task: clearml.Task | None):
    train_dataloader = create_dataloader(Path(train_cfg["data_root"]), "train")
    val_dataloader = create_dataloader(Path(train_cfg["data_root"]), "val")
    model = create_model(train_cfg["model"])
    optimizer = create_optimizer(train_cfg["optimizer"], model)
    criterion = create_criterion(train_cfg["loss"])
    device = torch.device(train_cfg["device"])

    # Create test environments for validation
    test_envs = []
    for env_config in dataset_cfg['test']['envs']:
        env_name = env_config['env_name']
        env_cfg = dataset_cfg['envs'][env_name]

        # Import the environment class dynamically
        env_class_path = env_cfg['env_class']
        module_path, class_name = env_class_path.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        env_class = getattr(module, class_name)

        # Instantiate parameters
        env_params = {
            k: hydra.utils.instantiate(v) if isinstance(v, dict) and '_target_' in v else v
            for k, v in env_cfg['params'].items()
        }

        # Create environment instance
        env = env_class(**env_params)
        test_envs.append((env_name, env))

    model = model.to(train_cfg["device"], dtype=torch.float64)
    model.train()

    for epoch in range(train_cfg["epochs"]):
        if task:
            task.set_progress(int(100 * epoch / train_cfg["epochs"]))

        total_loss = 0.0
        for states, actions, task_ids in train_dataloader:
            states = states.to(device, dtype=torch.float64)
            actions = actions.to(device, dtype=torch.float64)
            task_ids = task_ids.to(device)

            optimizer.zero_grad()
            predicted_actions = model(states, task_ids)

            loss = criterion(predicted_actions, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * states.size(0)

        avg_loss = total_loss / len(train_dataloader.dataset)
        logger.info(f"Epoch: {epoch}, Loss: {avg_loss:.4f}")
        if task:
            task.get_logger().report_scalar("loss", "train_loss", value=avg_loss, iteration=epoch)

        if (epoch + 1) % train_cfg["val_freq"] == 0:
            # Regular validation
            validate_model(model, criterion, val_dataloader, device, task, epoch=epoch)

            # Environment-based validation
            logger.info("Starting environment-based validation")
            for env_name, env in test_envs:
                logger.info(f"Validating on environment: {env_name}")
                avg_reward = validate_model_with_env(
                    model=model,
                    device=device,
                    task=task,
                    env=env,
                    num_episodes=train_cfg.get("val_episodes", 10)
                )
                if task:
                    task.get_logger().report_scalar(
                        f"environment_performance/{env_name}",
                        "avg_reward",
                        value=avg_reward,
                        iteration=epoch
                    )

            # Save model checkpoint
            torch.save(model.state_dict(), f"models/model_{epoch}.pth")
            model.train()



@hydra.main(config_name='pipeline.yaml', config_path="configs", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg.metadata.run_id = get_run_id()
    cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    metadata = cfg['metadata']
    track = metadata["track"]
    task = None
    if track:
        task = clearml.Task.init(
            project_name=metadata["project"],
            task_name=metadata["run_id"],
            task_type=clearml.Task.TaskTypes.training
        )
        task = typing.cast(clearml.Task, task)
        task.set_parameters_as_dict(cfg)

    set_global_seed(metadata['seed'])

    create_dataset_node(cfg['dataset'])
    create_train_node(cfg['train'], cfg['dataset'], task)

    if track:
        task.close()


if __name__ == '__main__':
    main()
