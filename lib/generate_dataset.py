import time
import json
import hydra
import hashlib
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import (
    Any,
    Dict,
)


def generate_hash(params: Dict) -> str:
    """
    Generate a hash from the sorted parameters.

    :param params: Dictionary of parameters
    :return: Short hash string
    """
    sorted_params = sorted(params.items())
    params_str = ','.join(f"{k}={v}" for k, v in sorted_params)
    return hashlib.md5(params_str.encode()).hexdigest()[:8]

def get_run_id():
    return hashlib.md5(str(time.time()).encode()).hexdigest()

def generate_env_data(env, num_steps: int = 1000) -> Dict:
    """
    Generate data from the given environment using its analytical solution.

    :param env: The environment instance
    :param num_steps: Number of steps to run the environment
    :param seed: Random seed for reproducibility
    :return: A dictionary containing:
        - 'env_params': The parameters of the environment.
        - 'tracks': A DataFrame containing the generated data with columns:
            - 'state': The state of the environment at each step.
            - 'reward': The reward received at each step.
            - 'done': A boolean indicating if the episode is done.
            - 'truncated': A boolean indicating if the episode was truncated.
            - 'info': Additional information from the environment at each step.
    """
    env.reset()

    data = []
    for _ in tqdm(range(num_steps)):
        state, reward, done, truncated, info = env.analytical_step()
        data.append({
            "state": state,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": info
        })

    return {
        'env_name': env.__class__.__name__,
        'env_params': env.params,
        'action_description': env.action_description,
        'state_description': env.state_description,
        'tracks': pd.DataFrame(data),
    }


def run_generation_batch(dataset_cfg: dict[str, Any], envs_cfg: dict[str, Any], workdir: Path):
    """
    Run batch data generation using Hydra config for all specified environments.
    """

    metadata = []
    idx = 1
    for env_config_metadata in dataset_cfg['envs']:
        num_steps = env_config_metadata["num_steps"]
        num_combinations = env_config_metadata["num_combinations"]
        env_config = envs_cfg[env_config_metadata["env_name"]]
        logger.info(f"Generating data for environment: {env_config['env_name']} ({num_combinations=}, {num_steps=})")

        # Generate parameter combinations for current environment
        params_list = []
        for _ in range(num_combinations):
            params = {}

            for param_name, param_spec in env_config["params"].items():
                params[param_name] = hydra.utils.instantiate(param_spec)

            params_list.append(params)

        # Run generation for each parameter combination
        logger.info(f"Generating {num_combinations} combinations")
        logger.info(f"Using {num_steps} steps per combination")

        for i, params in enumerate(params_list, 1):
            logger.info(f"Running combination {i}/{num_combinations}")
            logger.info(f"Parameters: {params}")

            env = hydra.utils.instantiate({"_target_": env_config["env_class"]} | params)
            try:
                env_data = generate_env_data(env, num_steps)
                params_hash = generate_hash(params)
                output_path = workdir / f"{idx}_{params_hash}.parquet"
                idx += 1

                env_data['tracks'].to_parquet(output_path)
                metadata.append({
                    'env_name': env_data['env_name'],
                    'env_params': env_data['env_params'],
                    'output_dir': str(output_path),
                })
            except Exception as e:
                logger.error(f"Error generating data, combination {i}")
                logger.exception(e)
                continue

    with open(workdir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
